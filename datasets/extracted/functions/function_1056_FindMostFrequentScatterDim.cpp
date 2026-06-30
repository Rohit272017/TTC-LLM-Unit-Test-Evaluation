#include "xla/service/reduce_scatter_combiner.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_reachability.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/all_reduce_key.h"
#include "xla/service/collective_combiner_utils.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/hlo_domain_map.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
namespace xla {
namespace {
int64_t FindMostFrequentScatterDim(
    absl::Span<HloInstruction* const> to_combine) {
  assert(!to_combine.empty());
  int64_t min_rank = std::numeric_limits<int64_t>::max();
  std::vector<int64_t> frequency;
  for (const HloInstruction* it : to_combine) {
    int64_t dim = Cast<HloReduceScatterInstruction>(it)->scatter_dimension();
    frequency.resize(std::max(dim + 1, static_cast<int64_t>(frequency.size())),
                     0);
    frequency[dim]++;
    min_rank = std::min(min_rank, it->shape().rank());
  }
  int64_t most_frequent_dim = std::distance(
      frequency.begin(), std::max_element(frequency.begin(), frequency.end()));
  return most_frequent_dim < min_rank ? most_frequent_dim : 0;
}
using ReduceScatterKey =
    std::tuple<AllReduceKey,  int64_t>;
absl::Status CombineReduceScatters(
    absl::Span<HloInstruction* const> to_combine) {
  if (to_combine.size() < 2) {
    return absl::OkStatus();
  }
  VLOG(1) << "Combined " << to_combine.size() << " reduce-scatter ops";
  HloComputation& computation = *to_combine.back()->parent();
  HloComputation* reduction = to_combine[0]->to_apply();
  std::optional<ReductionKind> first_reduction_kind =
      MatchReductionComputation(reduction);
  TF_RET_CHECK(first_reduction_kind);
  std::vector<HloInstruction*> operands;
  std::vector<std::optional<std::vector<int64_t>>> operand_permutations;
  std::vector<Shape> output_shapes;
  int64_t most_frequent_dim = FindMostFrequentScatterDim(to_combine);
  VLOG(1) << "Combining set";
  for (HloInstruction* hlo : to_combine) {
    VLOG(1) << "Set element: " << hlo->ToString();
    TF_RET_CHECK(hlo->opcode() == HloOpcode::kReduceScatter);
    const auto* rs = Cast<HloReduceScatterInstruction>(hlo);
    TF_RET_CHECK(hlo->operands().size() == 1);
    std::optional<ReductionKind> reduction_kind =
        MatchReductionComputation(hlo->to_apply());
    TF_RET_CHECK(reduction_kind);
    TF_RET_CHECK(*reduction_kind == *first_reduction_kind);
    TF_RET_CHECK(hlo->shape().IsArray());
    HloInstruction* operand = hlo->operands().front();
    operands.push_back(operand);
    operand_permutations.emplace_back();
    output_shapes.push_back(hlo->shape());
    if (rs->scatter_dimension() != most_frequent_dim) {
      const Shape& operand_shape = operand->shape();
      auto& perm = operand_permutations.back();
      perm = std::vector<int64_t>(operand_shape.rank());
      std::iota(perm->begin(), perm->end(), 0);
      std::swap((*perm)[most_frequent_dim], (*perm)[rs->scatter_dimension()]);
      operands.back() =
          computation.AddInstruction(HloInstruction::CreateBitcast(
              ShapeUtil::PermuteDimensions(*perm, operand_shape), operand));
      output_shapes.back() = ShapeUtil::PermuteDimensions(*perm, hlo->shape());
    }
  }
  HloInstruction* combined;
  TF_RET_CHECK(operands.size() >= 2);
  combined = computation.AddInstruction(HloInstruction::CreateReduceScatter(
      ShapeUtil::MakeTupleShape(output_shapes), operands, reduction,
      to_combine.front()->device_list(),
      false, to_combine.front()->channel_id(),
      Cast<HloReduceScatterInstruction>(to_combine.front())
          ->use_global_device_ids(),
      most_frequent_dim));
  if (to_combine.front()->has_sharding()) {
    combined->set_sharding(to_combine.front()->sharding());
  }
  VLOG(1) << "Replacing with : " << combined->ToString();
  for (int64_t i = 0; i < to_combine.size(); ++i) {
    HloInstruction* replacement = computation.AddInstruction(
        HloInstruction::CreateGetTupleElement(combined, i));
    if (operand_permutations[i]) {
      replacement = computation.AddInstruction(HloInstruction::CreateBitcast(
          ShapeUtil::PermuteDimensions(*operand_permutations[i],
                                       replacement->shape()),
          replacement));
    }
    TF_RETURN_IF_ERROR(
        computation.ReplaceInstruction(to_combine[i], replacement));
  }
  return absl::OkStatus();
}
}  
ReduceScatterCombiner::ReduceScatterCombiner(int64_t combine_threshold_in_bytes,
                                             int64_t combine_threshold_count,
                                             bool combine_by_dim)
    : combine_threshold_in_bytes_(combine_threshold_in_bytes),
      combine_threshold_count_(combine_threshold_count),
      combine_by_dim_(combine_by_dim) {}
absl::StatusOr<bool> ReduceScatterCombiner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(1) << "Running ReduceScatterCombiner with threshold of "
          << combine_threshold_in_bytes_ << " bytes";
  if (combine_threshold_in_bytes_ <= 0 || combine_threshold_count_ <= 0) {
    VLOG(1) << "Skip ReduceScatterCombiner because the threshold is zero";
    return false;
  }
  if (hlo_query::ContainsLayoutConstrainedCollective(
          *module, HloOpcode::kReduceScatter)) {
    VLOG(1) << "Skip ReduceScatterCombiner because the module contains "
               "reduce-scatter with constrained layouts";
    return false;
  }
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(auto domain_map, HloDomainMap::Create(computation, ""));
    auto key_fn = [&domain_map, this](const HloInstruction* instruction)
        -> std::optional<ReduceScatterKey> {
      auto* rs = DynCast<HloReduceScatterInstruction>(instruction);
      std::optional<AllReduceKey> key =
          GetAllReduceKey(instruction, domain_map.get());
      if (!rs || !key) {
        return std::nullopt;
      }
      if (!MatchReductionComputation(rs->to_apply())) {
        return std::nullopt;
      }
      int64_t rs_dim_key = this->combine_by_dim_ ? rs->scatter_dimension() : -1;
      return ReduceScatterKey{std::move(*key), rs_dim_key};
    };
    TF_ASSIGN_OR_RETURN(
        bool computation_changed,
        CombineInstructionsByKey<ReduceScatterKey>(
            computation, key_fn, &CombineReduceScatters,
            combine_threshold_in_bytes_, combine_threshold_count_));
    changed |= computation_changed;
  }
  return changed;
}
}  