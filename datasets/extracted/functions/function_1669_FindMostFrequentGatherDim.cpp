#include "xla/service/all_gather_combiner.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/hlo/utils/hlo_sharding_util.h"
#include "xla/layout.h"
#include "xla/service/collective_combiner_utils.h"
#include "xla/service/hlo_domain_map.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace {
int64_t FindMostFrequentGatherDim(
    absl::Span<HloInstruction* const> to_combine) {
  assert(!to_combine.empty());
  int64_t min_rank = std::numeric_limits<int64_t>::max();
  std::vector<int64_t> frequency;
  for (const HloInstruction* it : to_combine) {
    int64_t dim = Cast<HloAllGatherInstruction>(it)->all_gather_dimension();
    frequency.resize(std::max(dim + 1, static_cast<int64_t>(frequency.size())),
                     0);
    frequency[dim]++;
    min_rank = std::min(min_rank, it->shape().rank());
  }
  int64_t most_frequent_dim = std::distance(
      frequency.begin(), std::max_element(frequency.begin(), frequency.end()));
  return most_frequent_dim < min_rank ? most_frequent_dim : 0;
}
absl::Status CombineAllGathers(absl::Span<HloInstruction* const> to_combine,
                               bool combine_by_dim) {
  if (to_combine.size() < 2) {
    return absl::OkStatus();
  }
  VLOG(1) << "Combined " << to_combine.size() << " AllGather ops";
  HloComputation& computation = *to_combine.back()->parent();
  std::vector<HloInstruction*> operands;
  std::vector<std::optional<std::vector<int64_t>>> operand_permutations;
  std::vector<Shape> output_shapes;
  int64_t most_frequent_dim = FindMostFrequentGatherDim(to_combine);
  VLOG(1) << "Combining set";
  for (HloInstruction* hlo : to_combine) {
    VLOG(1) << "Set element: " << hlo->ToString();
    TF_RET_CHECK(hlo->opcode() == HloOpcode::kAllGather);
    const auto* ag = Cast<HloAllGatherInstruction>(hlo);
    TF_RET_CHECK(hlo->operand_count() == 1);
    TF_RET_CHECK(hlo->shape().IsArray());
    TF_RET_CHECK(!combine_by_dim ||
                 ag->all_gather_dimension() == most_frequent_dim);
    HloInstruction* operand = hlo->operands().front();
    operands.push_back(operand);
    operand_permutations.emplace_back();
    output_shapes.push_back(hlo->shape());
    if (ag->all_gather_dimension() != most_frequent_dim) {
      const Shape& operand_shape = operand->shape();
      auto& perm = operand_permutations.back();
      perm = std::vector<int64_t>(operand_shape.rank());
      std::iota(perm->begin(), perm->end(), 0);
      std::swap((*perm)[most_frequent_dim],
                (*perm)[ag->all_gather_dimension()]);
      operands.back() =
          computation.AddInstruction(HloInstruction::CreateBitcast(
              ShapeUtil::PermuteDimensions(*perm, operand_shape), operand));
      output_shapes.back() = ShapeUtil::PermuteDimensions(*perm, hlo->shape());
    }
  }
  HloInstruction* combined;
  combined = computation.AddInstruction(HloInstruction::CreateAllGather(
      ShapeUtil::MakeTupleShape(output_shapes), operands, most_frequent_dim,
      to_combine.front()->device_list(),
      false, to_combine.front()->channel_id(),
      Cast<HloAllGatherInstruction>(to_combine.front())
          ->use_global_device_ids()));
  combined->set_sharding(
      hlo_sharding_util::CreateTupleSharding(combined->shape(), to_combine));
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
 std::string& AllGatherCombiner::GetGroupKeyExtraArgs(
    AllGatherCombiner::GroupKey& key) {
  return std::get<6>(key);
}
 std::optional<AllGatherCombiner::GroupKey>
AllGatherCombiner::CombineKey(const HloInstruction* instruction,
                              const HloDomainMap& domain_map,
                              bool combine_by_dim,
                              bool combine_different_dtypes) {
  if (instruction->opcode() != HloOpcode::kAllGather) {
    return std::nullopt;
  }
  std::vector<std::vector<int64_t>> replica_groups;
  const auto* ag = Cast<HloAllGatherInstruction>(instruction);
  replica_groups.reserve(ag->replica_groups().size());
  for (const ReplicaGroup& replica_group : ag->replica_groups()) {
    replica_groups.push_back(
        std::vector<int64_t>(replica_group.replica_ids().begin(),
                             replica_group.replica_ids().end()));
  }
  int64_t ag_dim_key = combine_by_dim ? ag->all_gather_dimension() : -1;
  PrimitiveType data_type = combine_different_dtypes
                                ? PRIMITIVE_TYPE_INVALID
                                : ag->shape().element_type();
  return GroupKey{ag_dim_key,
                  domain_map.GetDomainMetadataId(ag),
                  ag->channel_id().has_value(),
                  ag->use_global_device_ids(),
                  data_type,
                  replica_groups,
                  ""};
}
AllGatherCombiner::AllGatherCombiner(int64_t combine_threshold_in_bytes,
                                     int64_t combine_threshold_count,
                                     bool combine_by_dim,
                                     bool combine_different_dtypes)
    : combine_threshold_in_bytes_(combine_threshold_in_bytes),
      combine_threshold_count_(combine_threshold_count),
      combine_by_dim_(combine_by_dim),
      combine_different_dtypes_(combine_different_dtypes) {}
absl::StatusOr<bool> AllGatherCombiner::RunWithKeyCombiner(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    absl::FunctionRef<std::optional<AllGatherCombiner::GroupKey>(
        const HloInstruction*, const HloDomainMap&, bool, bool)>
        combine_key) {
  VLOG(1) << "Running AllGatherCombiner with threshold of "
          << combine_threshold_in_bytes_ << " bytes";
  if (combine_threshold_in_bytes_ <= 0 || combine_threshold_count_ <= 0) {
    VLOG(1) << "Skip AllGatherCombiner because the threshold is zero";
    return false;
  }
  if (hlo_query::ContainsLayoutConstrainedCollective(*module,
                                                     HloOpcode::kAllGather)) {
    VLOG(1) << "Skip AllGatherCombiner because the module contains "
               "all-gather with constrained layouts";
    return false;
  }
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(auto domain_map, HloDomainMap::Create(computation, ""));
    auto key_fn = [&](const HloInstruction* instruction) {
      return combine_key(instruction, *domain_map, combine_by_dim_,
                         combine_different_dtypes_);
    };
    auto combine_fn =
        [&](absl::Span<HloInstruction* const> to_combine) -> absl::Status {
      return CombineAllGathers(to_combine, combine_by_dim_);
    };
    TF_ASSIGN_OR_RETURN(
        bool computation_changed,
        CombineInstructionsByKey<GroupKey>(computation, key_fn, combine_fn,
                                           combine_threshold_in_bytes_,
                                           combine_threshold_count_));
    changed |= computation_changed;
  }
  return changed;
}
absl::StatusOr<bool> AllGatherCombiner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_ASSIGN_OR_RETURN(
      bool changed, RunWithKeyCombiner(module, execution_threads, CombineKey));
  return changed;
}
}  