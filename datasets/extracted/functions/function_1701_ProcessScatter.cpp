#include "xla/service/spmd/spmd_prepare.h"
#include <memory>
#include <optional>
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_sharding_util.h"
#include "xla/service/call_graph.h"
#include "xla/service/pattern_matcher.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace spmd {
namespace {
absl::StatusOr<bool> ProcessScatter(HloInstruction* hlo,
                                    const CallGraph& call_graph) {
  if (hlo->opcode() != HloOpcode::kScatter) {
    return false;
  }
  HloScatterInstruction* scatter = Cast<HloScatterInstruction>(hlo);
  HloComputation* computation = hlo->parent();
  if (scatter->scatter_operand_count() > 1) {
    return false;
  }
  HloInstruction* operand = scatter->scatter_operands()[0];
  HloInstruction* indices = scatter->scatter_indices();
  HloInstruction* updates = scatter->scatter_updates()[0];
  if (operand->opcode() != HloOpcode::kAdd ||
      indices->opcode() != HloOpcode::kConcatenate ||
      indices->operand_count() != 2 ||
      updates->opcode() != HloOpcode::kConcatenate ||
      updates->operand_count() != 2 ||
      !Match(scatter->to_apply()->root_instruction(),
             match::AddAnyOrder(match::Parameter(0), match::Parameter(1)))) {
    return false;
  }
  const auto& dnums = scatter->scatter_dimension_numbers();
  auto get_parallel_dims_for_scatter = [&dnums, &call_graph](
                                           const HloInstruction* operand,
                                           const HloInstruction* indices,
                                           const HloInstruction* updates) {
    std::vector<int64_t> slice_sizes = hlo_sharding_util::GetScatterSliceSize(
        operand->shape(), updates->shape(), dnums);
    int64_t index_vector_dim = dnums.index_vector_dim();
    const auto& index_map = dnums.scatter_dims_to_operand_dims();
    return hlo_sharding_util::GetGatherScatterBatchParallelDims(
        operand, indices, slice_sizes, index_vector_dim, index_map, call_graph);
  };
  if (get_parallel_dims_for_scatter(operand, indices, updates).has_value()) {
    return false;
  }
  HloInstruction* lhs_indices = indices->mutable_operand(0);
  HloInstruction* rhs_indices = indices->mutable_operand(1);
  HloInstruction* lhs_updates = updates->mutable_operand(0);
  HloInstruction* rhs_updates = updates->mutable_operand(1);
  std::optional<hlo_sharding_util::GatherScatterParallelDims> lhs_parallel_dims;
  std::optional<hlo_sharding_util::GatherScatterParallelDims> rhs_parallel_dims;
  lhs_parallel_dims =
      get_parallel_dims_for_scatter(operand, lhs_indices, lhs_updates);
  if (!lhs_parallel_dims.has_value()) {
    return false;
  }
  rhs_parallel_dims =
      get_parallel_dims_for_scatter(operand, rhs_indices, rhs_updates);
  if (!rhs_parallel_dims.has_value()) {
    return false;
  }
  if (lhs_parallel_dims->operand_parallel_dims !=
          rhs_parallel_dims->operand_parallel_dims ||
      lhs_parallel_dims->indices_parallel_dims !=
          rhs_parallel_dims->indices_parallel_dims) {
    return false;
  }
  if (lhs_parallel_dims->operand_parallel_dims.size() !=
      lhs_parallel_dims->indices_parallel_dims.size()) {
    return false;
  }
  HloInstruction* lhs_operand = operand->mutable_operand(0);
  HloInstruction* rhs_operand = operand->mutable_operand(1);
  bool any_sharded_parallel_dim = false;
  if (!lhs_operand->has_sharding() || !rhs_operand->has_sharding() ||
      !lhs_indices->has_sharding() || !rhs_indices->has_sharding()) {
    return false;
  }
  for (int i = 0; i < lhs_parallel_dims->operand_parallel_dims.size(); ++i) {
    if (lhs_operand->sharding().IsTiled() &&
        lhs_operand->sharding().tile_assignment().dim(
            lhs_parallel_dims->operand_parallel_dims[i]) != 1 &&
        lhs_indices->sharding().tile_assignment().dim(
            lhs_parallel_dims->indices_parallel_dims[i]) != 1) {
      any_sharded_parallel_dim = true;
      break;
    }
  }
  if (!any_sharded_parallel_dim) {
    return false;
  }
  HloInstruction* scatter0 =
      computation->AddInstruction(HloInstruction::CreateScatter(
          scatter->shape(), operand, lhs_indices, lhs_updates,
          scatter->to_apply(), dnums, false, false));
  scatter0->set_metadata(scatter->metadata());
  scatter0->set_sharding(scatter->sharding());
  HloInstruction* scatter1 =
      computation->AddInstruction(HloInstruction::CreateScatter(
          scatter->shape(), scatter0, rhs_indices, rhs_updates,
          scatter->to_apply(), dnums, false, false));
  scatter1->set_metadata(scatter->metadata());
  scatter1->set_sharding(scatter->sharding());
  TF_RETURN_IF_ERROR(scatter->ReplaceAllUsesWith(scatter1));
  return true;
}
absl::StatusOr<bool> RunOnComputation(HloComputation* computation,
                                      const CallGraph& call_graph) {
  bool changed = false;
  for (HloInstruction* hlo : computation->MakeInstructionPostOrder()) {
    if (!hlo->has_sharding()) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(bool scatter_changed, ProcessScatter(hlo, call_graph));
    if (scatter_changed) {
      changed = true;
      continue;
    }
  }
  return changed;
}
}  
absl::StatusOr<bool> SpmdPrepare::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  for (auto comp : module->computations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool comp_changed, RunOnComputation(comp, *call_graph));
    changed |= comp_changed;
  }
  return changed;
}
}  
}  