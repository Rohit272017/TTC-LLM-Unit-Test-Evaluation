#include "xla/service/spmd/whole_graph_manual_pass.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace {
bool ShouldClearInstruction(HloInstruction* inst) {
  return inst->opcode() != HloOpcode::kParameter &&
         inst != inst->parent()->root_instruction() &&
         inst->opcode() != HloOpcode::kPartitionId &&
         DynCast<HloCollectiveInstruction>(inst) == nullptr &&
         !inst->HasSideEffectNoRecurse();
}
absl::StatusOr<bool> RunOnComputation(HloComputation* computation) {
  bool changed = false;
  for (HloInstruction* inst : computation->instructions()) {
    if (ShouldClearInstruction(inst)) {
      inst->clear_sharding();
      changed = true;
      continue;
    }
    if (inst->shape().IsTuple()) {
      inst->set_sharding(
          HloSharding::SingleTuple(inst->shape(), HloSharding::Manual()));
      changed = true;
    } else {
      inst->set_sharding(HloSharding::Manual());
      changed = true;
    }
  }
  return changed;
}
}  
absl::StatusOr<bool> WholeGraphManualPass::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (auto* comp : module->computations()) {
    TF_ASSIGN_OR_RETURN(bool comp_changed, RunOnComputation(comp));
    changed |= comp_changed;
  }
  return changed;
}
}  