#include "xla/service/gpu/transforms/scheduling_instruction_annotator.h"
#include <string>
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "tsl/platform/statusor.h"
namespace xla::gpu {
namespace {
absl::StatusOr<bool> AnnotateSchedulingInstructionNames(
    HloComputation& computation) {
  bool changed = false;
  for (HloInstruction* inst : computation.instructions()) {
    if (!inst->metadata().scheduling_name().empty()) {
      continue;
    }
    if (inst->opcode() == HloOpcode::kConstant) {
      continue;
    }
    inst->set_metadata_scheduling_name(inst->name());
    changed = true;
  }
  return changed;
}
}  
absl::StatusOr<bool> SchedulingInstructionAnnotator::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  CHECK(module->has_schedule())
      << "The pass is supposed to run in the beginning of post-scheduling!";
  bool changed = false;
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result,
                        AnnotateSchedulingInstructionNames(*computation));
    changed |= result;
  }
  return changed;
}
}  