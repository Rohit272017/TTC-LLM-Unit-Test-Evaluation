#include "xla/service/gpu/transforms/sanitize_constant_names.h"
#include <string>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/llvm_ir/buffer_assignment_util.h"
#include "xla/service/name_uniquer.h"
#include "tsl/platform/logging.h"
namespace xla {
namespace gpu {
absl::StatusOr<bool> SanitizeConstantNames::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  NameUniquer instr_name_uniquer("_");
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() == HloOpcode::kConstant) {
        continue;
      }
      instr_name_uniquer.GetUniqueName(instr->name());
    }
  }
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() != HloOpcode::kConstant) {
        continue;
      }
      std::string sanitized_name = llvm_ir::SanitizeConstantName(*instr);
      instr->SetAndSanitizeName(sanitized_name);
      instr->UniquifyName(&instr_name_uniquer);
      module->instruction_name_uniquer().GetUniqueName(instr->name());
      changed = true;
    }
  }
  return changed;
}  
}  
}  