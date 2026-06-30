#include "xla/service/sharding_remover.h"
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/spmd/shardy/constants.h"
#include "tsl/platform/errors.h"
namespace xla {
absl::StatusOr<bool> ShardingRemover::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  const absl::flat_hash_set<absl::string_view> to_remove_sharding_ops = {
      "Sharding", "SPMDShardToFullShape", "SPMDFullToShardShape",
      sdy::kFuncResultShardingTargetName};
  for (HloComputation* computation : module->computations(execution_threads)) {
    auto instructions = computation->MakeInstructionPostOrder();
    std::reverse(instructions.begin(), instructions.end());
    for (HloInstruction* instruction : instructions) {
      if (instruction->opcode() != HloOpcode::kCustomCall) {
        continue;
      }
      if (!to_remove_sharding_ops.contains(instruction->custom_call_target())) {
        continue;
      }
      CHECK(instruction->operand_count() == 1)
          << "Sharding instruction must have exactly one operand";
      TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(
          instruction->mutable_operand(0), name()));
      changed = true;
      if (instruction->custom_call_target() == "Sharding" ||
          instruction->custom_call_target() ==
              sdy::kFuncResultShardingTargetName) {
        auto copy = computation->AddInstruction(
            HloInstruction::CreateUnary(instruction->shape(), HloOpcode::kCopy,
                                        instruction->mutable_operand(0)));
        TF_RETURN_IF_ERROR(computation->ReplaceInstruction(instruction, copy));
        instruction = copy;
      }
    }
  }
  return changed;
}
}  