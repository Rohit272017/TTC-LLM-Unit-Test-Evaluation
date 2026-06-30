#include "xla/service/hlo_constant_folding.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/service/slow_operation_alarm.h"
#include "xla/shape_util.h"
#include "tsl/platform/errors.h"
namespace xla {
static bool IsOrContainsIllegalInstr(const HloInstruction* instr) {
  if (instr->opcode() == HloOpcode::kAfterAll ||
      instr->opcode() == HloOpcode::kRng) {
    return true;
  }
  for (const HloComputation* c : instr->called_computations()) {
    if (absl::c_any_of(c->instructions(), IsOrContainsIllegalInstr)) {
      return true;
    }
  }
  return false;
}
 std::atomic<int64_t> HloConstantFolding::slow_op_counter_{0};
absl::StatusOr<bool> HloConstantFolding::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto evaluator = std::make_unique<HloEvaluator>(0);
  evaluator->set_use_fast_path(true);
  std::vector<HloInstruction*> dead_instructions;
  for (auto* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->IsDead()) {
        continue;
      }
      if (!absl::c_any_of(instruction->operands(),
                          HloPredicateIsOp<HloOpcode::kConstant>) ||
          !absl::c_all_of(
              instruction->operands(), [](const HloInstruction* operand) {
                return operand->opcode() == HloOpcode::kConstant ||
                       (operand->opcode() == HloOpcode::kBroadcast &&
                        operand->operand(0)->opcode() == HloOpcode::kConstant);
              })) {
        continue;
      }
      if (instruction->opcode() == HloOpcode::kParameter ||
          instruction->opcode() == HloOpcode::kConstant ||
          instruction->opcode() == HloOpcode::kTuple) {
        continue;
      }
      if (instruction->opcode() == HloOpcode::kBroadcast ||
          instruction->opcode() == HloOpcode::kIota) {
        continue;
      }
      if (instruction->IsAsynchronous() &&
          instruction->async_execution_thread() !=
              instruction->parent()->execution_thread()) {
        continue;
      }
      if (instruction->opcode() == HloOpcode::kFft) {
        continue;
      }
      if (IsOrContainsIllegalInstr(instruction)) {
        continue;
      }
      if (instruction->HasSideEffect()) {
        continue;
      }
      if (instruction->opcode() == HloOpcode::kPad &&
          instruction->operand(0)->opcode() == HloOpcode::kBroadcast &&
          instruction->operand(1)->opcode() == HloOpcode::kConstant) {
        continue;
      }
      if (instruction->shape().IsArray()) {
        int64_t elements_in_operands = 0;
        for (HloInstruction* operand : instruction->operands()) {
          if (operand->shape().IsArray()) {
            elements_in_operands += ShapeUtil::ElementsIn(operand->shape());
          }
        }
        int64_t elements_in_constant =
            ShapeUtil::ElementsIn(instruction->shape());
        static const int64_t kMaximumConstantSizeElements = 45 * 1000 * 1000;
        if (std::max(elements_in_constant, elements_in_operands) >
            kMaximumConstantSizeElements) {
          VLOG(2) << "Ignore constant folding: result shape size is "
                  << elements_in_constant << " total size of arguments is "
                  << elements_in_operands;
          continue;
        }
      }
      VLOG(5) << "Constant folding: " << instruction->ToString();
      absl::Duration slow_timeout =
          absl::Seconds(uint64_t{1} << slow_op_counter_.load());
      SlowOperationAlarm slow_alarm(slow_timeout, [instruction, slow_timeout] {
        const bool ndebug =
#if NDEBUG
            true;
#else
            false;
#endif
        absl::string_view explanation_msg =
            ndebug
                ? "This isn't necessarily a bug; constant-folding is "
                  "inherently a trade-off between compilation time and speed "
                  "at runtime. XLA has some guards that attempt to keep "
                  "constant folding from taking too long, but fundamentally "
                  "you'll always be able to come up with an input program that "
                  "takes a long time.\n\n"
                  "If you'd like to file a bug, run with envvar "
                  "XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results."
                : "XLA was built without compiler optimizations, which can be "
                  "slow. Try rebuilding with -c opt.";
        return absl::StrFormat(
            "Constant folding an instruction is taking > %s:\n\n"
            "  %s\n\n"  
            "%s",       
            absl::FormatDuration(slow_timeout), instruction->ToString(),
            explanation_msg);
      });
      Literal result;
      if (!evaluator->TryEvaluate(
              instruction, &result,
              true)) {
        VLOG(2) << "Constant folding failed for instruction: "
                << instruction->ToString();
        continue;
      }
      slow_alarm.cancel();
      if (slow_alarm.fired()) {
        slow_op_counter_++;
      }
      VLOG(4) << "Constant folded: " << instruction->ToString();
      dead_instructions.push_back(instruction);
      HloInstruction* new_constant = instruction->AddInstruction(
          HloInstruction::CreateConstant(std::move(result)));
      if (new_constant->shape().has_layout()) {
        new_constant->mutable_shape()
            ->mutable_layout()
            ->set_element_size_in_bits(
                instruction->shape().layout().element_size_in_bits());
      }
      TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(new_constant));
    }
  }
  const bool changed = !dead_instructions.empty();
  for (HloInstruction* dead_instruction : dead_instructions) {
    CHECK(dead_instruction->IsDead());
    HloComputation* computation = dead_instruction->parent();
    TF_RETURN_IF_ERROR(computation->RemoveInstruction(dead_instruction));
  }
  return changed;
}
}  