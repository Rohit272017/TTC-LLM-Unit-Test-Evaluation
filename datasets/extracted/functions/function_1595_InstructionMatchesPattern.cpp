#include "xla/service/real_imag_expander.h"
#include "xla/literal_util.h"
namespace xla {
bool RealImagExpander::InstructionMatchesPattern(HloInstruction* inst) {
  return (inst->opcode() == HloOpcode::kReal ||
          inst->opcode() == HloOpcode::kImag) &&
         !ShapeUtil::ElementIsComplex(inst->operand(0)->shape());
}
absl::StatusOr<HloInstruction*> RealImagExpander::ExpandInstruction(
    HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kReal) {
    return inst->mutable_operand(0);
  } else {
    HloComputation* comp = inst->parent();
    auto zero = comp->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::Zero(inst->operand(0)->shape().element_type())));
    zero = comp->AddInstruction(
        HloInstruction::CreateBroadcast(inst->shape(), zero, {}));
    return zero;
  }
}
}  