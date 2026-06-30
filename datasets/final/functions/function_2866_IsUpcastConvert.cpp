#include "xla/service/convert_operand_folding.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
namespace xla {
namespace {
bool IsUpcastConvert(const HloInstruction* hlo) {
  if (!hlo->shape().IsArray()) {
    return false;
  }
  switch (hlo->opcode()) {
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kGather:
    case HloOpcode::kReshape:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose: {
      return IsUpcastConvert(hlo->operand(0));
    }
    case HloOpcode::kReduce: {
      if (ShapeUtil::ElementsIn(hlo->shape()) ==
          ShapeUtil::ElementsIn(hlo->operand(0)->shape())) {
        return IsUpcastConvert(hlo->operand(0));
      }
      return false;
    }
    case HloOpcode::kConvert:
      return primitive_util::CastPreservesValues(
          hlo->operand(0)->shape().element_type(), hlo->shape().element_type());
    default:
      return false;
  }
}
HloInstruction* EffectiveOperand(HloInstruction* hlo) {
  switch (hlo->opcode()) {
    case HloOpcode::kBroadcast:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kGather:
    case HloOpcode::kReshape:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose: {
      HloInstruction* operand = EffectiveOperand(hlo->mutable_operand(0));
      HloInstruction* clone = hlo->AddInstruction(hlo->Clone());
      *(clone->mutable_shape()) = ShapeUtil::ChangeElementType(
          clone->shape(), operand->shape().element_type());
      clone->ReplaceOperandWithDifferentShape(0, operand).IgnoreError();
      return clone;
    }
    case HloOpcode::kReduce: {
      HloInstruction* operand = EffectiveOperand(hlo->mutable_operand(0));
      return hlo->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::ChangeElementType(hlo->shape(),
                                       operand->shape().element_type()),
          operand));
    }
    case HloOpcode::kConvert:
      return hlo->mutable_operand(0);
    default:
      return nullptr;
  }
}
}  
bool ConvertOperandFolding::InstructionMatchesPattern(
    HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kDot &&
      instruction->opcode() != HloOpcode::kConvolution) {
    return false;
  }
  for (auto* operand : instruction->operands()) {
    if (IsUpcastConvert(operand)) {
      return true;
    }
  }
  return false;
}
absl::StatusOr<HloInstruction*> ConvertOperandFolding::ExpandInstruction(
    HloInstruction* instruction) {
  for (int i = 0; i < instruction->operand_count(); ++i) {
    auto* operand = instruction->mutable_operand(i);
    if (IsUpcastConvert(operand)) {
      TF_RETURN_IF_ERROR(instruction->ReplaceOperandWithDifferentShape(
          i, EffectiveOperand(operand)));
    }
  }
  return nullptr;
}
}  