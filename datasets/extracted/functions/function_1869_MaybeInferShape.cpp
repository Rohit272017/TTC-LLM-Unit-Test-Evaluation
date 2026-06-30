#include "xla/service/operand_upcaster.h"
#include <optional>
#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace {
absl::StatusOr<std::optional<Shape>> MaybeInferShape(
    const HloInstruction* instruction) {
  switch (instruction->opcode()) {
    case HloOpcode::kDot:
      return ShapeInference::InferDotOpShape(
          instruction->operand(0)->shape(), instruction->operand(1)->shape(),
          instruction->dot_dimension_numbers(),
          std::nullopt,
          Cast<HloDotInstruction>(instruction)->sparsity());
    case HloOpcode::kConvolution:
      return ShapeInference::InferConvolveShape(
          instruction->operand(0)->shape(), instruction->operand(1)->shape(),
          instruction->feature_group_count(), instruction->batch_group_count(),
          instruction->window(), instruction->convolution_dimension_numbers(),
          std::nullopt);
    default:
      return std::optional<Shape>(std::nullopt);
  }
}
}  
bool OperandUpcaster::InstructionMatchesPattern(HloInstruction* instruction) {
  auto status_or_inferred_shape = MaybeInferShape(instruction);
  if (!status_or_inferred_shape.ok() ||
      !status_or_inferred_shape->has_value()) {
    return false;
  }
  if (absl::c_count(instruction->precision_config().operand_precision(),
                    PrecisionConfig::PACKED_NIBBLE) == 2) {
    return true;
  }
  PrimitiveType inferred_type = (*status_or_inferred_shape)->element_type();
  if (instruction->shape().element_type() == inferred_type &&
      instruction->operand(0)->shape().element_type() == inferred_type &&
      instruction->operand(1)->shape().element_type() == inferred_type) {
    return false;
  }
  return ShapeUtil::ElementCanUpcast(**status_or_inferred_shape,
                                     instruction->shape());
}
absl::StatusOr<HloInstruction*> OperandUpcaster::ExpandInstruction(
    HloInstruction* instruction) {
  const bool packed_nibble =
      absl::c_count(instruction->precision_config().operand_precision(),
                    PrecisionConfig::PACKED_NIBBLE) == 2;
  auto type = instruction->shape().element_type();
  if (packed_nibble) {
    HloInstruction *lhs_n0 = instruction->mutable_operand(0), *lhs_n1 = lhs_n0,
                   *rhs_n0 = instruction->mutable_operand(1), *rhs_n1 = rhs_n0;
    TF_ASSIGN_OR_RETURN(lhs_n0, MakeBinaryHlo(HloOpcode::kShiftLeft, lhs_n0,
                                              MakeScalarLike(lhs_n0, 4)));
    HloOpcode lhs_shift = ShapeUtil::ElementIsSigned(lhs_n0->shape())
                              ? HloOpcode::kShiftRightArithmetic
                              : HloOpcode::kShiftRightLogical;
    TF_ASSIGN_OR_RETURN(
        lhs_n0, MakeBinaryHlo(lhs_shift, lhs_n0, MakeScalarLike(lhs_n0, 4)));
    lhs_n0 = MakeConvertToHlo(lhs_n0, type);
    TF_ASSIGN_OR_RETURN(
        lhs_n1, MakeBinaryHlo(lhs_shift, lhs_n1, MakeScalarLike(lhs_n1, 4)));
    lhs_n1 = MakeConvertToHlo(lhs_n1, type);
    TF_ASSIGN_OR_RETURN(rhs_n0, MakeBinaryHlo(HloOpcode::kShiftLeft, rhs_n0,
                                              MakeScalarLike(rhs_n0, 4)));
    HloOpcode rhs_shift = ShapeUtil::ElementIsSigned(rhs_n0->shape())
                              ? HloOpcode::kShiftRightArithmetic
                              : HloOpcode::kShiftRightLogical;
    TF_ASSIGN_OR_RETURN(
        rhs_n0, MakeBinaryHlo(rhs_shift, rhs_n0, MakeScalarLike(rhs_n0, 4)));
    rhs_n0 = MakeConvertToHlo(rhs_n0, type);
    TF_ASSIGN_OR_RETURN(
        rhs_n1, MakeBinaryHlo(rhs_shift, rhs_n1, MakeScalarLike(rhs_n1, 4)));
    rhs_n1 = MakeConvertToHlo(rhs_n1, type);
    HloInstruction* linear_n0 =
        instruction->parent()->AddInstruction(instruction->CloneWithNewOperands(
            instruction->shape(), {lhs_n0, rhs_n0}));
    linear_n0->mutable_precision_config()->mutable_operand_precision()->Set(
        0, PrecisionConfig::DEFAULT);
    linear_n0->mutable_precision_config()->mutable_operand_precision()->Set(
        1, PrecisionConfig::DEFAULT);
    HloInstruction* linear_n1 =
        instruction->parent()->AddInstruction(linear_n0->CloneWithNewOperands(
            instruction->shape(), {lhs_n1, rhs_n1}));
    return MakeBinaryHlo(HloOpcode::kAdd, linear_n0, linear_n1);
  }
  for (int i = 0; i < HloDotInstruction::kOperands; ++i) {
    auto* operand = instruction->mutable_operand(i);
    if (operand->shape().element_type() == type) {
      continue;
    }
    auto upcast_shape = operand->shape();
    upcast_shape.set_element_type(type);
    auto* convert_inst = instruction->AddInstruction(
        HloInstruction::CreateConvert(upcast_shape, operand));
    TF_RETURN_IF_ERROR(
        instruction->ReplaceOperandWithDifferentShape(i, convert_inst));
  }
  return nullptr;
}
}  