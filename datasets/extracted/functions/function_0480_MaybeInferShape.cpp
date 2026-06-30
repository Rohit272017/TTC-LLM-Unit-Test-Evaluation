#include "xla/service/result_caster.h"
#include <optional>
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
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
bool ResultCaster::InstructionMatchesPattern(HloInstruction* instruction) {
  auto status_or_inferred_shape = MaybeInferShape(instruction);
  if (!status_or_inferred_shape.ok() ||
      !status_or_inferred_shape->has_value()) {
    return false;
  }
  const Shape& inferred_shape = status_or_inferred_shape.value().value();
  return inferred_shape.element_type() != instruction->shape().element_type() &&
         ShapeUtil::HigherPrecisionElementType(inferred_shape,
                                               instruction->shape()) ==
             inferred_shape.element_type();
}
absl::StatusOr<HloInstruction*> ResultCaster::ExpandInstruction(
    HloInstruction* instruction) {
  auto* computation = instruction->parent();
  Shape inferred_shape = MaybeInferShape(instruction).value().value();
  *inferred_shape.mutable_layout() = instruction->shape().layout();
  auto clone = computation->AddInstruction(
      instruction->CloneWithNewShape(inferred_shape));
  return computation->AddInstruction(
      HloInstruction::CreateConvert(instruction->shape(), clone));
}
}  