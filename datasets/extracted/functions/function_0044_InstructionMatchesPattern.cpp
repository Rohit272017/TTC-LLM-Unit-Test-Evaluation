#include "xla/service/convolution_pred_expander.h"
#include <iterator>
#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
namespace xla {
namespace m = match;
bool ConvolutionPredExpander::InstructionMatchesPattern(
    HloInstruction* instruction) {
  return Match(instruction, m::Convolution(m::Op().WithElementType(PRED),
                                           m::Op().WithElementType(PRED))
                                .WithElementType(PRED));
}
absl::StatusOr<HloInstruction*> ConvolutionPredExpander::ExpandInstruction(
    HloInstruction* instruction) {
  HloComputation* computation = instruction->parent();
  absl::InlinedVector<HloInstruction*, 2> new_operands;
  absl::c_transform(instruction->operands(), std::back_inserter(new_operands),
                    [&](HloInstruction* operand) {
                      CHECK_EQ(operand->shape().element_type(), PRED);
                      return MakeConvertToHlo(operand, F16);
                    });
  Shape new_shape = ShapeUtil::ChangeElementType(instruction->shape(), F16);
  HloInstruction* new_instruction = computation->AddInstruction(
      instruction->CloneWithNewOperands(new_shape, new_operands));
  return MakeConvertToHlo(new_instruction, PRED);
}
}  