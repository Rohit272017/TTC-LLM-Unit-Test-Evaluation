#include "xla/service/slice_sinker.h"
#include <algorithm>
#include <optional>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "xla/shape_util.h"
namespace xla {
namespace {
bool SameSliceConfiguration(const HloInstruction* slice_1,
                            const HloInstruction* slice_2) {
  CHECK_EQ(slice_1->opcode(), HloOpcode::kSlice);
  CHECK_EQ(slice_2->opcode(), HloOpcode::kSlice);
  CHECK(slice_1->operand(0)->shape().dimensions() ==
        slice_2->operand(0)->shape().dimensions());
  return slice_1->slice_starts() == slice_2->slice_starts() &&
         slice_1->slice_limits() == slice_2->slice_limits() &&
         slice_1->slice_strides() == slice_2->slice_strides();
}
bool IsElementwiseOperationOnSimilarSlices(const HloInstruction* inst) {
  CHECK(inst->IsElementwise());
  if (absl::c_any_of(inst->operands(), [](const HloInstruction* operand) {
        return operand->opcode() != HloOpcode::kSlice;
      })) {
    return false;
  }
  const HloInstruction* slice0 = inst->operand(0);
  return absl::c_all_of(absl::MakeSpan(inst->operands()).subspan(1),
                        [slice0](const HloInstruction* slice) {
                          return ShapeUtil::CompatibleIgnoringElementType(
                                     slice0->operand(0)->shape(),
                                     slice->operand(0)->shape()) &&
                                 SameSliceConfiguration(slice0, slice);
                        });
}
bool IsSimilarOperationOnSlices(const HloInstruction* operation_on_slices,
                                const HloInstruction* candidate) {
  if (candidate->user_count() == 0) {
    return false;
  }
  if (!candidate->SameOp(*operation_on_slices) ||
      operation_on_slices->shape().element_type() !=
          candidate->shape().element_type()) {
    return false;
  }
  const HloInstruction* operand_slice0 = candidate->operand(0);
  for (int64_t i = 0; i < candidate->operand_count(); ++i) {
    const HloInstruction* operand_slice = candidate->operand(i);
    if (operand_slice->opcode() != HloOpcode::kSlice ||
        operand_slice->operand(0) !=
            operation_on_slices->operand(i)->operand(0) ||
        !SameSliceConfiguration(operand_slice0, operand_slice)) {
      return false;
    }
  }
  return true;
}
bool ShouldTransform(const std::vector<HloInstruction*>& operations_on_slices) {
  int64_t sum = 0;
  for (HloInstruction* user : operations_on_slices) {
    sum += ShapeUtil::ElementsIn(user->shape());
  }
  return sum >= xla::ShapeUtil::ElementsIn(
                    operations_on_slices[0]->operand(0)->operand(0)->shape());
}
std::optional<std::vector<HloInstruction*>> FindElementwiseOperationGroup(
    const HloInstruction* operation_on_slices) {
  std::vector<HloInstruction*> operations;
  const HloInstruction* slice_source0 =
      operation_on_slices->operand(0)->operand(0);
  for (const HloInstruction* operand_slice0 : slice_source0->users()) {
    if (operand_slice0->opcode() != HloOpcode::kSlice) {
      continue;
    }
    for (HloInstruction* user : operand_slice0->users()) {
      if (IsSimilarOperationOnSlices(operation_on_slices, user)) {
        operations.push_back(user);
      }
    }
  }
  return ShouldTransform(operations) ? std::make_optional(operations)
                                     : std::nullopt;
}
absl::Status SinkSlices(
    const std::vector<HloInstruction*>& slice_sources,
    const std::vector<HloInstruction*>& operation_on_slices) {
  const Shape shape = slice_sources[0]->shape();
  PrimitiveType element_type = operation_on_slices[0]->shape().element_type();
  Shape new_shape = ShapeUtil::ChangeElementType(shape, element_type);
  HloComputation* computation = operation_on_slices[0]->parent();
  auto operation_on_slice_sources = computation->AddInstruction(
      operation_on_slices[0]->CloneWithNewOperands(new_shape, slice_sources));
  VLOG(10) << "Adding operation_on_slice_sources: "
           << operation_on_slice_sources->ToString();
  for (HloInstruction* user : operation_on_slices) {
    const HloInstruction* operand_slice = user->operand(0);
    auto user_slice =
        computation->AddInstruction(operand_slice->CloneWithNewOperands(
            user->shape(), {operation_on_slice_sources}));
    VLOG(10) << "Adding new slice: " << user_slice->ToString()
             << " to replace: " << user->ToString();
    TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(user_slice));
  }
  return absl::OkStatus();
}
}  
absl::StatusOr<bool> SliceSinker::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (!instruction->IsElementwise() || instruction->operand_count() == 0 ||
          instruction->user_count() == 0) {
        continue;
      }
      VLOG(10) << "Processing instruction : " << instruction->ToString();
      if (!IsElementwiseOperationOnSimilarSlices(instruction)) {
        continue;
      }
      std::optional<std::vector<HloInstruction*>> similar_operations =
          FindElementwiseOperationGroup(instruction);
      if (!similar_operations.has_value()) {
        continue;
      }
      std::vector<HloInstruction*> slice_sources;
      absl::c_transform(
          instruction->operands(), std::back_inserter(slice_sources),
          [](HloInstruction* slice) { return slice->mutable_operand(0); });
      TF_RETURN_IF_ERROR(SinkSlices(slice_sources, similar_operations.value()));
      changed = true;
    }
  }
  return changed;
}
}  