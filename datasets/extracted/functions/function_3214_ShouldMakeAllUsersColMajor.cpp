#include "xla/service/cpu/cpu_layout_assignment.h"
#include <cstdint>
#include <numeric>
#include <optional>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/map_util.h"
#include "xla/service/cpu/dot_op_emitter.h"
#include "xla/service/cpu/ir_emission_utils.h"
#include "xla/shape_util.h"
#include "tsl/platform/errors.h"
namespace xla {
namespace cpu {
namespace {
using std::nullopt;
using std::optional;
using ShouldMakeOperandColMajorCache =
    absl::flat_hash_map<const HloInstruction*, bool>;
}  
static bool ShouldMakeAllUsersColMajor(const HloInstruction* instruction) {
  for (auto* user : instruction->users()) {
    optional<int64_t> operand_idx =
        ProfitableToMakeDotOperandColumnMajor(*user);
    if (!operand_idx || user->operand(*operand_idx) != instruction ||
        absl::c_count(user->operands(), instruction) != 1) {
      return false;
    }
  }
  return true;
}
static optional<int64_t> ShouldMakeOperandColumnMajor(
    ShouldMakeOperandColMajorCache* cache, const HloInstruction& instruction) {
  optional<int64_t> operand_idx =
      ProfitableToMakeDotOperandColumnMajor(instruction);
  if (!operand_idx) {
    return nullopt;
  }
  const HloInstruction* operand = instruction.operand(*operand_idx);
  if (operand->opcode() != HloOpcode::kConstant) {
    return nullopt;
  }
  auto it = cache->find(operand);
  if (it == cache->end()) {
    auto insert_result =
        cache->insert({operand, ShouldMakeAllUsersColMajor(operand)});
    CHECK(insert_result.second);
    it = insert_result.first;
  }
  return it->second ? operand_idx : nullopt;
}
static Shape RowMajorShape(Shape shape) {
  ShapeUtil::ForEachMutableSubshape(
      &shape, [](Shape* subshape, const ShapeIndex& index) {
        if (!subshape->IsArray()) {
          return;
        }
        std::vector<int64_t> dimension_order(subshape->dimensions_size());
        std::iota(dimension_order.rbegin(), dimension_order.rend(), 0);
        *subshape->mutable_layout() = LayoutUtil::MakeLayout(dimension_order);
      });
  return shape;
}
static Shape ColMajorShape(const Shape& old_shape) {
  Shape new_shape(old_shape);
  std::vector<int64_t> dimension_order(new_shape.dimensions_size());
  std::iota(dimension_order.begin(), dimension_order.end(), 0);
  *new_shape.mutable_layout() = LayoutUtil::MakeLayout(dimension_order);
  return new_shape;
}
static bool OperandsAndResultMustHaveRowMajorLayout(
    const HloInstruction& instr,
    const TargetMachineFeatures& target_machine_features) {
  if (instr.opcode() == HloOpcode::kConvolution) {
    return PotentiallyImplementedAsEigenConvolution(instr,
                                                    target_machine_features);
  } else if (instr.opcode() == HloOpcode::kDot) {
    return DotOperandsAndResultMustHaveRowMajorLayout(instr,
                                                      target_machine_features);
  } else if (instr.opcode() == HloOpcode::kCustomCall) {
    return instr.custom_call_target() == "TopK";
  }
  return false;
}
absl::Status CpuLayoutAssignment::AddBackendConstraints(
    LayoutConstraints* constraints) {
  ShouldMakeOperandColMajorCache cache;
  const HloComputation* computation = constraints->computation();
  for (auto* instruction : computation->instructions()) {
    if (OperandsAndResultMustHaveRowMajorLayout(*instruction,
                                                target_machine_features_)) {
      TF_RETURN_IF_ERROR(SetInstructionLayout(
          RowMajorShape(instruction->shape()), instruction));
      for (int i = 0; i < instruction->operand_count(); i++) {
        TF_RETURN_IF_ERROR(SetOperandLayout(
            RowMajorShape(instruction->operand(i)->shape()), instruction, i));
      }
    } else if (optional<int64_t> op_idx =
                   ShouldMakeOperandColumnMajor(&cache, *instruction)) {
      const HloInstruction* op = instruction->operand(*op_idx);
      TF_RETURN_IF_ERROR(
          SetOperandLayout(ColMajorShape(op->shape()), instruction, *op_idx));
    } else if (instruction->opcode() == HloOpcode::kReduceScatter) {
      auto ars = Cast<HloReduceScatterInstruction>(instruction);
      TF_RETURN_IF_ERROR(SetInstructionLayout(
          ShapeUtil::MoveDimToMajor(ars->shape(), ars->scatter_dimension()),
          ars));
    } else if (instruction->opcode() == HloOpcode::kAllGather) {
      auto ag = Cast<HloAllGatherInstruction>(instruction);
      TF_RETURN_IF_ERROR(SetInstructionLayout(
          ShapeUtil::MoveDimToMajor(ag->shape(), ag->all_gather_dimension()),
          ag));
    } else {
      for (int64_t operand_no = 0; operand_no < instruction->operand_count();
           ++operand_no) {
        if (constraints->OperandLayout(instruction, operand_no) != nullptr) {
          continue;
        }
        if (AnyOperandBufferForwarded(instruction, operand_no)) {
          continue;
        }
        if (!instruction->operand(operand_no)->shape().IsArray()) {
          continue;
        }
        Shape operand_shape(
            RowMajorShape(instruction->operand(operand_no)->shape()));
        TF_RETURN_IF_ERROR(
            SetOperandLayout(operand_shape, instruction, operand_no));
      }
      if (computation->parent()->entry_computation() == computation &&
          computation->root_instruction() == instruction) {
        continue;
      }
      if (!instruction->shape().IsArray()) {
        continue;
      }
    }
  }
  return absl::OkStatus();
}
}  
}  