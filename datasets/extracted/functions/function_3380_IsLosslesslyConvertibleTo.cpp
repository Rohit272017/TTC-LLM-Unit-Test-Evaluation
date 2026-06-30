#include "xla/service/convert_mover.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace {
static bool IsLosslesslyConvertibleTo(const Literal& literal,
                                      PrimitiveType dst_ty) {
  PrimitiveType orig_ty = literal.shape().element_type();
  absl::StatusOr<Literal> converted1 = literal.Convert(dst_ty);
  if (!converted1.ok()) {
    return false;
  }
  absl::StatusOr<Literal> converted2 = converted1->Convert(orig_ty);
  if (!converted2.ok()) {
    return false;
  }
  return literal == *converted2;
}
bool OpCommutesWithConvert(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kConcatenate:
    case HloOpcode::kPad:
    case HloOpcode::kReshape:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
      return true;
    default:
      return false;
  }
}
absl::StatusOr<bool> MoveConvertPrecisionOps(HloComputation* comp) {
  bool changed = false;
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    if (!OpCommutesWithConvert(instr->opcode()) ||
        instr->operand_count() == 0 ||
        !absl::c_all_of(instr->operands(), [](const HloInstruction* operand) {
          return (operand->opcode() == HloOpcode::kConvert &&
                  operand->user_count() == 1) ||
                 operand->opcode() == HloOpcode::kConstant;
        })) {
      continue;
    }
    auto convert_op_it = absl::c_find_if(instr->operands(),
                                         HloPredicateIsOp<HloOpcode::kConvert>);
    if (convert_op_it == instr->operands().end()) {
      continue;
    }
    const HloInstruction* convert_op = *convert_op_it;
    if (!absl::c_all_of(instr->operands(), [&](const HloInstruction* operand) {
          return operand->opcode() != HloOpcode::kConvert ||
                 operand->operand(0)->shape().element_type() ==
                     convert_op->operand(0)->shape().element_type();
        })) {
      continue;
    }
    PrimitiveType src_ty = convert_op->operand(0)->shape().element_type();
    PrimitiveType dst_ty = convert_op->shape().element_type();
    if (primitive_util::BitWidth(src_ty) >= primitive_util::BitWidth(dst_ty)) {
      continue;
    }
    if (absl::c_any_of(instr->operands(), [&](const HloInstruction* operand) {
          return operand->opcode() == HloOpcode::kConstant &&
                 !IsLosslesslyConvertibleTo(operand->literal(), src_ty);
        })) {
      continue;
    }
    if (primitive_util::IsSubByteNonPredType(src_ty)) {
      continue;
    }
    VLOG(2) << "Moving increase-precision convert op " << convert_op->ToString()
            << " down the graph: " << instr->ToString();
    absl::InlinedVector<HloInstruction*, 8> new_operands;
    new_operands.reserve(instr->operand_count());
    for (HloInstruction* operand : instr->operands()) {
      switch (operand->opcode()) {
        case HloOpcode::kConvert:
          new_operands.push_back(operand->mutable_operand(0));
          break;
        case HloOpcode::kConstant:
          new_operands.push_back(MakeConvertToHlo(operand, src_ty));
          break;
        default:
          LOG(FATAL) << "Unexpected opcode in " << operand->ToString();
      }
    }
    Shape new_shape = instr->shape();
    new_shape.set_element_type(src_ty);
    HloInstruction* new_instr = comp->AddInstruction(
        instr->CloneWithNewOperands(new_shape, new_operands));
    TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
        instr, HloInstruction::CreateConvert(instr->shape(), new_instr)));
    changed = true;
  }
  std::deque<HloInstruction*> work_queue;
  std::vector<HloInstruction*> instrs = comp->MakeInstructionPostOrder();
  work_queue.insert(work_queue.end(), instrs.rbegin(), instrs.rend());
  while (!work_queue.empty()) {
    HloInstruction* instr = work_queue.front();
    work_queue.pop_front();
    if (instr->opcode() != HloOpcode::kConvert ||
        instr->operand(0)->user_count() != 1 ||
        !OpCommutesWithConvert(instr->operand(0)->opcode())) {
      continue;
    }
    PrimitiveType src_ty = instr->operand(0)->shape().element_type();
    PrimitiveType dst_ty = instr->shape().element_type();
    if (primitive_util::BitWidth(src_ty) <= primitive_util::BitWidth(dst_ty)) {
      continue;
    }
    if (primitive_util::IsSubByteNonPredType(dst_ty)) {
      continue;
    }
    VLOG(2) << "Moving decrease-precision convert up the graph: "
            << instr->ToString();
    HloInstruction* to_convert = instr->mutable_operand(0);
    absl::InlinedVector<HloInstruction*, 8> new_operands;
    new_operands.reserve(to_convert->operand_count());
    for (HloInstruction* operand : to_convert->operands()) {
      work_queue.push_front(MakeConvertToHlo(operand, dst_ty));
      new_operands.push_back(work_queue.front());
    }
    Shape new_shape = to_convert->shape();
    new_shape.set_element_type(dst_ty);
    TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
        instr, to_convert->CloneWithNewOperands(new_shape, new_operands)));
    changed = true;
  }
  return changed;
}
}  
absl::StatusOr<bool> ConvertMover::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool changed_computation,
                        MoveConvertPrecisionOps(comp));
    changed |= changed_computation;
  }
  return changed;
}
}  