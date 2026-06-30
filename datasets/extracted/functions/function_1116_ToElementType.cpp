#include "xla/service/hlo_element_type_converter.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/shape_util.h"
#include "xla/types.h"
#include "tsl/platform/errors.h"
namespace xla {
namespace {
HloInstruction* ToElementType(HloInstruction* hlo, PrimitiveType type) {
  if (hlo->shape().element_type() != type) {
    Shape shape = ShapeUtil::ChangeElementType(hlo->shape(), type);
    hlo = hlo->parent()->AddInstruction(
        HloInstruction::CreateConvert(shape, hlo));
  }
  CHECK_EQ(hlo->shape().element_type(), type);
  return hlo;
}
bool HasOperandType(HloInstruction* hlo, PrimitiveType type) {
  for (HloInstruction* operand : hlo->operands()) {
    if (operand->shape().element_type() == type) {
      return true;
    }
  }
  return false;
}
Shape GetConvertedTupleShape(const Shape& shape, PrimitiveType from_type,
                             PrimitiveType to_type) {
  std::vector<Shape> new_tuple_subshapes;
  const int64_t n = ShapeUtil::TupleElementCount(shape);
  new_tuple_subshapes.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    Shape subshape = ShapeUtil::GetTupleElementShape(shape, i);
    CHECK(!subshape.IsTuple());
    if (subshape.element_type() == from_type) {
      subshape = ShapeUtil::ChangeElementType(subshape, to_type);
    }
    new_tuple_subshapes.push_back(subshape);
  }
  return ShapeUtil::MakeTupleShape(new_tuple_subshapes);
}
HloInstruction* ConvertTupleElements(HloInstruction* hlo,
                                     const Shape& to_shape) {
  const Shape& shape = hlo->shape();
  HloComputation* computation = hlo->parent();
  std::vector<HloInstruction*> tuple_elements;
  for (int64_t i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
    const Shape& ele_shape = ShapeUtil::GetTupleElementShape(shape, i);
    HloInstruction* element = computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(ele_shape, hlo, i));
    const Shape& to_ele_shape = ShapeUtil::GetTupleElementShape(to_shape, i);
    CHECK(!ele_shape.IsTuple());
    if (ele_shape.element_type() != to_ele_shape.element_type()) {
      element = computation->AddInstruction(
          HloInstruction::CreateConvert(to_ele_shape, element));
    }
    tuple_elements.push_back(element);
  }
  return computation->AddInstruction(
      HloInstruction::CreateTuple(tuple_elements));
}
}  
HloElementTypeConverter::HloElementTypeConverter(
    PrimitiveType eliminate_type, PrimitiveType replace_with_type)
    : eliminate_type_(eliminate_type), replace_with_type_(replace_with_type) {}
absl::StatusOr<bool> HloElementTypeConverter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      3, "HloElementTypeConverter::Run(), before:\n" + module->ToString());
  if (eliminate_type_ == replace_with_type_) {
    return false;
  }
  HloCloneContext context(module);
  bool changed = false;
  for (auto* computation : module->computations(execution_threads)) {
    for (auto* hlo : computation->MakeInstructionPostOrder()) {
      const auto opcode = hlo->opcode();
      if (opcode == HloOpcode::kParameter || opcode == HloOpcode::kConstant ||
          opcode == HloOpcode::kTuple || opcode == HloOpcode::kConvert ||
          opcode == HloOpcode::kBitcastConvert ||
          opcode == HloOpcode::kGetTupleElement ||
          opcode == HloOpcode::kInfeed || opcode == HloOpcode::kOutfeed) {
        continue;
      }
      if (opcode == HloOpcode::kCustomCall) {
        continue;
      }
      if (opcode == HloOpcode::kWhile || opcode == HloOpcode::kCall ||
          opcode == HloOpcode::kAllReduce ||
          opcode == HloOpcode::kReduceScatter ||
          opcode == HloOpcode::kAllReduceStart ||
          opcode == HloOpcode::kFusion || opcode == HloOpcode::kMap ||
          opcode == HloOpcode::kReduce || opcode == HloOpcode::kReduceWindow ||
          opcode == HloOpcode::kScatter ||
          opcode == HloOpcode::kSelectAndScatter ||
          opcode == HloOpcode::kSort || opcode == HloOpcode::kConditional) {
        continue;
      }
      TF_RET_CHECK(hlo->called_computations().empty()) << hlo->ToString();
      bool nullary = hlo->operands().empty();
      bool wrong_element_type = hlo->shape().element_type() == eliminate_type_;
      bool should_eliminate_type = (nullary && wrong_element_type) ||
                                   HasOperandType(hlo, eliminate_type_);
      if (!should_eliminate_type) {
        TF_RET_CHECK(hlo->shape().element_type() != eliminate_type_);
        continue;
      }
      std::vector<HloInstruction*> new_operands;
      const auto& operands = hlo->operands();
      new_operands.reserve(operands.size());
      for (HloInstruction* operand : operands) {
        if (operand->shape().element_type() == eliminate_type_) {
          operand = ToElementType(operand, replace_with_type_);
        }
        new_operands.push_back(operand);
      }
      HloInstruction* new_hlo;
      if (hlo->shape().element_type() == eliminate_type_) {
        Shape shape =
            ShapeUtil::ChangeElementType(hlo->shape(), replace_with_type_);
        new_hlo = computation->AddInstruction(
            hlo->CloneWithNewOperands(shape, new_operands, &context));
        TF_RETURN_IF_ERROR(new_hlo->CopyAllControlDepsFrom(hlo));
        new_hlo = ToElementType(new_hlo, eliminate_type_);
      } else if (hlo->shape().IsTuple()) {
        Shape old_shape = hlo->shape();
        Shape new_shape = GetConvertedTupleShape(hlo->shape(), eliminate_type_,
                                                 replace_with_type_);
        new_hlo = computation->AddInstruction(
            hlo->CloneWithNewOperands(new_shape, new_operands, &context));
        TF_RETURN_IF_ERROR(new_hlo->CopyAllControlDepsFrom(hlo));
        new_hlo = ConvertTupleElements(new_hlo, old_shape);
      } else {
        new_hlo = computation->AddInstruction(
            hlo->CloneWithNewOperands(hlo->shape(), new_operands, &context));
        TF_RETURN_IF_ERROR(new_hlo->CopyAllControlDepsFrom(hlo));
      }
      TF_RETURN_IF_ERROR(hlo->ReplaceAllUsesWith(new_hlo));
      TF_RETURN_IF_ERROR(hlo->DropAllControlDeps());
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(hlo));
      changed = true;
    }
  }
  XLA_VLOG_LINES(
      2, "HloElementTypeConverter::Run(), after:\n" + module->ToString());
  return changed;
}
}  