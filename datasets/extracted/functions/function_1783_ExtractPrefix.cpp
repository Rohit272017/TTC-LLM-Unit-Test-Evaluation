#include "xla/service/tuple_util.h"
#include <cstdint>
#include <string>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "tsl/platform/statusor.h"
namespace xla {
 HloInstruction* TupleUtil::ExtractPrefix(HloInstruction* input_tuple,
                                                    int64_t elements,
                                                    absl::string_view name) {
  CHECK(input_tuple->shape().IsTuple());
  HloComputation* computation = input_tuple->parent();
  const Shape& input_shape = input_tuple->shape();
  std::vector<HloInstruction*> tuple_elements;
  tuple_elements.reserve(elements);
  for (int i = 0; i < elements; i++) {
    std::string element_name;
    if (!name.empty()) {
      element_name = absl::StrCat(name, ".element.", i);
    }
    tuple_elements.push_back(computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(input_shape.tuple_shapes(i),
                                              input_tuple, i),
        element_name));
  }
  return computation->AddInstruction(
      HloInstruction::CreateTuple(tuple_elements), name);
}
 HloInstruction* TupleUtil::AppendSuffix(
    HloInstruction* input_tuple,
    absl::Span<HloInstruction* const> trailing_values) {
  CHECK(input_tuple->shape().IsTuple());
  HloComputation* computation = input_tuple->parent();
  const Shape& input_shape = input_tuple->shape();
  std::vector<HloInstruction*> tuple_elements;
  tuple_elements.reserve(input_shape.tuple_shapes_size());
  for (int i = 0; i < input_shape.tuple_shapes_size(); i++) {
    tuple_elements.push_back(
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            input_shape.tuple_shapes(i), input_tuple, i)));
  }
  tuple_elements.insert(tuple_elements.end(), trailing_values.begin(),
                        trailing_values.end());
  return computation->AddInstruction(
      HloInstruction::CreateTuple(tuple_elements));
}
 absl::StatusOr<HloInstruction*> TupleUtil::ReplaceTupleWith(
    HloInstruction* new_instruction, HloInstruction* tuple,
    ShapeIndex shape_index, bool insert_bitcast_if_different_shape) {
  const Shape& tuple_shape = tuple->shape();
  CHECK(tuple->shape().IsTuple())
      << "ReplaceTupleWith was called for a non-tuple. Tuple = "
      << tuple->ToString()
      << ", new_instruction = " << new_instruction->ToString()
      << ", shape_index = " << shape_index.ToString();
  const HloInstruction* instruction = new_instruction;
  bool equivalent = true;
  for (int i = shape_index.size() - 1; i >= 0; --i) {
    int index = shape_index[i];
    if (instruction->opcode() != HloOpcode::kGetTupleElement ||
        instruction->tuple_index() != index) {
      equivalent = false;
      break;
    }
    instruction = instruction->operand(0);
  }
  if (equivalent && instruction == tuple) {
    VLOG(4) << "Instruction " << new_instruction->ToShortString()
            << " already exists at index " << shape_index.ToString() << " of "
            << tuple->ToShortString();
    return tuple;
  }
  HloComputation* computation = new_instruction->parent();
  std::vector<HloInstruction*> tuple_args(tuple_shape.tuple_shapes_size());
  CHECK_GE(tuple_shape.tuple_shapes_size(), shape_index[0]);
  for (int i = 0; i < tuple_shape.tuple_shapes_size(); ++i) {
    const Shape& subshape = tuple_shape.tuple_shapes(i);
    auto get_operand = [&]() {
      if (tuple->opcode() == HloOpcode::kTuple) {
        return tuple->mutable_operand(i);
      } else {
        return computation->AddInstruction(
            HloInstruction::CreateGetTupleElement(subshape, tuple, i));
      }
    };
    if (i == shape_index[0]) {
      if (subshape.IsTuple()) {
        TF_ASSIGN_OR_RETURN(tuple_args[i],
                            ReplaceTupleWith(new_instruction, get_operand(),
                                             ShapeIndex(shape_index.begin() + 1,
                                                        shape_index.end())));
      } else {
        if (subshape != new_instruction->shape() &&
            insert_bitcast_if_different_shape) {
          VLOG(4) << "Old shape = " << subshape.ToString()
                  << ", new shape = " << new_instruction->shape().ToString()
                  << "; inserting a bitcast.";
          new_instruction = computation->AddInstruction(
              HloInstruction::CreateBitcast(subshape, new_instruction));
        } else if (tuple->opcode() == HloOpcode::kTuple &&
                   tuple->operand(i) == new_instruction) {
          VLOG(4) << "Tuple already contains the new instruction = "
                  << new_instruction->ToShortString()
                  << " tuple = " << tuple->ToShortString();
          return tuple;
        }
        tuple_args[i] = new_instruction;
      }
    } else {
      tuple_args[i] = get_operand();
    }
  }
  if (shape_index[0] == tuple_shape.tuple_shapes_size()) {
    tuple_args.push_back(new_instruction);
  }
  return computation->AddInstruction(HloInstruction::CreateTuple(tuple_args));
}
 HloInstruction* TupleUtil::AddGetTupleElements(
    const HloPosition& position) {
  HloInstruction* instruction = position.instruction;
  HloComputation* computation = instruction->parent();
  for (int64_t index : position.index) {
    auto gte_it = absl::c_find_if(
        instruction->users(), [index](const HloInstruction* use) {
          return use != use->parent()->root_instruction() &&
                 use->opcode() == HloOpcode::kGetTupleElement &&
                 use->tuple_index() == index;
        });
    if (gte_it != instruction->users().end()) {
      instruction = *gte_it;
    } else {
      instruction =
          computation->AddInstruction(HloInstruction::CreateGetTupleElement(
              instruction->shape().tuple_shapes(index), instruction, index));
    }
  }
  return instruction;
}
ShapeTree<HloInstruction*> TupleUtil::DisassembleTupleInstruction(
    HloInstruction* tuple) {
  const Shape& shape = tuple->shape();
  ShapeTree<HloInstruction*> result(shape);
  result.ForEachMutableElement([&](ShapeIndexView index,
                                   HloInstruction** element) {
    if (index.empty()) {
      *element = tuple;
    } else {
      ShapeIndexView parent_index = index.subspan(0, index.size() - 1);
      HloInstruction* parent = result.element(parent_index);
      std::string name = absl::StrCat(tuple->name(), ".disassembled.",
                                      absl::StrJoin(index, "."));
      *element = tuple->parent()->AddInstruction(
          HloInstruction::CreateGetTupleElement(parent, index.back()), name);
    }
  });
  return result;
}
HloInstruction* TupleUtil::AssembleTupleInstruction(
    HloComputation* computation, ShapeTree<HloInstruction*> elements,
    absl::string_view name) {
  elements.ForEachMutableElementPostOrder(
      [&](const ShapeIndex& index, HloInstruction** element) {
        const Shape& subshape = ShapeUtil::GetSubshape(elements.shape(), index);
        if (subshape.IsTuple()) {
          absl::InlinedVector<HloInstruction*, 2> children;
          ShapeIndex child_index = index;
          for (int i = 0; i < subshape.tuple_shapes_size(); ++i) {
            child_index.push_back(i);
            children.push_back(elements.element(child_index));
            child_index.pop_back();
          }
          std::string new_name;
          if (!name.empty()) {
            if (index.empty()) {
              new_name = std::string(name);
            } else {
              new_name =
                  absl::StrCat(name, ".assembled.", absl::StrJoin(index, "."));
            }
          }
          *element = computation->AddInstruction(
              HloInstruction::CreateTuple(children), new_name);
        }
      });
  return elements.element({});
}
}  