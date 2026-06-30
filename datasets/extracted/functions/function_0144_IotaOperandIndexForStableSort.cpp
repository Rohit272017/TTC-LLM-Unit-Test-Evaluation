#include "xla/service/stable_sort_expander.h"
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
namespace xla {
int64_t StableSortExpander::IotaOperandIndexForStableSort(
    const HloSortInstruction& sort) {
  for (const HloInstruction* operand : sort.operands()) {
    if (operand->opcode() == HloOpcode::kIota &&
        Cast<HloIotaInstruction>(operand)->iota_dimension() ==
            sort.sort_dimension() &&
        operand->shape().element_type() == S32) {
      return sort.operand_index(operand);
    }
  }
  return -1;
}
absl::StatusOr<HloInstruction*> StableSortExpander::ExpandInstruction(
    HloInstruction* instruction) {
  auto* sort = Cast<HloSortInstruction>(instruction);
  HloComputation* computation = sort->parent();
  HloInstruction* expanded_sort = nullptr;
  int64_t iota_index = IotaOperandIndexForStableSort(*sort);
  if (iota_index == -1) {
    Shape iota_shape = sort->operand(0)->shape();
    if (iota_shape.dimensions(sort->sort_dimension()) >
        std::numeric_limits<int32_t>::max()) {
      return Unimplemented(
          "Stable sorting of more than 2^31-1 elements is not implemented");
    }
    iota_shape.set_element_type(S32);
    auto iota = computation->AddInstruction(
        HloInstruction::CreateIota(iota_shape, sort->sort_dimension()));
    auto comparator = sort->to_apply();
    absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
        replacements;
    std::vector<std::unique_ptr<HloInstruction>> extra_parameters;
    std::vector<HloInstruction*> extra_parameter_ptrs;
    Shape scalar_shape = ShapeUtil::MakeShape(S32, {});
    extra_parameters.push_back(HloInstruction::CreateParameter(
        sort->operand_count() * 2, scalar_shape,
        absl::StrCat("p.", sort->operand_count(), ".lhs")));
    extra_parameter_ptrs.push_back(extra_parameters.back().get());
    extra_parameters.push_back(HloInstruction::CreateParameter(
        sort->operand_count() * 2 + 1, scalar_shape,
        absl::StrCat("p.", sort->operand_count(), ".rhs")));
    extra_parameter_ptrs.push_back(extra_parameters.back().get());
    sort->set_to_apply(sort->GetModule()->AddEmbeddedComputation(
        comparator->CloneWithReplacements(&replacements,
                                          extra_parameter_ptrs)));
    std::vector<HloInstruction*> new_operands(sort->operands().begin(),
                                              sort->operands().end());
    new_operands.push_back(iota);
    std::vector<Shape> new_shapes = sort->operand_count() == 1
                                        ? std::vector<Shape>{sort->shape()}
                                        : sort->shape().tuple_shapes();
    new_shapes.push_back(iota_shape);
    Shape new_sort_shape = ShapeUtil::MakeTupleShape(new_shapes);
    HloInstruction* new_sort = computation->AddInstruction(
        sort->CloneWithNewOperands(new_sort_shape, new_operands));
    std::vector<HloInstruction*> tuple_elements;
    tuple_elements.reserve(sort->operand_count());
    for (int64_t i = 0; i < sort->operand_count(); ++i) {
      tuple_elements.push_back(
          computation->AddInstruction(HloInstruction::CreateGetTupleElement(
              sort->operand(i)->shape(), new_sort, i)));
    }
    expanded_sort = tuple_elements[0];
    if (tuple_elements.size() > 1) {
      expanded_sort = computation->AddInstruction(
          HloInstruction::CreateTuple(tuple_elements));
    }
    sort = Cast<HloSortInstruction>(new_sort);
    iota_index = sort->operand_count() - 1;
  }
  auto comparator = sort->to_apply();
  std::vector<HloInstruction*> instructions_postorder =
      comparator->MakeInstructionPostOrder();
  absl::flat_hash_map<HloInstruction*, HloInstruction*> replacements;
  auto replace = [&](HloInstruction* instr) {
    auto it = replacements.find(instr);
    if (it == replacements.end()) {
      return instr;
    }
    return it->second;
  };
  HloInstruction* old_root = comparator->root_instruction();
  for (int64_t i = 0; i < comparator->num_parameters(); ++i) {
    replacements[comparator->parameter_instruction(i)] =
        comparator->parameter_instruction(i ^ 1);
  }
  HloInstruction* cloned_root = nullptr;
  for (HloInstruction* inst : instructions_postorder) {
    if (inst->operand_count() == 0) {
      continue;
    }
    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(inst->operand_count());
    for (HloInstruction* operand : inst->operands()) {
      new_operands.push_back(replace(operand));
    }
    auto new_instruction =
        inst->CloneWithNewOperands(inst->shape(), new_operands);
    replacements[inst] = new_instruction.get();
    if (inst == old_root) {
      cloned_root = new_instruction.get();
    }
    comparator->AddInstruction(std::move(new_instruction));
  }
  CHECK_NE(cloned_root, nullptr);
  Shape scalar_pred = ShapeUtil::MakeShape(PRED, {});
  HloInstruction* same =
      comparator->AddInstruction(HloInstruction::CreateCompare(
          scalar_pred, old_root, cloned_root, ComparisonDirection::kEq));
  HloInstruction* tie_breaker =
      comparator->AddInstruction(HloInstruction::CreateCompare(
          scalar_pred, comparator->parameter_instruction(2 * iota_index),
          comparator->parameter_instruction(2 * iota_index + 1),
          ComparisonDirection::kLt));
  HloInstruction* new_root =
      comparator->AddInstruction(HloInstruction::CreateTernary(
          ShapeUtil::MakeShape(PRED, {}), HloOpcode::kSelect, same, tie_breaker,
          old_root));
  comparator->set_root_instruction(new_root);
  return expanded_sort;
}
bool StableSortExpander::InstructionMatchesPattern(
    HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kSort &&
         Cast<HloSortInstruction>(instruction)->is_stable();
}
}  