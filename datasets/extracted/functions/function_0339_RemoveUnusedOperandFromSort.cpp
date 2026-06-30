#include "xla/service/sort_simplifier.h"
#include <memory>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
namespace xla {
namespace {
absl::StatusOr<bool> RemoveUnusedOperandFromSort(HloInstruction* sort) {
  if (!sort->shape().IsTuple()) {
    return false;
  }
  HloComputation* computation = sort->parent();
  if (computation->root_instruction() == sort) {
    return false;
  }
  absl::flat_hash_set<int64_t> used_indices;
  for (const HloInstruction* user : sort->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      return false;
    }
    used_indices.insert(user->tuple_index());
  }
  auto comparator = sort->to_apply();
  for (int64_t i = 0; i < sort->operand_count() * 2; ++i) {
    if (comparator->parameter_instruction(i)->user_count() > 0) {
      used_indices.insert(i / 2);
    }
  }
  if (used_indices.size() == sort->operand_count()) {
    return false;
  }
  std::vector<HloInstruction*> operands;
  std::vector<const Shape*> new_shapes;
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
    if (used_indices.contains(i)) {
      operands.push_back(sort->mutable_operand(i));
      new_shapes.push_back(&sort->operand(i)->shape());
    }
  }
  Shape new_sort_shape = new_shapes.size() == 1
                             ? *new_shapes[0]
                             : ShapeUtil::MakeTupleShapeWithPtrs(new_shapes);
  HloInstruction* new_sort = computation->AddInstruction(
      sort->CloneWithNewOperands(new_sort_shape, operands));
  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      replacements;
  int64_t parameter_number = 0;
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
    auto* old_lhs_parameter = comparator->parameter_instruction(i * 2);
    auto* old_rhs_parameter = comparator->parameter_instruction(i * 2 + 1);
    if (used_indices.contains(i)) {
      Shape scalar_shape =
          ShapeUtil::MakeShape(sort->operand(i)->shape().element_type(), {});
      replacements[old_lhs_parameter] = HloInstruction::CreateParameter(
          parameter_number, scalar_shape,
          absl::StrCat("p.", parameter_number / 2, ".lhs"));
      ++parameter_number;
      replacements[old_rhs_parameter] = HloInstruction::CreateParameter(
          parameter_number, scalar_shape,
          absl::StrCat("p.", parameter_number / 2, ".rhs"));
      ++parameter_number;
    } else {
      replacements[old_lhs_parameter] = nullptr;
      replacements[old_rhs_parameter] = nullptr;
    }
  }
  HloModule* module = sort->GetModule();
  HloComputation* new_compare = module->AddEmbeddedComputation(
      comparator->CloneWithReplacements(&replacements));
  new_sort->set_to_apply(new_compare);
  absl::flat_hash_map<int64_t, HloInstruction*> result_map;
  if (new_sort->shape().IsTuple()) {
    int64_t new_index = 0;
    for (int64_t i = 0; i < sort->operand_count(); ++i) {
      if (used_indices.count(i)) {
        result_map[i] =
            computation->AddInstruction(HloInstruction::CreateGetTupleElement(
                *new_shapes[new_index], new_sort, new_index));
        ++new_index;
      }
    }
  } else {
    CHECK_EQ(used_indices.size(), 1);
    result_map[*used_indices.begin()] = new_sort;
  }
  std::vector<HloInstruction*> users(sort->users().begin(),
                                     sort->users().end());
  for (HloInstruction* user : users) {
    TF_RETURN_IF_ERROR(
        user->ReplaceAllUsesWith(result_map.at(user->tuple_index())));
    TF_RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(user));
  }
  return true;
}
}  
absl::StatusOr<bool> SortSimplifier::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(2) << "HLO module before SortSimplifier:";
  XLA_VLOG_LINES(2, module->ToString());
  bool changed = false;
  std::vector<HloInstruction*> sort_instrs;
  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(sort_instrs),
                    HloPredicateIsOp<HloOpcode::kSort>);
  }
  for (HloInstruction* sort_instr : sort_instrs) {
    TF_ASSIGN_OR_RETURN(bool result, RemoveUnusedOperandFromSort(sort_instr));
    changed |= result;
  }
  if (changed) {
    VLOG(2) << "HLO module after SortSimplifier:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "HLO module unchanged after SortSimplifier";
  }
  return changed;
}
}  