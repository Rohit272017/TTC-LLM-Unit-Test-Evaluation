#include "xla/service/hlo_dce.h"
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <set>
#include <tuple>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace {
bool IsRemovableWhile(HloInstruction* instruction,
                      bool remove_cross_partition_collective_ops) {
  if (instruction->opcode() != HloOpcode::kWhile) {
    return false;
  }
  for (HloComputation* computation : instruction->called_computations()) {
    for (HloInstruction* called_instr : computation->instructions()) {
      auto maybe_collective_op =
          DynCast<HloCollectiveInstruction>(called_instr);
      if (called_instr->HasSideEffect() &&
          (!remove_cross_partition_collective_ops || !maybe_collective_op ||
           maybe_collective_op->constrain_layout())) {
        return false;
      }
    }
  }
  return true;
}
absl::StatusOr<bool> RemoveMultiOutputFusionsUnusedOutputs(
    HloComputation* computation) {
  HloInstruction* fusion_instruction = computation->FusionInstruction();
  if (!fusion_instruction) {
    return false;
  }
  if (computation->root_instruction()->opcode() != HloOpcode::kTuple ||
      computation->root_instruction()->has_sharding() ||
      !fusion_instruction->output_operand_aliasing().empty() ||
      fusion_instruction->HasControlDependencies() ||
      fusion_instruction->IsCustomFusion()) {
    return false;
  }
  std::set<int64_t> used_tuple_elements;
  if (fusion_instruction->users().empty()) {
    return false;
  }
  for (HloInstruction* gte : fusion_instruction->users()) {
    if (gte->opcode() != HloOpcode::kGetTupleElement) {
      return false;
    }
    used_tuple_elements.insert(gte->tuple_index());
  }
  if (used_tuple_elements.size() ==
      computation->root_instruction()->operand_count()) {
    return false;
  }
  std::vector<Shape> tuple_shapes;
  tuple_shapes.reserve(used_tuple_elements.size());
  for (int64_t tuple_index : used_tuple_elements) {
    tuple_shapes.push_back(
        fusion_instruction->shape().tuple_shapes(tuple_index));
  }
  Shape new_shape = tuple_shapes.size() == 1
                        ? tuple_shapes[0]
                        : ShapeUtil::MakeTupleShape(tuple_shapes);
  *fusion_instruction->mutable_shape() = std::move(new_shape);
  if (tuple_shapes.size() > 1) {
    for (HloInstruction* gte : fusion_instruction->users()) {
      auto it = used_tuple_elements.lower_bound(gte->tuple_index());
      int64_t new_tuple_index = std::distance(used_tuple_elements.begin(), it);
      gte->set_tuple_index(new_tuple_index);
    }
  } else {
    std::vector<HloInstruction*> users(fusion_instruction->users());
    for (HloInstruction* gte : users) {
      TF_ASSIGN_OR_RETURN(std::ignore, gte->parent()->ReplaceInstruction(
                                           gte, fusion_instruction,
                                           true,
                                           true));
    }
  }
  if (tuple_shapes.size() > 1) {
    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(used_tuple_elements.size());
    for (int64_t tuple_index : used_tuple_elements) {
      new_operands.push_back(
          computation->root_instruction()->mutable_operand(tuple_index));
    }
    auto new_tuple =
        computation->AddInstruction(HloInstruction::CreateTuple(new_operands));
    TF_RETURN_IF_ERROR(computation->ReplaceInstructionWithDifferentShape(
        computation->root_instruction(), new_tuple));
  } else {
    TF_RETURN_IF_ERROR(
        computation->root_instruction()->ReplaceAllUsesWithDifferentShape(
            computation->root_instruction()->mutable_operand(
                *used_tuple_elements.begin())));
  }
  return true;
}
}  
 absl::StatusOr<bool> HloDCE::RunOnComputation(
    HloComputation* computation, bool remove_cross_partition_collective_ops) {
  TF_ASSIGN_OR_RETURN(bool changed,
                      RemoveMultiOutputFusionsUnusedOutputs(computation));
  std::vector<HloInstruction*> dead_roots;
  for (auto* instruction : computation->instructions()) {
    auto maybe_collective_op = DynCast<HloCollectiveInstruction>(instruction);
    if (instruction->IsDead() && computation->IsSafelyRemovable(instruction) &&
        (!instruction->IsCustomCall("Sharding") ||
         (!instruction->operand(0)->IsRoot() &&
          instruction->operand(0)->opcode() != HloOpcode::kParameter &&
          instruction->operand(0)->user_count() == 1)) &&
        (!instruction->HasSideEffect() ||
         (remove_cross_partition_collective_ops && maybe_collective_op &&
          !maybe_collective_op->constrain_layout()) ||
         IsRemovableWhile(instruction,
                          remove_cross_partition_collective_ops))) {
      dead_roots.push_back(instruction);
    }
  }
  for (HloInstruction* dead_root : dead_roots) {
    VLOG(1) << "Removing dead root " << dead_root->ToString()
            << " and its unused operands";
    TF_RETURN_IF_ERROR(
        computation->RemoveInstructionAndUnusedOperands(dead_root));
    changed = true;
  }
  return changed;
}
absl::Status HloDCE::RecursivelyRemoveDeadComputation(
    HloModule* module, HloComputation* computation,
    absl::flat_hash_map<HloComputation*, int>& live_call_counts) {
  std::vector<HloComputation*> to_be_deleted;
  for (HloInstruction* instruction : computation->instructions()) {
    for (HloComputation* subcomp : instruction->called_computations()) {
      auto iter = live_call_counts.find(subcomp);
      if (iter == live_call_counts.end()) {
        return tsl::errors::Internal(
            "called computation %s not found in live_call_counts table during "
            "HloDCE",
            subcomp->name());
      }
      int live_call_count = --iter->second;
      CHECK_GE(live_call_count, 0);
      if (live_call_count == 0) {
        to_be_deleted.push_back(subcomp);
        live_call_counts.erase(iter);
      }
    }
  }
  VLOG(1) << "Removing dead computation " << computation->name();
  TF_RETURN_IF_ERROR(module->RemoveEmbeddedComputation(computation));
  for (HloComputation* subcomp : to_be_deleted) {
    TF_RETURN_IF_ERROR(
        RecursivelyRemoveDeadComputation(module, subcomp, live_call_counts));
  }
  return absl::OkStatus();
}
absl::StatusOr<bool> HloDCE::RecursivelyRemoveDeadComputations(
    HloModule* module) {
  bool module_contains_dead_code = false;
  absl::flat_hash_map<HloComputation*, int> live_computation_call_count;
  if (HloComputation* entry_computation = module->entry_computation()) {
    ++live_computation_call_count[entry_computation];
  }
  for (auto* computation : module->MakeComputationPostOrder()) {
    for (auto* instruction : computation->instructions()) {
      for (auto* subcomp : instruction->called_computations()) {
        ++live_computation_call_count[subcomp];
      }
    }
  }
  for (auto* computation : module->MakeComputationPostOrder()) {
    if (!live_computation_call_count.contains(computation)) {
      TF_RETURN_IF_ERROR(RecursivelyRemoveDeadComputation(
          module, computation, live_computation_call_count));
      module_contains_dead_code = true;
    }
  }
  return module_contains_dead_code;
}
absl::StatusOr<bool> HloDCE::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  VLOG(2) << "Before dce:";
  XLA_VLOG_LINES(2, module->ToString());
  auto computations = module->MakeComputationPostOrder(execution_threads);
  std::reverse(computations.begin(), computations.end());
  for (auto* computation : computations) {
    TF_ASSIGN_OR_RETURN(
        bool changed_for_computation,
        RunOnComputation(computation, remove_cross_partition_collective_ops_));
    changed |= changed_for_computation;
  }
  TF_ASSIGN_OR_RETURN(bool module_contains_dead_code,
                      RecursivelyRemoveDeadComputations(module));
  changed |= module_contains_dead_code;
  if (changed) {
    VLOG(2) << "After dce:";
    XLA_VLOG_LINES(2, module->ToString());
  }
  return changed;
}
}  