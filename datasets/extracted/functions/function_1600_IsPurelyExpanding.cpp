#include "xla/service/while_loop_fusible_sinking.h"
#include <cstdint>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/while_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace {
bool IsPurelyExpanding(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kBroadcast ||
         (instr->opcode() == HloOpcode::kConstant &&
          instr->shape().rank() == 0) ||
         instr->opcode() == HloOpcode::kIota;
}
bool IsFusionCandidate(const HloInstruction* instr) {
  return instr->opcode() != HloOpcode::kRng &&
         (instr->IsElementwise() || instr->opcode() == HloOpcode::kReshape ||
          instr->opcode() == HloOpcode::kTranspose);
}
}  
bool WhileLoopFusibleSinking::IsSinkableFusion(HloInstruction* while_operand) {
  absl::InlinedVector<HloInstruction*, 8> worklist;
  absl::flat_hash_set<int> visited;
  worklist.push_back(while_operand);
  while (!worklist.empty()) {
    HloInstruction* to_process = worklist.back();
    worklist.pop_back();
    if (!to_process->IsFusible()) {
      return false;
    }
    if (!visited.insert(to_process->unique_id()).second) {
      if (visited.size() > 100) {
        return false;
      }
      continue;
    }
    if (IsPurelyExpanding(to_process)) {
      continue;
    }
    if (IsFusionCandidate(to_process)) {
      for (auto* op : to_process->operands()) {
        worklist.push_back(op);
      }
      continue;
    }
    return false;
  }
  return true;
}
HloInstruction* WhileLoopFusibleSinking::CreateSinkableFusion(
    HloInstruction* while_operand) {
  HloInstruction* fusion =
      while_operand->AddInstruction(while_operand->CreateFusion(
          while_operand->shape(), HloInstruction::FusionKind::kLoop,
          while_operand));
  bool did_fuse = IsFusionCandidate(while_operand);
  while (did_fuse) {
    did_fuse = false;
    for (int64_t i = fusion->operand_count() - 1; i >= 0; --i) {
      HloInstruction* op = fusion->mutable_operand(i);
      if (IsPurelyExpanding(op)) {
        continue;
      }
      fusion->FuseInstruction(op);
      did_fuse = true;
      break;
    }
  }
  did_fuse = true;
  while (did_fuse) {
    did_fuse = false;
    for (int64_t i = fusion->operand_count() - 1; i >= 0; --i) {
      HloInstruction* op = fusion->mutable_operand(i);
      if (IsPurelyExpanding(op)) {
        fusion->FuseInstruction(op);
        did_fuse = true;
        break;
      }
    }
  }
  return fusion;
}
absl::StatusOr<bool> WhileLoopFusibleSinking::TrySinkingFusiblesIntoWhileLoop(
    HloInstruction* while_instr) {
  HloComputation* while_cond = while_instr->while_condition();
  HloComputation* while_body = while_instr->while_body();
  if (call_counts_[while_body] > 1 || call_counts_[while_cond] > 1) {
    return false;
  }
  HloInstruction* init_value = while_instr->mutable_operand(0);
  if (init_value->opcode() != HloOpcode::kTuple) {
    return false;
  }
  bool changed = false;
  std::vector<HloInstruction*> invariant_body_gtes =
      WhileUtil::GetInvariantGTEsForWhileBody(*while_body);
  std::vector<int64_t> tuple_indices;
  std::vector<HloInstruction*> new_operands;
  for (HloInstruction* invariant_body_gte : invariant_body_gtes) {
    int64_t index = invariant_body_gte->tuple_index();
    if (while_instr->operand_count() == 0 || init_value->operand_count() == 0) {
      CHECK_EQ(while_instr->user_count(), 0);
      VLOG(3) << "Each element in the operand tuple of the while instruction '"
              << while_instr->name()
              << "' was an invariant value, whose usage has been replaced "
                 " directly by the value.";
      break;
    }
    HloInstruction* invariant_value = init_value->mutable_operand(index);
    if (absl::c_any_of(invariant_body_gte->users(),
                       [](const HloInstruction* use) {
                         switch (use->opcode()) {
                           case HloOpcode::kDynamicSlice:
                           case HloOpcode::kGather:
                           case HloOpcode::kSlice:
                             return true;
                           default:
                             return false;
                         }
                       })) {
      continue;
    }
    if (init_value->IsRoot() || init_value->user_count() > 1) {
      init_value = init_value->AddInstruction(init_value->Clone());
      TF_RETURN_IF_ERROR(while_instr->ReplaceOperandWith(0, init_value));
    }
    if (!IsSinkableFusion(invariant_value)) {
      continue;
    }
    HloInstruction* fusion = CreateSinkableFusion(invariant_value);
    changed = true;
    if (fusion->operand_count() > 0 &&
        (while_instr->IsRoot() ||
         absl::c_any_of(while_instr->users(), [&](HloInstruction* use) {
           return use->opcode() != HloOpcode::kGetTupleElement;
         }))) {
      auto uses = while_instr->users();
      std::vector<HloInstruction*> gtes(init_value->operand_count());
      for (int64_t i = 0; i < gtes.size(); ++i) {
        gtes[i] = while_instr->AddInstruction(
            HloInstruction::CreateGetTupleElement(while_instr, i));
      }
      HloInstruction* tuple =
          while_instr->AddInstruction(HloInstruction::CreateTuple(gtes));
      if (while_instr->IsRoot()) {
        while_instr->parent()->set_root_instruction(tuple);
      }
      if (!uses.empty()) {
        TF_RETURN_IF_ERROR(while_instr->ReplaceUsesWith(uses, tuple));
      }
    }
    absl::InlinedVector<HloInstruction*, 2> invariant_output_uses;
    for (auto use : while_instr->users()) {
      if (use->opcode() == HloOpcode::kGetTupleElement &&
          use->tuple_index() == index) {
        invariant_output_uses.push_back(use);
      }
    }
    for (auto use : invariant_output_uses) {
      TF_RETURN_IF_ERROR(
          while_instr->parent()->ReplaceInstruction(use, invariant_value));
    }
    HloInstruction* root = while_body->root_instruction();
    HloInstruction* parameter = while_body->parameter_instruction(0);
    tuple_indices.resize(fusion->operand_count());
    int64_t next_index = init_value->operand_count();
    new_operands.resize(fusion->operand_count());
    for (int64_t i = 0; i < fusion->operand_count(); ++i) {
      init_value->AppendOperand(fusion->mutable_operand(i));
      parameter->mutable_shape()->mutable_tuple_shapes()->push_back(
          fusion->mutable_operand(i)->shape());
      new_operands[i] = root->AddInstruction(
          HloInstruction::CreateGetTupleElement(parameter, next_index++));
      root->AppendOperand(new_operands[i]);
    }
    *(init_value->mutable_shape()) = parameter->shape();
    *(while_instr->mutable_shape()) = parameter->shape();
    *(while_cond->parameter_instruction(0)->mutable_shape()) =
        parameter->shape();
    *(root->mutable_shape()) = parameter->shape();
    auto cloned_fusion = while_body->AddInstruction(
        fusion->CloneWithNewOperands(fusion->shape(), new_operands));
    TF_RETURN_IF_ERROR(fusion->parent()->RemoveInstruction(fusion));
    TF_RETURN_IF_ERROR(
        while_body->ReplaceInstruction(invariant_body_gte, cloned_fusion));
    TF_RETURN_IF_ERROR(cloned_fusion->Defuse());
  }
  return changed;
}
absl::StatusOr<bool> WhileLoopFusibleSinking::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  call_counts_.clear();
  bool changed = false;
  std::vector<HloInstruction*> while_instrs;
  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(while_instrs),
                    HloPredicateIsOp<HloOpcode::kWhile>);
  }
  for (HloInstruction* while_instr : while_instrs) {
    call_counts_[while_instr->while_body()]++;
    call_counts_[while_instr->while_condition()]++;
  }
  for (HloInstruction* while_instr : while_instrs) {
    TF_ASSIGN_OR_RETURN(bool result,
                        TrySinkingFusiblesIntoWhileLoop(while_instr));
    changed |= result;
  }
  return changed;
}
}  