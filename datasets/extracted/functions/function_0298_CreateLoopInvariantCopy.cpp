#include "xla/service/while_loop_invariant_code_motion.h"
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/map_util.h"
#include "xla/service/compile_time_cap.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/while_loop_analysis.h"
#include "xla/service/while_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
using absl::flat_hash_map;
using absl::flat_hash_set;
using absl::InlinedVector;
static void CreateLoopInvariantCopy(
    flat_hash_map<HloInstruction*, HloInstruction*>* hoisted_instructions,
    flat_hash_set<HloInstruction*>* unhoisted_invariant_instructions,
    HloInstruction* while_instr, HloInstruction* to_hoist) {
  HloComputation* parent_of_while = while_instr->parent();
  HloComputation* while_body = while_instr->while_body();
  struct DFSFrame {
    HloInstruction* instruction;
    int64_t operand_index;
  };
  InlinedVector<DFSFrame, 8> dfs_stack;
  dfs_stack.push_back({to_hoist, 0});
  HloInstruction* while_body_param = while_body->parameter_instruction(0);
  HloInstruction* while_operand = while_instr->mutable_operand(0);
  do {
    DFSFrame* frame = &dfs_stack.back();
    if (frame->operand_index == frame->instruction->operand_count()) {
      HloInstruction* old_instruction = frame->instruction;
      auto get_new_operand = [&](HloInstruction* old_operand) {
        return old_operand == while_body_param
                   ? while_operand
                   : FindOrDie(*hoisted_instructions, old_operand);
      };
      InlinedVector<HloInstruction*, 4> new_operands;
      absl::c_transform(old_instruction->operands(),
                        std::back_inserter(new_operands), get_new_operand);
      HloInstruction* new_instruction =
          parent_of_while->AddInstruction(old_instruction->CloneWithNewOperands(
              old_instruction->shape(), new_operands));
      InsertOrDie(hoisted_instructions, old_instruction, new_instruction);
      CHECK_EQ(unhoisted_invariant_instructions->erase(old_instruction),
               to_hoist != old_instruction &&
                   old_instruction->opcode() != HloOpcode::kConstant);
      dfs_stack.pop_back();
      continue;
    }
    HloInstruction* next_operand =
        frame->instruction->mutable_operand(frame->operand_index++);
    if (hoisted_instructions->contains(next_operand) ||
        next_operand == while_body_param) {
      continue;
    }
    dfs_stack.push_back({next_operand, 0});
  } while (!dfs_stack.empty());
}
bool WhileLoopInvariantCodeMotion::NotWorthHoistingIndividually(
    const HloInstruction& instruction) {
  if (instruction.IsCustomCall("Sharding")) {
    return true;
  }
  switch (instruction.opcode()) {
    default:
      return false;
    case HloOpcode::kConstant:
      return !hoist_constants_;
    case HloOpcode::kReshape:
      return !hoist_reshapes_;
    case HloOpcode::kBitcast:
    case HloOpcode::kBroadcast:
    case HloOpcode::kIota:
    case HloOpcode::kReverse:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
    case HloOpcode::kTuple:
      return true;
  }
}
absl::StatusOr<bool>
WhileLoopInvariantCodeMotion::TryHoistingInvariantInstructionsFromWhileBody(
    HloInstruction* while_instr, BoundNonLinearCompilerAnalysis* allowance) {
  auto print_no_metadata = HloPrintOptions{}.set_print_metadata(false);
  if (!while_instr->shape().IsTuple()) {
    return false;
  }
  std::string while_instr_name = while_instr->ToString(print_no_metadata);
  VLOG(2) << "Trying to hoist from " << while_instr_name;
  auto maybe_upper_bound = ComputeWhileLoopTripCountUpperBound(while_instr);
  if (maybe_upper_bound && *maybe_upper_bound <= 1) {
    VLOG(2) << "Loop has a trip count of at most 1, skipping.";
    return false;
  }
  HloComputation* while_body = while_instr->while_body();
  flat_hash_map<HloInstruction*, HloInstruction*> hoisted_instructions;
  flat_hash_set<HloInstruction*> unhoisted_invariant_instructions;
  for (auto* instr : WhileUtil::GetInvariantGTEsForWhileBody(*while_body)) {
    if (instr->shape().IsArray()) {
      InsertOrDie(&unhoisted_invariant_instructions, instr);
    }
  }
  if (unhoisted_invariant_instructions.empty() && !hoist_constants_) {
    return false;
  }
  for (auto* instruction : while_body->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kDomain ||
        instruction->IsCustomCall("SPMDFullToShardShape") ||
        instruction->IsCustomCall("SPMDShardShapeToFull")) {
      return false;
    }
  }
  std::vector<HloInstruction*> instructions_to_replace;
  std::vector<HloInstruction*> replacement_instructions;
  for (auto* instruction : while_body->MakeInstructionPostOrder()) {
    allowance->DeductCost(1);
    if (!allowance->ContinueAnalysis()) {
      return false;
    }
    if (instruction->HasSideEffect() ||
        instruction->opcode() == HloOpcode::kAfterAll ||
        instruction->opcode() == HloOpcode::kParameter ||
        !instruction->control_predecessors().empty() ||
        !instruction->control_successors().empty()) {
      continue;
    }
    if (!hoist_other_ && instruction->opcode() != HloOpcode::kConstant &&
        instruction->opcode() != HloOpcode::kReshape) {
      continue;
    }
    if (hoist_size_inflation_ratio_ &&
        instruction->opcode() != HloOpcode::kConstant) {
      int64_t input_size = 0, output_size = 0;
      for (auto* operand : instruction->operands()) {
        ShapeUtil::ForEachSubshape(
            operand->shape(), [&input_size, this](const Shape& subshape,
                                                  const ShapeIndex& ) {
              if (subshape.IsArray()) {
                input_size += shape_size_function_(subshape);
              }
            });
      }
      ShapeUtil::ForEachSubshape(
          instruction->shape(),
          [&output_size, this](const Shape& subshape,
                               const ShapeIndex& ) {
            if (subshape.IsArray()) {
              output_size += shape_size_function_(subshape);
            }
          });
      if (output_size > input_size * *hoist_size_inflation_ratio_) {
        continue;
      }
    }
    auto is_invariant = [&](HloInstruction* op) {
      return hoisted_instructions.find(op) != hoisted_instructions.end() ||
             unhoisted_invariant_instructions.contains(op) ||
             op->opcode() == HloOpcode::kConstant;
    };
    if (!absl::c_all_of(instruction->operands(), is_invariant)) {
      continue;
    }
    if (NotWorthHoistingIndividually(*instruction)) {
      VLOG(2) << "Adding " << instruction->ToString(print_no_metadata)
              << " to unhoisted invariant set.";
      if (instruction->opcode() != HloOpcode::kConstant) {
        InsertOrDie(&unhoisted_invariant_instructions, instruction);
      }
      continue;
    }
    VLOG(2) << "Hoisting " << instruction->ToString(print_no_metadata);
    CreateLoopInvariantCopy(&hoisted_instructions,
                            &unhoisted_invariant_instructions, while_instr,
                            instruction);
    instructions_to_replace.push_back(instruction);
    replacement_instructions.push_back(
        FindOrDie(hoisted_instructions, instruction));
  }
  if (instructions_to_replace.empty()) {
    return false;
  }
  TF_ASSIGN_OR_RETURN(
      WhileUtil::MakeInstructionsLiveInResult live_in_instructions_result,
      WhileUtil::MakeInstructionsLiveIn(while_instr, replacement_instructions));
  HloComputation* new_while_body =
      live_in_instructions_result.new_while_instr->while_body();
  for (int i = 0; i < instructions_to_replace.size(); i++) {
    HloInstruction* instruction_to_replace_in_new_while =
        FindOrDie(live_in_instructions_result.while_body_instruction_map,
                  instructions_to_replace[i]);
    TF_RETURN_IF_ERROR(new_while_body->ReplaceInstruction(
        instruction_to_replace_in_new_while,
        live_in_instructions_result.while_body_live_in_values[i]));
  }
  VLOG(1) << "Hoisted " << instructions_to_replace.size()
          << " instructions from " << while_instr_name;
  return true;
}
absl::StatusOr<bool> WhileLoopInvariantCodeMotion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(2) << "HLO module before WhileLoopInvariantCodeMotion:";
  XLA_VLOG_LINES(2, module->ToString());
  bool changed = false;
  std::vector<HloInstruction*> while_instrs;
  for (auto* comp : module->MakeComputationPostOrder(execution_threads)) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(while_instrs),
                    HloPredicateIsOp<HloOpcode::kWhile>);
  }
  BoundNonLinearCompilerAnalysis allowance(module, name(), 10);
  for (HloInstruction* while_instr : while_instrs) {
    if (!allowance.ContinueAnalysis()) {
      break;
    }
    TF_ASSIGN_OR_RETURN(
        bool result,
        TryHoistingInvariantInstructionsFromWhileBody(while_instr, &allowance));
    changed |= result;
  }
  if (changed) {
    HloDCE dce;
    TF_RETURN_IF_ERROR(dce.Run(module).status());
  }
  if (changed) {
    VLOG(2) << "HLO module after WhileLoopInvariantCodeMotion:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "HLO module unchanged after WhileLoopInvariantCodeMotion";
  }
  return changed;
}
}  