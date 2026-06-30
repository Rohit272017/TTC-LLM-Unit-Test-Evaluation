#include "xla/service/while_loop_expensive_invariant_code_motion.h"
#include <iterator>
#include <string>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "xla/service/while_loop_analysis.h"
#include "xla/service/while_util.h"
#include "xla/shape_util.h"
#include "xla/util.h"
namespace xla {
namespace {
using absl::flat_hash_map;
using absl::flat_hash_set;
using absl::InlinedVector;
struct InvariantInfo {
  explicit InvariantInfo(int64_t user_count)
      : remaining_user_count(user_count) {}
  int64_t transitive_input_size = 0;
  int64_t remaining_user_count;
  HloInstruction* hoisted_copy = nullptr;
  InlinedVector<HloInstruction*, 2> blocked_users;
};
static void CreateLoopInvariantCopy(
    flat_hash_map<HloInstruction*, InvariantInfo>* invariant_instructions,
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
      InvariantInfo& info = FindOrDie(*invariant_instructions, old_instruction);
      if (info.hoisted_copy == nullptr) {
        auto get_new_operand = [&](HloInstruction* old_operand) {
          return old_operand == while_body_param
                     ? while_operand
                     : FindOrDie(*invariant_instructions, old_operand)
                           .hoisted_copy;
        };
        InlinedVector<HloInstruction*, 4> new_operands;
        absl::c_transform(old_instruction->operands(),
                          std::back_inserter(new_operands), get_new_operand);
        HloInstruction* new_instruction = parent_of_while->AddInstruction(
            old_instruction->CloneWithNewOperands(old_instruction->shape(),
                                                  new_operands));
        info.hoisted_copy = new_instruction;
      }
      dfs_stack.pop_back();
      continue;
    }
    HloInstruction* next_operand =
        frame->instruction->mutable_operand(frame->operand_index++);
    if (next_operand == while_body_param ||
        FindOrDie(*invariant_instructions, next_operand).hoisted_copy !=
            nullptr) {
      continue;
    }
    dfs_stack.push_back({next_operand, 0});
  } while (!dfs_stack.empty());
}
}  
absl::StatusOr<bool> WhileLoopExpensiveInvariantCodeMotion::
    TryHoistingInvariantInstructionsFromWhileBody(HloInstruction* while_instr) {
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
  flat_hash_map<HloInstruction*, InvariantInfo> invariant_instructions;
  flat_hash_map<HloInstruction*, int64_t> to_hoist_when_ready;
  for (auto* instr : WhileUtil::GetInvariantGTEsForWhileBody(*while_body)) {
    if (instr->shape().IsArray()) {
      auto emplace_result = invariant_instructions.emplace(
          instr, InvariantInfo(instr->user_count() - 1));
      CHECK(emplace_result.second);
      InvariantInfo& info = emplace_result.first->second;
      info.transitive_input_size = shape_size_function_(instr->shape());
    }
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
  auto hoist = [&](HloInstruction* instruction, const InvariantInfo& info) {
    if (info.hoisted_copy) {
      return;
    }
    VLOG(2) << "Hoisting " << instruction->ToString(print_no_metadata);
    CreateLoopInvariantCopy(&invariant_instructions, while_instr, instruction);
    instructions_to_replace.push_back(instruction);
    replacement_instructions.push_back(info.hoisted_copy);
  };
  flat_hash_set<HloInstruction*> checked_operands;
  for (auto* instruction : while_body->MakeInstructionPostOrder()) {
    if (instruction->HasSideEffect() ||
        instruction->opcode() == HloOpcode::kParameter ||
        !instruction->control_predecessors().empty() ||
        !instruction->control_successors().empty() ||
        instruction == while_body->root_instruction()) {
      continue;
    }
    auto is_invariant = [&](HloInstruction* op) {
      return invariant_instructions.find(op) != invariant_instructions.end();
    };
    if (!absl::c_all_of(instruction->operands(), is_invariant)) {
      continue;
    }
    auto emplace_result = invariant_instructions.emplace(
        instruction, InvariantInfo(instruction->user_count()));
    CHECK(emplace_result.second);
    InvariantInfo& instr_info = emplace_result.first->second;
    for (auto* user : instruction->users()) {
      if (user == while_body->root_instruction()) {
        --instr_info.remaining_user_count;
        break;
      }
    }
    int64_t num_blocking_operands = 0;
    int64_t output_size = 0;
    for (auto* operand : instruction->operands()) {
      auto& operand_info = invariant_instructions.at(operand);
      if (!checked_operands.contains(operand)) {
        instr_info.transitive_input_size += operand_info.transitive_input_size;
        --operand_info.remaining_user_count;
        checked_operands.insert(operand);
      }
      if (operand_info.remaining_user_count == 0) {
        for (auto* user : operand_info.blocked_users) {
          auto it = to_hoist_when_ready.find(user);
          if (it != to_hoist_when_ready.end()) {
            auto& num_blocking = it->second;
            CHECK_GT(num_blocking, 0);
            --num_blocking;
            if (num_blocking == 0) {
              hoist(user, invariant_instructions.at(user));
              to_hoist_when_ready.erase(it);
            }
          }
        }
        operand_info.blocked_users.clear();
      } else if (operand_info.remaining_user_count > 0) {
        ++num_blocking_operands;
        if (operand_info.blocked_users.empty() ||
            operand_info.blocked_users.back() != instruction) {
          operand_info.blocked_users.push_back(instruction);
        }
      } else {
        LOG(FATAL)
            << "An instruction should not have number of negative users.";
      }
    }
    checked_operands.erase(checked_operands.begin(), checked_operands.end());
    ShapeUtil::ForEachSubshape(
        instruction->shape(),
        [&output_size, this](const Shape& subshape,
                             const ShapeIndex& ) {
          if (subshape.IsArray()) {
            output_size += shape_size_function_(subshape);
          }
        });
    if (output_size > instr_info.transitive_input_size) {
      continue;
    }
    if (!worth_hoisting_individually_(instruction)) {
      continue;
    }
    if (num_blocking_operands > 0) {
      to_hoist_when_ready.emplace(instruction, num_blocking_operands);
      continue;
    }
    hoist(instruction, instr_info);
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
absl::StatusOr<bool> WhileLoopExpensiveInvariantCodeMotion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(2) << "HLO module before WhileLoopExpensiveInvariantCodeMotion:";
  XLA_VLOG_LINES(2, module->ToString());
  bool changed = false;
  std::vector<HloInstruction*> while_instrs;
  for (auto* comp : module->computations(execution_threads)) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(while_instrs),
                    HloPredicateIsOp<HloOpcode::kWhile>);
  }
  for (HloInstruction* while_instr : while_instrs) {
    TF_ASSIGN_OR_RETURN(
        bool result,
        TryHoistingInvariantInstructionsFromWhileBody(while_instr));
    changed |= result;
  }
  if (changed) {
    VLOG(2) << "HLO module after WhileLoopExpensiveInvariantCodeMotion:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2)
        << "HLO module unchanged after WhileLoopExpensiveInvariantCodeMotion";
  }
  return changed;
}
}  