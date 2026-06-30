#include "xla/service/scan_loop_accumulator_input_unification.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/call_graph.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/hlo_dataflow_analysis.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/service/while_loop_simplifier.h"
#include "xla/service/while_loop_unroller.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace {
bool LoopIndexIsReadOnly(const HloAliasAnalysis& alias_analysis,
                         HloInstruction* while_instr, int64_t idx) {
  const HloDataflowAnalysis& dataflow_analysis =
      alias_analysis.dataflow_analysis();
  return !(
      dataflow_analysis.GetValueSet(while_instr->while_init(), {idx})
              .values()
              .size() > 1 ||
      dataflow_analysis.GetValueSet(while_instr, {idx}).values().size() > 1 ||
      dataflow_analysis.GetUniqueValueAt(while_instr, {idx}) !=
          dataflow_analysis.GetUniqueValueAt(while_instr->while_init(), {idx}));
}
std::vector<std::pair<HloInstruction*, HloInstruction*>>
FindAccumulatorInputPairs(const HloAliasAnalysis& alias_analysis,
                          HloInstruction* while_instr,
                          const WhileLoopConfig& config) {
  HloComputation* computation = while_instr->while_body();
  HloInstruction* body_param = computation->parameter_instruction(0);
  std::vector<HloInstruction*> possible_acc;
  for (int64_t param_idx = 0;
       param_idx < while_instr->while_init()->operand_count(); ++param_idx) {
    for (HloInstruction* gte : body_param->users()) {
      if (!Match(gte, match::GetTupleElement().WithTupleIndex(param_idx))) {
        continue;
      }
      if (gte->operand(0) != body_param) {
        continue;
      }
      if (gte->user_count() > 1 || gte->user_count() == 0) {
        continue;
      }
      HloInstruction* gte_user = gte->users().at(0);
      if (MatchShapeCoveringDynamicIndexInstruction(
              gte_user, gte, HloOpcode::kDynamicUpdateSlice, config)
              .has_value()) {
        if (computation->root_instruction()->mutable_operand(param_idx) ==
            gte_user) {
          possible_acc.push_back(gte);
          VLOG(3) << "accumulator index: " << param_idx << " = " << gte->name();
        }
      }
    }
  }
  auto operand_index = [](HloInstruction* instr,
                          HloInstruction* operand) -> int64_t {
    for (int64_t i = 0; i < instr->operand_count(); ++i) {
      if (operand == instr->operand(i)) {
        return i;
      }
    }
    return -1;
  };
  auto find_gte_instr = [](HloInstruction* tuple,
                           int64_t idx) -> HloInstruction* {
    for (HloInstruction* instr : tuple->parent()->MakeInstructionPostOrder()) {
      HloInstruction* operand;
      if (Match(instr, match::GetTupleElement()
                           .WithOperand(0, match::Op(&operand))
                           .WithTupleIndex(idx))) {
        if (operand != tuple) {
          continue;
        }
        return instr;
      }
    }
    return nullptr;
  };
  auto check_single_user_not_null = [](HloInstruction* instr) -> bool {
    if (instr == nullptr || instr->user_count() != 1) {
      return false;
    }
    return true;
  };
  std::vector<std::pair<HloInstruction*, HloInstruction*>> acc_input_pairs;
  HloComputation* outer_while_body = while_instr->parent();
  for (HloInstruction* acc : possible_acc) {
    VLOG(3) << "Looking for corresponding input for " << acc->name();
    HloInstruction* acc_gte_outer_body =
        find_gte_instr(while_instr, acc->tuple_index());
    if (acc_gte_outer_body == nullptr) {
      continue;
    }
    int64_t idx =
        operand_index(outer_while_body->root_instruction(), acc_gte_outer_body);
    VLOG(3) << "Accumulator output of the scan in the outer body = "
            << acc_gte_outer_body->name() << ", index = " << idx;
    if (idx == -1) {
      continue;
    }
    HloInstruction* input_gte_outer =
        find_gte_instr(outer_while_body->parameter_instruction(0), idx);
    if (!check_single_user_not_null(input_gte_outer)) {
      continue;
    }
    if (input_gte_outer->users().at(0) != while_instr->while_init()) {
      continue;
    }
    VLOG(3) << "Input parameter outer body = " << input_gte_outer->name()
            << ", index = " << input_gte_outer->tuple_index();
    int64_t input_idx_inner =
        operand_index(while_instr->while_init(), input_gte_outer);
    HloInstruction* input_gte_inner =
        find_gte_instr(computation->parameter_instruction(0), input_idx_inner);
    if (!LoopIndexIsReadOnly(alias_analysis, while_instr, input_idx_inner)) {
      continue;
    }
    VLOG(3) << "Input parameter scan body = " << input_gte_inner->name()
            << ", index = " << input_gte_inner->tuple_index();
    if (input_gte_inner->user_count() != 2) {
      continue;
    }
    HloInstruction* gte_user = input_gte_inner->users().at(0);
    VLOG(3) << "User of the inner loop input = " << gte_user->ToString();
    if (MatchShapeCoveringDynamicIndexInstruction(
            gte_user, input_gte_inner, HloOpcode::kDynamicSlice, config)
            .has_value()) {
      acc_input_pairs.emplace_back(acc, input_gte_inner);
    }
  }
  return acc_input_pairs;
}
absl::StatusOr<bool> UnifyAccumulatorWithInput(
    const HloAliasAnalysis& alias_analysis,
    std::vector<std::pair<HloInstruction*, WhileLoopConfig>> unrollable_loops) {
  std::unique_ptr<CallGraph> call_graph =
      CallGraph::Build(&alias_analysis.dataflow_analysis().module());
  auto is_while_body = [&](HloComputation* comp) {
    std::vector<HloInstruction*> callers =
        call_graph->GetComputationCallers(comp);
    return !callers.empty() && callers.at(0)->opcode() == HloOpcode::kWhile;
  };
  std::vector<HloInstruction*> changed_loops;
  bool unified = false;
  for (auto& [while_instr, loop_config] : unrollable_loops) {
    if (!is_while_body(while_instr->parent())) {
      continue;
    }
    auto acc_input_pairs =
        FindAccumulatorInputPairs(alias_analysis, while_instr, loop_config);
    for (const auto& [acc, input] : acc_input_pairs) {
      if (Match(while_instr->while_init()->mutable_operand(acc->tuple_index()),
                match::GetTupleElement(match::Parameter()))) {
        continue;
      }
      VLOG(3) << while_instr->name() << " -> " << "<accumulator_@"
              << acc->tuple_index() << ": " << acc->name() << ", " << "input_@"
              << input->tuple_index() << ": " << input->name() << ">";
      TF_RETURN_IF_ERROR(input->ReplaceAllUsesWith(acc));
      TF_RETURN_IF_ERROR(while_instr->while_init()->ReplaceOperandWith(
          acc->tuple_index(),
          while_instr->while_init()->mutable_operand(input->tuple_index())));
      if (input->user_count() == 0) {
        TF_RETURN_IF_ERROR(while_instr->while_body()->RemoveInstruction(input));
      }
      unified = true;
    }
  }
  return unified;
}
}  
absl::StatusOr<bool> ScanLoopAccumulatorInputUnification::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(2) << "HLO module before ScanLoopAccumulatorInputUnification:";
  XLA_VLOG_LINES(2, module->ToString());
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(module));
  std::vector<std::pair<HloInstruction*, WhileLoopConfig>> unrollable_loops =
      WhileLoopUnroller::GetUnrollableLoops(module, execution_threads,
                                            std::nullopt);
  TF_ASSIGN_OR_RETURN(bool changed, UnifyAccumulatorWithInput(
                                        *alias_analysis, unrollable_loops));
  if (changed) {
    for (auto& [while_instr, loop_config] : unrollable_loops) {
      TF_RETURN_IF_ERROR(TryRemoveDeadWhileParams(while_instr).status());
    }
    TF_RETURN_IF_ERROR(TupleSimplifier{}.Run(module).status());
    TF_RETURN_IF_ERROR(module->RemoveUnusedComputations());
    VLOG(2) << "HLO module after ScanLoopAccumulatorInputUnification:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "HLO module unchanged after ScanLoopAccumulatorInputUnification";
  }
  return changed;
}
}  