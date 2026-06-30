#include "xla/service/while_loop_unroller.h"
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>
#include "absl/algorithm/algorithm.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/overflow_util.h"
#include "xla/service/call_inliner.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/flatten_call_graph.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/service/while_loop_analysis.h"
#include "xla/service/while_loop_constant_sinking.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace {
using hlo_query::ContainsInstrWithOpcode;
std::unique_ptr<HloComputation> MakeTrivialLoopCondition(
    HloInstruction* while_op, std::string_view name, int64_t induction_idx,
    int64_t init_value) {
  auto condition_builder = HloComputation::Builder(name);
  absl::StatusOr<HloInstruction*> param_instruction =
      condition_builder.AddParameter(
          while_op->while_condition()->parameter_instruction(0)->Clone());
  HloInstruction* indvar_instruction =
      condition_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          param_instruction.value(), induction_idx));
  HloInstruction* init_value_constant = condition_builder.AddInstruction(
      MakeScalarConstantWithShape(indvar_instruction->shape(), init_value));
  return condition_builder.Build(
      condition_builder.AddInstruction(HloInstruction::CreateCompare(
          ShapeUtil::MakeShape(PrimitiveType::PRED, {}), indvar_instruction,
          init_value_constant, ComparisonDirection::kLe)));
}
absl::Status HandleDynamicGteOrTuple(HloInstruction* instr) {
  if (instr->IsCustomCall("DynamicGte")) {
    HloEvaluator evaluator(0);
    TF_ASSIGN_OR_RETURN(
        Literal index_lit,
        evaluator.Evaluate(instr->mutable_operand(1),
                           {},
                           true));
    auto index = LiteralUtil::LiteralAsScalarInt64(std::move(index_lit));
    TF_RET_CHECK(index.has_value());
    return instr->parent()->ReplaceInstruction(
        instr, instr->AddInstruction(HloInstruction::CreateGetTupleElement(
                   instr->mutable_operand(0), index.value())));
  } else if (instr->IsCustomCall("DynamicTuple")) {
    HloEvaluator evaluator(0);
    std::vector<HloInstruction*> tuple_operands;
    TF_ASSIGN_OR_RETURN(
        Literal index_lit,
        evaluator.Evaluate(instr->mutable_operand(2),
                           {},
                           true));
    auto index = LiteralUtil::LiteralAsScalarInt64(std::move(index_lit));
    TF_RET_CHECK(index.has_value());
    for (int64_t i = 0; i < instr->operand(0)->shape().tuple_shapes_size();
         i++) {
      if (i == index.value()) {
        tuple_operands.push_back(instr->mutable_operand(1));
      } else {
        HloInstruction* slice =
            instr->AddInstruction(HloInstruction::CreateGetTupleElement(
                instr->mutable_operand(0), i));
        tuple_operands.push_back(slice);
      }
    }
    return instr->parent()->ReplaceInstruction(
        instr,
        instr->AddInstruction(HloInstruction::CreateTuple(tuple_operands)));
  }
  return absl::OkStatus();
}
absl::Status ReplaceInductionVarUses(HloComputation* body,
                                     HloInstruction* induction_value_constant,
                                     int64_t induction_var_idx) {
  for (HloInstruction* body_inst : body->instructions()) {
    if (!Match(body_inst,
               match::GetTupleElement(match::Parameter().WithParameterNum(0))
                   .WithTupleIndex(induction_var_idx))) {
      continue;
    }
    std::vector<HloInstruction*> indvar_uses;
    indvar_uses.reserve(body_inst->users().size());
    for (HloInstruction* indvar_use : body_inst->users()) {
      indvar_uses.push_back(indvar_use);
    }
    for (HloInstruction* indvar_use : indvar_uses) {
      if (Match(indvar_use, match::Add(match::GetTupleElement().WithTupleIndex(
                                           induction_var_idx),
                                       match::Constant()))) {
        continue;
      }
      for (int64_t i = 0; i < indvar_use->operand_count(); ++i) {
        const HloInstruction* indvar_use_operand = indvar_use->operand(i);
        if (indvar_use_operand == body_inst) {
          TF_RETURN_IF_ERROR(
              indvar_use->ReplaceOperandWith(i, induction_value_constant));
        }
      }
    }
  }
  return absl::OkStatus();
}
absl::StatusOr<std::unique_ptr<HloComputation>>
UnrollSingleIterationOfTrivialLoop(HloInstruction* while_op,
                                   WhileLoopConfig config,
                                   const int64_t induction_value) {
  std::unique_ptr<HloComputation> while_body_clone =
      while_op->while_body()->Clone(
          absl::StrCat(while_op->name(), induction_value));
  HloInstruction* induction_var_hlo =
      while_op->mutable_operand(0)->mutable_operand(config.induction_var_idx);
  int64_t unique_channel_id = hlo_query::NextChannelId(*while_op->GetModule());
  HloInstruction* induction_value_constant = while_body_clone->AddInstruction(
      MakeScalarConstantWithShape(induction_var_hlo->shape(), induction_value));
  TF_RETURN_IF_ERROR(ReplaceInductionVarUses(while_body_clone.get(),
                                             induction_value_constant,
                                             config.induction_var_idx));
  for (HloInstruction* body_inst : while_body_clone->instructions()) {
    HloInstruction* collective = IsOrHasCollectiveWithChannelId(body_inst);
    if (collective != nullptr) {
      collective->set_channel_id(unique_channel_id++);
    }
    TF_RETURN_IF_ERROR(HandleDynamicGteOrTuple(body_inst));
  }
  return while_body_clone;
}
bool InitialFeasibilityCheck(const HloInstruction* while_op,
                             const WhileLoopConfig config,
                             const UnrollConfig unroll_config) {
  CHECK_EQ(while_op->opcode(), HloOpcode::kWhile);
  VLOG(5) << "Trying to unroll " << while_op->ToShortString();
  if (while_op->while_body()->instruction_count() >
      unroll_config.instruction_count_threshold) {
    VLOG(5) << absl::StrCat(
        "Cannot unroll while loop. Too many instructions in the body: ",
        while_op->while_body()->instruction_count());
    return false;
  }
  if (config.trip_count > unroll_config.trip_count_threshold) {
    VLOG(5) << absl::StrCat(
        "Cannot unroll while loop. The trip count is greater "
        "than the threshold: ",
        config.trip_count, " vs ", unroll_config.trip_count_threshold);
    return false;
  }
  if (config.trip_count * while_op->while_body()->instruction_count() >
      unroll_config.expand_factor_threshold) {
    VLOG(5) << absl::StrCat(
        "Not attempting to unroll due to instruction count "
        "increase explosion. New instruction count: ",
        config.trip_count * while_op->while_body()->instruction_count(), " vs ",
        unroll_config.expand_factor_threshold);
    return false;
  }
  return true;
}
absl::StatusOr<bool> UnrollInternal(HloInstruction* while_op,
                                    WhileLoopConfig config) {
  VLOG(3) << "Unrolling while instruction " << while_op->ToShortString()
          << " with body instruction count "
          << while_op->while_body()->instruction_count();
  HloModule* module = while_op->GetModule();
  HloComputation* computation = while_op->parent();
  HloInstruction* unrolled_body_call_op;
  std::vector<HloInstruction*> call_operands = {while_op->operands().at(0)};
  for (int64_t i = config.init; i < config.trip_count + config.init; ++i) {
    CHECK(OverflowSafeAdd(i, (int64_t)1).has_value());
    HloComputation* unrolled_body = module->AddEmbeddedComputation(
        UnrollSingleIterationOfTrivialLoop(while_op, config, i).value());
    unrolled_body_call_op =
        computation->AddInstruction(HloInstruction::CreateCall(
            while_op->shape(), call_operands, unrolled_body));
    call_operands.clear();
    call_operands.emplace_back(unrolled_body_call_op);
  }
  TF_RETURN_IF_ERROR(
      computation->ReplaceInstruction(while_op, unrolled_body_call_op));
  TF_RETURN_IF_ERROR(FlattenCallGraph().Run(module).status());
  return true;
}
absl::StatusOr<UnrollResult> UnrollInternalWrappedAndReturnReplacement(
    HloInstruction* while_op, WhileLoopConfig config) {
  VLOG(3) << "Unrolling (wrapped) while instruction "
          << while_op->ToShortString() << " with body instruction count "
          << while_op->while_body()->instruction_count();
  HloModule* module = while_op->GetModule();
  HloComputation* computation = while_op->parent();
  HloInstruction* unrolled_body_call_op;
  std::vector<HloInstruction*> call_operands;
  auto body_builder =
      HloComputation::Builder(absl::StrCat("unrolled-body-", while_op->name()));
  absl::StatusOr<HloInstruction*> p = body_builder.AddParameter(
      while_op->while_body()->parameter_instruction(0)->Clone());
  call_operands.emplace_back(std::move(p.value()));
  for (int64_t i = config.init; i < config.trip_count + config.init; ++i) {
    CHECK(OverflowSafeAdd(i, (int64_t)1).has_value());
    HloComputation* unrolled_body = module->AddEmbeddedComputation(
        UnrollSingleIterationOfTrivialLoop(while_op, config, i).value());
    unrolled_body_call_op = body_builder.AddInstruction(
        HloInstruction::CreateCall(while_op->shape(), call_operands,
                                   unrolled_body),
        absl::StrCat(while_op->name(), "-unrolled-body-call-", i));
    call_operands.clear();
    call_operands.emplace_back(unrolled_body_call_op);
  }
  HloComputation* new_body =
      module->AddEmbeddedComputation(body_builder.Build(unrolled_body_call_op));
  HloComputation* new_cond =
      module->AddEmbeddedComputation(MakeTrivialLoopCondition(
          while_op, absl::StrCat("unrolled", while_op->name(), "-cond"),
          config.induction_var_idx, config.init));
  HloInstruction* new_while_op =
      computation->AddInstruction(HloInstruction::CreateWhile(
          while_op->shape(), new_cond, new_body, while_op->mutable_operand(0)));
  while_op->SetupDerivedInstruction(new_while_op);
  CHECK_OK(computation->ReplaceInstruction(while_op, new_while_op));
  TF_RETURN_IF_ERROR(FlattenCallGraph().Run(module).status());
  UnrollResult result;
  result.unrolled = true;
  result.new_while_op = new_while_op;
  return result;
}
absl::StatusOr<bool> UnrollInternalWrapped(HloInstruction* while_op,
                                           WhileLoopConfig config) {
  TF_ASSIGN_OR_RETURN(
      UnrollResult result,
      UnrollInternalWrappedAndReturnReplacement(while_op, config));
  return result.unrolled;
}
};  
bool IsLoopInductionVar(const HloInstruction* instr,
                        const WhileLoopConfig& config) {
  if (!instr->parent()->IsFusionComputation()) {
    return Match(instr, match::GetTupleElement(match::Parameter(),
                                               config.induction_var_idx));
  } else {
    if (!Match(instr, match::Parameter())) {
      return false;
    }
    HloInstruction* caller_fusion = instr->parent()->FusionInstruction();
    return IsLoopInductionVar(caller_fusion->operand(instr->parameter_number()),
                              config);
  }
}
bool IsEffectivelyStatic(const HloInstruction* instr,
                         const WhileLoopConfig& config) {
  switch (instr->opcode()) {
    case HloOpcode::kConstant:
      return true;
    case HloOpcode::kParameter: {
      if (instr->parent()->IsFusionComputation()) {
        HloInstruction* caller_fusion = instr->parent()->FusionInstruction();
        return IsEffectivelyStatic(
            caller_fusion->operand(instr->parameter_number()), config);
      }
      return false;
    }
    case HloOpcode::kGetTupleElement: {
      if (instr->parent() != config.while_instr->while_body()) {
        return false;
      }
      if (!Match(instr, match::GetTupleElement(match::Parameter(),
                                               config.induction_var_idx))) {
        return false;
      }
      return true;
    }
    default: {
      for (int64_t i = 0; i < instr->operand_count(); ++i) {
        if (!IsEffectivelyStatic(instr->operand(i), config)) {
          return false;
        }
      }
      return true;
    }
  }
}
std::optional<int64_t> MatchEffectivelyStaticDynamicSliceInsideLoop(
    const HloInstruction* instr, const HloInstruction* input,
    const WhileLoopConfig& config) {
  if (instr->opcode() != HloOpcode::kDynamicSlice) {
    return std::nullopt;
  }
  int64_t start_indices_offset = 1;
  const HloInstruction* operand = instr->operand(0);
  if (operand != input) {
    VLOG(3) << "Input of dynamic index instruction is not the given operand.";
    return std::nullopt;
  }
  int64_t dynamic_index = -1;
  for (int64_t start_index = start_indices_offset;
       start_index < instr->operand_count(); ++start_index) {
    const HloInstruction* index = instr->operand(start_index);
    if (Match(index, match::ConstantScalar())) {
      std::optional<int64_t> offset =
          LiteralUtil::LiteralAsScalarInt64(index->literal());
      if (offset.has_value() && offset.value() != 0) {
        VLOG(3) << "Constant index " << start_index << " is not zero.";
        return std::nullopt;
      }
      continue;
    }
    if (IsEffectivelyStatic(index, config)) {
      if (dynamic_index != -1) {
        VLOG(3) << "Multiple non-constant indices.";
        return std::nullopt;
      }
      dynamic_index = start_index - start_indices_offset;
    }
  }
  if (dynamic_index == -1) {
    VLOG(3) << "No dynamic index found.";
    return std::nullopt;
  }
  return dynamic_index;
}
std::optional<int64_t> MatchShapeCoveringDynamicIndexInstruction(
    const HloInstruction* instr, const HloInstruction* input, HloOpcode opcode,
    const WhileLoopConfig& config) {
  if (instr->opcode() != opcode) {
    return std::nullopt;
  }
  int64_t start_indices_offset;
  if (instr->opcode() == HloOpcode::kDynamicSlice) {
    start_indices_offset = 1;
  } else if (instr->opcode() == HloOpcode::kDynamicUpdateSlice) {
    start_indices_offset = 2;
  } else {
    return std::nullopt;
  }
  const HloInstruction* operand = instr->operand(0);
  if (input != nullptr && operand != input) {
    VLOG(3) << "Input of dynamic index instruction is not the given operand.";
    return std::nullopt;
  }
  int64_t dynamic_index = -1;
  for (int64_t start_index = start_indices_offset;
       start_index < instr->operand_count(); ++start_index) {
    const HloInstruction* index = instr->operand(start_index);
    if (Match(index, match::ConstantScalar())) {
      std::optional<int64_t> offset =
          LiteralUtil::LiteralAsScalarInt64(index->literal());
      if (offset.has_value() && offset.value() != 0) {
        VLOG(3) << "Constant index " << start_index << " is not zero.";
        return std::nullopt;
      }
      continue;
    }
    if (IsLoopInductionVar(index, config)) {
      if (dynamic_index != -1) {
        VLOG(3) << "Multiple non-constant indices.";
        return std::nullopt;
      }
      dynamic_index = start_index - start_indices_offset;
    }
  }
  if (dynamic_index == -1) {
    VLOG(3) << "No dynamic index found.";
    return std::nullopt;
  }
  if (operand->shape().dimensions(dynamic_index) != config.trip_count) {
    VLOG(3) << "The shape's broadcast_dim must be exactly equal to the loop "
               "trip count.";
    return std::nullopt;
  }
  return dynamic_index;
}
 std::optional<WhileLoopConfig> WhileLoopUnroller::IsLoopUnrollable(
    HloInstruction* while_op) {
  CHECK_EQ(while_op->opcode(), HloOpcode::kWhile);
  CHECK_EQ(while_op->operands().size(), 1);
  if (while_op->operands().size() != 1) {
    VLOG(5) << absl::StrCat(
        "Cannot unroll while loop ", while_op->name(),
        ". While loop must have a single "
        "tuple operand, instead has more than one operand: ",
        while_op->operands().size());
    return std::nullopt;
  }
  std::vector<HloInstruction*> while_dependees;
  for (HloComputation* comp : while_op->GetModule()->computations()) {
    for (HloInstruction* instr : comp->instructions()) {
      for (HloInstruction* control_dep : instr->control_predecessors()) {
        if (control_dep->opcode() == HloOpcode::kWhile) {
          while_dependees.push_back(control_dep);
        }
      }
    }
  }
  if (absl::linear_search(while_dependees.begin(), while_dependees.end(),
                          while_op)) {
    VLOG(2) << "Not attempting to unroll " << while_op->name()
            << " due to control dependency: " << while_op->ToShortString();
    return std::nullopt;
  }
  if (ContainsInstrWithOpcode(while_op->while_body(),
                              {HloOpcode::kSend, HloOpcode::kSendDone,
                               HloOpcode::kRecv, HloOpcode::kRecvDone}) ||
      ContainsInstrWithOpcode(while_op->while_condition(),
                              {HloOpcode::kSend, HloOpcode::kSendDone,
                               HloOpcode::kRecv, HloOpcode::kRecvDone})) {
    VLOG(2) << "Not attempting to unroll " << while_op->name()
            << " because it contains a send/recv node: "
            << while_op->ToShortString();
    return std::nullopt;
  }
  if (while_op->operand(0)->opcode() != HloOpcode::kTuple) {
    VLOG(2) << "Not attempting to unroll " << while_op->name()
            << " because the operand is not a tuple: "
            << while_op->ToShortString();
    return std::nullopt;
  }
  if (while_op->while_condition()->HasSideEffect()) {
    VLOG(2) << "Not attempting to remove while loop whose condition contains "
               "side-effecting instructions: "
            << while_op->ToShortString();
    return std::nullopt;
  }
  std::optional<int64_t> indvar_tuple_idx =
      GetLoopInductionVarTupleIdx(while_op);
  if (!indvar_tuple_idx.has_value()) {
    VLOG(2) << "Not attempting to unroll because induction variable could not "
               "be found.";
    return std::nullopt;
  }
  HloEvaluator evaluator(0);
  const HloInstruction* while_init = while_op->operand(0);
  const HloInstruction* indvar_init = while_init->operand(*indvar_tuple_idx);
  absl::StatusOr<Literal> indvar_init_result = evaluator.Evaluate(indvar_init);
  if (!indvar_init_result.ok()) {
    VLOG(2) << "Couldn't evaluate induction variable init, "
            << indvar_init_result.status() << ", " << indvar_init->ToString();
    return std::nullopt;
  }
  Literal indvar_iter_val = std::move(indvar_init_result).value();
  std::optional<int64_t> trip_count =
      MatchTrivialLoopTripCount(while_op, *indvar_tuple_idx, indvar_iter_val);
  if (!trip_count.has_value()) {
    VLOG(3) << "Loop doesn't have trivial trip count";
    return std::nullopt;
  }
  VLOG(3) << "Loop trip count " << trip_count.value();
  WhileLoopConfig config;
  config.while_instr = while_op;
  config.init =
      LiteralUtil::LiteralAsScalarInt64(std::move(indvar_iter_val)).value();
  config.trip_count = trip_count.value();
  config.induction_var_idx = *indvar_tuple_idx;
  return config;
}
 absl::StatusOr<bool> WhileLoopUnroller::PrepareModuleForUnrolling(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  TF_ASSIGN_OR_RETURN(
      bool applied_cse,
      HloCSE(true, false,
             false, true)
          .Run(module, execution_threads));
  if (applied_cse) {
    changed = true;
    VLOG(3) << "Applied hlo cse to module " << module->name();
  }
  TF_ASSIGN_OR_RETURN(bool applied_tuple_simplifier,
                      TupleSimplifier{}.Run(module, execution_threads));
  if (applied_tuple_simplifier) {
    changed = true;
    VLOG(3) << "Applied tuple simplifier to module " << module->name();
  }
  HloPassFix<WhileLoopConstantSinking> constant_sinking(
      true,
      true);
  TF_ASSIGN_OR_RETURN(bool applied_constant_sinking,
                      constant_sinking.Run(module, execution_threads));
  if (applied_constant_sinking) {
    changed = true;
    VLOG(3) << "Applied constant sinking to module " << module->name();
  }
  return changed;
}
 std::vector<std::pair<HloInstruction*, WhileLoopConfig>>
WhileLoopUnroller::GetUnrollableLoops(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    std::optional<UnrollConfig> unroll_config) {
  std::vector<HloInstruction*> all_while_ops;
  for (auto* comp : module->MakeComputationPostOrder(execution_threads)) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(all_while_ops),
                    HloPredicateIsOp<HloOpcode::kWhile>);
  }
  std::vector<std::pair<HloInstruction*, WhileLoopConfig>> while_loop_configs;
  for (HloInstruction* instr : all_while_ops) {
    std::optional<WhileLoopConfig> config = IsLoopUnrollable(instr);
    if (!config.has_value()) {
      continue;
    }
    if (unroll_config.has_value() &&
        !InitialFeasibilityCheck(instr, config.value(),
                                 unroll_config.value())) {
      VLOG(3) << "Initial feasibility check failed for " << instr->name();
      continue;
    }
    while_loop_configs.emplace_back(instr, config.value());
  }
  return while_loop_configs;
}
 absl::StatusOr<UnrollResult>
WhileLoopUnroller::UnrollAndReturnReplacement(
    HloInstruction* while_op, int64_t unroll_factor, bool wrap_in_trivial_loop,
    bool force_unroll, bool prepare, const UnrollConfig& unroll_config) {
  UnrollResult result;
  HloModule* module = while_op->GetModule();
  if (unroll_factor != -1) {
    VLOG(5) << absl::StrCat(
        "Currently, only full unrolling is supported, unroll factor: ",
        unroll_factor);
    return result;
  }
  if (prepare) {
    TF_RETURN_IF_ERROR(
        PrepareModuleForUnrolling(module, {}).status());
  }
  std::optional<WhileLoopConfig> config = IsLoopUnrollable(while_op);
  if (!config.has_value()) {
    VLOG(5) << "Not attempting to unroll " << while_op->name()
            << " because it is not unrollable.";
    return result;
  }
  if (!force_unroll &&
      !InitialFeasibilityCheck(while_op, config.value(), unroll_config)) {
    return result;
  }
  if (wrap_in_trivial_loop) {
    TF_ASSIGN_OR_RETURN(result, UnrollInternalWrappedAndReturnReplacement(
                                    while_op, config.value()));
  } else {
    TF_ASSIGN_OR_RETURN(result.unrolled,
                        UnrollInternal(while_op, config.value()));
  }
  if (result.unrolled) {
    TF_RETURN_IF_ERROR(CallInliner().Run(module).status());
  }
  return result;
}
absl::StatusOr<bool> WhileLoopUnroller::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (unroll_factor_ != -1) {
    return false;
  }
  XLA_VLOG_LINES(3, "WhileLoopUnroller::Run(), before:\n" + module->ToString());
  bool changed = false;
  TF_ASSIGN_OR_RETURN(changed,
                      PrepareModuleForUnrolling(module, execution_threads));
  std::vector<HloInstruction*> all_while_ops;
  for (auto* comp : module->MakeComputationPostOrder(execution_threads)) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(all_while_ops),
                    HloPredicateIsOp<HloOpcode::kWhile>);
  }
  std::vector<std::pair<HloInstruction*, WhileLoopConfig>>
      unrollable_while_ops = GetUnrollableLoops(
          module, execution_threads, unroll_config_);
  VLOG(3) << "Number of while instructions in the module to unroll: "
          << unrollable_while_ops.size();
  bool unrolled = false;
  for (auto& [while_op, config] : unrollable_while_ops) {
    if (wrap_in_trivial_loop_) {
      TF_ASSIGN_OR_RETURN(unrolled, UnrollInternalWrapped(while_op, config));
    } else {
      TF_ASSIGN_OR_RETURN(unrolled, UnrollInternal(while_op, config));
    }
    changed |= unrolled;
  }
  if (changed) {
    TF_RETURN_IF_ERROR(CallInliner().Run(module, execution_threads).status());
  }
  XLA_VLOG_LINES(3, "WhileLoopUnroller::Run(), after:\n" + module->ToString());
  return changed;
}
}  