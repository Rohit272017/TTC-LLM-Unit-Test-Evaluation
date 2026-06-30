#include "xla/service/while_loop_analysis.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "xla/comparison_util.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_reachability.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"
namespace xla {
using std::nullopt;
using std::optional;
namespace m = match;
static const HloInstruction* NonConstantOperand(const HloInstruction* instr) {
  const HloInstruction* result = nullptr;
  for (const HloInstruction* operand : instr->operands()) {
    if (!operand->IsConstant()) {
      if (result != nullptr) {
        CHECK_EQ(result, operand);
      }
      result = operand;
    }
  }
  CHECK_NE(result, nullptr);
  return result;
}
static optional<int64_t> GetGTEOperandIndex(const HloInstruction* instr,
                                            const HloInstruction* gte_operand) {
  VLOG(2) << "GetGTEOperandIndex(" << instr->ToString()
          << ", GTE Operand: " << gte_operand->ToString() << ")";
  optional<int64_t> tuple_idx;
  for (const HloInstruction* operand : instr->operands()) {
    if (Match(operand, m::Constant())) {
      continue;
    }
    auto possibly_gte_operand = operand;
    if (operand->opcode() == HloOpcode::kCopy) {
      possibly_gte_operand = operand->operand(0);
    }
    if (possibly_gte_operand->opcode() != HloOpcode::kGetTupleElement) {
      return nullopt;
    }
    if (!Match(possibly_gte_operand,
               m::GetTupleElement(m::Op().Is(gte_operand))) &&
        !Match(possibly_gte_operand,
               m::GetTupleElement(m::CustomCall(m::Op().Is(gte_operand))))) {
      return nullopt;
    }
    int64_t operand_tuple_idx = possibly_gte_operand->tuple_index();
    if (!tuple_idx.has_value()) {
      tuple_idx = operand_tuple_idx;
    } else {
      if (operand_tuple_idx != tuple_idx) {
        return nullopt;
      }
    }
  }
  return tuple_idx;
}
std::vector<const HloInstruction*> GetAuxiliaryLoopInductionVars(
    const HloInstruction* while_op) {
  std::vector<const HloInstruction*> aux_ind_gte;
  CHECK_EQ(while_op->opcode(), HloOpcode::kWhile);
  auto* while_body = while_op->while_body();
  auto* while_body_param = while_body->parameter_instruction(0);
  VLOG(2) << "Aux Induction Variables for loop:" << while_op->ToShortString();
  VLOG(2) << "the parameter instr:" << while_body_param->ToShortString();
  VLOG(2) << "the parameter user count:" << while_body_param->users().size();
  if (while_body_param == nullptr) return aux_ind_gte;
  std::map<int64_t, const HloInstruction*> extractions;
  for (const HloInstruction* indx_instr : while_body_param->users()) {
    if (indx_instr->opcode() != HloOpcode::kGetTupleElement) {
      continue;
    }
    auto it = extractions.find(indx_instr->tuple_index());
    if (it != extractions.end()) {
      it->second = nullptr;
      VLOG(2) << "two extractions at same index:" << indx_instr->ToString();
    } else {
      extractions.insert(std::make_pair(indx_instr->tuple_index(), indx_instr));
      VLOG(2) << "inserting extraction :" << indx_instr->ToString();
    }
  }
  VLOG(2) << "total extractions size:" << extractions.size() << std::endl;
  if (extractions.empty()) {
    return aux_ind_gte;
  }
  auto* while_body_root = while_body->root_instruction();
  if (while_body_root->opcode() != HloOpcode::kTuple) {
    VLOG(2) << "While body root is not a tuple:" << while_body_root->ToString();
    return aux_ind_gte;
  }
  int64_t index = -1;
  std::map<int64_t, const HloInstruction*> insertions;
  for (const HloInstruction* operand : while_body_root->operands()) {
    index++;
    if (!operand->IsConstant()) {
      auto it = insertions.find(index);
      if (it != insertions.end()) {
        it->second = nullptr;
        VLOG(2) << "two insertions at same index:" << operand->ToString();
      } else {
        insertions.insert(std::make_pair(index, operand));
        VLOG(2) << "inserting insertions:" << operand->ToString();
      }
    }
  }
  if (insertions.empty()) {
    return aux_ind_gte;
  }
  std::map<int64_t, std::pair<const HloInstruction*, const HloInstruction*>>
      candidate_pairs;
  for (; index >= 0; --index) {
    const HloInstruction *ext, *inst;
    ext = (extractions.find(index) != extractions.end())
              ? extractions.find(index)->second
              : nullptr;
    inst = (insertions.find(index) != insertions.end())
               ? insertions.find(index)->second
               : nullptr;
    if (ext != nullptr && inst != nullptr) {
      if (ext != inst) {
        candidate_pairs.insert(
            std::make_pair(index, std::make_pair(ext, inst)));
      }
    }
  }
  VLOG(2) << "total candidate pairs:" << candidate_pairs.size() << std::endl;
  const auto add_dependencies = [](const HloInstruction* hlo,
                                   std::vector<HloInstruction*>* inputs) {
    HloInstruction* non_const_operand = nullptr;
    int num_non_constants = 0;
    for (HloInstruction* operand : hlo->operands()) {
      if (!operand->IsConstant()) {
        num_non_constants++;
        non_const_operand = operand;
      }
    }
    if (num_non_constants == 1 &&
        (hlo->opcode() == HloOpcode::kGetTupleElement ||
         hlo->opcode() == HloOpcode::kAdd ||
         hlo->opcode() == HloOpcode::kMultiply ||
         hlo->opcode() == HloOpcode::kDivide ||
         hlo->opcode() == HloOpcode::kSubtract)) {
      inputs->push_back(non_const_operand);
    }
  };
  std::unique_ptr<HloReachabilityMap> hrm =
      HloReachabilityMap::BuildWithRestrictions(
          while_body,
          absl::FunctionRef<void(const HloInstruction* hlo,
                                 std::vector<HloInstruction*>* inputs)>(
              add_dependencies));
  for (auto candidates : candidate_pairs) {
    VLOG(2) << "are reachable?:" << (candidates.second.first)->ToString()
            << "*************" << (candidates.second.second)->ToString()
            << std::endl;
    if (hrm->IsReachable(candidates.second.first, candidates.second.second)) {
      aux_ind_gte.push_back(candidates.second.first);
      VLOG(2) << "YES";
    } else {
      VLOG(2) << "NO";
    }
  }
  VLOG(2) << "num auxiliary candidates :" << aux_ind_gte.size();
  return aux_ind_gte;
}
optional<int64_t> GetLoopInductionVarTupleIdx(const HloInstruction* while_op) {
  CHECK_EQ(while_op->opcode(), HloOpcode::kWhile);
  VLOG(2) << "Finding induction variable for loop "
          << while_op->ToShortString();
  auto* while_cond = while_op->while_condition();
  auto* while_cond_root = while_cond->root_instruction();
  auto* while_cond_param = while_cond->parameter_instruction(0);
  optional<int64_t> indvar_tuple_idx =
      GetGTEOperandIndex(while_cond_root, while_cond_param);
  if (!indvar_tuple_idx) {
    VLOG(2) << "Induction variable not found in loop condition: "
            << while_cond->root_instruction()->ToString();
    return nullopt;
  }
  auto* while_body = while_op->while_body();
  auto* while_body_root = while_body->root_instruction();
  if (while_body_root->opcode() != HloOpcode::kTuple &&
      while_body_root->opcode() != HloOpcode::kCustomCall) {
    VLOG(2) << "While body's root is not a tuple or custom-call instruction: "
            << while_body_root->ToString();
    return nullopt;
  }
  const HloInstruction* while_body_inc;
  if (while_body_root->opcode() == HloOpcode::kTuple) {
    while_body_inc = while_body_root->operand(*indvar_tuple_idx);
  } else {
    if (while_body_root->operand_count() == 1 &&
        while_body_root->operand(0)->opcode() == HloOpcode::kTuple) {
      auto* while_body_root_input_tuple = while_body_root->operand(0);
      if (*indvar_tuple_idx >= while_body_root_input_tuple->operand_count()) {
        VLOG(2) << "Cannot find the induction variable in the output root "
                   "custom-call "
                << while_body_root->ToString();
        return std::nullopt;
      }
      while_body_inc = while_body_root_input_tuple->operand(*indvar_tuple_idx);
    } else {
      if (*indvar_tuple_idx >= while_body_root->operand_count()) {
        VLOG(2) << "Cannot find the induction variable in the output root "
                   "custom-call "
                << while_body_root->ToString();
        return std::nullopt;
      }
      while_body_inc = while_body_root->operand(*indvar_tuple_idx);
    }
  }
  auto* while_body_param = while_body->parameter_instruction(0);
  optional<int64_t> while_body_indvar_tuple_idx =
      GetGTEOperandIndex(while_body_inc, while_body_param);
  if (!while_body_indvar_tuple_idx) {
    VLOG(2)
        << "Induction variable not found in while body increment instruction: "
        << while_body_inc->ToString();
    return nullopt;
  }
  if (while_body_indvar_tuple_idx != indvar_tuple_idx) {
    VLOG(2) << "Tuple index of induction variable does not match between loop "
               "condition ("
            << *indvar_tuple_idx << ") and while body ("
            << *while_body_indvar_tuple_idx << ")";
    return nullopt;
  }
  auto* while_init = while_op->operand(0);
  if (while_init->opcode() != HloOpcode::kTuple) {
    VLOG(2) << "While init expected to be a tuple: " << while_init->ToString();
    return nullopt;
  }
  VLOG(2) << "Induction variable's tuple index: " << *indvar_tuple_idx;
  return indvar_tuple_idx;
}
optional<int64_t> CheckedAdd(int64_t a, int64_t b) {
  uint64_t aa = absl::bit_cast<uint64_t>(a);
  uint64_t bb = absl::bit_cast<uint64_t>(b);
  int64_t result = absl::bit_cast<int64_t>(aa + bb);
  if (a >= 0 == b >= 0 && result >= 0 != a >= 0) {
    return nullopt;
  }
  return result;
}
optional<int64_t> CheckedSubtract(int64_t a, int64_t b) {
  uint64_t aa = absl::bit_cast<uint64_t>(a);
  uint64_t bb = absl::bit_cast<uint64_t>(b);
  int64_t result = absl::bit_cast<int64_t>(aa - bb);
  if (a >= 0 != b >= 0 && result >= 0 == b >= 0) {
    return nullopt;
  }
  return result;
}
optional<int64_t> MatchTrivialLoopTripCount(const HloInstruction* while_op,
                                            int64_t indvar_tuple_idx,
                                            const Literal& indvar_init) {
  optional<int64_t> indvar_init_val =
      LiteralUtil::LiteralAsScalarInt64(indvar_init);
  if (!indvar_init_val) {
    VLOG(2) << "Pattern-match failed: induction variable init is not a "
               "constant scalar representable as an int64_t: "
            << indvar_init.ToString();
    return nullopt;
  }
  auto* while_body = while_op->while_body();
  auto* while_body_root = while_body->root_instruction();
  HloInstruction* while_body_indvar_update;
  if (while_body_root->opcode() == HloOpcode::kCustomCall) {
    if (while_body_root->operand_count() == 1 &&
        while_body_root->operand(0)->opcode() == HloOpcode::kTuple) {
      auto* while_body_root_input_tuple = while_body_root->mutable_operand(0);
      while_body_indvar_update =
          while_body_root_input_tuple->mutable_operand(indvar_tuple_idx);
    } else {
      while_body_indvar_update =
          while_body_root->mutable_operand(indvar_tuple_idx);
    }
  } else {
    while_body_indvar_update =
        while_body_root->mutable_operand(indvar_tuple_idx);
  }
  auto* while_body_indvar = NonConstantOperand(while_body_indvar_update);
  HloInstruction* trip_count_increase_step_instr = nullptr;
  int64_t trip_count_step = 0;
  if (!Match(while_body_indvar_update,
             m::AddAnyOrder(m::Op().Is(while_body_indvar),
                            m::Op(&trip_count_increase_step_instr)))) {
    if (trip_count_increase_step_instr == nullptr) {
      VLOG(2) << "Pattern-match failed: induction variable is not getting "
                 "updated by an add operation: "
              << while_body_indvar_update->ToString();
      return nullopt;
    }
    if (!trip_count_increase_step_instr->IsConstant() ||
        !ShapeUtil::IsEffectiveScalar(
            trip_count_increase_step_instr->shape())) {
      VLOG(2) << "Pattern-match failed: induction variable is not getting "
                 "incremented by constant: "
              << while_body_indvar_update->ToString();
      return nullopt;
    }
    if (!LiteralUtil::LiteralAsScalarInt64(
             trip_count_increase_step_instr->literal())
             .has_value()) {
      VLOG(2)
          << "Pattern-match failed: trip count step is not an integral type: "
          << trip_count_increase_step_instr->shape().ToString();
      return nullopt;
    }
    VLOG(2) << "Pattern-match for trip count step failed: "
            << trip_count_increase_step_instr->ToString();
  }
  trip_count_step = LiteralUtil::LiteralAsScalarInt64(
                        trip_count_increase_step_instr->literal())
                        .value();
  if (trip_count_step <= 0) {
    VLOG(2) << "Pattern-match failed: trip count step is not a natural number: "
            << trip_count_step;
    return nullopt;
  }
  auto* while_cond = while_op->while_condition();
  auto* while_cond_root = while_cond->root_instruction();
  auto* while_cond_indvar = NonConstantOperand(while_cond_root);
  HloInstruction* while_cond_bound = nullptr;
  if (!Match(while_cond_root,
             m::Op().WithBinaryOperandsAnyOrder(
                 m::Op().Is(while_cond_indvar),
                 m::ConstantEffectiveScalar(&while_cond_bound)))) {
    VLOG(2) << "Pattern-match failed: while condition is not of the form "
               "op(i, N) or op(N, i).";
    return nullopt;
  }
  optional<int64_t> while_cond_bound_val =
      LiteralUtil::LiteralAsScalarInt64(while_cond_bound->literal());
  if (!while_cond_bound_val) {
    VLOG(2) << "Pattern-match failed: while condition induction variable is "
               "not a constant scalar representable as an int64_t.";
    return nullopt;
  }
  if (Match(while_cond_root,
            m::Op()
                .WithComparisonDirection(ComparisonDirection::kLt)
                .WithOperand(0, m::Op().Is(while_cond_indvar)))) {
    VLOG(2) << "Pattern-match succeeded: loop condition is i < N: "
            << while_cond_root->ToString();
    optional<int64_t> trips =
        CheckedSubtract(*while_cond_bound_val, *indvar_init_val);
    if (trips) {
      const int64_t remainder = std::remainder(*trips, trip_count_step);
      const int64_t div = std::floor(*trips / trip_count_step);
      if (remainder == 0) {
        return std::max(int64_t{0}, div);
      }
      trips = CheckedAdd(div, 1);
      if (!trips) {
        VLOG(2) << "Pattern-match failed: Trip count exceeds INT64_MAX.";
        return nullopt;
      }
      if (*trips < *while_cond_bound_val) {
        return std::max(int64_t{0}, *trips);
      }
      return std::max(int64_t{0}, div);
    }
    VLOG(2) << "Pattern-match failed: Trip count exceeds INT64_MAX.";
    return nullopt;
  }
  if (Match(while_cond_root,
            m::Op()
                .WithComparisonDirection(ComparisonDirection::kLe)
                .WithOperand(0, m::Op().Is(while_cond_indvar)))) {
    VLOG(2) << "Pattern-match succeeded: loop condition is i <= N: "
            << while_cond_root->ToString();
    optional<int64_t> trips =
        CheckedSubtract(*while_cond_bound_val, *indvar_init_val);
    if (!trips) {
      VLOG(2) << "Pattern-match failed: Trip count exceeds INT64_MAX";
      return nullopt;
    }
    trips = CheckedAdd(std::floor(*trips / trip_count_step), 1);
    if (!trips) {
      VLOG(2) << "Pattern-match failed: Trip count exceeds INT64_MAX";
      return nullopt;
    }
    return std::max<int64_t>(0, *trips);
  }
  VLOG(2) << "Pattern-match failed: while condition follows unknown pattern: "
          << while_cond_root->ToString();
  return nullopt;
}
optional<int64_t> ComputeWhileLoopTripCount(const HloInstruction* while_op,
                                            int64_t max_brute_force_iters) {
  VLOG(2) << "Getting trip count for loop " << while_op->ToString();
  optional<int64_t> indvar_tuple_idx = GetLoopInductionVarTupleIdx(while_op);
  if (!indvar_tuple_idx) {
    return nullopt;
  }
  HloEvaluator evaluator(0);
  auto* while_init = while_op->operand(0);
  auto* indvar_init = while_init->operand(*indvar_tuple_idx);
  absl::StatusOr<Literal> indvar_init_result = evaluator.Evaluate(indvar_init);
  if (!indvar_init_result.ok()) {
    VLOG(2) << "Couldn't evaluate induction variable init, "
            << indvar_init_result.status() << ", " << indvar_init->ToString();
    return nullopt;
  }
  Literal indvar_iter_val = std::move(indvar_init_result).value();
  if (auto trip_count = MatchTrivialLoopTripCount(while_op, *indvar_tuple_idx,
                                                  indvar_iter_val)) {
    return trip_count;
  }
  auto* while_body = while_op->while_body();
  auto* while_body_indvar_update =
      while_body->root_instruction()->operand(*indvar_tuple_idx);
  auto* while_body_indvar = NonConstantOperand(while_body_indvar_update);
  auto* while_cond = while_op->while_condition();
  auto* while_cond_root = while_cond->root_instruction();
  auto* while_cond_indvar = NonConstantOperand(while_cond_root);
  for (int64_t trip_count = 0; trip_count != max_brute_force_iters + 1;
       ++trip_count) {
    absl::StatusOr<Literal> result = evaluator.EvaluateWithSubstitutions(
        while_cond_root, {{while_cond_indvar, &indvar_iter_val}});
    if (!result.ok()) {
      VLOG(2) << "Couldn't evaluate while cond: " << result.status();
      return nullopt;
    }
    if (result.value().data<bool>() == absl::Span<const bool>{false}) {
      VLOG(2) << "Loop has static trip count of " << trip_count;
      return trip_count;
    }
    absl::StatusOr<Literal> indvar_next_result =
        evaluator.EvaluateWithSubstitutions(
            while_body_indvar_update, {{while_body_indvar, &indvar_iter_val}});
    if (!indvar_next_result.ok()) {
      VLOG(2) << "Couldn't evaluate induction variable update: "
              << indvar_next_result.status();
      return nullopt;
    }
    indvar_iter_val = std::move(indvar_next_result).value();
  }
  VLOG(2) << "Loop has unknown trip count.";
  return nullopt;
}
static HloInstruction* GetOnlyGTE(HloInstruction* inst) {
  if (inst->user_count() != 1) {
    return nullptr;
  }
  HloInstruction* user = inst->users().back();
  if (user->opcode() != HloOpcode::kGetTupleElement) {
    return nullptr;
  }
  return user;
}
optional<int64_t> ComputeWhileLoopTripCountUpperBound(
    const HloInstruction* while_op) {
  auto exact_trip_count = ComputeWhileLoopTripCount(while_op);
  if (exact_trip_count) {
    VLOG(2) << "Loop has exact trip count.";
    return exact_trip_count;
  }
  auto* while_cond = while_op->while_condition();
  auto* while_cond_param = while_cond->parameter_instruction(0);
  auto* cond_gte = GetOnlyGTE(while_cond_param);
  if (!cond_gte) {
    VLOG(2) << "Induction variable not found in loop condition: "
            << while_cond->root_instruction()->ToString();
    return nullopt;
  }
  auto* while_body = while_op->while_body();
  auto* while_body_root = while_body->root_instruction();
  if (while_body_root->opcode() != HloOpcode::kTuple) {
    VLOG(3) << "While body's root is not a tuple instruction: "
            << while_body_root->ToString();
    return nullopt;
  }
  int64_t indvar_index = cond_gte->tuple_index();
  auto* while_body_indvar = while_body_root->operand(indvar_index);
  if (while_body_indvar->opcode() != HloOpcode::kConstant) {
    VLOG(3) << "While body does not set the IV to a constant: "
            << while_body_indvar->ToString();
    return nullopt;
  }
  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      replacements;
  auto new_param = HloInstruction::CreateParameter(
      0, ShapeUtil::MakeTupleShape({cond_gte->shape()}), "temp");
  replacements[cond_gte] =
      HloInstruction::CreateGetTupleElement(new_param.get(), 0);
  replacements[while_cond_param] = std::move(new_param);
  auto new_module = std::make_unique<HloModule>("temp_mod", HloModuleConfig{});
  auto* new_computation = new_module->AddEmbeddedComputation(
      while_cond->CloneWithReplacements(&replacements));
  HloEvaluator evaluator(0);
  Literal fake_input = Literal::CreateFromShape(
      new_computation->parameter_instruction(0)->shape());
  TF_CHECK_OK(fake_input.CopyFrom(while_body_indvar->literal(),
                                  {0},
                                  {}));
  absl::StatusOr<Literal> eval_result =
      evaluator.Evaluate(*new_computation, {std::move(fake_input)});
  if (!eval_result.ok()) {
    VLOG(2) << "Couldn't evaluate while loop condition.";
    return nullopt;
  }
  Literal cond_result_pred = std::move(eval_result.value());
  CHECK(Shape::Equal().IgnoreLayout()(cond_result_pred.shape(),
                                      ShapeUtil::MakeShape(PRED, {})));
  bool cond_returns_true = cond_result_pred.GetFirstElement<bool>();
  if (!cond_returns_true) {
    VLOG(2) << "Upper bound on the trip count is 1";
    return 1;
  }
  VLOG(2) << "Loop has no known upper bound on the trip count.";
  return nullopt;
}
}  