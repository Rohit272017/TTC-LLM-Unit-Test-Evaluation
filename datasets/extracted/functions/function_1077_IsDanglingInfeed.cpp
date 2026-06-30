#include "xla/service/infeed_token_propagation.h"
#include <cstdint>
#include <string_view>
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/call_graph.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace {
bool IsDanglingInfeed(HloInstruction* infeed) {
  CHECK(infeed->opcode() == HloOpcode::kInfeed);
  if (infeed->has_sharding()) {
    return false;
  }
  if (const HloInstruction* after_all = infeed->operand(0);
      after_all->opcode() != HloOpcode::kAfterAll ||
      after_all->operand_count() != 0) {
    return false;
  }
  for (const HloInstruction* user : infeed->users()) {
    if (user->opcode() == HloOpcode::kGetTupleElement &&
        user->tuple_index() == 1) {
      return false;
    }
  }
  return true;
}
bool IsDanglingOutfeed(HloInstruction* outfeed) {
  CHECK(outfeed->opcode() == HloOpcode::kOutfeed);
  if (outfeed->has_sharding()) {
    return false;
  }
  if (const HloInstruction* after_all = outfeed->operand(1);
      after_all->opcode() != HloOpcode::kAfterAll ||
      after_all->operand_count() != 0) {
    return false;
  }
  if (outfeed->user_count() != 0) {
    return false;
  }
  return true;
}
HloInstruction* ReconstructTuple(HloInstruction* tuple) {
  CHECK(tuple->shape().IsTuple());
  HloComputation* computation = tuple->parent();
  std::vector<HloInstruction*> gtes;
  gtes.resize(tuple->shape().tuple_shapes_size());
  for (int64_t idx = 0; idx < gtes.size(); ++idx) {
    gtes[idx] = computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(tuple, idx));
  }
  return computation->AddInstruction(HloInstruction::CreateTuple(gtes));
}
absl::StatusOr<HloInstruction*> InsertTokenIntoTuple(HloInstruction* tuple,
                                                     bool add_token_operand) {
  CHECK(tuple->shape().IsTuple());
  HloComputation* computation = tuple->parent();
  std::vector<HloInstruction*> original_users = tuple->users();
  HloInstruction* original_tuple = ReconstructTuple(tuple);
  for (HloInstruction* original_user : original_users) {
    for (int64_t idx : original_user->operand_indices(tuple)) {
      TF_RETURN_IF_ERROR(
          original_user->ReplaceOperandWith(idx, original_tuple));
    }
  }
  *tuple->mutable_shape()->add_tuple_shapes() = ShapeUtil::MakeTokenShape();
  if (add_token_operand) {
    tuple->AppendOperand(
        computation->AddInstruction(HloInstruction::CreateToken()));
  }
  HloInstruction* input_token_gte =
      computation->AddInstruction(HloInstruction::CreateGetTupleElement(
          tuple, tuple->shape().tuple_shapes_size() - 1));
  return input_token_gte;
}
}  
absl::Status CanonicalizeConditionalInstruction(HloInstruction* conditional) {
  CHECK_EQ(conditional->opcode(), HloOpcode::kConditional);
  for (HloComputation* branch : conditional->branch_computations()) {
    HloInstruction* parameter = branch->parameter_instruction(0);
    if (!parameter->shape().IsTuple()) {
      *parameter->mutable_shape() =
          ShapeUtil::MakeTupleShape({parameter->shape()});
      HloInstruction* original = branch->AddInstruction(
          HloInstruction::CreateGetTupleElement(parameter, 0));
      TF_RETURN_IF_ERROR(parameter->ReplaceAllUsesWithDifferentShape(original));
    }
    int64_t branch_operand_idx = conditional->branch_index(branch) + 1;
    HloInstruction* branch_tuple =
        conditional->mutable_operand(branch_operand_idx);
    if (!branch_tuple->shape().IsTuple()) {
      branch_tuple = conditional->parent()->AddInstruction(
          HloInstruction::CreateTuple({branch_tuple}));
      TF_RETURN_IF_ERROR(conditional->ReplaceOperandWithDifferentShape(
          branch_operand_idx, branch_tuple));
    }
    if (branch_tuple->opcode() == HloOpcode::kParameter) {
      branch_tuple = ReconstructTuple(branch_tuple);
      TF_RETURN_IF_ERROR(
          conditional->ReplaceOperandWith(branch_operand_idx, branch_tuple));
    }
    HloInstruction* root = branch->root_instruction();
    if (root->opcode() != HloOpcode::kTuple) {
      root = ReconstructTuple(root);
      branch->set_root_instruction(root);
    }
  }
  CHECK(conditional->shape().IsTuple());
  if (conditional->IsRoot()) {
    HloInstruction* new_root = ReconstructTuple(conditional);
    conditional->parent()->set_root_instruction(new_root);
  }
  return absl::OkStatus();
}
absl::Status CanonicalizeWhileInstruction(HloInstruction* loop) {
  CHECK_EQ(loop->opcode(), HloOpcode::kWhile);
  HloComputation* body = loop->while_body();
  HloComputation* cond = loop->while_condition();
  HloInstruction* body_parameter = body->parameter_instruction(0);
  if (!body_parameter->shape().IsTuple()) {
    *body_parameter->mutable_shape() =
        ShapeUtil::MakeTupleShape({body_parameter->shape()});
    HloInstruction* original = body->AddInstruction(
        HloInstruction::CreateGetTupleElement(body_parameter, 0));
    TF_RETURN_IF_ERROR(
        body_parameter->ReplaceAllUsesWithDifferentShape(original));
  }
  HloInstruction* root = body->root_instruction();
  if (!root->shape().IsTuple()) {
    root = body->AddInstruction(HloInstruction::CreateTuple({root}));
    body->set_root_instruction(root, true);
  }
  HloInstruction* cond_parameter = cond->parameter_instruction(0);
  if (!cond_parameter->shape().IsTuple()) {
    *cond_parameter->mutable_shape() =
        ShapeUtil::MakeTupleShape({cond_parameter->shape()});
    HloInstruction* original = cond->AddInstruction(
        HloInstruction::CreateGetTupleElement(cond_parameter, 0));
    TF_RETURN_IF_ERROR(
        cond_parameter->ReplaceAllUsesWithDifferentShape(original));
  }
  if (!loop->shape().IsTuple()) {
    *loop->mutable_shape() = ShapeUtil::MakeTupleShape({loop->shape()});
    HloInstruction* original = loop->parent()->AddInstruction(
        HloInstruction::CreateGetTupleElement(loop, 0));
    TF_RETURN_IF_ERROR(loop->ReplaceAllUsesWithDifferentShape(original));
  }
  HloInstruction* loop_tuple = loop->mutable_operand(0);
  if (!loop_tuple->shape().IsTuple()) {
    loop_tuple = loop->parent()->AddInstruction(
        HloInstruction::CreateTuple({loop_tuple}));
    TF_RETURN_IF_ERROR(loop->ReplaceOperandWithDifferentShape(0, loop_tuple));
  }
  if (loop_tuple->opcode() == HloOpcode::kParameter) {
    loop_tuple = ReconstructTuple(loop_tuple);
    TF_RETURN_IF_ERROR(loop->ReplaceOperandWith(0, loop_tuple));
  }
  if (root->opcode() != HloOpcode::kTuple) {
    root = ReconstructTuple(root);
    body->set_root_instruction(root);
  }
  if (loop->IsRoot()) {
    HloInstruction* new_root = ReconstructTuple(loop);
    loop->parent()->set_root_instruction(new_root);
  }
  return absl::OkStatus();
}
absl::Status InfeedTokenPropagation::PropagateTokenThroughConditionalBranch() {
  HloComputation* comp = dangling_instruction_->parent();
  dangling_instruction_ = call_graph_->GetComputationCallers(comp)[0];
  CHECK_EQ(dangling_instruction_->opcode(), HloOpcode::kConditional);
  for (HloComputation* branch : dangling_instruction_->branch_computations()) {
    HloInstruction* root = branch->root_instruction();
    if (branch == comp) {
      TF_RETURN_IF_ERROR(
          InsertTokenIntoTuple(root, false).status());
      root->AppendOperand(output_token_);
    } else {
      TF_RETURN_IF_ERROR(
          InsertTokenIntoTuple(root, true).status());
    }
  }
  HloInstruction* parameter = comp->parameter_instruction(0);
  TF_ASSIGN_OR_RETURN(
      HloInstruction * input_token_gte,
      InsertTokenIntoTuple(parameter, false));
  TF_RETURN_IF_ERROR(input_token_->ReplaceAllUsesWith(input_token_gte));
  int64_t branch_operand_idx = dangling_instruction_->branch_index(comp) + 1;
  HloInstruction* branch_tuple =
      dangling_instruction_->mutable_operand(branch_operand_idx);
  TF_ASSIGN_OR_RETURN(
      HloInstruction * next_input_token_gte,
      InsertTokenIntoTuple(branch_tuple, true));
  TF_RETURN_IF_ERROR(dangling_instruction_->ReplaceOperandWithDifferentShape(
      branch_operand_idx, branch_tuple));
  input_token_ =
      branch_tuple->mutable_operand(next_input_token_gte->tuple_index());
  TF_ASSIGN_OR_RETURN(
      output_token_,
      InsertTokenIntoTuple(dangling_instruction_, false));
  return absl::OkStatus();
}
absl::Status InfeedTokenPropagation::PropagateTokenThroughWhileBody() {
  HloComputation* comp = dangling_instruction_->parent();
  dangling_instruction_ = call_graph_->GetComputationCallers(comp)[0];
  CHECK_EQ(dangling_instruction_->opcode(), HloOpcode::kWhile);
  HloInstruction* root = comp->root_instruction();
  TF_RETURN_IF_ERROR(
      InsertTokenIntoTuple(root, false).status());
  root->AppendOperand(output_token_);
  HloInstruction* body_parameter = comp->parameter_instruction(0);
  TF_ASSIGN_OR_RETURN(
      HloInstruction * input_token_gte,
      InsertTokenIntoTuple(body_parameter, false));
  TF_RETURN_IF_ERROR(input_token_->ReplaceAllUsesWith(input_token_gte));
  HloComputation* cond = dangling_instruction_->while_condition();
  HloInstruction* cond_parameter = cond->parameter_instruction(0);
  TF_RETURN_IF_ERROR(
      InsertTokenIntoTuple(cond_parameter, false)
          .status());
  HloInstruction* while_tuple = dangling_instruction_->mutable_operand(0);
  TF_ASSIGN_OR_RETURN(
      input_token_,
      InsertTokenIntoTuple(while_tuple, true));
  TF_RETURN_IF_ERROR(
      dangling_instruction_->ReplaceOperandWithDifferentShape(0, while_tuple));
  TF_ASSIGN_OR_RETURN(
      output_token_,
      InsertTokenIntoTuple(dangling_instruction_, false));
  return absl::OkStatus();
}
absl::Status InfeedTokenPropagation::PropagateToken() {
  HloComputation* comp = dangling_instruction_->parent();
  if (comp->IsEntryComputation()) {
    return absl::OkStatus();
  }
  VLOG(2) << "Propagating tokens for: " << dangling_instruction_->name();
  HloInstruction* caller = call_graph_->GetComputationCallers(comp)[0];
  if (caller->has_sharding()) {
    return absl::OkStatus();
  }
  if (caller->opcode() == HloOpcode::kConditional) {
    TF_RETURN_IF_ERROR(CanonicalizeConditionalInstruction(caller));
    TF_RETURN_IF_ERROR(PropagateTokenThroughConditionalBranch());
  } else if (caller->opcode() == HloOpcode::kWhile &&
             comp == caller->while_body()) {
    TF_RETURN_IF_ERROR(CanonicalizeWhileInstruction(caller));
    TF_RETURN_IF_ERROR(PropagateTokenThroughWhileBody());
  } else {
    VLOG(2) << "Unhandled computation: " << comp->name();
    return absl::OkStatus();
  }
  return PropagateToken();
}
absl::StatusOr<bool> InfeedTokenPropagation::Run(
    HloModule* module,
    const absl::flat_hash_set<std::string_view>& execution_threads) {
  VLOG(5) << "Before InfeedTokenPropagation:";
  XLA_VLOG_LINES(5, module->ToString());
  std::vector<HloInstruction*> dangling_infeeds;
  std::vector<HloInstruction*> dangling_outfeeds;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    if (!computation->IsEntryComputation()) {
      for (HloInstruction* instruction : computation->instructions()) {
        if (instruction->opcode() == HloOpcode::kInfeed &&
            IsDanglingInfeed(instruction)) {
          VLOG(1) << "Found dangling infeed: " << instruction->ToString();
          dangling_infeeds.push_back(instruction);
        } else if (instruction->opcode() == HloOpcode::kOutfeed &&
                   IsDanglingOutfeed(instruction)) {
          VLOG(1) << "Found dangling outfeed: " << instruction->ToString();
          dangling_outfeeds.push_back(instruction);
        }
      }
    }
  }
  bool changed = !dangling_infeeds.empty() || !dangling_outfeeds.empty();
  if (changed) {
    call_graph_ = CallGraph::Build(module);
    if (!call_graph_->IsFlattened()) {
      return FailedPrecondition(
          "Call graph must be flattened before infeed token propagation.");
    }
  }
  for (HloInstruction* dangling_infeed : dangling_infeeds) {
    dangling_instruction_ = dangling_infeed;
    input_token_ = dangling_infeed->mutable_operand(0);
    output_token_ = dangling_infeed->AddInstruction(
        HloInstruction::CreateGetTupleElement(dangling_infeed, 1));
    TF_RETURN_IF_ERROR(PropagateToken());
  }
  for (HloInstruction* dangling_outfeed : dangling_outfeeds) {
    dangling_instruction_ = dangling_outfeed;
    input_token_ = dangling_outfeed->mutable_operand(1);
    output_token_ = dangling_outfeed;
    TF_RETURN_IF_ERROR(PropagateToken());
  }
  if (changed) {
    TF_RETURN_IF_ERROR(
        TupleSimplifier().Run(module, execution_threads).status());
    TF_RETURN_IF_ERROR(HloDCE().Run(module, execution_threads).status());
  }
  VLOG(5) << "After InfeedTokenPropagation:";
  XLA_VLOG_LINES(5, module->ToString());
  return changed;
}
}  