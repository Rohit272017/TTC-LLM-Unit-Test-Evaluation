#include "xla/tools/hlo_control_flow_flattening.h"
#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/call_graph.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/tuple_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace {
HloInstruction* CreateConstant(const Shape& shape,
                               HloComputation* computation) {
  if (shape.IsTuple()) {
    std::vector<HloInstruction*> tuple_arguments(shape.tuple_shapes_size());
    for (int index = 0; index < shape.tuple_shapes_size(); ++index) {
      tuple_arguments[index] =
          CreateConstant(shape.tuple_shapes(index), computation);
    }
    return computation->AddInstruction(
        HloInstruction::CreateTuple(tuple_arguments));
  } else {
    return computation->AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateFromShape(shape)));
  }
}
void PrintSubexpression(HloInstruction* inst, int depth) {
  if (depth == 0) {
    return;
  }
  for (auto* operand : inst->operands()) {
    PrintSubexpression(operand, depth - 1);
  }
  VLOG(2) << inst->ToString();
}
bool IsConstantScalarInt(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kConstant &&
         ShapeUtil::IsEffectiveScalar(inst->shape()) &&
         inst->shape().IsInteger();
}
bool IsNotContainedInLoop(const HloInstruction& while_hlo,
                          const CallGraph& call_graph) {
  const HloComputation* computation = while_hlo.parent();
  while (!computation->IsEntryComputation()) {
    auto& node = call_graph.GetNode(computation);
    CHECK_EQ(node.caller_callsites().size(), 1)
        << "The module is not flattened!";
    auto& callsite = node.caller_callsites()[0];
    if (callsite.instruction()->opcode() == HloOpcode::kWhile) {
      return false;
    }
    computation = callsite.instruction()->parent();
  }
  return true;
}
}  
int GetLoopBound(const HloInstruction& while_hlo, const int default_loop_count,
                 const int max_loop_count) {
  HloInstruction* condition = while_hlo.while_condition()->root_instruction();
  if (condition->opcode() == HloOpcode::kCompare) {
    int64_t value = 0;
    Comparison::Direction cmp = condition->comparison_direction();
    if ((cmp == Comparison::Direction::kLt ||
         cmp == Comparison::Direction::kLe ||
         cmp == Comparison::Direction::kNe) &&
        IsConstantScalarInt(condition->operand(1))) {
      value = *condition->operand(1)->literal().GetFirstInteger();
    } else if ((cmp == Comparison::Direction::kGt ||
                cmp == Comparison::Direction::kGe ||
                cmp == Comparison::Direction::kNe) &&
               IsConstantScalarInt(condition->operand(0))) {
      value = *condition->operand(0)->literal().GetFirstInteger();
    }
    if (value > 0) {
      return std::min(value, static_cast<int64_t>(max_loop_count));
    }
  }
  return default_loop_count;
}
int GetLoopBoundWithOuterLoopMax(const HloInstruction& while_hlo,
                                 const CallGraph& call_graph,
                                 const int default_loop_count,
                                 const int max_outer_loop_count,
                                 const int max_loop_count) {
  int loop_bound = GetLoopBound(while_hlo, default_loop_count, max_loop_count);
  if (loop_bound > max_outer_loop_count) {
    if (IsNotContainedInLoop(while_hlo, call_graph)) {
      return max_outer_loop_count;
    }
  }
  return loop_bound;
}
absl::Status HloControlFlowFlattening::FlattenWhileLoop(
    HloInstruction* while_hlo, const CallGraph& call_graph) const {
  CHECK_EQ(while_hlo->opcode(), HloOpcode::kWhile);
  HloComputation* computation = while_hlo->parent();
  HloInstruction* initialization = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int>(0)));
  HloInstruction* old_tuple = while_hlo->mutable_operand(0);
  HloInstruction* new_tuple =
      TupleUtil::AppendSuffix(old_tuple, {initialization});
  int new_tuple_size = new_tuple->shape().tuple_shapes().size();
  TF_RETURN_IF_ERROR(while_hlo->ReplaceOperandWithDifferentShape(0, new_tuple));
  auto change_op_shape = [&](HloInstruction* instruction) {
    Shape* shape = instruction->mutable_shape();
    CHECK(shape->IsTuple());
    CHECK_EQ(shape->tuple_shapes().size(), new_tuple_size - 1);
    Shape* subshape = shape->add_tuple_shapes();
    return ShapeUtil::PopulateShape(S32, {}, subshape);
  };
  auto replace_non_gte_users =
      [](HloInstruction* new_tuple) -> absl::StatusOr<HloInstruction*> {
    CHECK(new_tuple->shape().IsTuple());
    HloInstruction* prefix = nullptr;
    std::vector<HloInstruction*> users(new_tuple->users());
    for (HloInstruction* user : users) {
      if (user->opcode() == HloOpcode::kGetTupleElement) {
        continue;
      }
      if (prefix == nullptr) {
        prefix = TupleUtil::ExtractPrefix(
            new_tuple, new_tuple->shape().tuple_shapes_size() - 1);
      }
      TF_RETURN_IF_ERROR(new_tuple->ReplaceUseWithDifferentShape(user, prefix));
    }
    return prefix;
  };
  {
    HloComputation* condition = while_hlo->while_condition();
    TF_RETURN_IF_ERROR(change_op_shape(condition->parameter_instruction(0)));
    TF_RETURN_IF_ERROR(
        replace_non_gte_users(condition->parameter_instruction(0)).status());
    if (VLOG_IS_ON(2)) {
      VLOG(2) << "Loop condition in " << while_hlo->parent()->name();
      PrintSubexpression(condition->root_instruction(), 3);
    }
    const int loop_bound = GetLoopBoundWithOuterLoopMax(
        *while_hlo, call_graph, while_execution_count_, max_outer_loop_count_,
        max_loop_count_);
    VLOG(1) << "loop_bound = " << loop_bound;
    HloInstruction* limit = condition->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int>(loop_bound)));
    Shape shape = initialization->shape();
    HloInstruction* induction_variable =
        condition->AddInstruction(HloInstruction::CreateGetTupleElement(
            shape, condition->parameter_instruction(0), new_tuple_size - 1));
    HloInstruction* compare =
        condition->AddInstruction(HloInstruction::CreateCompare(
            ShapeUtil::MakeShape(PRED, {}), induction_variable, limit,
            ComparisonDirection::kLt));
    TF_RETURN_IF_ERROR(
        condition->ReplaceInstruction(condition->root_instruction(), compare));
  }
  {
    HloComputation* body = while_hlo->while_body();
    TF_RETURN_IF_ERROR(change_op_shape(body->parameter_instruction(0)));
    TF_RETURN_IF_ERROR(
        replace_non_gte_users(body->parameter_instruction(0)).status());
    HloInstruction* old_root = body->root_instruction();
    Shape shape = initialization->shape();
    HloInstruction* induction_variable =
        body->AddInstruction(HloInstruction::CreateGetTupleElement(
            shape, body->parameter_instruction(0), new_tuple_size - 1));
    HloInstruction* increment = body->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int>(1)));
    induction_variable = body->AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kAdd, induction_variable, increment));
    HloInstruction* new_root =
        TupleUtil::AppendSuffix(old_root, {induction_variable});
    body->set_root_instruction(new_root, true);
  }
  std::vector<HloInstruction*> while_users(while_hlo->users().begin(),
                                           while_hlo->users().end());
  TF_RETURN_IF_ERROR(change_op_shape(while_hlo));
  TF_ASSIGN_OR_RETURN(HloInstruction * prefix,
                      replace_non_gte_users(while_hlo));
  if (while_hlo->parent()->root_instruction() == while_hlo) {
    if (prefix == nullptr) {
      prefix = TupleUtil::ExtractPrefix(while_hlo, new_tuple_size - 1);
    }
    while_hlo->parent()->set_root_instruction(prefix,
                                              true);
  }
  return absl::OkStatus();
}
absl::Status HloControlFlowFlattening::RemoveInfeed(
    HloInstruction* infeed_hlo) const {
  CHECK_EQ(infeed_hlo->opcode(), HloOpcode::kInfeed);
  HloComputation* computation = infeed_hlo->parent();
  CHECK_EQ(infeed_hlo->shape().tuple_shapes_size(), 2);
  const Shape& infeed_shape = ShapeUtil::GetSubshape(infeed_hlo->shape(), {0});
  HloInstruction* custom_call = computation->AddInstruction(
      HloInstruction::CreateCustomCall(infeed_shape, {}, kNopCustomCallTarget));
  auto new_tuple = HloInstruction::CreateTuple(
      {custom_call, infeed_hlo->mutable_operand(0)});
  TF_RETURN_IF_ERROR(
      computation->ReplaceWithNewInstruction(infeed_hlo, std::move(new_tuple)));
  custom_call->SetAndSanitizeName(infeed_hlo->name());
  return absl::OkStatus();
}
absl::StatusOr<std::pair<HloInstruction*, HloInstruction*>>
HloControlFlowFlattening::RemoveRecvAndRecvDone(
    HloInstruction* recv_done,
    absl::flat_hash_set<HloInstruction*>* additional_removed) const {
  CHECK_EQ(recv_done->opcode(), HloOpcode::kRecvDone);
  CHECK_EQ(recv_done->operand_count(), 1);
  HloInstruction* recv = recv_done->mutable_operand(0);
  CHECK_EQ(recv->opcode(), HloOpcode::kRecv);
  HloComputation* computation = recv_done->parent();
  CHECK_EQ(recv_done->shape().tuple_shapes_size(), 2);
  HloModule* module = computation->parent();
  HloInstruction* custom_call_recv =
      computation->AddInstruction(HloInstruction::CreateCustomCall(
          recv->shape(), recv->operands(), kNopCustomCallTarget));
  std::string original_recv_name(recv->name());
  if (module->has_schedule() &&
      module->schedule().is_computation_scheduled(computation)) {
    module->schedule().replace_instruction(computation, recv, custom_call_recv);
  }
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(recv, custom_call_recv));
  custom_call_recv->SetAndSanitizeName(original_recv_name);
  std::string original_recv_done_name(recv_done->name());
  HloInstruction* custom_call_recv_done = computation->AddInstruction(
      HloInstruction::CreateCustomCall(
          recv_done->shape(), recv_done->operands(), kNopCustomCallTarget),
      recv_done->name());
  if (module->has_schedule() &&
      module->schedule().is_computation_scheduled(computation)) {
    module->schedule().replace_instruction(computation, recv_done,
                                           custom_call_recv_done);
  }
  TF_RETURN_IF_ERROR(
      computation->ReplaceInstruction(recv_done, custom_call_recv_done));
  custom_call_recv_done->SetAndSanitizeName(original_recv_done_name);
  return std::make_pair(custom_call_recv, custom_call_recv_done);
}
absl::Status HloControlFlowFlattening::RemoveOutfeed(
    HloInstruction* outfeed_hlo) const {
  CHECK_EQ(outfeed_hlo->opcode(), HloOpcode::kOutfeed);
  HloComputation* computation = outfeed_hlo->parent();
  HloInstruction* custom_call =
      computation->AddInstruction(HloInstruction::CreateCustomCall(
          outfeed_hlo->shape(), outfeed_hlo->operands(),
          kNopReturnTokenCustomCallTarget));
  Cast<HloCustomCallInstruction>(custom_call)
      ->set_custom_call_has_side_effect(true);
  custom_call->set_sharding(HloSharding::Manual());
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(outfeed_hlo, custom_call));
  custom_call->SetAndSanitizeName(outfeed_hlo->name());
  return absl::OkStatus();
}
absl::StatusOr<std::pair<HloInstruction*, HloInstruction*>>
HloControlFlowFlattening::RemoveSendAndSendDone(
    HloInstruction* send_done,
    absl::flat_hash_set<HloInstruction*>* additional_removed) const {
  CHECK_EQ(send_done->opcode(), HloOpcode::kSendDone);
  CHECK_EQ(send_done->operand_count(), 1);
  HloInstruction* send = send_done->mutable_operand(0);
  CHECK_EQ(send->opcode(), HloOpcode::kSend);
  HloComputation* computation = send_done->parent();
  HloModule* module = computation->parent();
  HloInstruction* custom_call_send =
      computation->AddInstruction(HloInstruction::CreateCustomCall(
          send->shape(), send->operands(), kNopCustomCallTarget));
  std::string original_send_name(send->name());
  if (module->has_schedule() &&
      module->schedule().is_computation_scheduled(computation)) {
    module->schedule().replace_instruction(computation, send, custom_call_send);
  }
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(send, custom_call_send));
  custom_call_send->SetAndSanitizeName(original_send_name);
  HloInstruction* custom_call_send_done =
      computation->AddInstruction(HloInstruction::CreateCustomCall(
          send_done->shape(), send_done->operands(),
          kNopReturnTokenCustomCallTarget));
  std::string original_send_done_name(send_done->name());
  Cast<HloCustomCallInstruction>(custom_call_send_done)
      ->set_custom_call_has_side_effect(true);
  if (module->has_schedule() &&
      module->schedule().is_computation_scheduled(computation)) {
    module->schedule().replace_instruction(computation, send_done,
                                           custom_call_send_done);
  }
  TF_RETURN_IF_ERROR(
      computation->ReplaceInstruction(send_done, custom_call_send_done));
  custom_call_send_done->SetAndSanitizeName(original_send_done_name);
  return std::make_pair(custom_call_send, custom_call_send_done);
}
absl::StatusOr<HloInstruction*> HloControlFlowFlattening::RemoveCollective(
    HloInstruction* hlo) const {
  HloComputation* computation = hlo->parent();
  HloInstruction* custom_call =
      computation->AddInstruction(HloInstruction::CreateCustomCall(
          hlo->shape(), hlo->operands(), kNopCustomCallTarget));
  custom_call->CopyBackendConfigFrom(hlo);
  HloModule* module = computation->parent();
  if (module->has_schedule() &&
      module->schedule().is_computation_scheduled(computation)) {
    module->schedule().replace_instruction(computation, hlo, custom_call);
  }
  std::string original_op_name(hlo->name());
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(hlo, custom_call));
  custom_call->SetAndSanitizeName(original_op_name);
  return custom_call;
}
absl::Status HloControlFlowFlattening::RemoveId(HloInstruction* hlo) const {
  HloComputation* computation = hlo->parent();
  HloInstruction* zero = CreateConstant(hlo->shape(), computation);
  std::string original_op_name(hlo->name());
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(hlo, zero));
  zero->SetAndSanitizeName(original_op_name);
  return absl::OkStatus();
}
absl::StatusOr<bool> HloControlFlowFlattening::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto call_graph = CallGraph::Build(module);
  bool changed = false;
  absl::flat_hash_set<HloInstruction*> removed;
  for (HloComputation* computation : module->computations(execution_threads)) {
    if (computation->IsAsyncComputation()) {
      continue;
    }
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (removed.contains(instruction)) {
        continue;
      }
      if (flatten_while_loop_ && instruction->opcode() == HloOpcode::kWhile) {
        VLOG(1) << "Remove " << instruction->name();
        TF_RETURN_IF_ERROR(FlattenWhileLoop(instruction, *call_graph));
        changed = true;
      } else if (remove_infeed_outfeed_ &&
                 instruction->opcode() == HloOpcode::kInfeed) {
        VLOG(1) << "Remove " << instruction->name();
        TF_RETURN_IF_ERROR(RemoveInfeed(instruction));
        changed = true;
      } else if (remove_infeed_outfeed_ &&
                 instruction->opcode() == HloOpcode::kOutfeed) {
        VLOG(1) << "Remove " << instruction->name();
        TF_RETURN_IF_ERROR(RemoveOutfeed(instruction));
        changed = true;
      } else if (instruction->opcode() == HloOpcode::kSendDone) {
        auto send_done_instruction =
            DynCast<HloSendDoneInstruction>(instruction);
        CHECK(send_done_instruction);
        if (remove_comm_ || (remove_host_transfer_ &&
                             send_done_instruction->is_host_transfer())) {
          VLOG(1) << "Remove " << instruction->name();
          TF_RETURN_IF_ERROR(
              RemoveSendAndSendDone(instruction, &removed).status());
          changed = true;
        }
      } else if (instruction->opcode() == HloOpcode::kRecvDone) {
        auto recv_done_instruction =
            DynCast<HloRecvDoneInstruction>(instruction);
        CHECK(recv_done_instruction);
        if (remove_comm_ || (remove_host_transfer_ &&
                             recv_done_instruction->is_host_transfer())) {
          VLOG(1) << "Remove " << instruction->name();
          TF_RETURN_IF_ERROR(
              RemoveRecvAndRecvDone(instruction, &removed).status());
          changed = true;
        }
      } else if (remove_comm_ && IsCollective(instruction) &&
                 !instruction->parent()->IsFusionComputation() &&
                 (instruction->opcode() != HloOpcode::kAsyncStart &&
                  instruction->opcode() != HloOpcode::kAsyncUpdate)) {
        if (instruction->opcode() == HloOpcode::kAsyncDone) {
          while (instruction->opcode() == HloOpcode::kAsyncDone ||
                 instruction->opcode() == HloOpcode::kAsyncUpdate ||
                 instruction->opcode() == HloOpcode::kAsyncStart) {
            HloInstruction* operand = instruction->mutable_operand(0);
            VLOG(1) << "Remove " << instruction->name();
            TF_RETURN_IF_ERROR(RemoveCollective(instruction).status());
            instruction = operand;
          }
        } else {
          VLOG(1) << "Remove " << instruction->name();
          TF_RETURN_IF_ERROR(RemoveCollective(instruction).status());
        }
        changed = true;
      } else if ((remove_comm_ || remove_id_) &&
                 (instruction->opcode() == HloOpcode::kPartitionId ||
                  instruction->opcode() == HloOpcode::kReplicaId ||
                  (instruction->opcode() == HloOpcode::kCustomCall &&
                   instruction->custom_call_target() == "SliceId"))) {
        VLOG(1) << "Remove " << instruction->name();
        TF_RETURN_IF_ERROR(RemoveId(instruction));
        changed = true;
      }
    }
  }
  HloDCE hlo_dce;
  TF_ASSIGN_OR_RETURN(bool dce_changed, hlo_dce.Run(module, execution_threads));
  changed |= dce_changed;
  if (changed && module->has_schedule()) {
    TF_RETURN_IF_ERROR(module->schedule().Update());
  }
  XLA_VLOG_LINES(3, module->ToString());
  return changed;
}
}  