#include "xla/service/gpu/transforms/double_buffer_loop_unrolling.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/flatten_call_graph.h"
#include "xla/service/hlo_parser.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace gpu {
namespace {
void SetChannelIdForNewCollective(HloInstruction* new_instr,
                                  const HloModule* module) {
  absl::flat_hash_map<int64_t, int64_t> old_to_new_channel_id_map;
  absl::flat_hash_map<int64_t, HloComputation*> channel_id_comp_map;
  if (new_instr->IsAsynchronous() && hlo_query::IsCollectiveCommunicationOp(
                                         new_instr->async_wrapped_opcode())) {
    HloInstruction* wrapped_instr =
        DynCast<HloAsyncInstruction>(new_instr)->async_wrapped_instruction();
    int64_t old_channel_id = *wrapped_instr->channel_id();
    int64_t new_channel_id = old_to_new_channel_id_map[old_channel_id];
    if (old_to_new_channel_id_map.find(old_channel_id) ==
        old_to_new_channel_id_map.end()) {
      new_channel_id = hlo_query::NextChannelId(*module);
      VLOG(2) << "Generated new channel id " << new_channel_id;
      old_to_new_channel_id_map[old_channel_id] = new_channel_id;
    }
    VLOG(2) << "Setting channel id to " << new_channel_id;
    wrapped_instr->set_channel_id(new_channel_id);
    if (channel_id_comp_map.find(new_channel_id) == channel_id_comp_map.end()) {
      channel_id_comp_map[new_channel_id] =
          new_instr->async_wrapped_computation();
    } else {
      channel_id_comp_map[new_channel_id]->AddAsyncStart(new_instr);
    }
  } else if (hlo_query::IsCollectiveCommunicationOp(new_instr->opcode()) ||
             hlo_query::IsAsyncCollectiveStartOp(new_instr)) {
    new_instr->set_channel_id(hlo_query::NextChannelId(*module));
  }
}
using Interval = std::pair<int64_t, int64_t>;
absl::StatusOr<std::vector<Interval>> ParseVectorOfPairs(
    absl::string_view str) {
  TF_ASSIGN_OR_RETURN(std::vector<ReplicaGroup> replica_groups,
                      ParseReplicaGroupsOnly(str));
  std::vector<Interval> res;
  res.reserve(replica_groups.size());
  for (const ReplicaGroup& replica_group : replica_groups) {
    TF_RET_CHECK(replica_group.replica_ids_size() == 2);
    int64_t a = replica_group.replica_ids(0);
    int64_t b = replica_group.replica_ids(1);
    res.emplace_back(a, b);
  }
  return res;
}
absl::Status SetSendRecvValidationForPeeledInstr(HloInstruction* new_instr,
                                                 HloInstruction* old_instr) {
  TF_RET_CHECK(
      new_instr->opcode() == old_instr->opcode() &&
      "cloned instruction and original instruction have different opcodes");
  if (!HloPredicateIsOp<HloOpcode::kCollectivePermute,
                        HloOpcode::kCollectivePermuteStart, HloOpcode::kSend,
                        HloOpcode::kRecv>(old_instr)) {
    return absl::OkStatus();
  }
  const auto& attribute_map = new_instr->frontend_attributes().map();
  if (!attribute_map.contains(kSendRecvValidationAttr)) {
    return absl::OkStatus();
  }
  VLOG(3) << "Original send-recv iterations: "
          << attribute_map.at(kSendRecvValidationAttr);
  TF_ASSIGN_OR_RETURN(
      auto send_recv_validation_attr,
      ParseVectorOfPairs(attribute_map.at(kSendRecvValidationAttr)));
  uint64_t n_pairs = send_recv_validation_attr.size();
  if (n_pairs == 0) {
    return absl::OkStatus();
  }
  std::vector<Interval> send_recv_validation_attr_updated(n_pairs, {1, 0});
  for (std::uint64_t i = 0; i < send_recv_validation_attr.size(); i++) {
    if (send_recv_validation_attr[i].first <= 0 &&
        send_recv_validation_attr[i].second >= 0) {
      send_recv_validation_attr_updated[i] = {0, 0};
    }
  }
  hlo_instruction_utils::AddOrUpdateVectorOfPairsAsAttribute(
      new_instr, kSendRecvValidationAttr,
      send_recv_validation_attr_updated);
  return absl::OkStatus();
}
absl::Status SetSendRecvValidation(HloInstruction* cp1, HloInstruction* cp2,
                                   bool is_peeled) {
  TF_RET_CHECK(
      cp2->opcode() == cp1->opcode() &&
      "cloned instruction and original instruction have different opcodes");
  if (!HloPredicateIsOp<HloOpcode::kCollectivePermute,
                        HloOpcode::kCollectivePermuteStart, HloOpcode::kSend,
                        HloOpcode::kRecv>(cp1)) {
    return absl::OkStatus();
  }
  const auto& attribute_map = cp2->frontend_attributes().map();
  if (!attribute_map.contains(kSendRecvValidationAttr)) {
    return absl::OkStatus();
  }
  VLOG(3) << "Original send-recv iterations: "
          << attribute_map.at(kSendRecvValidationAttr);
  TF_ASSIGN_OR_RETURN(
      auto send_recv_validation_attr,
      ParseVectorOfPairs(attribute_map.at(kSendRecvValidationAttr)));
  if (send_recv_validation_attr.size() == 0) {
    return absl::OkStatus();
  }
  std::vector<Interval> send_recv_iterations_new_instr1,
      send_recv_iterations_new_instr2;
  send_recv_iterations_new_instr1.reserve(send_recv_validation_attr.size());
  send_recv_iterations_new_instr2.reserve(send_recv_validation_attr.size());
  for (const Interval& pair : send_recv_validation_attr) {
    int64_t a = pair.first;
    int64_t b = pair.second;
    if (is_peeled) {
      send_recv_iterations_new_instr1.emplace_back(
          std::floor(a / 2.0), std::max(0.0, std::floor((b - 1) / 2.0)));
      send_recv_iterations_new_instr2.emplace_back(
          std::max(0.0, std::floor((a - 1) / 2.0)),
          std::max(0.0, std::floor((b - 2) / 2.0)));
    } else {
      send_recv_iterations_new_instr1.emplace_back(std::floor((a + 1) / 2.0),
                                                   std::floor(b / 2.0));
      send_recv_iterations_new_instr2.emplace_back(
          std::floor(a / 2.0), std::max(0.0, std::floor((b - 1) / 2.0)));
    }
  }
  hlo_instruction_utils::AddOrUpdateVectorOfPairsAsAttribute(
      cp1, kSendRecvValidationAttr,
      send_recv_iterations_new_instr1);
  hlo_instruction_utils::AddOrUpdateVectorOfPairsAsAttribute(
      cp2, kSendRecvValidationAttr,
      send_recv_iterations_new_instr2);
  VLOG(3) << "Updated send-recv iterations for " << cp1->name() << " : "
          << cp1->frontend_attributes().map().at(kSendRecvValidationAttr);
  VLOG(3) << "Updated send-recv iterations for " << cp2->name() << " : "
          << cp2->frontend_attributes().map().at(kSendRecvValidationAttr);
  return absl::OkStatus();
}
absl::Status HandleControlDependencies(
    const HloComputation* while_body,
    const absl::flat_hash_map<HloInstruction*, HloInstruction*>& old_to_new_map,
    HloInstruction::InstructionVector* old_loop_roots,
    HloInstruction* input_parameter,
    const absl::flat_hash_set<HloInstruction*>& skip_control_dep_injection) {
  for (HloInstruction* old_instr : while_body->MakeInstructionPostOrder()) {
    if (old_to_new_map.find(old_instr) != old_to_new_map.end()) {
      HloInstruction* new_instr = old_to_new_map.at(old_instr);
      VLOG(2) << "Processing control predecessors for "
              << new_instr->ToString();
      std::vector<HloInstruction*> new_control_pred;
      new_control_pred.reserve(old_instr->control_predecessors().size());
      for (HloInstruction* pred : old_instr->control_predecessors()) {
        if (!old_to_new_map.contains(pred)) {
          continue;
        }
        new_control_pred.push_back(old_to_new_map.at(pred));
      }
      TF_RETURN_IF_ERROR(new_instr->DropAllControlDeps());
      for (HloInstruction* new_pred : new_control_pred) {
        TF_RETURN_IF_ERROR(new_pred->AddControlDependencyTo(new_instr));
        VLOG(2) << "Adding " << new_pred->ToString()
                << " to control dependency of " << new_instr->ToString();
      }
    }
  }
  for (HloInstruction* input_consumer : input_parameter->users()) {
    for (HloInstruction* old_input : input_consumer->users()) {
      if (old_to_new_map.find(old_input) != old_to_new_map.end()) {
        HloInstruction* new_input = old_to_new_map.at(old_input);
        if (skip_control_dep_injection.find(old_input) ==
                skip_control_dep_injection.end() &&
            !IsCollective(old_input)) {
          for (HloInstruction* old_root : *old_loop_roots) {
            TF_RETURN_IF_ERROR(old_root->AddControlDependencyTo(new_input));
          }
        }
      }
    }
  }
  return absl::OkStatus();
}
absl::StatusOr<bool> FullyUnroll(HloInstruction* while_instr,
                                 HloModule* module) {
  HloComputation* while_body = while_instr->while_body();
  bool changed = false;
  VLOG(2) << "Processing root " << while_body->root_instruction()->ToString();
  auto loop_roots = while_body->root_instruction()->mutable_operands();
  HloInstruction* input_parameter = while_body->parameter_instruction(0);
  VLOG(2) << "Processing input parameter " << input_parameter->ToString();
  absl::flat_hash_map<HloInstruction*, HloInstruction*> old_to_new_map;
  absl::flat_hash_set<HloInstruction*> skip_control_dep_injection;
  std::string clone_suffix = "full_unroll_clone";
  TF_ASSIGN_OR_RETURN(WhileLoopBackendConfig config,
                      while_instr->backend_config<WhileLoopBackendConfig>());
  std::vector<HloInstruction*> ops_to_clone;
  ops_to_clone.reserve(while_body->MakeInstructionPostOrder().size());
  HloInstruction* old_input_parameter = input_parameter;
  HloInstruction* new_input_parameter = while_body->root_instruction();
  absl::flat_hash_set<HloInstruction*> seen_ops;
  for (HloInstruction* old_instr : while_body->MakeInstructionPostOrder()) {
    if (seen_ops.contains(old_instr)) {
      continue;
    }
    ops_to_clone.push_back(old_instr);
    seen_ops.insert(old_instr);
  }
  int n = config.known_trip_count().n();
  while (--n) {
    std::vector<HloInstruction*> new_ops_to_clone;
    old_to_new_map[old_input_parameter] = new_input_parameter;
    for (HloInstruction* old_instr : ops_to_clone) {
      if (old_to_new_map.contains(old_instr)) {
        continue;
      }
      VLOG(2) << "Cloning instruction " << old_instr->ToString();
      std::vector<HloInstruction*> new_operands;
      for (HloInstruction* old_operand : old_instr->mutable_operands()) {
        new_operands.push_back(old_to_new_map[old_operand]);
      }
      HloInstruction* new_instr =
          while_body->AddInstruction(old_instr->CloneWithNewOperands(
              old_instr->shape(), new_operands, clone_suffix));
      if (old_instr->IsElementwiseBinary() && old_instr->HasConstantOperand()) {
        skip_control_dep_injection.insert(old_instr);
      }
      SetChannelIdForNewCollective(new_instr, module);
      old_to_new_map[old_instr] = new_instr;
      new_ops_to_clone.push_back(new_instr);
      VLOG(2) << "Added instruction " << new_instr->ToString();
    }
    while_body->set_root_instruction(
        old_to_new_map[while_body->root_instruction()]);
    VLOG(2) << "Replaced with new root "
            << while_body->root_instruction()->ToString();
    TF_RETURN_IF_ERROR(HandleControlDependencies(
        while_body, old_to_new_map, &loop_roots, old_input_parameter,
        skip_control_dep_injection));
    old_to_new_map.clear();
    skip_control_dep_injection.clear();
    loop_roots = while_body->root_instruction()->mutable_operands();
    old_input_parameter = new_input_parameter;
    new_input_parameter = while_body->root_instruction();
    ops_to_clone = std::move(new_ops_to_clone);
    changed = true;
  }
  WhileLoopBackendConfig new_config;
  new_config.mutable_known_trip_count()->set_n(1);
  TF_RETURN_IF_ERROR(while_instr->set_backend_config(new_config));
  return changed;
}
absl::Status PeelInstructionsForOddTripCount(HloModule* module,
                                             HloInstruction* while_instr) {
  std::string suffix = "peeled_double_buffer";
  absl::flat_hash_map<HloInstruction*, HloInstruction*> old_to_new_map;
  HloComputation* while_body = while_instr->while_body();
  HloInstruction* input_parameter = while_body->parameter_instruction(0);
  HloInstruction* input_tuple = while_instr->mutable_operand(0);
  auto old_loop_roots = while_body->root_instruction()->mutable_operands();
  HloComputation* parent_comp = while_instr->parent();
  old_to_new_map[input_parameter] = input_tuple;
  for (HloInstruction* old_instr : while_body->MakeInstructionPostOrder()) {
    if (old_to_new_map.find(old_instr) != old_to_new_map.end()) {
      continue;
    }
    VLOG(2) << "Peeling instruction " << old_instr->ToString();
    std::vector<HloInstruction*> new_operands(old_instr->operand_count());
    for (int64_t i = 0; i < old_instr->operand_count(); i++) {
      new_operands[i] = old_to_new_map[old_instr->mutable_operand(i)];
    }
    HloInstruction* new_instr =
        parent_comp->AddInstruction(old_instr->CloneWithNewOperands(
            old_instr->shape(), new_operands, suffix));
    SetChannelIdForNewCollective(new_instr, module);
    TF_CHECK_OK(SetSendRecvValidationForPeeledInstr(new_instr, old_instr));
    old_to_new_map[old_instr] = new_instr;
    VLOG(2) << "Added instruction " << new_instr->ToString()
            << " to parent computation.";
  }
  std::vector<HloInstruction*> new_roots;
  for (HloInstruction* instr : old_loop_roots) {
    new_roots.push_back(old_to_new_map[instr]);
  }
  TF_RETURN_IF_ERROR(while_instr->ReplaceOperandWith(
      0, old_to_new_map[while_body->root_instruction()]));
  VLOG(2) << "Replaced with new input tuple "
          << while_instr->operand(0)->ToString();
  for (HloInstruction* old_instr : while_body->MakeInstructionPostOrder()) {
    if (old_to_new_map.find(old_instr) != old_to_new_map.end()) {
      HloInstruction* new_instr = old_to_new_map[old_instr];
      VLOG(2) << "Processing control predecessors for peeled instruction "
              << new_instr->ToString();
      std::vector<HloInstruction*> new_control_pred(
          old_instr->control_predecessors().size());
      for (HloInstruction* pred : old_instr->control_predecessors()) {
        new_control_pred.push_back(old_to_new_map[pred]);
      }
      TF_RETURN_IF_ERROR(new_instr->DropAllControlDeps());
      for (HloInstruction* new_pred : new_control_pred) {
        TF_RETURN_IF_ERROR(new_pred->AddControlDependencyTo(new_instr));
        VLOG(2) << "Adding " << new_pred->ToString()
                << " to control dependency of peeled instruction: "
                << new_instr->ToString();
      }
    }
  }
  return absl::OkStatus();
}
absl::StatusOr<bool> DoubleBufferingUnroll(HloInstruction* while_instr,
                                           HloModule* module) {
  TF_ASSIGN_OR_RETURN(auto config,
                      while_instr->backend_config<WhileLoopBackendConfig>());
  CHECK(config.has_known_trip_count())
      << "Only loops with known trip count are supported.";
  int64_t exact_trip_count = config.known_trip_count().n();
  VLOG(2) << "Processing while loop " << while_instr->ToString()
          << " with trip count: " << exact_trip_count;
  HloComputation* while_body = while_instr->while_body();
  VLOG(2) << "Processing root " << while_body->root_instruction()->ToString();
  auto old_loop_roots = while_body->root_instruction()->mutable_operands();
  HloInstruction* input_parameter = while_body->parameter_instruction(0);
  VLOG(2) << "Processing input parameter " << input_parameter->ToString();
  absl::flat_hash_map<HloInstruction*, HloInstruction*> old_to_new_map;
  absl::flat_hash_set<HloInstruction*> skip_control_dep_injection;
  bool peel_one_iteration = exact_trip_count % 2;
  if (peel_one_iteration) {
    VLOG(2) << "Found loops with odd trip count, 1 iteration will be peeled "
               "outside of the main body.";
    TF_RETURN_IF_ERROR(PeelInstructionsForOddTripCount(module, while_instr));
    exact_trip_count -= 1;
  }
  std::string suffix = "double_buffer_clone";
  old_to_new_map[input_parameter] = while_body->root_instruction();
  for (HloInstruction* old_instr : while_body->MakeInstructionPostOrder()) {
    if (old_to_new_map.find(old_instr) != old_to_new_map.end()) {
      continue;
    }
    VLOG(2) << "Cloning instruction " << old_instr->ToString();
    std::vector<HloInstruction*> new_operands;
    for (HloInstruction* old_operand : old_instr->mutable_operands()) {
      new_operands.push_back(old_to_new_map[old_operand]);
    }
    HloInstruction* new_instr =
        while_body->AddInstruction(old_instr->CloneWithNewOperands(
            old_instr->shape(), new_operands, suffix));
    if (old_instr->IsElementwiseBinary() && old_instr->HasConstantOperand()) {
      skip_control_dep_injection.insert(old_instr);
    }
    SetChannelIdForNewCollective(new_instr, module);
    TF_CHECK_OK(SetSendRecvValidation(old_instr, new_instr,
                                      peel_one_iteration));
    old_to_new_map[old_instr] = new_instr;
    VLOG(2) << "Added instruction " << new_instr->ToString();
  }
  while_body->set_root_instruction(
      old_to_new_map[while_body->root_instruction()]);
  VLOG(2) << "Replaced with new root "
          << while_body->root_instruction()->ToString();
  TF_RETURN_IF_ERROR(HandleControlDependencies(while_body, old_to_new_map,
                                               &old_loop_roots, input_parameter,
                                               skip_control_dep_injection));
  WhileLoopBackendConfig new_config;
  new_config.mutable_known_trip_count()->set_n(exact_trip_count / 2);
  TF_RETURN_IF_ERROR(while_instr->set_backend_config(new_config));
  return true;  
}
absl::StatusOr<bool> AutoUnroll(HloInstruction* while_instr,
                                HloModule* module) {
  CHECK_EQ(while_instr->opcode(), HloOpcode::kWhile);
  bool any_collective_present = absl::c_any_of(
      while_instr->while_body()->MakeInstructionPostOrder(),
      [](HloInstruction* instr) {
        return hlo_query::IsCollectiveCommunicationOp(instr->opcode());
      });
  if (any_collective_present) {
    return DoubleBufferingUnroll(while_instr, module);
  }
  return false;  
}
}  
absl::StatusOr<bool> DoubleBufferLoopUnrolling::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  std::vector<HloInstruction*> while_instrs;
  for (auto comp : module->MakeNonfusionComputations()) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(while_instrs),
                    HloPredicateIsOp<HloOpcode::kWhile>);
  }
  VLOG(2) << "Processing " << while_instrs.size() << " while loops.";
  for (HloInstruction* while_instr : while_instrs) {
    TF_ASSIGN_OR_RETURN(WhileLoopBackendConfig config,
                        while_instr->backend_config<WhileLoopBackendConfig>());
    if (!config.has_known_trip_count()) {
      VLOG(2) << while_instr->ToString()
              << " doesn't have exact trip count, skipping loop unrolling.";
      continue;
    }
    if (config.known_trip_count().n() == 1) {
      VLOG(2) << while_instr->ToString()
              << " has an iteration count of one, skipping unrolling.";
      continue;
    }
    if (unroll_strategy_ == UnrollStrategy::kFullUnroll) {
      TF_ASSIGN_OR_RETURN(changed, FullyUnroll(while_instr, module));
    } else if (unroll_strategy_ == UnrollStrategy::kDoubleBuffer) {
      TF_ASSIGN_OR_RETURN(changed, DoubleBufferingUnroll(while_instr, module));
    } else if (unroll_strategy_ == UnrollStrategy::kAuto) {
      TF_ASSIGN_OR_RETURN(changed, AutoUnroll(while_instr, module));
    } else {
      LOG(FATAL) << absl::StrCat("Unhandled unrolling strategy: ",
                                 unroll_strategy_);
    }
  }
  VLOG(2) << "LoopDoubleBufferTransformer output: " << module->ToString();
  if (changed) {
    TF_RETURN_IF_ERROR(
        FlattenCallGraph().Run(module, execution_threads).status());
  }
  return changed;
}
}  
}  