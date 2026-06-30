#include "xla/service/p2p_schedule_preparation.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_reachability.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace {
bool IsP2POp(const HloInstruction* op) {
  auto p2p = DynCast<HloSendRecvInstruction>(op);
  return p2p != nullptr && !p2p->is_host_transfer();
}
bool IsCollectiveOp(const HloInstruction* op) {
  HloOpcode opcode = op->opcode();
  if (opcode == HloOpcode::kCustomCall) {
    return true;
  }
  return hlo_query::IsAsyncCollectiveDoneOp(op, true) ||
         (hlo_query::IsCollectiveCommunicationOp(opcode) &&
          !hlo_query::IsAsyncCollectiveStartOp(op, true));
}
HloInstruction* GetStartOpForDoneOp(HloInstruction* op) {
  switch (op->opcode()) {
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kSendDone:
    case HloOpcode::kRecvDone:
      return op->mutable_operand(0);
    default:
      return op;
  }
}
enum P2PGroupKind { kUnpipelined = 0, kPipelined = 1, kUnrecognized = 2 };
enum P2PRuntimeStream { kUnknown = 0, kStream0 = 1, kStream1 = 2 };
struct P2PGroupNode {
  bool RecordParentComputation(HloComputation* parent) {
    if (computation == nullptr) {
      computation = parent;
      return true;
    }
    return computation == parent;
  }
  bool RecordP2POp(HloSendRecvInstruction* p2p) {
    if (!RecordParentComputation(p2p->parent())) {
      return false;
    }
    switch (p2p->opcode()) {
      case HloOpcode::kRecvDone:
        if (recv_done == nullptr) {
          recv_done = Cast<HloRecvDoneInstruction>(p2p);
          return true;
        }
        break;
      case HloOpcode::kSendDone:
        if (send_done == nullptr) {
          send_done = Cast<HloSendDoneInstruction>(p2p);
          return true;
        }
        break;
      case HloOpcode::kRecv:
        if (recv == nullptr) {
          recv = Cast<HloRecvInstruction>(p2p);
          return true;
        }
        break;
      case HloOpcode::kSend:
        if (send == nullptr) {
          send = Cast<HloSendInstruction>(p2p);
          return true;
        }
        break;
      default:
        break;
    }
    return false;
  }
  bool RecordWhileOp(HloInstruction* while_op) {
    if (while_loop != nullptr) {
      return false;
    }
    if (!RecordParentComputation(while_op->parent())) {
      return false;
    }
    while_loop = while_op;
    return true;
  }
  bool Incomplete() const {
    return recv_done == nullptr || send_done == nullptr || recv == nullptr ||
           send == nullptr;
  }
  bool IncompletePipelinedParent() const {
    return Incomplete() || while_loop == nullptr;
  }
  P2PRuntimeStream GetRuntimeStream(const HloInstruction* start) const {
    auto it = start->frontend_attributes().map().find(kSendRecvPipelineAttr);
    if (it != start->frontend_attributes().map().end()) {
      if (it->second == "0") {
        return kStream0;
      }
      if (it->second == "1") {
        return kStream1;
      }
    }
    return kUnknown;
  }
  P2PRuntimeStream GetRuntimeStream() const {
    P2PRuntimeStream send_stream = GetRuntimeStream(send);
    P2PRuntimeStream recv_stream = GetRuntimeStream(recv);
    if (send_stream != recv_stream) {
      return kUnknown;
    }
    return send_stream;
  }
  int64_t GetChannel() const { return recv->channel_id().value(); }
  HloRecvDoneInstruction* recv_done = nullptr;
  HloSendDoneInstruction* send_done = nullptr;
  HloRecvInstruction* recv = nullptr;
  HloSendInstruction* send = nullptr;
  HloComputation* computation = nullptr;
  HloInstruction* while_loop = nullptr;
};
struct P2PGroup;
using P2PGroupMap = absl::flat_hash_map<int64_t, P2PGroup>;
using P2PInComputation =
    absl::flat_hash_map<const HloComputation*, std::set<int64_t>>;
using CollectiveInComputation =
    absl::flat_hash_map<const HloComputation*, bool>;
using ChainStartEnd =
    std::pair<HloSendRecvInstruction*, HloSendRecvInstruction*>;
static constexpr int kUnpipelinedNodeIdx = 0;
static constexpr int kPipelinedChildNodeIdx = 0;
static constexpr int kPipelinedParentNodeIdx = 1;
struct P2PGroup {
  absl::Status RecordP2POpForUnpipelinedGroup(HloSendRecvInstruction* p2p) {
    if (kind == kUnrecognized) {
      return absl::OkStatus();
    }
    if (kind != kUnpipelined) {
      return Internal("Expected unpipelined group");
    }
    P2PGroupNode& node = nodes[kUnpipelinedNodeIdx];
    if (!node.RecordP2POp(p2p)) {
      kind = kUnrecognized;
    }
    return absl::OkStatus();
  }
  absl::Status RecordP2POpForPipelinedGroup(HloSendRecvInstruction* p2p) {
    if (kind == kUnrecognized) {
      return absl::OkStatus();
    }
    if (kind == kUnpipelined) {
      if (nodes[kPipelinedParentNodeIdx].computation != nullptr) {
        return Internal("Expected unpipelined group");
      }
      kind = kPipelined;
    }
    P2PGroupNode& node = nodes[kPipelinedParentNodeIdx];
    if (!node.RecordP2POp(p2p)) {
      kind = kUnrecognized;
    }
    return absl::OkStatus();
  }
  absl::Status RecordWhileOpToPipelinedGroup(HloInstruction* while_op) {
    if (kind == kUnrecognized) {
      return absl::OkStatus();
    }
    if (kind == kUnpipelined) {
      return Internal("Expected pipelined group");
    }
    P2PGroupNode& node = nodes[kPipelinedParentNodeIdx];
    if (!node.RecordWhileOp(while_op)) {
      kind = kUnrecognized;
    }
    return absl::OkStatus();
  }
  bool RecordRuntimeStream() {
    P2PRuntimeStream child_stream =
        nodes[kPipelinedChildNodeIdx].GetRuntimeStream();
    if (kind == kPipelined) {
      P2PRuntimeStream parent_stream =
          nodes[kPipelinedParentNodeIdx].GetRuntimeStream();
      if (child_stream != parent_stream || child_stream == kUnknown) {
        return false;
      }
    }
    runtime_stream = child_stream;
    return true;
  }
  absl::Status RecordComplementGroup(P2PGroupMap& p2p_group_map) {
    CHECK(!complement_group_channel.has_value() && runtime_stream == kStream1);
    for (auto& [channel, p2p_group] : p2p_group_map) {
      if (&p2p_group == this ||
          p2p_group.ChildComputation() != ChildComputation()) {
        continue;
      }
      if (p2p_group.kind == kPipelined &&
          p2p_group.ParentComputation() == ParentComputation()) {
        if (p2p_group.runtime_stream != kStream0) {
          return Internal(
              "Expected different pipeline stream for complement group");
        }
        complement_group_channel = channel;
        p2p_group.complement_group_channel = GetChannel();
      } else if (p2p_group.kind == kUnpipelined &&
                 p2p_group.runtime_stream == kStream0) {
        complement_group_channel = channel;
        p2p_group.complement_group_channel = GetChannel();
      }
    }
    return absl::OkStatus();
  }
  HloComputation* ParentComputation() const { return GetParent().computation; }
  HloComputation* ChildComputation() const { return GetChild().computation; }
  int64_t GetChannel() const { return nodes[kUnpipelinedNodeIdx].GetChannel(); }
  P2PGroupNode& GetChild() { return nodes[kPipelinedChildNodeIdx]; }
  P2PGroupNode& GetParent() { return nodes[kPipelinedParentNodeIdx]; }
  const P2PGroupNode& GetChild() const { return nodes[kPipelinedChildNodeIdx]; }
  const P2PGroupNode& GetParent() const {
    return nodes[kPipelinedParentNodeIdx];
  }
  ChainStartEnd GetChainStartEnd(const HloComputation* computation,
                                 const P2PGroupMap& p2p_group_map) const {
    if (computation == ChildComputation()) {
      if (!InCycle()) {
        return std::make_pair(GetChild().recv, GetChild().send_done);
      }
      if (runtime_stream == kStream1) {
        return std::make_pair(
            GetComplementGroup(p2p_group_map)->GetChild().recv,
            GetChild().send_done);
      }
      return std::make_pair(
          GetChild().recv,
          GetComplementGroup(p2p_group_map)->GetChild().send_done);
    }
    CHECK(kind == kPipelined && computation == ParentComputation());
    if (!InCycle()) {
      return std::make_pair(GetParent().recv, GetParent().send_done);
    }
    if (runtime_stream == kStream1) {
      return std::make_pair(GetComplementGroup(p2p_group_map)->GetParent().recv,
                            GetParent().send_done);
    }
    return std::make_pair(
        GetParent().recv,
        GetComplementGroup(p2p_group_map)->GetParent().send_done);
  }
  HloInstruction* GetWhileOp() const {
    return nodes[kPipelinedParentNodeIdx].while_loop;
  }
  bool InCycle() const { return complement_group_channel.has_value(); }
  P2PGroup* GetComplementGroup(P2PGroupMap& p2p_group_map) const {
    CHECK(InCycle());
    return &p2p_group_map.at(*complement_group_channel);
  }
  const P2PGroup* GetComplementGroup(const P2PGroupMap& p2p_group_map) const {
    CHECK(InCycle());
    return &p2p_group_map.at(*complement_group_channel);
  }
  P2PGroupKind kind = kUnpipelined;
  P2PGroupNode nodes[2];
  P2PRuntimeStream runtime_stream = kUnknown;
  std::optional<int64_t> complement_group_channel = std::nullopt;
};
bool MayInvokeCollectiveOp(
    const HloInstruction* hlo,
    const CollectiveInComputation& collective_in_computation) {
  if (IsCollectiveOp(hlo)) {
    return true;
  }
  for (auto callee : hlo->called_computations()) {
    auto collective_in_comp = collective_in_computation.find(callee);
    if (collective_in_comp != collective_in_computation.end() &&
        collective_in_comp->second) {
      return true;
    }
  }
  return false;
}
absl::Status MayAddWhileOpToPipelinedGroup(HloInstruction* while_op,
                                           P2PInComputation& p2p_in_computation,
                                           P2PGroupMap& p2p_group_map) {
  if (while_op->while_init()->opcode() != HloOpcode::kTuple) {
    return absl::OkStatus();
  }
  HloComputation* body = while_op->called_computations()[0];
  auto p2p_in_while = p2p_in_computation.find(body);
  if (p2p_in_while == p2p_in_computation.end()) {
    return absl::OkStatus();
  }
  int pipelined_group = 0;
  for (auto hlo : while_op->while_init()->operands()) {
    if (hlo->opcode() != HloOpcode::kSendDone) {
      continue;
    }
    int64_t channel_id = hlo->channel_id().value();
    if (p2p_in_while->second.find(channel_id) == p2p_in_while->second.end()) {
      continue;
    }
    auto group = p2p_group_map.find(channel_id);
    if (group == p2p_group_map.end() || group->second.kind != kPipelined) {
      continue;
    }
    pipelined_group++;
    if (pipelined_group > 2) {
      return Internal(
          "Expecting up to two pipelined P2P groups for each while-loop");
    }
    TF_RETURN_IF_ERROR(group->second.RecordWhileOpToPipelinedGroup(while_op));
  }
  return absl::OkStatus();
}
absl::Status OrderBefore(HloInstruction* i1, HloInstruction* i2) {
  TF_RETURN_IF_ERROR(i1->AddControlDependencyTo(i2));
  VLOG(10) << "Add control predecessor " << i2->ToString();
  return absl::OkStatus();
}
absl::Status ConnectP2P1NodeChain(const P2PGroupNode& node) {
  HloRecvDoneInstruction* recv_done = node.recv_done;
  HloRecvInstruction* recv = node.recv;
  HloSendDoneInstruction* send_done = node.send_done;
  HloSendInstruction* send = node.send;
  TF_RETURN_IF_ERROR(OrderBefore(recv, send));
  TF_RETURN_IF_ERROR(OrderBefore(send, recv_done));
  TF_RETURN_IF_ERROR(OrderBefore(recv_done, send_done));
  return absl::OkStatus();
}
absl::Status ConnectUnpipelinedP2P(const P2PGroup& p2p_group) {
  return ConnectP2P1NodeChain(p2p_group.GetChild());
}
absl::Status ConnectPipelined1P2PChild(const P2PGroup& p2p_group) {
  return ConnectP2P1NodeChain(p2p_group.GetChild());
}
absl::Status ConnectP2P2NodeChain(const P2PGroupNode& node0,
                                  const P2PGroupNode& node1) {
  HloSendRecvInstruction* recv_done0 = node0.recv_done;
  HloRecvInstruction* recv0 = node0.recv;
  HloSendRecvInstruction* send_done0 = node0.send_done;
  HloSendInstruction* send0 = node0.send;
  HloSendRecvInstruction* recv_done1 = node1.recv_done;
  HloRecvInstruction* recv1 = node1.recv;
  HloSendRecvInstruction* send_done1 = node1.send_done;
  HloSendInstruction* send1 = node1.send;
  TF_RETURN_IF_ERROR(OrderBefore(recv_done0, recv_done1));
  TF_RETURN_IF_ERROR(OrderBefore(recv_done1, send_done0));
  TF_RETURN_IF_ERROR(OrderBefore(send_done0, send_done1));
  TF_RETURN_IF_ERROR(OrderBefore(recv0, send0));
  TF_RETURN_IF_ERROR(OrderBefore(send0, recv1));
  TF_RETURN_IF_ERROR(OrderBefore(recv1, send1));
  TF_RETURN_IF_ERROR(OrderBefore(send1, recv_done0));
  return absl::OkStatus();
}
absl::Status ConnectPipelined2P2PChild(const P2PGroup& p2p_group,
                                       const P2PGroupMap& p2p_group_map) {
  return ConnectP2P2NodeChain(
      p2p_group.GetComplementGroup(p2p_group_map)->GetChild(),
      p2p_group.GetChild());
}
absl::Status ConnectPipelined1P2PParent(const P2PGroup& p2p_group) {
  return ConnectP2P1NodeChain(p2p_group.GetParent());
}
absl::Status ConnectPipelined2P2PParent(const P2PGroup& p2p_group,
                                        const P2PGroupMap& p2p_group_map) {
  return ConnectP2P2NodeChain(
      p2p_group.GetComplementGroup(p2p_group_map)->GetParent(),
      p2p_group.GetParent());
}
absl::Status ConnectUnpipelined2P2P(const P2PGroup& p2p_group,
                                    const P2PGroupMap& p2p_group_map) {
  CHECK(p2p_group.runtime_stream == kStream1);
  return ConnectP2P2NodeChain(
      p2p_group.GetComplementGroup(p2p_group_map)->GetChild(),
      p2p_group.GetChild());
}
absl::Status GatherP2PGroupsAndCollectiveInfo(
    const HloComputation* computation, P2PInComputation& p2p_in_computation,
    P2PGroupMap& p2p_group_map,
    CollectiveInComputation& collective_in_computation) {
  collective_in_computation[computation] = false;
  std::vector<HloInstruction*> while_ops;
  for (auto hlo : computation->MakeInstructionPostOrder()) {
    if (MayInvokeCollectiveOp(hlo, collective_in_computation)) {
      collective_in_computation[computation] = true;
    }
    if (hlo->opcode() == HloOpcode::kWhile) {
      while_ops.push_back(hlo);
      continue;
    }
    if (!IsP2POp(hlo)) {
      continue;
    }
    HloSendRecvInstruction* p2p = Cast<HloSendRecvInstruction>(hlo);
    int64_t channel = p2p->channel_id().value();
    auto p2p_group = p2p_group_map.find(channel);
    if (p2p_group == p2p_group_map.end()) {
      P2PGroup group;
      TF_RETURN_IF_ERROR(group.RecordP2POpForUnpipelinedGroup(p2p));
      p2p_group_map[channel] = group;
    } else {
      P2PGroup& group = p2p_group->second;
      if (group.ChildComputation() == computation) {
        TF_RETURN_IF_ERROR(group.RecordP2POpForUnpipelinedGroup(p2p));
      } else {
        TF_RETURN_IF_ERROR(group.RecordP2POpForPipelinedGroup(p2p));
      }
    }
    auto p2p_in_comp = p2p_in_computation.find(computation);
    if (p2p_in_comp == p2p_in_computation.end()) {
      p2p_in_computation[computation] = {channel};
    } else {
      p2p_in_comp->second.insert(channel);
    }
  }
  for (auto hlo : while_ops) {
    TF_RETURN_IF_ERROR(
        MayAddWhileOpToPipelinedGroup(hlo, p2p_in_computation, p2p_group_map));
  }
  for (auto& [channel, p2p_group] : p2p_group_map) {
    if (p2p_group.kind == kUnpipelined) {
      if (p2p_group.nodes[kUnpipelinedNodeIdx].Incomplete() ||
          !p2p_group.RecordRuntimeStream()) {
        p2p_group.kind = kUnrecognized;
      }
    } else if (p2p_group.kind == kPipelined) {
      if (p2p_group.nodes[kPipelinedChildNodeIdx].Incomplete() ||
          p2p_group.nodes[kPipelinedParentNodeIdx]
              .IncompletePipelinedParent() ||
          !p2p_group.RecordRuntimeStream()) {
        p2p_group.kind = kUnrecognized;
      }
    }
  }
  absl::erase_if(p2p_group_map, [](const auto& p2p_group) {
    return p2p_group.second.kind == kUnrecognized;
  });
  for (auto& [channel, p2p_group] : p2p_group_map) {
    if ((p2p_group.kind == kPipelined &&
         p2p_group.ParentComputation() != computation) ||
        p2p_group.InCycle() || p2p_group.runtime_stream != kStream1) {
      continue;
    }
    TF_RETURN_IF_ERROR(p2p_group.RecordComplementGroup(p2p_group_map));
  }
  return absl::OkStatus();
}
absl::StatusOr<std::pair<int, const P2PGroup*>> ConnectP2PChain(
    HloComputation* computation, const P2PGroupMap& p2p_group_map,
    const std::set<int64_t>& p2p_channels) {
  const P2PGroup* pipelined_group = nullptr;
  int num_p2p_chains = 0;
  for (int64_t channel : p2p_channels) {
    auto it = p2p_group_map.find(channel);
    if (it == p2p_group_map.end()) {
      continue;
    }
    num_p2p_chains++;
    const P2PGroup& p2p_group = it->second;
    P2PGroupKind kind = p2p_group.kind;
    if (kind == P2PGroupKind::kUnpipelined) {
      if (!p2p_group.InCycle()) {
        TF_RETURN_IF_ERROR(ConnectUnpipelinedP2P(p2p_group));
      } else if (p2p_group.runtime_stream == kStream1) {
        TF_RETURN_IF_ERROR(ConnectUnpipelined2P2P(p2p_group, p2p_group_map));
      }
      continue;
    }
    if (!p2p_group.InCycle()) {
      if (computation == p2p_group.ParentComputation()) {
        TF_RETURN_IF_ERROR(ConnectPipelined1P2PParent(p2p_group));
      } else {
        if (pipelined_group != nullptr) {
          return Internal("Expected <=1 pipelined group in a while-body");
        }
        pipelined_group = &p2p_group;
        TF_RETURN_IF_ERROR(ConnectPipelined1P2PChild(p2p_group));
      }
      continue;
    }
    if (p2p_group.runtime_stream != kStream1) {
      continue;
    }
    if (computation == p2p_group.ParentComputation()) {
      TF_RETURN_IF_ERROR(ConnectPipelined2P2PParent(p2p_group, p2p_group_map));
    } else {
      if (pipelined_group != nullptr) {
        return Internal(
            "Expected only two pipelined groups forming a cycle in a "
            "while-body");
      }
      pipelined_group = &p2p_group;
      TF_RETURN_IF_ERROR(ConnectPipelined2P2PChild(p2p_group, p2p_group_map));
    }
  }
  return std::make_pair(num_p2p_chains, pipelined_group);
}
absl::Status OrderBefore(HloReachabilityMap* reachability, HloInstruction* a,
                         HloInstruction* b) {
  VLOG(10) << "OrderBefore " << a->ToString() << " " << b->ToString();
  if (!reachability->IsReachable(a, b)) {
    TF_RETURN_IF_ERROR(a->AddControlDependencyTo(b));
    VLOG(10) << "add control predecessor " << b->ToString();
    reachability->UpdateReachabilityThroughInstruction(b);
  }
  return absl::OkStatus();
}
absl::Status LinearizeCollectivesWithOtherP2P(
    const P2PGroupMap& p2p_group_map, const P2PGroup& group,
    const CollectiveInComputation& collective_in_computation,
    const std::vector<HloInstruction*>::iterator& chain_start_iter,
    const std::vector<HloInstruction*>::iterator& begin_iter,
    const std::vector<HloInstruction*>::iterator& end_iter,
    HloReachabilityMap* reachability) {
  HloComputation* computation = (*chain_start_iter)->parent();
  ChainStartEnd start_end = group.GetChainStartEnd(computation, p2p_group_map);
  for (auto it = begin_iter; it != end_iter; ++it) {
    HloInstruction* hlo = *it;
    if (IsP2POp(hlo)) {
      auto group_it = p2p_group_map.find(hlo->channel_id().value());
      if (group_it == p2p_group_map.end()) {
        continue;
      }
      const P2PGroup& cur_group = group_it->second;
      P2PGroupKind kind = cur_group.kind;
      if (kind == P2PGroupKind::kPipelined &&
          computation == cur_group.ChildComputation()) {
        continue;
      }
      ChainStartEnd cur_start_end =
          cur_group.GetChainStartEnd(computation, p2p_group_map);
      if (cur_start_end.first != hlo) {
        continue;
      }
      if (it <= chain_start_iter) {
        continue;
      }
      if (reachability->IsReachable(start_end.first, cur_start_end.second)) {
        TF_RETURN_IF_ERROR(
            OrderBefore(reachability, start_end.second, cur_start_end.first));
      } else {
        TF_RETURN_IF_ERROR(
            OrderBefore(reachability, cur_start_end.second, start_end.first));
      }
      continue;
    }
    if (!MayInvokeCollectiveOp(hlo, collective_in_computation)) {
      continue;
    }
    if (hlo->opcode() == HloOpcode::kWhile &&
        group.kind == P2PGroupKind::kPipelined && group.GetWhileOp() == hlo) {
      continue;
    }
    if (hlo_query::IsAsyncCollectiveDoneOp(hlo, false)) {
      if (reachability->IsReachable(start_end.first, hlo)) {
        TF_RETURN_IF_ERROR(OrderBefore(reachability, start_end.second,
                                       GetStartOpForDoneOp(hlo)));
      } else {
        TF_RETURN_IF_ERROR(OrderBefore(reachability, hlo, start_end.first));
      }
    }
    if (reachability->IsReachable(start_end.first, hlo)) {
      TF_RETURN_IF_ERROR(OrderBefore(reachability, start_end.second, hlo));
    } else {
      TF_RETURN_IF_ERROR(OrderBefore(reachability, hlo, start_end.first));
    }
  }
  return absl::OkStatus();
}
absl::Status LinearizeCollectivesWithPipelinedP2PChild(
    const P2PGroupMap& p2p_group_map, const P2PGroup& group,
    const CollectiveInComputation& collective_in_computation,
    HloComputation* computation, HloReachabilityMap* reachability) {
  ChainStartEnd start_end = group.GetChainStartEnd(computation, p2p_group_map);
  for (HloInstruction* hlo : computation->MakeInstructionPostOrder()) {
    if (!MayInvokeCollectiveOp(hlo, collective_in_computation)) {
      continue;
    }
    HloOpcode opcode = hlo->opcode();
    if (IsP2POp(hlo) && opcode != HloOpcode::kSendDone) {
      continue;
    }
    if (hlo->opcode() == HloOpcode::kSendDone) {
      auto group_it = p2p_group_map.find(hlo->channel_id().value());
      if (group_it == p2p_group_map.end()) {
        continue;
      }
      const P2PGroup& cur_group = group_it->second;
      P2PGroupKind kind = cur_group.kind;
      if (kind == P2PGroupKind::kPipelined &&
          computation == cur_group.ChildComputation()) {
        continue;
      }
      ChainStartEnd cur_start_end =
          cur_group.GetChainStartEnd(computation, p2p_group_map);
      TF_RETURN_IF_ERROR(
          OrderBefore(reachability, cur_start_end.second, start_end.first));
      continue;
    }
    TF_RETURN_IF_ERROR(OrderBefore(reachability, hlo, start_end.first));
  }
  return absl::OkStatus();
}
}  
absl::StatusOr<bool> P2PSchedulePreparation::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  P2PGroupMap p2p_group_map;
  P2PInComputation p2p_in_computation;
  CollectiveInComputation collective_in_computation;
  std::vector<HloComputation*> all_computations =
      module->MakeComputationPostOrder(execution_threads);
  for (auto iter = all_computations.begin(); iter != all_computations.end();
       ++iter) {
    VLOG(10) << "Gathering P2P groups and collective info for computation "
             << (*iter)->name();
    TF_RETURN_IF_ERROR(GatherP2PGroupsAndCollectiveInfo(
        *iter, p2p_in_computation, p2p_group_map, collective_in_computation));
  }
  if (p2p_group_map.empty()) {
    return false;
  }
  for (auto iter = all_computations.rbegin(); iter != all_computations.rend();
       ++iter) {
    HloComputation* computation = *iter;
    auto p2p_in_comp = p2p_in_computation.find(computation);
    if (p2p_in_comp == p2p_in_computation.end()) {
      continue;
    }
    std::set<int64_t>& p2p_channels = p2p_in_comp->second;
    TF_ASSIGN_OR_RETURN(
        auto result, ConnectP2PChain(computation, p2p_group_map, p2p_channels));
    if (result.first == 0) {
      continue;
    }
    VLOG(10) << "Processing computation " << computation->name()
             << " num_p2p_chains " << result.first;
    std::unique_ptr<HloReachabilityMap> reachability =
        HloReachabilityMap::Build(computation);
    if (result.second != nullptr) {
      TF_RETURN_IF_ERROR(LinearizeCollectivesWithPipelinedP2PChild(
          p2p_group_map, *result.second, collective_in_computation, computation,
          reachability.get()));
    }
    std::vector<HloInstruction*> all_instructions =
        computation->MakeInstructionPostOrder();
    std::vector<HloInstruction*>::iterator begin = all_instructions.begin();
    std::vector<HloInstruction*>::iterator end = all_instructions.end();
    for (auto instr_it = begin; instr_it != end; ++instr_it) {
      HloInstruction* hlo = *instr_it;
      if (!IsP2POp(hlo)) {
        continue;
      }
      HloSendRecvInstruction* p2p = Cast<HloSendRecvInstruction>(hlo);
      int64_t channel = p2p->channel_id().value();
      auto group_it = p2p_group_map.find(channel);
      if (group_it == p2p_group_map.end()) {
        continue;
      }
      P2PGroup& group = group_it->second;
      P2PGroupKind kind = group.kind;
      if (kind == P2PGroupKind::kPipelined &&
          computation == group.ChildComputation()) {
        continue;
      }
      ChainStartEnd start_end =
          group.GetChainStartEnd(computation, p2p_group_map);
      if (start_end.first != hlo) {
        continue;
      }
      VLOG(10) << "linearize other collectives with respect to channel "
               << hlo->ToString();
      TF_RETURN_IF_ERROR(LinearizeCollectivesWithOtherP2P(
          p2p_group_map, group, collective_in_computation, instr_it, begin, end,
          reachability.get()));
      VLOG(10) << "finish connect other collectives with channel ";
    }
  }
  return true;
}
}  