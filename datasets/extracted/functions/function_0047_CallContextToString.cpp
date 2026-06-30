#include "xla/service/call_graph.h"
#include <deque>
#include <memory>
#include <queue>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/map_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
namespace xla {
using absl::StrAppendFormat;
using absl::StrCat;
std::string CallContextToString(CallContext context) {
  switch (context) {
    case CallContext::kNone:
      return "kNone";
    case CallContext::kControlFlow:
      return "kControlFlow";
    case CallContext::kEmbedded:
      return "kEmbedded";
    case CallContext::kBoth:
      return "kBoth";
  }
}
std::ostream& operator<<(std::ostream& out, const CallContext& context) {
  out << CallContextToString(context);
  return out;
}
CallContext GetInstructionCallContext(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kWhile:
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:
      return CallContext::kControlFlow;
    case HloOpcode::kAllReduce:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kMap:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kScatter:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kSort:
    case HloOpcode::kTopK:
    case HloOpcode::kFusion:
    case HloOpcode::kCustomCall:
      return CallContext::kEmbedded;
    default:
      return CallContext::kNone;
  }
}
std::string CallSite::ToString() const {
  return StrCat(
      instruction()->name(), " calls in context ",
      CallContextToString(context()), ": ",
      absl::StrJoin(called_computations(), ", ",
                    [](std::string* out, const HloComputation* computation) {
                      absl::StrAppend(out, computation->name());
                    }));
}
CallGraphNode::CallGraphNode(HloComputation* computation)
    : computation_(computation) {}
const CallSite* CallGraphNode::GetCallSite(
    const HloInstruction* instruction) const {
  auto it = callsite_instructions_.find(instruction);
  if (it == callsite_instructions_.end()) {
    return nullptr;
  }
  return &callsites_[it->second];
}
absl::string_view CallGraphNode::ToString() const {
  return computation_->name();
}
void CallGraphNode::AddCallerCallSite(const CallSite& caller_callsite) {
  caller_callsites_.push_back(caller_callsite);
  HloComputation* caller = caller_callsite.instruction()->parent();
  if (!ContainsKey(caller_set_, caller)) {
    callers_.push_back(caller);
    caller_set_.insert(caller);
  }
}
void CallGraphNode::AddCallSiteForInstruction(
    HloInstruction* instruction,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  CHECK_EQ(instruction->parent(), computation());
  const CallContext context = GetInstructionCallContext(instruction->opcode());
  if (!instruction->called_computations().empty()) {
    CHECK(context == CallContext::kControlFlow ||
          context == CallContext::kEmbedded);
    callsite_instructions_.insert({instruction, callsites_.size()});
    callsites_.push_back(
        CallSite(instruction, instruction->called_computations(), context));
    for (auto* callee : callsites_.back().called_computations()) {
      if (HloInstruction::IsThreadIncluded(callee->execution_thread(),
                                           execution_threads) &&
          !ContainsKey(callee_set_, callee)) {
        callees_.push_back(callee);
        callee_set_.insert(callee);
      }
    }
  }
}
CallGraph::CallGraph(
    const HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads)
    : module_(module), execution_threads_(execution_threads) {}
const CallGraphNode& CallGraph::GetNode(
    const HloComputation* computation) const {
  DCHECK(node_indices_.contains(computation));
  return nodes_[node_indices_.find(computation)->second];
}
CallGraphNode& CallGraph::GetNode(const HloComputation* computation) {
  DCHECK(node_indices_.contains(computation));
  return nodes_[node_indices_.find(computation)->second];
}
bool CallGraph::DominatesHelper(
    const HloComputation* a, const HloComputation* b,
    absl::flat_hash_set<const HloComputation*>* visited) const {
  if (a == b || ContainsKey(*visited, b)) {
    return true;
  }
  const CallGraphNode& b_node = GetNode(b);
  if (b_node.callers().empty()) {
    return false;
  }
  visited->insert(b);
  for (const HloComputation* b_caller : b_node.callers()) {
    if (!DominatesHelper(a, b_caller, visited)) {
      return false;
    }
  }
  return true;
}
bool CallGraph::Dominates(const HloComputation* a,
                          const HloComputation* b) const {
  absl::flat_hash_set<const HloComputation*> visited;
  return DominatesHelper(a, b, &visited);
}
bool CallGraph::CanReach(const HloComputation* a,
                         const HloComputation* b) const {
  if (a == b) {
    return true;
  }
  const CallGraphNode& b_node = GetNode(b);
  for (const HloComputation* b_caller : b_node.callers()) {
    if (CanReach(a, b_caller)) {
      return true;
    }
  }
  return false;
}
namespace {
CallContext UnionContexts(CallContext a, CallContext b) {
  if (a == CallContext::kNone) {
    return b;
  } else if (b == CallContext::kNone) {
    return a;
  } else if (a == b) {
    return a;
  } else {
    return CallContext::kBoth;
  }
}
}  
void CallGraph::SetCallContexts() {
  std::queue<CallGraphNode*> worklist;
  for (const HloComputation* computation :
       module_->computations(execution_threads_)) {
    CallGraphNode& node = GetNode(computation);
    if (node.callers().empty()) {
      node.set_context(CallContext::kControlFlow);
      worklist.push(&node);
    }
  }
  while (!worklist.empty()) {
    CallGraphNode* node = worklist.front();
    worklist.pop();
    for (const CallSite& callsite : node->callsites()) {
      for (const HloComputation* callee : callsite.called_computations()) {
        if (!HloInstruction::IsThreadIncluded(callee->execution_thread(),
                                              execution_threads_)) {
          continue;
        }
        CallGraphNode& callee_node = GetNode(callee);
        CallContext context_to_add;
        if (callsite.context() == CallContext::kEmbedded) {
          context_to_add = CallContext::kEmbedded;
        } else {
          CHECK_EQ(callsite.context(), CallContext::kControlFlow);
          context_to_add = node->context();
        }
        CallContext new_context =
            UnionContexts(context_to_add, callee_node.context());
        if (new_context != callee_node.context()) {
          callee_node.set_context(new_context);
          worklist.push(&callee_node);
        }
      }
    }
  }
  for (const HloComputation* computation :
       module_->computations(execution_threads_)) {
    CHECK_NE(GetNode(computation).context(), CallContext::kNone);
  }
}
void CallGraph::SetNodeDepths() {
  std::queue<CallGraphNode*> worklist;
  for (CallGraphNode& node : nodes_) {
    node.set_depth(-1);
  }
  for (const HloComputation* computation :
       module_->computations(execution_threads_)) {
    CallGraphNode& node = GetNode(computation);
    if (node.callers().empty()) {
      node.set_depth(0);
      worklist.push(&node);
    }
  }
  while (!worklist.empty()) {
    CallGraphNode* node = worklist.front();
    worklist.pop();
    for (const HloComputation* callee : node->callees()) {
      CallGraphNode& callee_node = GetNode(callee);
      if (callee_node.depth() < node->depth() + 1) {
        callee_node.set_depth(node->depth() + 1);
        worklist.push(&callee_node);
      }
    }
  }
  for (CallGraphNode& node : nodes_) {
    CHECK_NE(node.depth(), -1);
  }
}
std::unique_ptr<CallGraph> CallGraph::Build(
    const HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto call_graph =
      absl::WrapUnique<CallGraph>(new CallGraph(module, execution_threads));
  VLOG(3) << "Building call graph for:";
  XLA_VLOG_LINES(3, module->ToString());
  for (HloComputation* computation : module->computations(execution_threads)) {
    auto it_added = call_graph->node_indices_.insert(
        {computation, call_graph->nodes_.size()});
    CHECK(it_added.second);
    call_graph->nodes_.emplace_back(computation);
    for (HloInstruction* instruction : computation->instructions()) {
      call_graph->nodes_.back().AddCallSiteForInstruction(instruction,
                                                          execution_threads);
    }
  }
  for (const HloComputation* computation :
       module->computations(execution_threads)) {
    for (const CallSite& callsite :
         call_graph->GetNode(computation).callsites()) {
      for (auto* callee : callsite.called_computations()) {
        if (!HloInstruction::IsThreadIncluded(callee->execution_thread(),
                                              execution_threads)) {
          continue;
        }
        call_graph->GetNode(callee).AddCallerCallSite(callsite);
      }
    }
  }
  call_graph->SetCallContexts();
  call_graph->SetNodeDepths();
  XLA_VLOG_LINES(2, call_graph->ToString());
  return call_graph;
}
absl::Status CallGraph::VisitNodesInternal(
    VisitorFunction visitor_func, const CallGraphNode& node,
    absl::flat_hash_set<const CallGraphNode*>* visited) const {
  auto pair = visited->insert(&node);
  if (!pair.second) {
    return absl::OkStatus();
  }
  for (const HloComputation* computation : node.callees()) {
    TF_RETURN_IF_ERROR(
        VisitNodesInternal(visitor_func, GetNode(computation), visited));
  }
  return visitor_func(node);
}
absl::Status CallGraph::VisitNodes(VisitorFunction visitor_func,
                                   bool visit_unreachable_nodes) const {
  absl::flat_hash_set<const CallGraphNode*> visited;
  if (visit_unreachable_nodes) {
    for (const CallGraphNode& node : nodes()) {
      if (node.callers().empty()) {
        TF_RETURN_IF_ERROR(VisitNodesInternal(visitor_func, node, &visited));
      }
    }
  } else {
    TF_RETURN_IF_ERROR(VisitNodesInternal(
        visitor_func, GetNode(module_->entry_computation()), &visited));
  }
  return absl::OkStatus();
}
bool CallGraph::IsFlattened() const {
  for (const CallGraphNode& node : nodes_) {
    if (node.context() == CallContext::kBoth) {
      return false;
    }
    if (node.context() == CallContext::kControlFlow &&
        !node.computation()->IsAsyncComputation() &&
        node.caller_callsites().size() > 1) {
      return false;
    }
  }
  return true;
}
std::vector<HloInstruction*> CallGraph::GetComputationCallers(
    const HloComputation* c) const {
  std::vector<HloInstruction*> callers;
  for (const auto& callsite : GetNode(c).caller_callsites()) {
    callers.push_back(callsite.instruction());
  }
  return callers;
}
std::pair<HloInstruction*, HloInstruction*>
CallGraph::NearestAncestorsInSameComputation(HloInstruction* a,
                                             HloInstruction* b) const {
  auto next_caller = [this](HloInstruction* instruction) -> HloInstruction* {
    const CallGraphNode& node = GetNode(instruction->parent());
    if (node.caller_callsites().size() != 1) {
      if (instruction->parent()->IsAsyncComputation()) {
        return node.caller_callsites()[0].instruction();
      }
      return nullptr;
    }
    return node.caller_callsites()[0].instruction();
  };
  HloInstruction* a_ancestor = a;
  HloInstruction* b_ancestor = b;
  int a_depth = GetNode(a->parent()).depth();
  int b_depth = GetNode(b->parent()).depth();
  if (a_depth > b_depth) {
    for (int i = 0; i < a_depth - b_depth; ++i) {
      a_ancestor = next_caller(a_ancestor);
      if (a_ancestor == nullptr) {
        return {nullptr, nullptr};
      }
    }
  } else if (b_depth > a_depth) {
    for (int i = 0; i < b_depth - a_depth; ++i) {
      b_ancestor = next_caller(b_ancestor);
      if (b_ancestor == nullptr) {
        return {nullptr, nullptr};
      }
    }
  }
  while ((a_ancestor != nullptr) && (b_ancestor != nullptr)) {
    if (a_ancestor->parent() == b_ancestor->parent()) {
      return {a_ancestor, b_ancestor};
    }
    a_ancestor = next_caller(a_ancestor);
    b_ancestor = next_caller(b_ancestor);
  }
  return {nullptr, nullptr};
}
template <typename T>
absl::flat_hash_set<const T*> CallGraph::NearestCommonAncestorsHelper(
    std::vector<const T*>& starting_nodes) {
  CHECK(
      (std::is_same_v<T, HloInstruction> || std::is_same_v<T, HloComputation>));
  if (starting_nodes.empty()) {
    return absl::flat_hash_set<const T*>();
  }
  if (starting_nodes.size() == 1) {
    return absl::flat_hash_set<const T*>({starting_nodes[0]});
  }
  absl::flat_hash_set<const T*> nearest_common_ancestors;
  std::vector<absl::flat_hash_set<const T*>> visited_ancestors;
  visited_ancestors.reserve(starting_nodes.size());
  for (int idx = 0; idx < starting_nodes.size(); ++idx) {
    visited_ancestors.push_back(
        absl::flat_hash_set<const T*>({starting_nodes[idx]}));
  }
  std::vector<std::deque<const T*>> bfs_queues;
  bfs_queues.reserve(starting_nodes.size());
  for (int idx = 0; idx < starting_nodes.size(); ++idx) {
    bfs_queues.push_back(std::deque<const T*>({starting_nodes[idx]}));
  }
  auto is_bfs_finished = [&bfs_queues]() -> bool {
    return absl::c_all_of(
        bfs_queues, [](std::deque<const T*> queue) { return queue.empty(); });
  };
  auto find_common_nodes = [&visited_ancestors,
                            &nearest_common_ancestors]() -> bool {
    absl::flat_hash_set<const T*> common_nodes(visited_ancestors[0]);
    for (int idx = 1; idx < visited_ancestors.size(); ++idx) {
      absl::erase_if(common_nodes, [&](auto k) {
        return !visited_ancestors[idx].contains(k);
      });
    }
    nearest_common_ancestors = common_nodes;
    return !nearest_common_ancestors.empty();
  };
  while (!is_bfs_finished() && !find_common_nodes()) {
    for (int idx = 0; idx < bfs_queues.size(); ++idx) {
      auto cur_queue = bfs_queues[idx];
      std::deque<const T*> next_queue;
      auto& visited_ancestor = visited_ancestors[idx];
      while (!cur_queue.empty()) {
        const T* node = cur_queue.back();
        cur_queue.pop_back();
        std::vector<T*> ancestors_to_visit;
        if constexpr (std::is_same_v<T, HloInstruction>) {
          ancestors_to_visit = node->users();
          ancestors_to_visit.insert(ancestors_to_visit.end(),
                                    node->control_successors().begin(),
                                    node->control_successors().end());
        } else if constexpr (std::is_same_v<T, HloComputation>) {
          for (auto caller_instruction : GetComputationCallers(node)) {
            ancestors_to_visit.push_back(caller_instruction->parent());
          }
        }
        for (auto ancestor : ancestors_to_visit) {
          if (!visited_ancestor.contains(ancestor)) {
            next_queue.push_back(ancestor);
            visited_ancestor.insert(ancestor);
          }
        }
      }
      bfs_queues[idx] = next_queue;
    }
  }
  CHECK(!nearest_common_ancestors.empty())
      << "At least one nearest_common_ancestor";
  if (absl::c_any_of(starting_nodes, [&nearest_common_ancestors](const T* nca) {
        return nearest_common_ancestors.contains(nca);
      })) {
    absl::erase_if(nearest_common_ancestors, [&starting_nodes](const T* nca) {
      return std::find(starting_nodes.begin(), starting_nodes.end(), nca) ==
             starting_nodes.end();
    });
  }
  return nearest_common_ancestors;
}
absl::flat_hash_set<const HloComputation*>
CallGraph::NearestCommonAncestorComputations(
    std::vector<const HloComputation*> computations) {
  return NearestCommonAncestorsHelper<HloComputation>(computations);
}
absl::flat_hash_set<const HloInstruction*>
CallGraph::NearestCommonAncestorInstructions(
    std::vector<const HloInstruction*> instructions) {
  if (instructions.empty()) {
    return absl::flat_hash_set<const HloInstruction*>();
  }
  auto computation = instructions[0]->parent();
  CHECK(absl::c_all_of(instructions, [&computation](
                                         const HloInstruction* instruction) {
    return instruction->parent() == computation;
  })) << "All provided instructions should be in the same computation";
  return NearestCommonAncestorsHelper<HloInstruction>(instructions);
}
std::string CallGraph::ToString() const {
  std::string out;
  StrAppendFormat(&out, "Call graph for module %s:\n", module_->name());
  for (const CallGraphNode& node : nodes()) {
    StrAppendFormat(&out, "Computation %s:\n", node.computation()->name());
    StrAppendFormat(&out, "  calls:\n");
    for (const HloComputation* callee : node.callees()) {
      StrAppendFormat(&out, "    %s\n", callee->name());
    }
    StrAppendFormat(&out, "  called by:\n");
    for (const HloComputation* caller : node.callers()) {
      StrAppendFormat(&out, "    %s\n", caller->name());
    }
    StrAppendFormat(&out, "  callsites:\n");
    for (const CallSite& callsite : node.callsites()) {
      StrAppendFormat(&out, "    %s\n", callsite.ToString());
    }
  }
  return out;
}
}  