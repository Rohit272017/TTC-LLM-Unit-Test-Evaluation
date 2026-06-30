#include "tensorflow/core/grappler/utils/traversal.h"
#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/graph_topology_view.h"
namespace tensorflow {
namespace grappler {
namespace {
struct DfsStackElem {
  DfsStackElem(int node, bool children_visited, int src)
      : node(node), children_visited(children_visited), src(src) {}
  explicit DfsStackElem(int node) : DfsStackElem(node, false, -1) {}
  int node;
  bool children_visited;
  int src;
};
enum class NodeState { kNotVisited, kVisiting, kDone };
}  
void DfsTraversal(const GraphTopologyView& graph_view,
                  const absl::Span<const NodeDef* const> from,
                  const TraversalDirection direction,
                  const DfsPredicates& predicates,
                  const DfsCallbacks& callbacks) {
  std::vector<DfsStackElem> stack;
  stack.reserve(from.size());
  for (const NodeDef* node : from) {
    const absl::optional<int> node_idx = graph_view.GetNodeIndex(*node);
    DCHECK(node_idx.has_value()) << "Illegal start node: " << node->name();
    if (node_idx.has_value()) {
      stack.emplace_back(node_idx.value());
    }
  }
  absl::flat_hash_map<int, NodeState> node_state;
  while (!stack.empty()) {
    DfsStackElem w = stack.back();
    stack.pop_back();
    NodeState& state = node_state[w.node];
    if (state == NodeState::kDone) continue;
    if (predicates.enter && !predicates.enter(graph_view.GetNode(w.node))) {
      state = NodeState::kDone;
      continue;
    }
    if (w.children_visited) {
      state = NodeState::kDone;
      if (callbacks.post_order) {
        callbacks.post_order(graph_view.GetNode(w.node));
      }
      continue;
    }
    if (state == NodeState::kVisiting) {
      if (callbacks.on_back_edge) {
        callbacks.on_back_edge(graph_view.GetNode(w.src),
                               graph_view.GetNode(w.node));
      }
      continue;
    }
    state = NodeState::kVisiting;
    if (callbacks.pre_order) {
      callbacks.pre_order(graph_view.GetNode(w.node));
    }
    stack.emplace_back(w.node, true, w.src);
    if (predicates.advance && !predicates.advance(graph_view.GetNode(w.node))) {
      continue;
    }
    if (direction == TraversalDirection::kFollowInputs) {
      for (const int fanin : graph_view.GetFanin(w.node)) {
        stack.emplace_back(fanin, false, w.node);
      }
    } else {
      for (const int fanout : graph_view.GetFanout(w.node)) {
        stack.emplace_back(fanout, false, w.node);
      }
    }
  }
}
void DfsTraversal(const GraphTopologyView& graph_view,
                  const absl::Span<const NodeDef* const> from,
                  TraversalDirection direction, const DfsCallbacks& callbacks) {
  DfsTraversal(graph_view, from, direction, {}, callbacks);
}
}  
}  