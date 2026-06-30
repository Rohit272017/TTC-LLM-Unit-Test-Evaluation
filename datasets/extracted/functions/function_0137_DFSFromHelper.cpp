#include "tensorflow/core/graph/algorithm.h"
#include <algorithm>
#include <deque>
#include <vector>
#include "tensorflow/core/platform/logging.h"
namespace tensorflow {
namespace {
template <typename T>
void DFSFromHelper(const Graph& g, gtl::ArraySlice<T> start,
                   const std::function<void(T)>& enter,
                   const std::function<void(T)>& leave,
                   const NodeComparator& stable_comparator,
                   const EdgeFilter& edge_filter) {
  struct Work {
    T node;
    bool leave;  
  };
  std::vector<Work> stack(start.size());
  for (int i = 0; i < start.size(); ++i) {
    stack[i] = Work{start[i], false};
  }
  std::vector<bool> visited(g.num_node_ids(), false);
  while (!stack.empty()) {
    Work w = stack.back();
    stack.pop_back();
    T n = w.node;
    if (w.leave) {
      leave(n);
      continue;
    }
    if (visited[n->id()]) continue;
    visited[n->id()] = true;
    if (enter) enter(n);
    if (leave) stack.push_back(Work{n, true});
    auto add_work = [&visited, &stack](Node* out) {
      if (!visited[out->id()]) {
        stack.push_back(Work{out, false});
      }
    };
    if (stable_comparator) {
      std::vector<Node*> nodes_sorted;
      for (const Edge* out_edge : n->out_edges()) {
        if (!edge_filter || edge_filter(*out_edge)) {
          nodes_sorted.emplace_back(out_edge->dst());
        }
      }
      std::sort(nodes_sorted.begin(), nodes_sorted.end(), stable_comparator);
      for (Node* out : nodes_sorted) {
        add_work(out);
      }
    } else {
      for (const Edge* out_edge : n->out_edges()) {
        if (!edge_filter || edge_filter(*out_edge)) {
          add_work(out_edge->dst());
        }
      }
    }
  }
}
}  
void DFS(const Graph& g, const std::function<void(Node*)>& enter,
         const std::function<void(Node*)>& leave,
         const NodeComparator& stable_comparator,
         const EdgeFilter& edge_filter) {
  DFSFromHelper(g, {g.source_node()}, enter, leave, stable_comparator,
                edge_filter);
}
void DFSFrom(const Graph& g, absl::Span<Node* const> start,
             const std::function<void(Node*)>& enter,
             const std::function<void(Node*)>& leave,
             const NodeComparator& stable_comparator,
             const EdgeFilter& edge_filter) {
  DFSFromHelper(g, start, enter, leave, stable_comparator, edge_filter);
}
void DFSFrom(const Graph& g, absl::Span<const Node* const> start,
             const std::function<void(const Node*)>& enter,
             const std::function<void(const Node*)>& leave,
             const NodeComparator& stable_comparator,
             const EdgeFilter& edge_filter) {
  DFSFromHelper(g, start, enter, leave, stable_comparator, edge_filter);
}
void ReverseDFS(const Graph& g, const std::function<void(Node*)>& enter,
                const std::function<void(Node*)>& leave,
                const NodeComparator& stable_comparator,
                const EdgeFilter& edge_filter) {
  ReverseDFSFrom(g, {g.sink_node()}, enter, leave, stable_comparator,
                 edge_filter);
}
namespace {
template <typename T>
void ReverseDFSFromHelper(const Graph& g, gtl::ArraySlice<T> start,
                          const std::function<void(T)>& enter,
                          const std::function<void(T)>& leave,
                          const NodeComparator& stable_comparator,
                          const EdgeFilter& edge_filter) {
  struct Work {
    T node;
    bool leave;  
  };
  std::vector<Work> stack(start.size());
  for (int i = 0; i < start.size(); ++i) {
    stack[i] = Work{start[i], false};
  }
  std::vector<bool> visited(g.num_node_ids(), false);
  while (!stack.empty()) {
    Work w = stack.back();
    stack.pop_back();
    T n = w.node;
    if (w.leave) {
      leave(n);
      continue;
    }
    if (visited[n->id()]) continue;
    visited[n->id()] = true;
    if (enter) enter(n);
    if (leave) stack.push_back(Work{n, true});
    auto add_work = [&visited, &stack](T out) {
      if (!visited[out->id()]) {
        stack.push_back(Work{out, false});
      }
    };
    if (stable_comparator) {
      std::vector<T> nodes_sorted;
      for (const Edge* in_edge : n->in_edges()) {
        if (!edge_filter || edge_filter(*in_edge)) {
          nodes_sorted.emplace_back(in_edge->src());
        }
      }
      std::sort(nodes_sorted.begin(), nodes_sorted.end(), stable_comparator);
      for (T in : nodes_sorted) {
        add_work(in);
      }
    } else {
      for (const Edge* in_edge : n->in_edges()) {
        if (!edge_filter || edge_filter(*in_edge)) {
          add_work(in_edge->src());
        }
      }
    }
  }
}
}  
void ReverseDFSFrom(const Graph& g, absl::Span<const Node* const> start,
                    const std::function<void(const Node*)>& enter,
                    const std::function<void(const Node*)>& leave,
                    const NodeComparator& stable_comparator,
                    const EdgeFilter& edge_filter) {
  ReverseDFSFromHelper(g, start, enter, leave, stable_comparator, edge_filter);
}
void ReverseDFSFrom(const Graph& g, absl::Span<Node* const> start,
                    const std::function<void(Node*)>& enter,
                    const std::function<void(Node*)>& leave,
                    const NodeComparator& stable_comparator,
                    const EdgeFilter& edge_filter) {
  ReverseDFSFromHelper(g, start, enter, leave, stable_comparator, edge_filter);
}
void GetPostOrder(const Graph& g, std::vector<Node*>* order,
                  const NodeComparator& stable_comparator,
                  const EdgeFilter& edge_filter) {
  order->clear();
  DFS(
      g, nullptr, [order](Node* n) { order->push_back(n); }, stable_comparator,
      edge_filter);
}
void GetReversePostOrder(const Graph& g, std::vector<Node*>* order,
                         const NodeComparator& stable_comparator,
                         const EdgeFilter& edge_filter) {
  GetPostOrder(g, order, stable_comparator, edge_filter);
  std::reverse(order->begin(), order->end());
}
bool PruneForReverseReachability(Graph* g,
                                 std::unordered_set<const Node*> start) {
  std::vector<bool> visited(g->num_node_ids());
  for (auto node : start) {
    visited[node->id()] = true;
  }
  std::deque<const Node*> queue(start.begin(), start.end());
  while (!queue.empty()) {
    const Node* n = queue.front();
    queue.pop_front();
    for (const Node* in : n->in_nodes()) {
      if (!visited[in->id()]) {
        visited[in->id()] = true;
        queue.push_back(in);
        VLOG(2) << "Reverse reach : " << n->name() << " from " << in->name();
      }
    }
  }
  bool any_removed = false;
  for (int i = 0; i < visited.size(); ++i) {
    if (!visited[i]) {
      Node* n = g->FindNodeId(i);
      if (n != nullptr && !n->IsSource() && !n->IsSink()) {
        g->RemoveNode(n);
        any_removed = true;
      }
    }
  }
  return any_removed;
}
bool FixupSourceAndSinkEdges(Graph* g) {
  bool changed = false;
  for (Node* n : g->nodes()) {
    if (!n->IsSource() && n->in_edges().empty()) {
      g->AddControlEdge(g->source_node(), n,
                        true );
      changed = true;
    }
    if (!n->IsSink() && n->out_edges().empty()) {
      g->AddControlEdge(n, g->sink_node(), true );
      changed = true;
    }
  }
  return changed;
}
namespace {
template <class T>
void BreadthFirstTraversalHelper(const Graph& g, gtl::ArraySlice<T> start,
                                 const std::function<void(T)>& visit,
                                 NodeComparator stable_comparator) {
  std::deque<T> stack;
  if (start.empty()) {
    for (T n : g.nodes()) {
      if (n->in_edges().empty()) {
        stack.push_back(n);
      }
    }
  }
  std::vector<bool> seen(g.num_node_ids(), false);
  while (!stack.empty()) {
    T n = stack.front();
    stack.pop_front();
    seen[n->id()] = true;
    visit(n);
    std::vector<T> nodes_sorted;
    for (const Edge* out_edge : n->out_edges()) {
      if (!seen[out_edge->dst()->id()]) {
        seen[out_edge->dst()->id()] = true;
        nodes_sorted.emplace_back(out_edge->dst());
      }
    }
    std::sort(nodes_sorted.begin(), nodes_sorted.end(), stable_comparator);
    for (T out : nodes_sorted) {
      stack.push_back(out);
    }
  }
}
}  
void BreadthFirstTraversal(const Graph& g, absl::Span<const Node* const> start,
                           const std::function<void(const Node*)>& visit,
                           NodeComparator stable_comparator) {
  return BreadthFirstTraversalHelper<const Node*>(g, start, visit,
                                                  stable_comparator);
}
void BreadthFirstTraversal(Graph& g, absl::Span<Node* const> start,
                           const std::function<void(Node*)>& visit,
                           NodeComparator stable_comparator) {
  return BreadthFirstTraversalHelper<Node*>(g, start, visit, stable_comparator);
}
}  