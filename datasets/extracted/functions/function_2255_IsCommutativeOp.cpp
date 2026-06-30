#include "tensorflow/core/grappler/utils/pattern_utils.h"
#include <algorithm>
#include <memory>
#include "absl/container/flat_hash_set.h"
namespace tensorflow {
namespace grappler {
namespace utils {
const bool IsCommutativeOp(const string& op) {
  std::vector<string> op_list = str_util::Split(op, '|');
  static const auto* commutative_ops = new absl::flat_hash_set<string>(
      {"Add", "AddV2", "Mul", "Maximum", "SquaredDifference"});
  for (const string& op_ : op_list) {
    if (commutative_ops->contains(op_)) return true;
  }
  return false;
}
bool IsSame(string op1, string op2) {
  if (op1 == "*") return true;
  std::vector<string> op1_list = str_util::Split(op1, '|');
  for (const string& op_1 : op1_list) {
    if (op_1 == op2) return true;
  }
  return false;
}
template <>
bool SubGraphMatcher<MatchingDirection::kFollowInputs>::DoesOpTypePatternMatch(
    const OpTypePattern& pattern, MutableNodeView* node_view,
    NodeViewMatch* match) {
  if ((node_view->NumControllingFanins() > 0 &&
       pattern.node_status != NodeStatus::kRemain) ||
      (node_view->NumControlledFanouts() > 0 &&
       pattern.node_status == NodeStatus::kRemove))
    return false;
  bool op_type_matched = false;
  if (pattern.op == "*") {
    op_type_matched = true;
  } else {
    std::vector<string> op_list = str_util::Split(pattern.op, '|');
    for (const string& op : op_list) {
      if (node_view->node()->op() == op) {
        op_type_matched = true;
        break;
      }
    }
  }
  if (op_type_matched) {
    if (node_label_to_index_.find(pattern.label) ==
        node_label_to_index_.end()) {
      node_label_to_index_[pattern.label] = node_view->node_index();
      matched_node_indices_.insert(node_view->node_index());
      if (pattern.node_status == NodeStatus::kRemove) {
        remove_node_indices_.insert(node_view->node_index());
      }
    } else if (node_label_to_index_[pattern.label] != node_view->node_index()) {
      return false;  
    } else {
      DCHECK(node_label_to_index_[pattern.label] == node_view->node_index());
    }
  } else {
    return false;
  }
  match->node_view = node_view;
  if (!pattern.children.empty()) {
    auto graph_children = node_view->GetRegularFanins();
    int num_children = graph_children.size();
    if (num_children != pattern.children.size()) {
      return false;
    } else {
      std::vector<int> pattern_child_indices(num_children);
      std::iota(pattern_child_indices.begin(), pattern_child_indices.end(), 0);
      string op_name = pattern.op;
      if (IsCommutativeOp(op_name) && num_children == 2) {
        MutableNodeView* graph_child0_node_view =
            graph_view_->GetNode(graph_children[0].node_index());
        MutableNodeView* graph_child1_node_view =
            graph_view_->GetNode(graph_children[1].node_index());
        if ((!IsSame(pattern.children[0].op, graph_child0_node_view->GetOp()) &&
             IsSame(pattern.children[1].op, graph_child0_node_view->GetOp())) ||
            (!IsSame(pattern.children[1].op, graph_child1_node_view->GetOp()) &&
             IsSame(pattern.children[0].op, graph_child1_node_view->GetOp())))
          std::swap(pattern_child_indices[0], pattern_child_indices[1]);
      }
      for (int i = 0; i < num_children; ++i) {
        auto child_node_index = graph_children[i].node_index();
        MutableNodeView* child_node_view =
            graph_view_->GetNode(child_node_index);
        const OpTypePattern& child_pattern =
            pattern.children[pattern_child_indices[i]];
        match->children.push_back(NodeViewMatch());
        NodeViewMatch* child_match = &(match->children.back());
        if (!DoesOpTypePatternMatch(child_pattern, child_node_view,
                                    child_match)) {
          return false;
        }
      }
    }
  }
  return true;
}
template <>
bool SubGraphMatcher<MatchingDirection::kFollowInputs>::GetMatchedNodes(
    const OpTypePattern& pattern,
    const std::unordered_set<string>& nodes_to_preserve,
    MutableNodeView* node_view, std::map<string, int>* matched_nodes_map,
    std::set<int>* remove_node_indices) {
  bool found_match = false;
  match_ = std::make_unique<NodeViewMatch>();
  if (DoesOpTypePatternMatch(pattern, node_view, match_.get())) {
    if (IsSafeNodesToRemove(nodes_to_preserve)) {
      found_match = true;
      *matched_nodes_map = this->node_label_to_index_;
      *remove_node_indices = this->remove_node_indices_;
    }
  } else {
    found_match = false;
  }
  match_->Clear();
  match_.reset(nullptr);
  matched_node_indices_.clear();
  node_label_to_index_.clear();
  remove_node_indices_.clear();
  return found_match;
}
}  
}  
}  