#ifndef AROLLA_DECISION_FOREST_POINTWISE_EVALUATION_POINTWISE_H_
#define AROLLA_DECISION_FOREST_POINTWISE_EVALUATION_POINTWISE_H_
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <type_traits>
#include <vector>
#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla {
namespace internal {
struct NodeId {
  static NodeId Split(int32_t split_node_id) { return {split_node_id}; }
  static NodeId Leaf(int32_t adjustment_id) { return {~adjustment_id}; }
  bool is_leaf() const { return val < 0; }
  int32_t split_node_id() const {
    DCHECK(!this->is_leaf());
    return val;
  }
  int32_t adjsutment_id() const {
    DCHECK(this->is_leaf());
    return ~val;
  }
  int32_t val;
};
template <class NodeTest>
struct CompactCondition {
  NodeTest test;
  std::array<NodeId, 2> next_node_ids;
};
template <class OutT, class NodeTest>
struct CompactDecisionTree {
  std::vector<CompactCondition<NodeTest>> splits;
  std::vector<OutT> adjustments;
  NodeId RootNodeId() const {
    return splits.empty() ? NodeId::Leaf(0) : NodeId::Split(0);
  }
};
template <class OutT, class NodeTest>
struct SingleTreeCompilationImpl {
  explicit SingleTreeCompilationImpl(size_t node_cnt)
      : nodes_(node_cnt), node_used_(node_cnt), node_used_as_child_(node_cnt) {
    if (node_cnt > 0) {
      node_used_as_child_[0] = true;
    }
  }
  absl::Status SetNode(size_t node_id, size_t left_id, size_t right_id,
                       const NodeTest& test) {
    RETURN_IF_ERROR(TestNode(node_id, &node_used_));
    RETURN_IF_ERROR(TestNode(left_id, &node_used_as_child_));
    RETURN_IF_ERROR(TestNode(right_id, &node_used_as_child_));
    nodes_[node_id].left_id = left_id;
    nodes_[node_id].right_id = right_id;
    nodes_[node_id].test = test;
    return absl::OkStatus();
  }
  absl::Status SetLeaf(size_t node_id, OutT leaf_value) {
    RETURN_IF_ERROR(TestNode(node_id, &node_used_));
    nodes_[node_id].leaf_id = leaf_values_.size();
    leaf_values_.push_back(leaf_value);
    return absl::OkStatus();
  }
  absl::StatusOr<CompactDecisionTree<OutT, NodeTest>> Compile() {
    if (nodes_.empty()) {
      return absl::Status(absl::StatusCode::kFailedPrecondition,
                          "Empty trees are not supported.");
    }
    for (size_t i = 0; i < nodes_.size(); ++i) {
      if (!node_used_as_child_[i] || !node_used_[i]) {
        return absl::Status(absl::StatusCode::kInvalidArgument,
                            "Id is not used");
      }
    }
    CompactDecisionTree<OutT, NodeTest> tree;
    const size_t inner_node_count = nodes_.size() - leaf_values_.size();
    leaf_values_.shrink_to_fit();
    tree.adjustments = std::move(leaf_values_);
    std::vector<NodeId> node_mapping(nodes_.size());
    int cur_new_node_id = 0;
    for (int id = 0; id < nodes_.size(); ++id) {
      const auto& node = nodes_[id];
      if (node.leaf_id == -1) {
        node_mapping[id] = NodeId::Split(cur_new_node_id++);
      } else {
        node_mapping[id] = NodeId::Leaf(node.leaf_id);
      }
    }
    tree.splits.resize(inner_node_count);
    for (int id = 0; id < nodes_.size(); ++id) {
      const auto& node = nodes_[id];
      if (node.leaf_id == -1) {
        auto& compact_node = tree.splits[node_mapping[id].split_node_id()];
        compact_node.test = node.test;
        compact_node.next_node_ids[1] = node_mapping[node.left_id];
        compact_node.next_node_ids[0] = node_mapping[node.right_id];
      }
    }
    return tree;
  }
 private:
  absl::Status TestNode(size_t id, std::vector<bool>* used) {
    if (id >= used->size()) {
      return absl::Status(absl::StatusCode::kOutOfRange, "Id out of range");
    }
    if ((*used)[id]) {
      return absl::Status(absl::StatusCode::kInvalidArgument, "Id duplicated");
    }
    (*used)[id] = true;
    return absl::OkStatus();
  }
  struct Node {
    size_t leaf_id = -1;  
    size_t left_id = -1;
    size_t right_id = -1;
    NodeTest test = {};
  };
  std::vector<Node> nodes_;
  std::vector<bool> node_used_;
  std::vector<bool> node_used_as_child_;
  std::vector<OutT> leaf_values_;
};
template <class OutT, class NodeTest>
class DecisionTreeTraverser {
 public:
  explicit DecisionTreeTraverser(
      const internal::CompactDecisionTree<OutT, NodeTest>& tree)
      : node_id_(tree.RootNodeId()), tree_(tree) {}
  bool CanStep() const { return !node_id_.is_leaf(); }
  template <class FeatureContainer>
  void MakeStep(const FeatureContainer& values) {
    const auto& split = tree_.splits[node_id_.split_node_id()];
    node_id_ = split.next_node_ids[split.test(values)];
  }
  OutT GetValue() const { return tree_.adjustments[node_id_.adjsutment_id()]; }
 private:
  NodeId node_id_;
  const internal::CompactDecisionTree<OutT, NodeTest>& tree_;
};
struct EmptyFilterTag {};
}  
template <class OutT, class NodeTest>
class SinglePredictor {
 public:
  explicit SinglePredictor(internal::CompactDecisionTree<OutT, NodeTest> tree)
      : tree_(std::move(tree)) {}
  template <class FeatureContainer>
  OutT Predict(const FeatureContainer& values) const {
    internal::DecisionTreeTraverser<OutT, NodeTest> traverser(tree_);
    while (ABSL_PREDICT_TRUE(traverser.CanStep())) {
      traverser.MakeStep(values);
    }
    return traverser.GetValue();
  }
 private:
  internal::CompactDecisionTree<OutT, NodeTest> tree_;
};
template <class TreeOutT, class NodeTest, class BinaryOp,
          class FilterTag = internal::EmptyFilterTag>
class BoostedPredictor {
 public:
  using OutT = std::decay_t<decltype(BinaryOp()(TreeOutT(), TreeOutT()))>;
  using NodeTestType = NodeTest;
  explicit BoostedPredictor(
      std::vector<internal::CompactDecisionTree<TreeOutT, NodeTest>> trees,
      std::vector<FilterTag> filter_tags, BinaryOp op)
      : trees_(std::move(trees)),
        filter_tags_(std::move(filter_tags)),
        op_(op) {}
  template <class FeatureContainer, class FilterFn>
  OutT Predict(const FeatureContainer& values, OutT start,
               FilterFn filter) const {
    constexpr int kBatchSize = 16;
    absl::InlinedVector<internal::DecisionTreeTraverser<TreeOutT, NodeTest>,
                        kBatchSize>
        traversers;
    uint32_t ids[kBatchSize];
    for (int first = 0; first < trees_.size(); first += kBatchSize) {
      int count = std::min<int>(trees_.size() - first, kBatchSize);
      traversers.clear();
      uint32_t* ids_end = ids;
      for (int i = 0; i < count; ++i) {
        if (!filter(filter_tags_[first + i])) continue;
        *(ids_end++) = traversers.size();
        traversers.emplace_back(trees_[first + i]);
      }
      while (ABSL_PREDICT_TRUE(ids != ids_end)) {
        auto out_it = ids;
        for (auto it = ids; it != ids_end; ++it) {
          auto& traverser = traversers[*it];
          if (ABSL_PREDICT_TRUE(traverser.CanStep())) {
            traverser.MakeStep(values);
            *(out_it++) = *it;
          } else {
            start = op_(start, traverser.GetValue());
          }
        }
        ids_end = out_it;
      }
    }
    return start;
  }
  template <class FeatureContainer>
  OutT Predict(const FeatureContainer& values, OutT start = OutT()) const {
    if (trees_.empty()) return start;
    return Predict(values, start, [](FilterTag tag) { return true; });
  }
 private:
  std::vector<internal::CompactDecisionTree<TreeOutT, NodeTest>> trees_;
  std::vector<FilterTag> filter_tags_;
  BinaryOp op_;
};
template <class OutT, class NodeTest>
class PredictorCompiler {
 public:
  explicit PredictorCompiler(size_t node_cnt) : impl_(node_cnt) {}
  absl::Status SetNode(size_t node_id, size_t left_id, size_t right_id,
                       const NodeTest& test) {
    return impl_.SetNode(node_id, left_id, right_id, test);
  }
  absl::Status SetLeaf(size_t node_id, OutT leaf_value) {
    return impl_.SetLeaf(node_id, leaf_value);
  }
  absl::StatusOr<SinglePredictor<OutT, NodeTest>> Compile() {
    if (compiled_) {
      return absl::Status(absl::StatusCode::kFailedPrecondition,
                          "Already compiled.");
    }
    compiled_ = true;
    auto tree_or = impl_.Compile();
    RETURN_IF_ERROR(tree_or.status());
    return SinglePredictor<OutT, NodeTest>(*std::move(tree_or));
  }
 private:
  bool compiled_ = false;
  internal::SingleTreeCompilationImpl<OutT, NodeTest> impl_;
};
template <class OutT, class NodeTest>
class OneTreeCompiler {
 public:
  explicit OneTreeCompiler(
      internal::SingleTreeCompilationImpl<OutT, NodeTest>* impl)
      : impl_(impl) {}
  absl::Status SetNode(size_t node_id, size_t left_id, size_t right_id,
                       const NodeTest& test) {
    return impl_->SetNode(node_id, left_id, right_id, test);
  }
  absl::Status SetLeaf(size_t node_id, OutT leaf_value) {
    return impl_->SetLeaf(node_id, leaf_value);
  }
 private:
  internal::SingleTreeCompilationImpl<OutT, NodeTest>* impl_;
};
template <class OutT, class NodeTest, class BinaryOp,
          class FilterTag = internal::EmptyFilterTag>
class BoostedPredictorCompiler {
 public:
  BoostedPredictorCompiler() {}
  explicit BoostedPredictorCompiler(BinaryOp op) : op_(op) {}
  OneTreeCompiler<OutT, NodeTest> AddTree(size_t node_count,
                                          FilterTag tag = FilterTag()) {
    impls_.emplace_back(node_count);
    filter_tags_.push_back(tag);
    return OneTreeCompiler<OutT, NodeTest>(&impls_.back());
  }
  absl::StatusOr<BoostedPredictor<OutT, NodeTest, BinaryOp, FilterTag>>
  Compile() {
    if (compiled_) {
      return absl::Status(absl::StatusCode::kFailedPrecondition,
                          "Already compiled.");
    }
    compiled_ = true;
    std::vector<internal::CompactDecisionTree<OutT, NodeTest>> trees;
    trees.reserve(impls_.size());
    for (auto& impl : impls_) {
      auto tree_or = impl.Compile();
      RETURN_IF_ERROR(tree_or.status());
      trees.push_back(*std::move(tree_or));
    }
    return BoostedPredictor<OutT, NodeTest, BinaryOp, FilterTag>(
        std::move(trees), std::move(filter_tags_), op_);
  }
 private:
  bool compiled_ = false;
  BinaryOp op_;
  std::deque<internal::SingleTreeCompilationImpl<OutT, NodeTest>> impls_;
  std::vector<FilterTag> filter_tags_;
};
}  
#endif  