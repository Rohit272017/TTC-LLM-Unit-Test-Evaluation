#ifndef TENSORFLOW_CORE_GRAPPLER_UTILS_GRAPH_VIEW_INTERNAL_H_
#define TENSORFLOW_CORE_GRAPPLER_UTILS_GRAPH_VIEW_INTERNAL_H_
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/map_util.h"
namespace tensorflow {
namespace grappler {
namespace utils {
namespace internal {
constexpr int kMissingSlot = -2;
constexpr int kMissingIndex = -1;
constexpr int kNodeNamePresent = -1;
template <typename NodeViewT, typename GraphViewT>
class NodeIndexAndPortIndex {
 public:
  NodeIndexAndPortIndex()
      : graph_view_(nullptr),
        node_index_(kMissingIndex),
        port_index_(kMissingSlot) {}
  NodeIndexAndPortIndex(GraphViewT* graph_view, int node_index, int port_index)
      : graph_view_(graph_view),
        node_index_(node_index),
        port_index_(port_index) {}
  bool operator==(const NodeIndexAndPortIndex& other) const {
    return port_index_ == other.port_index_ &&
           node_index_ == other.node_index_ && graph_view_ == other.graph_view_;
  }
  template <typename Hash>
  friend Hash AbslHashValue(Hash h, const NodeIndexAndPortIndex& n) {
    return Hash::combine(std::move(h), n.node_index_, n.port_index_);
  }
  NodeViewT* node_view() const {
    if (graph_view_ == nullptr) {
      return nullptr;
    }
    return graph_view_->GetNode(node_index_);
  }
  int node_index() const { return node_index_; }
  int index() const { return port_index_; }
 protected:
  GraphViewT* graph_view_;
  int node_index_;
  int port_index_;
};
class NodeDefAndPortIndex {
 public:
  NodeDefAndPortIndex(const NodeDef* node_def, int port_index)
      : node_def_(node_def), port_index_(port_index) {}
  bool operator==(const NodeDefAndPortIndex& other) const {
    return node_def_ == other.node_def_ && port_index_ == other.port_index_;
  }
  template <typename Hash>
  friend Hash AbslHashValue(Hash h, const NodeDefAndPortIndex& n) {
    return Hash::combine(std::move(h), n.node_def_, n.port_index_);
  }
 private:
  const NodeDef* node_def_;
  int port_index_;
};
template <typename FaninViewT, typename FanoutViewT, typename GraphViewT,
          bool IsConst>
class NodeViewInternal {
 private:
  using NodeDefT =
      typename std::conditional<IsConst, const NodeDef, NodeDef>::type;
 public:
  explicit NodeViewInternal(GraphViewT* graph_view, int node_index)
      : graph_view_(graph_view),
        node_index_(node_index),
        attrs_(AttrSlice(graph_view->graph()->node(node_index))) {}
  NodeViewInternal()
      : graph_view_(nullptr), node_index_(kMissingIndex), attrs_(AttrSlice()) {}
  virtual ~NodeViewInternal() {}
  NodeViewInternal(NodeViewInternal&&) = default;
  NodeViewInternal& operator=(NodeViewInternal&&) = default;
  bool operator==(const NodeViewInternal& other) const {
    return node_index_ == other.node_index_ && graph_view_ == other.graph_view_;
  }
  template <typename Hash>
  friend Hash AbslHashValue(Hash h, const NodeViewInternal& n) {
    return Hash::combine(std::move(h), n.node_index_);
  }
  virtual NodeDefT* node() const = 0;
  int node_index() const { return node_index_; }
  const string& GetName() const { return node()->name(); }
  const string& GetOp() const { return node()->op(); }
  const string& GetDevice() const { return node()->device(); }
  const std::vector<FanoutViewT>& GetRegularFanins() const {
    return regular_fanins_;
  }
  const FanoutViewT& GetRegularFanin(int i) const {
    int regular_fanins_size = regular_fanins_.size();
    if (i < 0 || i >= regular_fanins_size) {
      return GetMissingFanin();
    }
    return regular_fanins_[i];
  }
  const std::vector<FanoutViewT>& GetControllingFanins() const {
    return controlling_fanins_;
  }
  const std::vector<std::vector<FaninViewT>>& GetRegularFanouts() const {
    return regular_fanouts_by_port_;
  }
  const std::vector<FaninViewT>& GetRegularFanout(int i) const {
    int regular_fanouts_by_port_size = regular_fanouts_by_port_.size();
    if (i < 0 || i >= regular_fanouts_by_port_size) {
      return GetMissingFanout();
    }
    return regular_fanouts_by_port_[i];
  }
  const std::vector<FaninViewT>& GetControlledFanouts() const {
    return controlled_fanouts_;
  }
  int NumRegularFanins() const { return regular_fanins_.size(); }
  int NumControllingFanins() const { return controlling_fanins_.size(); }
  int NumRegularFanouts() const { return num_regular_fanouts_; }
  int NumControlledFanouts() const { return controlled_fanouts_.size(); }
  virtual bool HasFanin(const FanoutViewT& fanin) const = 0;
  virtual bool HasFanout(const FaninViewT& fanout) const = 0;
  const AttrValue* GetAttr(absl::string_view attr_name) const {
    return attrs_.Find(attr_name);
  }
  const AttrSlice& GetAttrs() const { return attrs_; }
  int NumAttrs() const { return attrs_.size(); }
  bool HasAttr(absl::string_view attr_name) const {
    return attrs_.Find(attr_name) != nullptr;
  }
 protected:
  virtual inline const FanoutViewT& GetMissingFanin() const = 0;
  virtual inline const std::vector<FaninViewT>& GetMissingFanout() const = 0;
  std::vector<FanoutViewT> regular_fanins_;
  std::vector<FanoutViewT> controlling_fanins_;
  std::vector<std::vector<FaninViewT>> regular_fanouts_by_port_;
  int num_regular_fanouts_ = 0;
  std::vector<FaninViewT> controlled_fanouts_;
  GraphViewT* graph_view_;
  int node_index_;
  AttrSlice attrs_;
};
template <typename NodeViewT, typename FaninViewT, typename FanoutViewT,
          bool IsConst>
class GraphViewInternal {
 private:
  using GraphDefT =
      typename std::conditional<IsConst, const GraphDef, GraphDef>::type;
 public:
  explicit GraphViewInternal(GraphDefT* graph) : graph_(graph) {}
  virtual ~GraphViewInternal() {}
  bool operator==(const GraphViewInternal& other) const {
    return graph_ == other.graph_;
  }
  GraphDefT* graph() const { return graph_; }
  const NodeViewT* GetNode(int node_index) const {
    int nodes_size = nodes_.size();
    if (node_index < 0 || node_index >= nodes_size) {
      return nullptr;
    }
    return &nodes_[node_index];
  }
  NodeViewT* GetNode(int node_index) {
    int nodes_size = nodes_.size();
    if (node_index < 0 || node_index >= nodes_size) {
      return nullptr;
    }
    return &nodes_[node_index];
  }
  const NodeViewT* GetNode(absl::string_view node_name) const {
    auto it = node_index_by_name_.find(node_name);
    if (it == node_index_by_name_.end()) {
      return nullptr;
    }
    return &nodes_[it->second];
  }
  NodeViewT* GetNode(absl::string_view node_name) {
    auto it = node_index_by_name_.find(node_name);
    if (it == node_index_by_name_.end()) {
      return nullptr;
    }
    return &nodes_[it->second];
  }
  const std::vector<NodeViewT>& GetNodes() const { return nodes_; }
  bool HasNode(absl::string_view node_name) const {
    return node_index_by_name_.contains(node_name);
  }
  int NumNodes() const { return nodes_.size(); }
 protected:
  void Reset() {
    std::vector<NodeViewT>().swap(nodes_);
    absl::flat_hash_map<absl::string_view, int>().swap(node_index_by_name_);
  }
  std::vector<NodeViewT> nodes_;
  absl::flat_hash_map<absl::string_view, int> node_index_by_name_;
  GraphDefT* graph_;
  const FanoutViewT missing_fanin_;
  const std::vector<FaninViewT> missing_fanout_;
};
inline SafeTensorId EmptyTensorId() {
  return SafeTensorId("", internal::kMissingSlot);
}
inline bool IsEmptyTensorId(const TensorId tensor_id) {
  return tensor_id.node().empty() &&
         tensor_id.index() == internal::kMissingSlot;
}
template <typename GraphViewT>
struct NodeViewDiff {
  explicit NodeViewDiff(GraphViewT* graph_view, int node_index)
      : graph_view(graph_view), node_index(node_index) {}
  GraphViewT* graph_view;
  int node_index;
  string name;
  bool update_name = false;
  string op;
  bool update_op = false;
  string device;
  bool update_device = false;
  std::vector<SafeTensorId> regular_inputs_to_add;
  int num_regular_inputs_to_add = 0;
  std::map<int, SafeTensorId> regular_inputs_to_update;
  std::vector<bool> regular_inputs_to_remove;
  int num_regular_inputs_to_remove = 0;
  absl::flat_hash_set<string> controlling_inputs_to_add;
  std::set<int> controlling_inputs_to_remove;
  absl::flat_hash_map<string, AttrValue> attrs_to_add;
  absl::flat_hash_set<string> attrs_to_remove;
  absl::optional<AttrValueMap> processed_attrs;
};
template <typename GraphViewT>
inline bool UpdateName(NodeViewDiff<GraphViewT>* diff, absl::string_view name) {
  if (diff->graph_view->GetNode(diff->node_index)->GetName() == name) {
    diff->name.clear();
    diff->update_name = false;
  } else {
    diff->name = string(name);
    diff->update_name = true;
  }
  return true;
}
template <typename GraphViewT>
inline bool UpdateOp(NodeViewDiff<GraphViewT>* diff, absl::string_view op) {
  if (diff->graph_view->GetNode(diff->node_index)->GetOp() == op) {
    diff->op.clear();
    diff->update_op = false;
  } else {
    diff->op = string(op);
    diff->update_op = true;
  }
  return true;
}
template <typename GraphViewT>
inline bool UpdateDevice(NodeViewDiff<GraphViewT>* diff,
                         absl::string_view device) {
  if (diff->graph_view->GetNode(diff->node_index)->GetDevice() == device) {
    diff->device.clear();
    diff->update_device = false;
  } else {
    diff->device = string(device);
    diff->update_device = true;
  }
  return true;
}
template <typename T, typename U>
inline bool AddOrUpdateAtIndex(std::vector<T>* v, int i, const U& value,
                               const T& default_value) {
  int v_size = v->size();
  if (i > v_size) {
    v->reserve(i + 1);
    v->resize(i, default_value);
    v->push_back({value});
  } else if (i == v_size) {
    v->push_back({value});
  } else {
    bool updated = (*v)[i] == default_value;
    (*v)[i] = {value};
    return updated;
  }
  return true;
}
template <typename GraphViewT>
inline bool CheckNodeNameExists(
    absl::string_view node_name,
    const absl::flat_hash_map<absl::string_view, int>& updated_node_names,
    const GraphViewT* graph_view) {
  auto it = updated_node_names.find(node_name);
  if (it != updated_node_names.end()) {
    return it->second == kNodeNamePresent;
  }
  return graph_view->HasNode(node_name);
}
template <typename GraphViewT>
inline bool AddOrUpdateRegularFanin(NodeViewDiff<GraphViewT>* diff, int index,
                                    const TensorId& fanin) {
  if (index < 0) {
    return false;
  }
  auto* node_view = diff->graph_view->GetNode(diff->node_index);
  const int num_regular_fanins = node_view->NumRegularFanins();
  if (index < num_regular_fanins) {  
    const int relative_removal_index = num_regular_fanins - index - 1;
    int diff_regular_inputs_to_remove_size =
        diff->regular_inputs_to_remove.size();
    if (relative_removal_index < diff_regular_inputs_to_remove_size &&
        diff->regular_inputs_to_remove[relative_removal_index]) {
      diff->regular_inputs_to_remove[relative_removal_index] = false;
      --diff->num_regular_inputs_to_remove;
    }
    const auto& existing_fanin = node_view->GetRegularFanin(index);
    if (existing_fanin.index() != fanin.index() ||
        existing_fanin.node_view()->GetName() != fanin.node()) {
      gtl::InsertOrUpdate(&diff->regular_inputs_to_update, index,
                          SafeTensorId(fanin));
    }
  } else {
    const int relative_add_index = index - num_regular_fanins;
    if (AddOrUpdateAtIndex(&diff->regular_inputs_to_add, relative_add_index,
                           fanin, EmptyTensorId())) {
      ++diff->num_regular_inputs_to_add;
    }
  }
  return true;
}
template <typename GraphViewT>
inline bool RemoveRegularFanin(NodeViewDiff<GraphViewT>* diff, int index) {
  if (index < 0) {
    return false;
  }
  auto* node_view = diff->graph_view->GetNode(diff->node_index);
  const int num_regular_fanins = node_view->NumRegularFanins();
  if (index < num_regular_fanins) {  
    diff->regular_inputs_to_update.erase(index);
    const int relative_removal_index = num_regular_fanins - index - 1;
    if (AddOrUpdateAtIndex(&diff->regular_inputs_to_remove,
                           relative_removal_index,
                           true, false)) {
      ++diff->num_regular_inputs_to_remove;
    }
  } else {
    const int relative_add_index = index - num_regular_fanins;
    int diff_regular_inputs_to_add_size = diff->regular_inputs_to_add.size();
    if (relative_add_index >= diff_regular_inputs_to_add_size ||
        IsEmptyTensorId(diff->regular_inputs_to_add[relative_add_index])) {
      return false;
    }
    diff->regular_inputs_to_add[relative_add_index] = EmptyTensorId();
    --diff->num_regular_inputs_to_add;
  }
  return true;
}
template <typename GraphViewT>
inline bool AddControllingFanin(NodeViewDiff<GraphViewT>* diff,
                                int control_index,
                                absl::string_view fanin_node_name) {
  if (control_index == kMissingIndex) {
    diff->controlling_inputs_to_add.emplace(fanin_node_name);
  } else {
    diff->controlling_inputs_to_remove.erase(control_index);
  }
  return true;
}
template <typename GraphViewT>
inline bool RemoveControllingFanin(NodeViewDiff<GraphViewT>* diff,
                                   int control_index,
                                   absl::string_view fanin_node_name) {
  if (control_index == kMissingIndex) {
    diff->controlling_inputs_to_add.erase(fanin_node_name);
  } else {
    diff->controlling_inputs_to_remove.emplace(control_index);
  }
  return true;
}
template <typename GraphViewT>
inline bool AddOrUpdateAttribute(NodeViewDiff<GraphViewT>* diff,
                                 absl::string_view attr_name,
                                 const AttrValue& attr_value) {
  diff->attrs_to_add.empty() ? 0 : diff->attrs_to_remove.erase(attr_name);
  gtl::InsertOrUpdate(&diff->attrs_to_add, string(attr_name), attr_value);
  return true;
}
template <typename GraphViewT>
inline bool RemoveAttribute(NodeViewDiff<GraphViewT>* diff,
                            absl::string_view attr_name) {
  const size_t num_erased =
      diff->attrs_to_add.empty() ? 0 : diff->attrs_to_add.erase(attr_name);
  auto* node_view = diff->graph_view->GetNode(diff->node_index);
  if (node_view->HasAttr(attr_name)) {
    diff->attrs_to_remove.emplace(attr_name);
    return true;
  }
  return num_erased > 0;
}
template <typename T>
inline void ResizeByTrimmingEndForValue(std::vector<T>* v, const T& value) {
  int curr_index = v->size();
  const int last_index = v->size() - 1;
  for (int i = last_index; i >= 0; --i) {
    if ((*v)[i] == value) {
      curr_index = i;
    } else {
      break;
    }
  }
  if (curr_index <= last_index) {
    v->resize(curr_index);
  }
}
template <typename GraphViewT>
inline bool IsEmpty(NodeViewDiff<GraphViewT>* diff) {
  ResizeByTrimmingEndForValue(&diff->regular_inputs_to_remove, false);
  ResizeByTrimmingEndForValue(&diff->regular_inputs_to_add, EmptyTensorId());
  return !diff->update_name && !diff->update_op && !diff->update_device &&
         diff->regular_inputs_to_add.empty() &&
         diff->regular_inputs_to_update.empty() &&
         diff->regular_inputs_to_remove.empty() &&
         diff->controlling_inputs_to_add.empty() &&
         diff->controlling_inputs_to_remove.empty() &&
         diff->attrs_to_add.empty() && diff->attrs_to_remove.empty();
}
template <typename GraphViewT>
inline void Reset(NodeViewDiff<GraphViewT>* diff) {
  diff->name.clear();
  diff->update_name = false;
  diff->op.clear();
  diff->update_op = false;
  diff->device.clear();
  diff->update_device = false;
  std::vector<SafeTensorId>().swap(diff->regular_inputs_to_add);
  diff->num_regular_inputs_to_add = false;
  std::map<int, SafeTensorId>().swap(diff->regular_inputs_to_update);
  std::vector<bool>().swap(diff->regular_inputs_to_remove);
  diff->num_regular_inputs_to_remove = 0;
  absl::flat_hash_set<string>().swap(diff->controlling_inputs_to_add);
  std::set<int>().swap(diff->controlling_inputs_to_remove);
  absl::flat_hash_map<string, AttrValue>().swap(diff->attrs_to_add);
  absl::flat_hash_set<string>().swap(diff->attrs_to_remove);
}
template <typename GraphViewT>
inline bool IsWellFormed(
    NodeViewDiff<GraphViewT>* diff,
    const absl::flat_hash_map<absl::string_view, int>& updated_node_names) {
  ResizeByTrimmingEndForValue(&diff->regular_inputs_to_remove, false);
  ResizeByTrimmingEndForValue(&diff->regular_inputs_to_add, EmptyTensorId());
  int diff_regular_inputs_to_add_size = diff->regular_inputs_to_add.size();
  if (diff_regular_inputs_to_add_size != diff->num_regular_inputs_to_add) {
    return false;
  } else if (diff->num_regular_inputs_to_add > 0 &&
             !diff->regular_inputs_to_remove.empty()) {
    return false;
  } else if (static_cast<int>(diff->regular_inputs_to_remove.size()) !=
             diff->num_regular_inputs_to_remove) {
    return false;
  }
  auto* node_view = diff->graph_view->GetNode(diff->node_index);
  const string& node_name =
      diff->update_name ? diff->name : node_view->GetName();
  auto invalid_node_name = [&](absl::string_view fanin_node_name) -> bool {
    return fanin_node_name == node_name ||
           !CheckNodeNameExists(fanin_node_name, updated_node_names,
                                diff->graph_view);
  };
  if (diff->update_name) {
    const int last_index =
        node_view->NumRegularFanins() - diff->num_regular_inputs_to_remove - 1;
    auto regular_to_update_it = diff->regular_inputs_to_update.begin();
    for (int i = 0; i <= last_index; ++i) {
      if (regular_to_update_it != diff->regular_inputs_to_update.end() &&
          regular_to_update_it->first < i) {
        ++regular_to_update_it;
      }
      if (regular_to_update_it != diff->regular_inputs_to_update.end() &&
          regular_to_update_it->first == i) {
        if (invalid_node_name(regular_to_update_it->second.node())) {
          return false;
        }
      } else {
        const string& regular_name =
            node_view->GetRegularFanin(i).node_view()->GetName();
        if (regular_name == node_name) {
          return false;
        }
      }
    }
    auto& controls = node_view->GetControllingFanins();
    const int num_controls = controls.size();
    auto control_to_remove_it = diff->controlling_inputs_to_remove.begin();
    for (int i = 0; i < num_controls; ++i) {
      if (control_to_remove_it != diff->controlling_inputs_to_remove.end() &&
          *control_to_remove_it < i) {
        ++control_to_remove_it;
      }
      if (control_to_remove_it != diff->controlling_inputs_to_remove.end() &&
          *control_to_remove_it == i) {
        continue;
      } else if (controls[i].node_view()->GetName() == node_name) {
        return false;
      }
    }
  } else {
    for (const auto& updated : diff->regular_inputs_to_update) {
      const string& fanin_name = updated.second.node();
      if (invalid_node_name(fanin_name)) {
        return false;
      }
    }
  }
  for (const auto& regular : diff->regular_inputs_to_add) {
    if (invalid_node_name(regular.node())) {
      return false;
    }
  }
  for (const auto& control : diff->controlling_inputs_to_add) {
    if (invalid_node_name(control)) {
      return false;
    }
  }
  return true;
}
template <typename GraphViewT>
struct NewNode {
  explicit NewNode(GraphViewT* graph_view, NodeDef&& node)
      : graph_view(graph_view), node(std::move(node)) {}
  GraphViewT* graph_view;
  NodeDef node;
  std::vector<SafeTensorId> regular_fanins;
  int num_regular_fanins = 0;
  absl::flat_hash_set<string> controlling_fanins;
};
template <typename GraphViewT>
inline void UpdateName(NewNode<GraphViewT>* new_node, absl::string_view name) {
  if (name.empty()) {
    new_node->node.clear_name();
  } else {
    new_node->node.set_name(string(name));
  }
}
template <typename GraphViewT>
inline void UpdateOp(NewNode<GraphViewT>* new_node, absl::string_view op) {
  if (op.empty()) {
    new_node->node.clear_op();
  } else {
    new_node->node.set_op(string(op));
  }
}
template <typename GraphViewT>
inline void UpdateDevice(NewNode<GraphViewT>* new_node,
                         absl::string_view device) {
  if (device.empty()) {
    new_node->node.clear_device();
  } else {
    new_node->node.set_device(string(device));
  }
}
template <typename GraphViewT>
inline void AddOrUpdateRegularFanin(NewNode<GraphViewT>* new_node, int index,
                                    const TensorId& fanin) {
  if (index < 0) {
    return;
  } else if (AddOrUpdateAtIndex(&new_node->regular_fanins, index, fanin,
                                EmptyTensorId())) {
    ++new_node->num_regular_fanins;
  }
}
template <typename GraphViewT>
inline void RemoveRegularFanin(NewNode<GraphViewT>* new_node, int index) {
  int new_node_regular_fanins_size = new_node->regular_fanins.size();
  if (index < 0 || index >= new_node_regular_fanins_size ||
      IsEmptyTensorId(new_node->regular_fanins[index])) {
    return;
  }
  new_node->regular_fanins[index] = EmptyTensorId();
  --new_node->num_regular_fanins;
}
template <typename GraphViewT>
inline void AddControllingFanin(NewNode<GraphViewT>* new_node,
                                absl::string_view fanin_node_name) {
  new_node->controlling_fanins.emplace(fanin_node_name);
}
template <typename GraphViewT>
inline void RemoveControllingFanin(NewNode<GraphViewT>* new_node,
                                   absl::string_view fanin_node_name) {
  new_node->controlling_fanins.erase(fanin_node_name);
}
template <typename GraphViewT>
inline void AddOrUpdateAttribute(NewNode<GraphViewT>* new_node,
                                 absl::string_view attr_name,
                                 const AttrValue& attr_value) {
  gtl::InsertOrUpdate(new_node->node.mutable_attr(), string(attr_name),
                      attr_value);
}
template <typename GraphViewT>
inline void RemoveAttribute(NewNode<GraphViewT>* new_node,
                            absl::string_view attr_name) {
  new_node->node.mutable_attr()->erase(string(attr_name));
}
template <typename GraphViewT>
inline bool IsWellFormed(
    NewNode<GraphViewT>* new_node,
    const absl::flat_hash_map<absl::string_view, int>& updated_node_names) {
  ResizeByTrimmingEndForValue(&new_node->regular_fanins, EmptyTensorId());
  int new_node_regular_fanins_size = new_node->regular_fanins.size();
  if (new_node_regular_fanins_size != new_node->num_regular_fanins) {
    return false;
  }
  const string& node_name = new_node->node.name();
  auto invalid_node_name = [new_node, updated_node_names,
                            node_name](absl::string_view fanin_node_name) {
    return fanin_node_name == node_name ||
           !CheckNodeNameExists(fanin_node_name, updated_node_names,
                                new_node->graph_view);
  };
  for (const auto& regular : new_node->regular_fanins) {
    if (invalid_node_name(regular.node())) {
      return false;
    }
  }
  for (const auto& control : new_node->controlling_fanins) {
    if (invalid_node_name(control)) {
      return false;
    }
  }
  return true;
}
}  
}  
}  
}  
#endif  