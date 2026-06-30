#include "tensorflow/core/grappler/optimizers/auto_mixed_precision.h"
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/virtual_placer.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/auto_mixed_precision_lists.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/util.h"
namespace tensorflow {
namespace grappler {
namespace {
bool ShouldSimulateGpu() {
  bool is_enabled = [] {
    bool ret = false;
    string var;
    TF_CHECK_OK(ReadStringFromEnvVar(
        "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_SIMULATE_GPU", "", &var));
    TF_CHECK_OK(
        ReadBoolFromEnvVar("TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_SIMULATE_GPU",
                           false, &ret));
    return ret;
  }();
  return is_enabled;
}
#if GOOGLE_CUDA
const std::pair<int, int> kMinGPUArch = {7, 0};
#else
const std::pair<int, int> kMinGPUArch = {0, 0};
#endif
const char kSuffix[] = "AutoMixedPrecision";
const char kCastToFp16[] = "CastToFp16";
const char kCastToBf16[] = "CastToBf16";
const char kCastToFp32[] = "CastToFp32";
#if GOOGLE_CUDA
std::pair<int, int> GetDeviceGPUArch(
    const DeviceProperties& device_properties) {
  if (device_properties.type() != "GPU") return {0, 0};
  string arch_str = device_properties.environment().at("architecture");
  std::vector<string> split_arch_str = str_util::Split(arch_str, '.');
  if (split_arch_str.empty()) {
    return {0, 0};
  }
  int major, minor;
  if (!strings::safe_strto32(split_arch_str[0], &major)) {
    return {0, 0};
  }
  if (split_arch_str.size() > 1) {
    if (strings::safe_strto32(split_arch_str[1], &minor)) {
      return {major, minor};
    } else {
      return {0, 0};
    }
  } else {
    return {major, 0};
  }
}
#endif
bool HasFastFP16Support(const DeviceProperties& props) {
#if GOOGLE_CUDA
  return GetDeviceGPUArch(props) >= kMinGPUArch;
#elif TENSORFLOW_USE_ROCM
  absl::flat_hash_set<std::string> FP16SupportedDevices = {{"gfx906"},
                                                           {"gfx908"}};
  std::string gcnArchName = props.environment().at("architecture");
  std::vector<std::string> gpu_arch = absl::StrSplit(gcnArchName, ":");
  return !gpu_arch.empty() && FP16SupportedDevices.contains(gpu_arch[0]);
#endif
  return ShouldSimulateGpu();
}
struct TypeAttrId {
  static constexpr int kSingleType = -1;
  explicit TypeAttrId(const string& _attr_name, int _type_index = kSingleType)
      : attr_name(_attr_name),
        type_index(_type_index),
        fixed_type(DT_INVALID) {}
  explicit TypeAttrId(DataType _fixed_type)
      : attr_name(), type_index(kSingleType), fixed_type(_fixed_type) {}
  bool operator==(const TypeAttrId& other) const {
    return attr_name == other.attr_name && type_index == other.type_index &&
           fixed_type == other.fixed_type;
  }
  bool operator<(const TypeAttrId& other) const {
    return std::make_tuple(attr_name, type_index, fixed_type) <
           std::make_tuple(other.attr_name, other.type_index, other.fixed_type);
  }
  template <typename H>
  friend H AbslHashValue(H h, const TypeAttrId& ta) {
    return H::combine(std::move(h), ta.attr_name, ta.type_index, ta.fixed_type);
  }
  string DebugString() const {
    if (!attr_name.empty()) {
      if (type_index == kSingleType) {
        return attr_name;
      } else {
        return strings::StrCat(attr_name, "[", type_index, "]");
      }
    } else {
      return tensorflow::DataTypeString(fixed_type);
    }
  }
  string attr_name;
  int type_index;
  DataType fixed_type;
};
DataType GetDataType(const NodeDef& node, const TypeAttrId& type_attr) {
  if (type_attr.attr_name.empty()) {
    return type_attr.fixed_type;
  }
  if (!node.attr().count(type_attr.attr_name)) {
    return DT_INVALID;
  }
  const AttrValue& attr_value = node.attr().at(type_attr.attr_name);
  if (type_attr.type_index == TypeAttrId::kSingleType) {
    return attr_value.type();
  } else {
    if (type_attr.type_index < 0 ||
        type_attr.type_index >= attr_value.list().type_size()) {
      return DT_INVALID;
    }
    return attr_value.list().type(type_attr.type_index);
  }
}
bool SetDataType(NodeDef* node, const TypeAttrId& type_attr, DataType type) {
  if (type_attr.attr_name.empty() || !node->attr().count(type_attr.attr_name)) {
    return false;
  }
  AttrValue& attr_value = node->mutable_attr()->at(type_attr.attr_name);
  if (type_attr.type_index == TypeAttrId::kSingleType) {
    attr_value.set_type(type);
  } else {
    if (type_attr.type_index < 0 ||
        type_attr.type_index >= attr_value.list().type_size()) {
      return false;
    }
    attr_value.mutable_list()->set_type(type_attr.type_index, type);
  }
  return true;
}
std::vector<std::pair<int, int>> ArgDefIndexes(const NodeDef& node, int arg_idx,
                                               const OpDef::ArgDef& arg_def) {
  std::vector<std::pair<int, int>> argdef_inds;
  if (!arg_def.type_list_attr().empty()) {
    int num_types = node.attr().at(arg_def.type_list_attr()).list().type_size();
    for (int type_idx = 0; type_idx < num_types; ++type_idx) {
      argdef_inds.push_back({arg_idx, type_idx});
    }
  } else {
    int num_repeat = 1;
    if (node.attr().count(arg_def.number_attr())) {
      num_repeat = node.attr().at(arg_def.number_attr()).i();
    }
    argdef_inds.insert(argdef_inds.end(), num_repeat, {arg_idx, -1});
  }
  return argdef_inds;
}
std::vector<std::pair<int, int>> InputPortArgDefIndexes(const NodeDef& node,
                                                        const OpDef& op_def) {
  std::vector<std::pair<int, int>> argdef_inds;
  argdef_inds.reserve(op_def.input_arg_size());  
  for (int arg_idx = 0; arg_idx < op_def.input_arg_size(); ++arg_idx) {
    const OpDef::ArgDef& arg_def = op_def.input_arg(arg_idx);
    auto arg_results = ArgDefIndexes(node, arg_idx, arg_def);
    argdef_inds.insert(argdef_inds.end(), arg_results.begin(),
                       arg_results.end());
  }
  return argdef_inds;
}
std::vector<std::pair<int, int>> OutputPortArgDefIndexes(const NodeDef& node,
                                                         const OpDef& op_def) {
  std::vector<std::pair<int, int>> argdef_inds;
  argdef_inds.reserve(op_def.output_arg_size());  
  for (int arg_idx = 0; arg_idx < op_def.output_arg_size(); ++arg_idx) {
    const OpDef::ArgDef& arg_def = op_def.output_arg(arg_idx);
    auto arg_results = ArgDefIndexes(node, arg_idx, arg_def);
    argdef_inds.insert(argdef_inds.end(), arg_results.begin(),
                       arg_results.end());
  }
  return argdef_inds;
}
TypeAttrId GetTypeAttrId(const OpDef::ArgDef& arg_def, int arg_type_index) {
  if (!arg_def.type_list_attr().empty()) {
    return TypeAttrId(arg_def.type_list_attr(), arg_type_index);
  } else if (!arg_def.type_attr().empty()) {
    return TypeAttrId(arg_def.type_attr());
  } else {
    return TypeAttrId(arg_def.type());
  }
}
std::vector<int> NonControlInputs(const NodeDef& node) {
  std::vector<int> pos;
  for (int i = 0; i < node.input_size(); i++) {
    if (!IsControlInput(node.input(i))) {
      pos.push_back(i);
    }
  }
  return pos;
}
class NodeTypeAttrMap {
 public:
  NodeTypeAttrMap() {}
  explicit NodeTypeAttrMap(const GraphDef& graph) { TF_CHECK_OK(Init(graph)); }
  Status Init(const GraphDef& graph) {
    if (graph_ != nullptr) {
      return errors::InvalidArgument("NodeTypeAttrMap is already initialized.");
    }
    graph_ = &graph;
    function_library_.reset(
        new FunctionLibraryDefinition(OpRegistry::Global(), graph.library()));
    for (const NodeDef& node : graph.node()) {
      TF_RETURN_IF_ERROR(AddNode(node));
    }
    return absl::OkStatus();
  }
  bool is_initialized() const { return graph_ != nullptr; }
  absl::flat_hash_set<TypeAttrId> GetTypeAttrs(const NodeDef& node) const {
    DCHECK(is_initialized()) << "NodeTypeAttrMap is not initialized";
    absl::flat_hash_set<TypeAttrId> type_attrs;
    const auto iter = type2io_.find(&node);
    CHECK(iter != type2io_.end());  
    for (const auto& key_value : iter->second) {
      type_attrs.insert(key_value.first);
    }
    return type_attrs;
  }
  const absl::flat_hash_set<int>& GetInputPorts(
      const NodeDef& node, const TypeAttrId& type_attr) const {
    DCHECK(is_initialized()) << "NodeTypeAttrMap is not initialized";
    return type2io_.at(&node).at(type_attr).first;
  }
  const absl::flat_hash_set<int>& GetOutputPorts(
      const NodeDef& node, const TypeAttrId& type_attr) const {
    DCHECK(is_initialized()) << "NodeTypeAttrMap is not initialized";
    return type2io_.at(&node).at(type_attr).second;
  }
  TypeAttrId GetInputTypeAttr(const NodeDef& node, int port) const {
    DCHECK(is_initialized()) << "NodeTypeAttrMap is not initialized";
    const auto iter = io2type_.find(&node);
    DCHECK(iter != io2type_.end())
        << "Node " << node.name() << " doesn't exist in a graph";
    auto type_vec = io2type_.at(&node).first;
    CHECK_GE(port, 0);                
    CHECK_LT(port, type_vec.size());  
    return type_vec[port];
  }
  TypeAttrId GetOutputTypeAttr(const NodeDef& node, int port) const {
    DCHECK(is_initialized()) << "NodeTypeAttrMap is not initialized";
    auto type_vec = io2type_.at(&node).second;
    CHECK_GE(port, 0);                
    CHECK_LT(port, type_vec.size());  
    return type_vec[port];
  }
 private:
  Status AddNode(const NodeDef& node) {
    const OpDef* op_def_ptr = nullptr;
    TF_RETURN_IF_ERROR(function_library_->LookUpOpDef(node.op(), &op_def_ptr));
    const OpDef& op_def = *op_def_ptr;
    auto& type2io_entry = type2io_[&node];
    auto& io2type_entry = io2type_[&node];
    auto input_arg_inds = InputPortArgDefIndexes(node, op_def);
    if (NonControlInputs(node).size() != input_arg_inds.size()) {
      return errors::InvalidArgument(
          "Expected ", node.op(), " node ", node.name(), " to have ",
          input_arg_inds.size(), " non-control input(s), but got ",
          node.input_size());
    }
    io2type_entry.first.reserve(input_arg_inds.size());
    for (int i = 0; i < static_cast<int>(input_arg_inds.size()); ++i) {
      const auto& arg_inds = input_arg_inds[i];
      const OpDef::ArgDef& arg_def = op_def.input_arg(arg_inds.first);
      TypeAttrId type_attr = GetTypeAttrId(arg_def, arg_inds.second);
      if (!type_attr.attr_name.empty() &&
          !node.attr().count(type_attr.attr_name)) {
        return errors::InvalidArgument("Type attribute ", type_attr.attr_name,
                                       " is not present in node ", node.name());
      }
      type2io_entry[type_attr].first.insert(i);
      io2type_entry.first.push_back(type_attr);
    }
    auto output_arg_inds = OutputPortArgDefIndexes(node, op_def);
    io2type_entry.second.reserve(output_arg_inds.size());
    for (int i = 0; i < static_cast<int>(output_arg_inds.size()); ++i) {
      const auto& arg_inds = output_arg_inds[i];
      const OpDef::ArgDef& arg_def = op_def.output_arg(arg_inds.first);
      TypeAttrId type_attr = GetTypeAttrId(arg_def, arg_inds.second);
      if (!type_attr.attr_name.empty() &&
          !node.attr().count(type_attr.attr_name)) {
        return errors::InvalidArgument("Type attribute ", type_attr.attr_name,
                                       " is not present in node ", node.name());
      }
      type2io_entry[type_attr].second.insert(i);
      io2type_entry.second.push_back(type_attr);
    }
    for (const auto& attr : node.attr()) {
      const string& attr_name = attr.first;
      if (!attr_name.empty() && attr_name[0] == '_') continue;
      const AttrValue& attr_value = attr.second;
      const OpDef::AttrDef* attr_def = FindAttr(attr_name, op_def);
      if (!attr_def) {
        return errors::InvalidArgument("AttrDef not found for attribute ",
                                       attr_name, " of node ", node.name());
      }
      if (attr_def->type() == "type") {
        type2io_entry[TypeAttrId(attr_name)];
      } else if (attr_def->type() == "list(type)") {
        for (int i = 0; i < attr_value.list().type_size(); ++i) {
          type2io_entry[TypeAttrId(attr_name, i)];
        }
      }
    }
    return absl::OkStatus();
  }
  const GraphDef* graph_ = nullptr;  
  std::unique_ptr<FunctionLibraryDefinition> function_library_;
  typedef absl::flat_hash_set<int> IntSet;
  typedef absl::flat_hash_map<TypeAttrId, std::pair<IntSet, IntSet>> Type2IOMap;
  absl::flat_hash_map<const NodeDef*, Type2IOMap> type2io_;
  typedef std::vector<TypeAttrId> TypeAttrIdVec;
  absl::flat_hash_map<const NodeDef*, std::pair<TypeAttrIdVec, TypeAttrIdVec>>
      io2type_;
};
struct NodeTypeId {
  NodeTypeId(const NodeDef* _node, const TypeAttrId& _type_attr)
      : node(_node), type_attr(_type_attr) {}
  const NodeDef* node;
  TypeAttrId type_attr;
  bool operator==(const NodeTypeId& other) const {
    return node == other.node && type_attr == other.type_attr;
  }
  template <typename H>
  friend H AbslHashValue(H h, const NodeTypeId& nt) {
    return H::combine(std::move(h), nt.node, nt.type_attr);
  }
};
struct NodeTypeIdEdge {
  NodeTypeIdEdge(const NodeTypeId& _src, const NodeTypeId& _dst)
      : src(_src), dst(_dst) {}
  NodeTypeId src;
  NodeTypeId dst;
};
class GraphTypeTopologyView {
 public:
  GraphTypeTopologyView() = default;
  explicit GraphTypeTopologyView(bool skip_invalid_edges)
      : skip_invalid_edges_(skip_invalid_edges) {}
  Status InitializeFromGraph(const GraphDef& graph,
                             const NodeTypeAttrMap& node_type_map);
  Status AddEphemeralEdges(absl::Span<const NodeTypeIdEdge> ephemeral_edges);
  bool is_initialized() const { return graph_ != nullptr; }
  int num_nodes() const { return num_nodes_; }
  const GraphDef* graph() const { return graph_; }
  bool HasNode(absl::string_view node_name, const TypeAttrId& type_attr) const;
  const NodeTypeId* GetNode(absl::string_view node_name,
                            const TypeAttrId& type_attr) const;
  const NodeTypeId* GetNode(int node_idx) const;
  const absl::optional<int> GetNodeIndex(absl::string_view node_name,
                                         const TypeAttrId& type_attr) const;
  const absl::optional<int> GetNodeIndex(const NodeTypeId& node) const;
  const absl::InlinedVector<int, 4>& GetFanin(int node_idx) const;
  const absl::InlinedVector<int, 2>& GetFanout(int node_idx) const;
 private:
  struct NodeTypeKey : public std::pair<absl::string_view, TypeAttrId> {
    typedef std::pair<absl::string_view, TypeAttrId> Base;
    using Base::pair;
    template <typename H>
    friend H AbslHashValue(H h, const NodeTypeKey& nt) {
      return H::combine(std::move(h), nt.first, nt.second);
    }
  };
  bool skip_invalid_edges_ = false;
  const GraphDef* graph_ = nullptr;  
  int num_nodes_ = 0;
  std::vector<NodeTypeId> node_type_attrs_;
  absl::flat_hash_map<absl::string_view, int> node_name_to_index_;
  absl::flat_hash_map<NodeTypeKey, int> node_type_name_to_index_;
  std::vector<absl::InlinedVector<int, 4>> fanins_;
  std::vector<absl::InlinedVector<int, 2>> fanouts_;
  absl::InlinedVector<int, 4> empty_fanin_;
  absl::InlinedVector<int, 2> empty_fanout_;
};
template <typename T>
inline void SortAndRemoveDuplicates(T* v) {
  std::sort(v->begin(), v->end());
  v->erase(std::unique(v->begin(), v->end()), v->end());
}
Status GraphTypeTopologyView::InitializeFromGraph(
    const GraphDef& graph, const NodeTypeAttrMap& node_type_map) {
  if (graph_ != nullptr) {
    return errors::InvalidArgument(
        "GraphTypeTopologyView is already initialized.");
  }
  graph_ = &graph;
  int num_nodedefs = graph.node_size();
  node_name_to_index_.rehash(num_nodedefs);
  node_type_attrs_.reserve(num_nodedefs);         
  node_type_name_to_index_.rehash(num_nodedefs);  
  for (int node_idx = 0; node_idx < num_nodedefs; ++node_idx) {
    const NodeDef& node = graph.node(node_idx);
    node_name_to_index_.emplace(node.name(), node_idx);
    for (const TypeAttrId& type_attr : node_type_map.GetTypeAttrs(node)) {
      int node_type_idx = node_type_attrs_.size();
      node_type_name_to_index_.emplace(NodeTypeKey(node.name(), type_attr),
                                       node_type_idx);
      node_type_attrs_.emplace_back(&node, type_attr);
    }
  }
  num_nodes_ = node_type_attrs_.size();
  fanins_.resize(num_nodes_);
  fanouts_.resize(num_nodes_);
  for (int node_type_idx = 0; node_type_idx < num_nodes_; ++node_type_idx) {
    const NodeTypeId& node_type = node_type_attrs_.at(node_type_idx);
    auto input_ports =
        node_type_map.GetInputPorts(*node_type.node, node_type.type_attr);
    fanins_[node_type_idx].reserve(input_ports.size());
    for (int port : input_ports) {
      const string& input = node_type.node->input(port);
      TensorId tensor = ParseTensorName(input);
      const auto it = node_name_to_index_.find(tensor.node());
      const bool valid_input = it != node_name_to_index_.end();
      if (!valid_input) {
        const string error_message = absl::StrCat(
            "Non-existent input ", input, " in node ", node_type.node->name());
        if (skip_invalid_edges_) {
          VLOG(3) << "Skip error: " << error_message;
        } else {
          return errors::InvalidArgument(error_message);
        }
      }
      if (valid_input) {
        const int input_idx = it->second;
        const NodeDef& input_node = graph_->node(input_idx);
        TypeAttrId input_type_attr =
            node_type_map.GetOutputTypeAttr(input_node, tensor.index());
        const auto it2 = node_type_name_to_index_.find(
            NodeTypeKey(input_node.name(), input_type_attr));
        if (it2 == node_type_name_to_index_.end()) {
          if (!skip_invalid_edges_) {
            return errors::InvalidArgument("Did not find type attr ",
                                           input_type_attr.DebugString(),
                                           " in node ", input_node.name());
          }
          continue;
        }
        int input_node_type_idx = it2->second;
        fanins_[node_type_idx].push_back(input_node_type_idx);
        fanouts_[input_node_type_idx].push_back(node_type_idx);
      }
    }
    SortAndRemoveDuplicates(&fanins_[node_type_idx]);
  }
  for (int node_type_idx = 0; node_type_idx < num_nodes_; ++node_type_idx) {
    SortAndRemoveDuplicates(&fanouts_[node_type_idx]);
  }
  return absl::OkStatus();
}
Status GraphTypeTopologyView::AddEphemeralEdges(
    absl::Span<const NodeTypeIdEdge> ephemeral_edges) {
  for (const NodeTypeIdEdge& edge : ephemeral_edges) {
    const auto src = node_name_to_index_.find(edge.src.node->name());
    const bool valid_src = src != node_name_to_index_.end();
    if (!valid_src) {
      const string error_message =
          absl::StrCat("Non-existent src node: ", edge.src.node->name());
      if (skip_invalid_edges_) {
        VLOG(0) << "Skip error: " << error_message;
      } else {
        return errors::InvalidArgument(error_message);
      }
    }
    const auto dst = node_name_to_index_.find(edge.dst.node->name());
    const bool valid_dst = dst != node_name_to_index_.end();
    if (!valid_dst) {
      const string error_message =
          absl::StrCat("Non-existent dst node: ", edge.dst.node->name());
      if (skip_invalid_edges_) {
        VLOG(0) << "Skip error: " << error_message;
      } else {
        return errors::InvalidArgument(error_message);
      }
    }
    if (valid_dst && valid_src) {
      int src_node_type_idx = node_type_name_to_index_.at(
          NodeTypeKey(edge.src.node->name(), edge.src.type_attr));
      int dst_node_type_idx = node_type_name_to_index_.at(
          NodeTypeKey(edge.dst.node->name(), edge.dst.type_attr));
      fanins_[dst_node_type_idx].push_back(src_node_type_idx);
      fanouts_[src_node_type_idx].push_back(dst_node_type_idx);
    }
  }
  for (int node_type_idx = 0; node_type_idx < num_nodes_; ++node_type_idx) {
    SortAndRemoveDuplicates(&fanins_[node_type_idx]);
    SortAndRemoveDuplicates(&fanouts_[node_type_idx]);
  }
  return absl::OkStatus();
}
bool GraphTypeTopologyView::HasNode(absl::string_view node_name,
                                    const TypeAttrId& type_attr) const {
  DCHECK(is_initialized()) << "GraphTypeTopologyView is not initialized";
  NodeTypeKey key(node_name, type_attr);
  const auto it = node_type_name_to_index_.find(key);
  return it != node_type_name_to_index_.end();
}
const NodeTypeId* GraphTypeTopologyView::GetNode(
    absl::string_view node_name, const TypeAttrId& type_attr) const {
  DCHECK(is_initialized()) << "GraphTypeTopologyView is not initialized";
  NodeTypeKey key(node_name, type_attr);
  const auto it = node_type_name_to_index_.find(key);
  return it == node_type_name_to_index_.end()
             ? nullptr
             : &node_type_attrs_.at(it->second);
}
const NodeTypeId* GraphTypeTopologyView::GetNode(int node_idx) const {
  DCHECK(is_initialized()) << "GraphTypeTopologyView is not initialized";
  DCHECK(node_idx >= 0 && node_idx < num_nodes_) << "node_idx is out of range";
  return &node_type_attrs_.at(node_idx);
}
const absl::optional<int> GraphTypeTopologyView::GetNodeIndex(
    absl::string_view node_name, const TypeAttrId& type_attr) const {
  DCHECK(is_initialized()) << "GraphTypeTopologyView is not initialized";
  NodeTypeKey key(node_name, type_attr);
  const auto it = node_type_name_to_index_.find(key);
  DCHECK(it != node_type_name_to_index_.end())
      << "Node doesn't exist in a graph";
  return it == node_type_name_to_index_.end() ? absl::nullopt
                                              : absl::make_optional(it->second);
}
const absl::optional<int> GraphTypeTopologyView::GetNodeIndex(
    const NodeTypeId& node) const {
  return GetNodeIndex(node.node->name(), node.type_attr);
}
const absl::InlinedVector<int, 4>& GraphTypeTopologyView::GetFanin(
    int node_idx) const {
  DCHECK(is_initialized()) << "GraphTypeTopologyView is not initialized";
  const bool is_valid_node_idx = node_idx >= 0 && node_idx < num_nodes_;
  DCHECK(is_valid_node_idx) << "node_idx is out of range";
  return is_valid_node_idx ? fanins_[node_idx] : empty_fanin_;
}
const absl::InlinedVector<int, 2>& GraphTypeTopologyView::GetFanout(
    int node_idx) const {
  DCHECK(is_initialized()) << "GraphTypeTopologyView is not initialized";
  const bool is_valid_node_idx = node_idx >= 0 && node_idx < num_nodes_;
  DCHECK(is_valid_node_idx) << "node_idx is out of range";
  return is_valid_node_idx ? fanouts_[node_idx] : empty_fanout_;
}
enum class TypeTraversalDirection {
  kFollowInputs,
  kFollowOutputs,
  kFollowInputsAndOutputs,
};
struct DfsTypeCallbacks {
  DfsTypeCallbacks() = default;
  DfsTypeCallbacks(std::function<void(int)> pre, std::function<void(int)> post,
                   std::function<void(int, int)> back_edge)
      : pre_order(std::move(pre)),
        post_order(std::move(post)),
        on_back_edge(std::move(back_edge)) {}
  static DfsTypeCallbacks PreOrder(std::function<void(int)> pre) {
    return DfsTypeCallbacks(std::move(pre), nullptr, nullptr);
  }
  static DfsTypeCallbacks PostOrder(std::function<void(int)> post) {
    return DfsTypeCallbacks(nullptr, std::move(post), nullptr);
  }
  std::function<void(int)> pre_order;
  std::function<void(int)> post_order;
  std::function<void(int, int)> on_back_edge;
};
struct DfsTypePredicates {
  DfsTypePredicates() = default;
  DfsTypePredicates(std::function<bool(int)> enter,
                    std::function<bool(int)> advance)
      : enter(std::move(enter)), advance(std::move(advance)) {}
  static DfsTypePredicates Enter(std::function<bool(int)> enter) {
    return DfsTypePredicates(std::move(enter), nullptr);
  }
  static DfsTypePredicates Advance(std::function<bool(int)> advance) {
    return DfsTypePredicates(nullptr, std::move(advance));
  }
  std::function<bool(int)> enter;
  std::function<bool(int)> advance;
};
struct DfsStackElem {
  DfsStackElem(int node, bool children_visited, int src)
      : node(node), children_visited(children_visited), src(src) {}
  explicit DfsStackElem(int node) : DfsStackElem(node, false, -1) {}
  int node;
  bool children_visited;
  int src;
};
enum class NodeState { kNotVisited, kVisiting, kDone };
void DfsTypeTraversal(const GraphTypeTopologyView& graph_type_view,
                      const absl::Span<const NodeTypeId* const> from,
                      const TypeTraversalDirection direction,
                      const DfsTypePredicates& predicates,
                      const DfsTypeCallbacks& callbacks) {
  std::vector<DfsStackElem> stack;
  stack.reserve(from.size());
  for (const NodeTypeId* node : from) {
    const absl::optional<int> node_idx = graph_type_view.GetNodeIndex(*node);
    DCHECK(node_idx.has_value())
        << "Illegal start node: " << node->node->name();
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
    if (predicates.enter && !predicates.enter(w.node)) {
      state = NodeState::kDone;
      continue;
    }
    if (w.children_visited) {
      state = NodeState::kDone;
      if (callbacks.post_order) {
        callbacks.post_order(w.node);
      }
      continue;
    }
    if (state == NodeState::kVisiting) {
      if (callbacks.on_back_edge) {
        callbacks.on_back_edge(w.src, w.node);
      }
      continue;
    }
    state = NodeState::kVisiting;
    if (callbacks.pre_order) {
      callbacks.pre_order(w.node);
    }
    stack.emplace_back(w.node, true, w.src);
    if (predicates.advance && !predicates.advance(w.node)) {
      continue;
    }
    if (direction == TypeTraversalDirection::kFollowInputs ||
        direction == TypeTraversalDirection::kFollowInputsAndOutputs) {
      for (const int fanin : graph_type_view.GetFanin(w.node)) {
        stack.emplace_back(fanin, false, w.node);
      }
    }
    if (direction == TypeTraversalDirection::kFollowOutputs ||
        direction == TypeTraversalDirection::kFollowInputsAndOutputs) {
      for (const int fanout : graph_type_view.GetFanout(w.node)) {
        stack.emplace_back(fanout, false, w.node);
      }
    }
  }
}
DataTypeSet AllowedDataTypes(const OpDef::AttrDef& attr_def) {
  const auto& allowed_types = attr_def.allowed_values().list().type();
  if (allowed_types.empty()) {
    return AllTypes();
  }
  uint32 dtype_mask = 0;
  for (int dtype : allowed_types) {
    dtype_mask |= 1u << dtype;
  }
  return DataTypeSet(dtype_mask);
}
DataTypeSet AllowedDataTypes(const OpDef& op_def, const TypeAttrId& t_attr_id) {
  if (t_attr_id.attr_name.empty()) {
    return ToSet(t_attr_id.fixed_type);
  }
  const OpDef::AttrDef* attr_def = FindAttr(t_attr_id.attr_name, op_def);
  CHECK(attr_def);  
  return AllowedDataTypes(*attr_def);
}
Status ValidateLists(const gtl::FlatSet<string>& allow_list,
                     const gtl::FlatSet<string>& deny_list,
                     const gtl::FlatSet<string>& infer_list,
                     const gtl::FlatSet<string>& clear_list) {
  std::vector<gtl::FlatSet<string>> lists{allow_list, deny_list, infer_list,
                                          clear_list};
  std::multiset<string> counts;
  for (const auto& list : lists) {
    counts.insert(list.begin(), list.end());
  }
  bool duplicates = false;
  for (const auto& s : counts) {
    if (counts.count(s) > 1) {
      duplicates = true;
      LOG(ERROR) << "Op present in multiple lists: " << s;
    }
  }
  if (duplicates) {
    return errors::InvalidArgument("Op lists have conflicting entries");
  } else {
    return absl::OkStatus();
  }
}
bool HasInputOrOutputRefs(const NodeDef& node) {
  const OpDef* op_def;
  Status status = OpRegistry::Global()->LookUpOpDef(node.op(), &op_def);
  if (!status.ok()) {
    return true;
  }
  for (const auto& input : op_def->input_arg()) {
    if (input.is_ref()) {
      return true;
    }
  }
  for (const auto& output : op_def->output_arg()) {
    if (output.is_ref()) {
      return true;
    }
  }
  return false;
}
bool CanForceFP16(const NodeDef& node) {
  return node.op() != "Const" && node.op() != "SoftmaxCrossEntropyWithLogits" &&
         !IsStateful(node) && !HasInputOrOutputRefs(node);
}
int GetCudaVersion(
    const std::unordered_map<string, DeviceProperties>& devices) {
  for (const auto& device : devices) {
    const DeviceProperties& device_properties = device.second;
    if (device_properties.type() == "GPU") {
      const auto& device_env = device_properties.environment();
      auto it = device_env.find("cuda");
      if (it != device_env.end()) {
        string cuda_version_str = it->second;
        return std::stoi(cuda_version_str);
      }
    }
  }
  return 0;
}
int GetCudnnVersion(
    const std::unordered_map<string, DeviceProperties>& devices) {
  for (const auto& device : devices) {
    const DeviceProperties& device_properties = device.second;
    if (device_properties.type() == "GPU") {
      const auto& device_env = device_properties.environment();
      auto it = device_env.find("cudnn");
      if (it != device_env.end()) {
        string cudnn_version_str = it->second;
        return std::stoi(cudnn_version_str);
      }
    }
  }
  return 0;
}
std::unordered_map<string, DeviceProperties> GetDevices(Cluster* cluster) {
  if (!ShouldSimulateGpu()) {
    return cluster->GetDevices();
  }
  bool has_gpu = false;
  for (const auto& device : cluster->GetDevices()) {
    const DeviceProperties& device_properties = device.second;
    if (device_properties.type() == "GPU") {
      has_gpu = true;
      break;
    }
  }
  if (has_gpu) {
    return cluster->GetDevices();
  }
  std::unordered_map<string, DeviceProperties> devices(cluster->GetDevices());
  DeviceProperties gpu_device_properies;
  gpu_device_properies.set_type("GPU");
#if GOOGLE_CUDA
  gpu_device_properies.set_vendor("NVIDIA");
  gpu_device_properies.mutable_environment()->insert({"architecture", "8.0"});
  gpu_device_properies.mutable_environment()->insert({"cuda", "11050"});
  gpu_device_properies.mutable_environment()->insert({"cudnn", "8302"});
#elif TENSORFLOW_USE_ROCM
  gpu_device_properies.set_vendor("Advanced Micro Devices, Inc");
  gpu_device_properies.mutable_environment()->insert(
      {"architecture", "gfx908"});
#endif
  devices.emplace(std::make_pair("/job:localhost/replica:0/task:0/device:GPU:0",
                                 gpu_device_properies));
  return devices;
}
class AutoMixedPrecisionImpl {
 public:
  enum class CastType { FP16, FP32, AUTO };
  AutoMixedPrecisionImpl(Cluster* cluster,
                         const std::unordered_set<string>& nodes_to_preserve,
                         GraphDef* graph, string id,
                         AutoMixedPrecisionMode mode)
      : devices_(GetDevices(cluster)),
        virtual_placer_(devices_),
        nodes_to_preserve_(nodes_to_preserve),
        graph_(graph),
        function_library_(OpRegistry::Global(), graph->library()),
        id_(id),
        graph_view_(graph),
        cuda_version_(GetCudaVersion(devices_)),
        cudnn_version_(GetCudnnVersion(devices_)),
        num_nonvar_casts_to_f16_(0),
        mode_(mode),
        target_dtype_((mode_ == AutoMixedPrecisionMode::CUDA ||
                       mode_ == AutoMixedPrecisionMode::CPU ||
                       mode_ == AutoMixedPrecisionMode::FP16_CPU)
                          ? DT_HALF
                          : DT_BFLOAT16) {}
  Status Optimize();
 private:
  typedef absl::flat_hash_set<NodeTypeId> NodeTypeIdSet;
  std::unique_ptr<AutoMixedPrecisionLists> get_mixed_precision_lists() const {
    switch (mode_) {
      case AutoMixedPrecisionMode::CUDA:
        return std::make_unique<AutoMixedPrecisionListsFp16>(
            cuda_version_, cudnn_version_, AutoMixedPrecisionMode::CUDA);
      case AutoMixedPrecisionMode::BF16:
        return std::make_unique<AutoMixedPrecisionListsMkl>();
      case AutoMixedPrecisionMode::CPU:
        return std::make_unique<AutoMixedPrecisionListsFp16>(
            10000,  
            8000,  
            AutoMixedPrecisionMode::CPU);
      case AutoMixedPrecisionMode::FP16_CPU:
        return std::make_unique<AutoMixedPrecisionListsFp16>(
            0, 0, AutoMixedPrecisionMode::FP16_CPU);
    }
  }
  Status PrintDebugLogs(bool preop, size_t timestamp);
  void LogSkippedNode(const NodeDef& node, const string& device_type) const;
  bool MustPreserve(const NodeDef& node) const;
  bool IsOnDevice(const NodeDef& node, const string& device_type) const;
  bool IsOnSuitableGPUArch(const NodeDef& node) const;
  bool ShouldProcess(const NodeDef& node) const;
  bool NodeHasF16KernelForTypeAttr(const NodeDef& node, TypeAttrId taid) const;
  bool NodeImplicitlyReadsNonResourceVariable(const NodeDef& node) const;
  void ConvertBatchNormOpsToV2();
  bool SupportsF16(const NodeTypeId& node_type) const;
  bool SupportsF16DataType(const NodeTypeId& node_type) const;
  bool IsQuantized(const NodeTypeId& node_type) const;
  const NodeTypeId* GetTensorListFloat32NodeTypeId(const NodeDef& node) const;
  bool IsSourceOrSinkOp(const string& op) const;
  void FindFloat32TensorListOpClustersAndDenylistUnsafe(
      std::vector<absl::flat_hash_set<const NodeDef*>>* clusters,
      absl::flat_hash_set<int>* deny_set) const;
  void FindTensorListImplicitFloat32Edges(
      const absl::flat_hash_set<const NodeDef*>& tensor_list_nodes,
      std::vector<NodeTypeIdEdge>* implicit_fp32_edges) const;
  void AddAllowlistOps(absl::flat_hash_set<int>* allow_set) const;
  void RemoveAllowsetWithFp32(absl::flat_hash_set<int>* allow_set) const;
  void PropagateDenyFwdThroughClearAndInfer(
      absl::flat_hash_set<int>* deny_set) const;
  void ForceColorMatchBetweenTensorListOps(
      const absl::flat_hash_set<const NodeDef*>& tensor_list_nodes,
      absl::flat_hash_set<int>* allow_set,
      absl::flat_hash_set<int>* deny_set) const;
  void AddClearAndInferToAllowIfBetweenAllow(
      const absl::flat_hash_set<int>& deny_set,
      absl::flat_hash_set<int>* allow_set) const;
  void AddInferToAllowIfFollowAllow(const absl::flat_hash_set<int>& deny_set,
                                    absl::flat_hash_set<int>* allow_set) const;
  void PropagateAllowThroughClear(const absl::flat_hash_set<int>& deny_set,
                                  absl::flat_hash_set<int>* allow_set) const;
  Status ForceColorMatchOnRecurrentEdges(
      absl::flat_hash_set<int>* allow_set) const;
  void MakeCastsAllowIfAllOutputsAllow(
      absl::flat_hash_set<int>* allow_set) const;
  NodeDef BuildCastNode(const MutableGraphView::OutputPort& src, bool to_f16,
                        const string& device) const;
  absl::StatusOr<NodeDef*> InsertCastNodeAtFanout(
      const absl::flat_hash_set<int>& allow_set, const bool src_is_allow,
      const CastType& cast_type, MutableGraphView::OutputPort& src);
  absl::StatusOr<DataType> GetCastToType(const NodeDef* node) const;
  void CollectOutputPorts(
      const TypeAttrId& type_attr, NodeDef* node,
      std::vector<MutableGraphView::OutputPort>& output_ports) const;
  Status ChangeTypeAttrsAndAddCasts(const absl::flat_hash_set<int>& allow_set);
  std::unordered_map<string, DeviceProperties> devices_;
  VirtualPlacer virtual_placer_;
  std::unordered_set<string> nodes_to_preserve_;
  GraphDef* graph_;
  FunctionLibraryDefinition function_library_;
  string id_;
  MutableGraphView graph_view_;
  int cuda_version_;
  int cudnn_version_;
  int num_nonvar_casts_to_f16_;
  NodeTypeAttrMap node_type_map_;
  GraphTypeTopologyView graph_type_view_;
  bool force_all_fp16_;
  bool treat_infer_as_deny_;
  AutoMixedPrecisionMode mode_;
  gtl::FlatSet<string> f16_allowlist_;
  gtl::FlatSet<string> f16_denylist_;
  gtl::FlatSet<string> f16_inferlist_;
  gtl::FlatSet<string> f16_clearlist_;
  absl::flat_hash_set<const NodeDef*> should_process_nodes_;
  DataType target_dtype_;  
};
NodeDef AutoMixedPrecisionImpl::BuildCastNode(
    const MutableGraphView::OutputPort& src, bool to_f16,
    const string& device) const {
  DataType src_type = to_f16 ? DT_FLOAT : target_dtype_;
  DataType dst_type = to_f16 ? target_dtype_ : DT_FLOAT;
  const char* cast_string = !to_f16                    ? kCastToFp32
                            : target_dtype_ == DT_HALF ? kCastToFp16
                                                       : kCastToBf16;
  int id = 0;
  std::string name;
  do {
    name = absl::StrCat(src.node->name(), "-", src.port_id, "-", cast_string,
                        "-", id, "-", kSuffix);
    ++id;
  } while (graph_view_.GetNode(name));
  NodeDef node;
  node.set_name(name);
  node.set_op("Cast");
  node.set_device(device);
  node.add_input(strings::StrCat(src.node->name(), ":", src.port_id));
  (*node.mutable_attr())["SrcT"].set_type(src_type);
  (*node.mutable_attr())["DstT"].set_type(dst_type);
  (*node.mutable_attr())["Truncate"].set_b(false);
  return node;
}
bool AutoMixedPrecisionImpl::NodeHasF16KernelForTypeAttr(
    const NodeDef& node, TypeAttrId taid) const {
  NodeDef node_copy(node);
  if (node.device().empty()) {
    string device_name = virtual_placer_.get_canonical_device_name(node);
    node_copy.set_device(device_name);
  }
  if (!SetDataType(&node_copy, taid, target_dtype_)) {
    return false;
  }
  return IsKernelRegisteredForNode(node_copy).ok();
}
Status AutoMixedPrecisionImpl::PrintDebugLogs(bool preop, size_t timestamp) {
  string prepend_path;
  TF_RETURN_IF_ERROR(ReadStringFromEnvVar(
      "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_LOG_PATH", "", &prepend_path));
  if (prepend_path.empty()) return absl::OkStatus();
  string suffix =
      strings::StrCat("_", preop ? "preop" : kSuffix, "_", id_, "_", timestamp);
  string fname =
      io::JoinPath(prepend_path, strings::StrCat("graphdef", suffix, ".pb"));
  std::fstream f;
  f.open(fname.c_str(), std::fstream::out | std::fstream::binary);
  f << graph_->SerializeAsString();
  f.close();
  LOG(INFO) << "Saved " << (preop ? "pre-optimization" : "post-optimization")
            << " graph as binary to " << fname;
  fname = io::JoinPath(prepend_path,
                       strings::StrCat("graphdef", suffix, ".pb.txt"));
  f.open(fname.c_str(), std::fstream::out);
  f << graph_->DebugString();
  f.close();
  LOG(INFO) << "Saved " << (preop ? "pre-optimization" : "post-optimization")
            << " graph as text to " << fname;
  if (!preop) {
    fname = io::JoinPath(prepend_path,
                         strings::StrCat("paintbuckets", suffix, ".txt"));
    f.open(fname.c_str(), std::fstream::out);
    std::unique_ptr<AutoMixedPrecisionLists> mp_lists =
        get_mixed_precision_lists();
    f << "AllowList:\n";
    for (const auto& x : mp_lists->AllowList()) {
      f << x << "\n";
    }
    f << "\nDenyList:\n";
    for (const auto& x : mp_lists->DenyList()) {
      f << x << "\n";
    }
    f << "\nInferList:\n";
    for (const auto& x : mp_lists->InferList()) {
      f << x << "\n";
    }
    f << "\nClearList:\n";
    for (const auto& x : mp_lists->ClearList()) {
      f << x << "\n";
    }
    f.close();
    LOG(INFO) << "Saved paint bucket info to " << fname;
  }
  return absl::OkStatus();
}
void AutoMixedPrecisionImpl::LogSkippedNode(const NodeDef& node,
                                            const string& device_type) const {
  VLOG(2) << "Skipping " << node.op() << " node " << node.name()
          << " because it "
          << (MustPreserve(node)
                  ? "must be preserved"
                  : absl::StrFormat(
                        "is not on the %s, or the %s arch is not suitable",
                        device_type, device_type));
}
bool AutoMixedPrecisionImpl::MustPreserve(const NodeDef& node) const {
  return nodes_to_preserve_.count(node.name());
}
bool AutoMixedPrecisionImpl::IsOnDevice(const NodeDef& node,
                                        const string& device_type) const {
  string device_name;
  if (node.device().empty()) {
    device_name = virtual_placer_.get_canonical_device_name(node);
  } else {
    device_name = node.device();
  }
  string device;
  string not_used;
  if (DeviceNameUtils::SplitDeviceName(device_name, &not_used, &device) &&
      absl::StrContains(absl::AsciiStrToLower(device),
                        absl::AsciiStrToLower(device_type))) {
    return true;
  }
  return false;
}
bool AutoMixedPrecisionImpl::IsOnSuitableGPUArch(const NodeDef& node) const {
  return HasFastFP16Support(virtual_placer_.get_device(node));
}
bool AutoMixedPrecisionImpl::ShouldProcess(const NodeDef& node) const {
  return should_process_nodes_.count(&node);
}
bool IsFloat32(const NodeTypeId& node_type) {
  return GetDataType(*node_type.node, node_type.type_attr) ==
         DataType::DT_FLOAT;
}
bool IsTensorListOp(const string& op) {
  return absl::StrContains(op, "TensorList");
}
bool IsTensorListReaderOp(const string& op) {
  static const gtl::FlatSet<string> tensor_list_reader_ops = {
      "TensorListConcat",  "TensorListConcatV2", "TensorListGather",
      "TensorListGetItem", "TensorListPopBack",  "TensorListStack"};
  return tensor_list_reader_ops.count(op);
}
bool IsTensorListWriterOp(const string& op) {
  static const gtl::FlatSet<string> tensor_list_writer_ops = {
      "TensorListFromTensor",    "TensorListPushBack",
      "TensorListPushBackBatch", "TensorListScatter",
      "TensorListScatterV2",     "TensorListScatterIntoExistingList",
      "TensorListSetItem",       "TensorListSplit"};
  return tensor_list_writer_ops.count(op);
}
bool AutoMixedPrecisionImpl::SupportsF16(const NodeTypeId& node_type) const {
  const OpDef* op_def;
  Status status =
      OpRegistry::Global()->LookUpOpDef(node_type.node->op(), &op_def);
  if (!status.ok()) return false;
  return AllowedDataTypes(*op_def, node_type.type_attr)
             .Contains(target_dtype_) &&
         NodeHasF16KernelForTypeAttr(*node_type.node, node_type.type_attr);
}
bool AutoMixedPrecisionImpl::SupportsF16DataType(
    const NodeTypeId& node_type) const {
  const OpDef* op_def;
  Status status =
      OpRegistry::Global()->LookUpOpDef(node_type.node->op(), &op_def);
  if (!status.ok()) return false;
  return AllowedDataTypes(*op_def, node_type.type_attr).Contains(target_dtype_);
}
bool AutoMixedPrecisionImpl::IsQuantized(const NodeTypeId& node_type) const {
  for (const TypeAttrId& type_attr :
       node_type_map_.GetTypeAttrs(*node_type.node)) {
    if (DataTypeIsQuantized(GetDataType(*node_type.node, type_attr))) {
      return true;
    }
  }
  return false;
}
void AutoMixedPrecisionImpl::ConvertBatchNormOpsToV2() {
  for (int node_idx = 0; node_idx < graph_->node_size(); ++node_idx) {
    NodeDef* node = graph_->mutable_node(node_idx);
    if (!ShouldProcess(*node)) continue;
    bool changed = false;
    if (node->op() == "FusedBatchNorm") {
      VLOG(2) << "Changing op of " << node->op() << " node " << node->name()
              << " to FusedBatchNormV2";
      node->set_op("FusedBatchNormV2");
      changed = true;
    } else if (node->op() == "FusedBatchNormGrad") {
      VLOG(2) << "Changing op of " << node->op() << " node " << node->name()
              << " to FusedBatchNormGradV2";
      node->set_op("FusedBatchNormGradV2");
      changed = true;
    }
    if (changed) {
      (*node->mutable_attr())["U"].set_type(DT_FLOAT);
    }
  }
}
bool ShouldIgnorePerformance() {
  static bool is_enabled = [] {
    bool ret = false;
    TF_CHECK_OK(ReadBoolFromEnvVar(
        "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE",
        false, &ret));
    return ret;
  }();
  return is_enabled;
}
Status AutoMixedPrecisionImpl::Optimize() {
  string optimization_level;
  TF_RETURN_IF_ERROR(ReadStringFromEnvVar(
      "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_LEVEL", "", &optimization_level));
  optimization_level = absl::AsciiStrToUpper(optimization_level);
  force_all_fp16_ = optimization_level == "UNSAFE_FORCE_ALL";
  if (force_all_fp16_ && (mode_ == AutoMixedPrecisionMode::BF16 ||
                          mode_ == AutoMixedPrecisionMode::FP16_CPU)) {
    return errors::InvalidArgument(
        "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_LEVEL cannot be set to "
        "UNSAFE_FORCE_ALL when oneDNN is used");
  }
  treat_infer_as_deny_ = optimization_level == "TREAT_INFER_AS_DENY";
  VLOG(2) << "Optimization Level: " << optimization_level;
  std::unique_ptr<AutoMixedPrecisionLists> mp_lists =
      get_mixed_precision_lists();
  f16_allowlist_ = mp_lists->AllowList();
  f16_denylist_ = mp_lists->DenyList();
  if (treat_infer_as_deny_) {
    for (const auto& op : mp_lists->InferList()) {
      f16_denylist_.insert(op);
    }
  } else {
    f16_inferlist_ = mp_lists->InferList();
  }
  f16_clearlist_ = mp_lists->ClearList();
  TF_RETURN_IF_ERROR(ValidateLists(f16_allowlist_, f16_denylist_,
                                   f16_inferlist_, f16_clearlist_));
  size_t timestamp = Env::Default()->NowMicros() / 1000;
  TF_RETURN_IF_ERROR(PrintDebugLogs( true, timestamp));
  VLOG(2) << "Identifying nodes that should be processed";
  for (const NodeDef& node : graph_->node()) {
    bool should_process;
    string device_type;
    switch (mode_) {
      case AutoMixedPrecisionMode::CUDA:
        device_type = DEVICE_GPU;
        should_process =
            !MustPreserve(node) && IsOnDevice(node, device_type) &&
            (ShouldIgnorePerformance() || IsOnSuitableGPUArch(node));
        break;
      case AutoMixedPrecisionMode::BF16:
      case AutoMixedPrecisionMode::CPU:
      case AutoMixedPrecisionMode::FP16_CPU:
        device_type = DEVICE_CPU;
        should_process = !MustPreserve(node) && IsOnDevice(node, device_type);
        break;
    }
    if (should_process) {
      should_process_nodes_.insert(&node);
    } else {
      LogSkippedNode(node, device_type);
    }
  }
  VLOG(2) << "Converting FusedBatchNorm* ops to V2";
  ConvertBatchNormOpsToV2();
  VLOG(2) << "Building node type map for graph";
  TF_RETURN_IF_ERROR(node_type_map_.Init(*graph_));
  VLOG(2) << "Constructing graph type attribute topology view";
  TF_RETURN_IF_ERROR(
      graph_type_view_.InitializeFromGraph(*graph_, node_type_map_));
  absl::flat_hash_set<int> deny_set;
  std::vector<absl::flat_hash_set<const NodeDef*>> tensor_list_clusters;
  FindFloat32TensorListOpClustersAndDenylistUnsafe(&tensor_list_clusters,
                                                   &deny_set);
  std::vector<NodeTypeIdEdge> ephemeral_edges;
  for (const auto& cluster : tensor_list_clusters) {
    VLOG(1) << "Found safe Tensor List cluster of size " << cluster.size();
    for (const NodeDef* node : cluster) {
      VLOG(2) << "  Cluster member: " << node->op() << " node " << node->name();
    }
    FindTensorListImplicitFloat32Edges(cluster, &ephemeral_edges);
  }
  TF_RETURN_IF_ERROR(graph_type_view_.AddEphemeralEdges(ephemeral_edges));
  absl::flat_hash_set<int> allow_set;
  VLOG(2) << "Beginning pass 1 to add allowlist ops";
  AddAllowlistOps(&allow_set);
  VLOG(2) << "Finished pass 1";
  if (allow_set.empty()) {
    LOG(INFO) << "No allowlist ops found, nothing to do";
    return absl::OkStatus();
  }
  VLOG(2) << "Beginning pass 2 to propagate deny forwards from denylist ops "
             "through clear/inferlist ops";
  PropagateDenyFwdThroughClearAndInfer(&deny_set);
  VLOG(2) << "Finished pass 2";
  VLOG(2) << "Forcing color match between data structure ops";
  for (const auto& cluster : tensor_list_clusters) {
    ForceColorMatchBetweenTensorListOps(cluster, &allow_set, &deny_set);
  }
  VLOG(2) << "Beginning pass 3 to set clear and infer nodes to allow if they "
             "are between allow ops";
  AddClearAndInferToAllowIfBetweenAllow(deny_set, &allow_set);
  VLOG(2) << "Finished pass 3";
  VLOG(2) << "Beginning pass 4 to add infer list ops to allow if they "
             "directly follow allow nodes";
  AddInferToAllowIfFollowAllow(deny_set, &allow_set);
  VLOG(2) << "Finished pass 4";
  VLOG(2) << "Beginning pass 5 to propagate allow from allow nodes through "
             "clearlist ops";
  PropagateAllowThroughClear(deny_set, &allow_set);
  VLOG(2) << "Finished pass 5";
  VLOG(2) << "Beginning pass 6 to remove some nodes which could not be changed "
             "to F16"
             "from allow set";
  RemoveAllowsetWithFp32(&allow_set);
  VLOG(2) << "Finished pass 6";
  VLOG(2) << "Forcing color match between data structure ops";
  for (const auto& cluster : tensor_list_clusters) {
    ForceColorMatchBetweenTensorListOps(cluster, &allow_set, &deny_set);
  }
  VLOG(2) << "Forcing color match on loop edges";
  TF_RETURN_IF_ERROR(ForceColorMatchOnRecurrentEdges(&allow_set));
  VLOG(2) << "Finding existing casts that can be made allow";
  MakeCastsAllowIfAllOutputsAllow(&allow_set);
  VLOG(2) << "Beginning final pass to change type attributes and insert Cast "
             "ops at paint boundaries";
  TF_RETURN_IF_ERROR(ChangeTypeAttrsAndAddCasts(allow_set));
  VLOG(2) << "Finished final pass";
  TF_RETURN_IF_ERROR(PrintDebugLogs( false, timestamp));
  return absl::OkStatus();
}
const NodeTypeId* AutoMixedPrecisionImpl::GetTensorListFloat32NodeTypeId(
    const NodeDef& node) const {
  if (!IsTensorListOp(node.op())) return nullptr;
  for (const TypeAttrId& type_attr : node_type_map_.GetTypeAttrs(node)) {
    const NodeTypeId* node_type =
        graph_type_view_.GetNode(node.name(), type_attr);
    if (node_type && node_type->type_attr.fixed_type == DT_INVALID &&
        node_type->type_attr.type_index == TypeAttrId::kSingleType &&
        IsFloat32(*node_type)) {
      return node_type;
    }
  }
  return nullptr;
}
bool AutoMixedPrecisionImpl::IsSourceOrSinkOp(const string& op) const {
  const gtl::FlatSet<string> source_and_sink_ops = {
      "_Arg",
      "_Retval",
      "OptionalFromValue",
      "OptionalGetValue",
      "PartitionedCall",
      "Placeholder",
      "StatefulPartitionedCall",
  };
  return source_and_sink_ops.count(op) || function_library_.Find(op);
}
void AutoMixedPrecisionImpl::FindFloat32TensorListOpClustersAndDenylistUnsafe(
    std::vector<absl::flat_hash_set<const NodeDef*>>* tensor_list_clusters,
    absl::flat_hash_set<int>* deny_set) const {
  absl::flat_hash_set<const NodeDef*> tensor_list_prop_set;
  for (int root_idx = 0; root_idx < graph_type_view_.num_nodes(); ++root_idx) {
    const NodeTypeId& root = *graph_type_view_.GetNode(root_idx);
    if (!ShouldProcess(*root.node) ||
        root.type_attr.fixed_type != DataType::DT_VARIANT ||
        !GetTensorListFloat32NodeTypeId(*root.node) ||
        tensor_list_prop_set.count(root.node)) {
      continue;
    }
    const NodeTypeId* root_fp32 = GetTensorListFloat32NodeTypeId(*root.node);
    const absl::optional<int> maybe_root_fp32_idx =
        graph_type_view_.GetNodeIndex(*root_fp32);
    DCHECK(maybe_root_fp32_idx.has_value())
        << "Type attribute " << root_fp32->type_attr.DebugString()
        << " of node " << root.node->name() << " not found in graph view";
    int root_fp32_idx = maybe_root_fp32_idx.value();
    absl::flat_hash_set<const NodeDef*> cluster({root.node});
    DfsTypeTraversal(graph_type_view_, {&root},
                     TypeTraversalDirection::kFollowInputsAndOutputs,
                     DfsTypePredicates::Enter([&](int idx) -> bool {
                       const NodeTypeId& item = *graph_type_view_.GetNode(idx);
                       return !tensor_list_prop_set.count(item.node);
                     }),
                     DfsTypeCallbacks::PreOrder([&](int idx) {
                       const NodeTypeId& item = *graph_type_view_.GetNode(idx);
                       const NodeDef* node = item.node;
                       if (GetTensorListFloat32NodeTypeId(*node)) {
                         cluster.insert(node);
                         if (!ShouldProcess(*node)) {
                           deny_set->insert(root_fp32_idx);
                         }
                       } else if (IsSourceOrSinkOp(node->op())) {
                         deny_set->insert(root_fp32_idx);
                       }
                     }));
    tensor_list_clusters->push_back(cluster);
  }
}
void AutoMixedPrecisionImpl::FindTensorListImplicitFloat32Edges(
    const absl::flat_hash_set<const NodeDef*>& tensor_list_nodes,
    std::vector<NodeTypeIdEdge>* implicit_fp32_edges) const {
  for (const NodeDef* root_node : tensor_list_nodes) {
    if (!IsTensorListReaderOp(root_node->op())) continue;
    NodeTypeId root(root_node, TypeAttrId(DataType::DT_VARIANT));
    const NodeTypeId* root_fp32 = GetTensorListFloat32NodeTypeId(*root.node);
    CHECK(root_fp32) << "No float32 type attribute found for "  
                     << root.node->op() << " node " << root.node->name();
    DfsTypeTraversal(
        graph_type_view_, {&root}, TypeTraversalDirection::kFollowInputs,
        DfsTypePredicates::Enter([&](int idx) -> bool {
          const NodeTypeId& item = *graph_type_view_.GetNode(idx);
          return ShouldProcess(*item.node);
        }),
        DfsTypeCallbacks::PreOrder([&](int idx) {
          const NodeTypeId& item = *graph_type_view_.GetNode(idx);
          if (IsTensorListWriterOp(item.node->op())) {
            const NodeTypeId* item_fp32 =
                GetTensorListFloat32NodeTypeId(*item.node);
            CHECK(item_fp32)  
                << "No float32 type attribute found for " << item.node->op()
                << " node " << item.node->name();
            VLOG(2) << "Adding ephemeral float32 edge from "
                    << item_fp32->node->op() << " node "
                    << item_fp32->node->name() << " to "
                    << root_fp32->node->op() << " node "
                    << root_fp32->node->name();
            implicit_fp32_edges->emplace_back(*item_fp32, *root_fp32);
          }
        }));
  }
}
void AutoMixedPrecisionImpl::AddAllowlistOps(
    absl::flat_hash_set<int>* allow_set) const {
  for (int root_idx = 0; root_idx < graph_type_view_.num_nodes(); ++root_idx) {
    const NodeTypeId& root = *graph_type_view_.GetNode(root_idx);
    if (!ShouldProcess(*root.node)) continue;
    bool force_allow = force_all_fp16_ && CanForceFP16(*root.node);
    if (f16_allowlist_.count(root.node->op()) || force_allow) {
      bool inserted = allow_set->insert(root_idx).second;
      if (VLOG_IS_ON(2) && inserted) {
        VLOG(2) << "Painting type " << root.type_attr.DebugString()
                << " of node " << root.node->name() << " ALLOW because its op "
                << root.node->op() << " is on the allowlist";
      }
    }
  }
}
void AutoMixedPrecisionImpl::PropagateDenyFwdThroughClearAndInfer(
    absl::flat_hash_set<int>* deny_set) const {
  if (force_all_fp16_) return;
  absl::flat_hash_set<int> upstream_of_deny_or_infer_set;
  for (int root_idx = 0; root_idx < graph_type_view_.num_nodes(); ++root_idx) {
    const NodeTypeId& root = *graph_type_view_.GetNode(root_idx);
    if (!(f16_denylist_.count(root.node->op()) ||
          f16_inferlist_.count(root.node->op()))) {
      continue;
    }
    DfsTypeTraversal(graph_type_view_, {&root},
                     TypeTraversalDirection::kFollowInputs,
                     DfsTypePredicates::Enter([&](int idx) -> bool {
                       const NodeTypeId& item = *graph_type_view_.GetNode(idx);
                       return idx == root_idx ||
                              (!upstream_of_deny_or_infer_set.count(idx) &&
                               f16_clearlist_.count(item.node->op()));
                     }),
                     DfsTypeCallbacks::PreOrder([&](int idx) {
                       upstream_of_deny_or_infer_set.insert(idx);
                     }));
  }
  for (int root_idx = 0; root_idx < graph_type_view_.num_nodes(); ++root_idx) {
    const NodeTypeId& root = *graph_type_view_.GetNode(root_idx);
    if (deny_set->count(root_idx) || !f16_denylist_.count(root.node->op())) {
      continue;
    }
    DfsTypeTraversal(
        graph_type_view_, {&root}, TypeTraversalDirection::kFollowOutputs,
        DfsTypePredicates::Enter([&](int idx) -> bool {
          return idx == root_idx || (!deny_set->count(idx) &&
                                     upstream_of_deny_or_infer_set.count(idx));
        }),
        DfsTypeCallbacks::PreOrder([&](int idx) {
          bool inserted = deny_set->insert(idx).second;
          if (VLOG_IS_ON(2) && inserted) {
            const NodeTypeId& item = *graph_type_view_.GetNode(idx);
            VLOG(2) << "Painting type " << item.type_attr.DebugString()
                    << " of " << item.node->op() << " node "
                    << item.node->name() << " DENY";
          }
        }));
  }
}
void AutoMixedPrecisionImpl::AddClearAndInferToAllowIfBetweenAllow(
    const absl::flat_hash_set<int>& deny_set,
    absl::flat_hash_set<int>* allow_set) const {
  absl::flat_hash_set<int> downstream_of_allow_set;
  for (int root_idx = 0; root_idx < graph_type_view_.num_nodes(); ++root_idx) {
    const NodeTypeId& root = *graph_type_view_.GetNode(root_idx);
    if (!ShouldProcess(*root.node) || !f16_allowlist_.count(root.node->op())) {
      continue;
    }
    DfsTypeTraversal(
        graph_type_view_, {&root}, TypeTraversalDirection::kFollowOutputs,
        DfsTypePredicates::Enter([&](int idx) -> bool {
          const NodeTypeId& item = *graph_type_view_.GetNode(idx);
          return idx == root_idx ||
                 (!downstream_of_allow_set.count(idx) &&
                  !f16_allowlist_.count(item.node->op()) &&
                  !deny_set.count(idx) && ShouldProcess(*item.node) &&
                  IsFloat32(item) && SupportsF16(item) &&
                  (f16_clearlist_.count(item.node->op()) ||
                   f16_inferlist_.count(item.node->op())));
        }),
        DfsTypeCallbacks::PreOrder(
            [&](int idx) { downstream_of_allow_set.insert(idx); }));
  }
  absl::flat_hash_set<int> upstream_of_allow_set;
  for (int root_idx = 0; root_idx < graph_type_view_.num_nodes(); ++root_idx) {
    const NodeTypeId& root = *graph_type_view_.GetNode(root_idx);
    if (!ShouldProcess(*root.node) || upstream_of_allow_set.count(root_idx) ||
        !f16_allowlist_.count(root.node->op())) {
      continue;
    }
    DfsTypeTraversal(
        graph_type_view_, {&root}, TypeTraversalDirection::kFollowInputs,
        DfsTypePredicates::Enter([&](int idx) -> bool {
          return idx == root_idx || (!upstream_of_allow_set.count(idx) &&
                                     downstream_of_allow_set.count(idx));
        }),
        DfsTypeCallbacks::PreOrder([&](int idx) {
          upstream_of_allow_set.insert(idx);
          bool inserted = allow_set->insert(idx).second;
          if (VLOG_IS_ON(2) && inserted) {
            const NodeTypeId& item = *graph_type_view_.GetNode(idx);
            VLOG(2) << "Painting type " << item.type_attr.DebugString()
                    << " of " << item.node->op() << " node "
                    << item.node->name() << " ALLOW";
          }
        }));
  }
}
void AutoMixedPrecisionImpl::PropagateAllowThroughClear(
    const absl::flat_hash_set<int>& deny_set,
    absl::flat_hash_set<int>* allow_set) const {
  absl::flat_hash_set<int> clear_prop_set;
  for (int root_idx = 0; root_idx < graph_type_view_.num_nodes(); ++root_idx) {
    const NodeTypeId& root = *graph_type_view_.GetNode(root_idx);
    if (!ShouldProcess(*root.node) || clear_prop_set.count(root_idx) ||
        !allow_set->count(root_idx)) {
      continue;
    }
    DfsTypeTraversal(
        graph_type_view_, {&root},
        TypeTraversalDirection::kFollowInputsAndOutputs,
        DfsTypePredicates::Enter([&](int idx) -> bool {
          const NodeTypeId& item = *graph_type_view_.GetNode(idx);
          return idx == root_idx ||
                 (!allow_set->count(idx) && !deny_set.count(idx) &&
                  ShouldProcess(*item.node) && IsFloat32(item) &&
                  SupportsF16(item) &&
                  (f16_clearlist_.count(item.node->op())) &&
                  !NodeImplicitlyReadsNonResourceVariable(*item.node));
        }),
        DfsTypeCallbacks::PreOrder([&](int idx) {
          clear_prop_set.insert(idx);
          bool inserted = allow_set->insert(idx).second;
          if (VLOG_IS_ON(2) && inserted) {
            const NodeTypeId& item = *graph_type_view_.GetNode(idx);
            VLOG(2) << "Painting type " << item.type_attr.DebugString()
                    << " of " << item.node->op() << " node "
                    << item.node->name() << " ALLOW";
          }
        }));
  }
}
void AutoMixedPrecisionImpl::AddInferToAllowIfFollowAllow(
    const absl::flat_hash_set<int>& deny_set,
    absl::flat_hash_set<int>* allow_set) const {
  if (mode_ != AutoMixedPrecisionMode::BF16) {
    return;
  }
  for (int item_idx = 0; item_idx < graph_type_view_.num_nodes(); ++item_idx) {
    const NodeTypeId& item = *graph_type_view_.GetNode(item_idx);
    if (!ShouldProcess(*item.node) || deny_set.count(item_idx) ||
        allow_set->count(item_idx) || !f16_inferlist_.count(item.node->op()) ||
        !IsFloat32(item) || !SupportsF16DataType(item)) {
      continue;
    }
    bool has_allow_fanin = false;
    for (const int fanin : graph_type_view_.GetFanin(item_idx)) {
      if (deny_set.count(fanin)) {
        has_allow_fanin = false;
        break;
      }
      if (allow_set->count(fanin)) {
        has_allow_fanin = true;
      }
    }
    if (has_allow_fanin) {
      bool inserted = allow_set->insert(item_idx).second;
      if (VLOG_IS_ON(2) && inserted) {
        VLOG(2) << "Painting type " << item.type_attr.DebugString() << " of "
                << item.node->op() << " node " << item.node->name() << " ALLOW";
      }
    }
  }
}
void AutoMixedPrecisionImpl::RemoveAllowsetWithFp32(
    absl::flat_hash_set<int>* allow_set) const {
  for (int root_idx = 0; root_idx < graph_type_view_.num_nodes(); ++root_idx) {
    const NodeTypeId& root = *graph_type_view_.GetNode(root_idx);
    if (f16_allowlist_.count(root.node->op()) && allow_set->count(root_idx) &&
        (!SupportsF16DataType(root) || IsQuantized(root))) {
      auto erased = allow_set->erase(root_idx);
      if (VLOG_IS_ON(2) && erased) {
        VLOG(2) << "UnPainting type " << root.type_attr.DebugString()
                << " of node " << root.node->name() << " ALLOW because its op "
                << root.node->op() << " is not support F16 DataType";
      }
    }
  }
}
Status AutoMixedPrecisionImpl::ForceColorMatchOnRecurrentEdges(
    absl::flat_hash_set<int>* allow_set) const {
  for (const NodeDef& node : graph_->node()) {
    if (node.op() == "NextIteration") {
      GraphView::OutputPort output_port(&node, 0);
      const auto& fanout = graph_view_.GetFanout(output_port);
      std::vector<int> merge_idxs;
      merge_idxs.reserve(fanout.size());
      bool any_merge_is_not_allow = false;
      for (const auto& output : fanout) {
        const NodeDef& merge_node = *output.node;
        if (merge_node.op() != "Merge") {
          return errors::FailedPrecondition(
              "Expected Merge node after NextIteration, got ", merge_node.op());
        }
        const absl::optional<int> maybe_merge_idx =
            graph_type_view_.GetNodeIndex(merge_node.name(), TypeAttrId("T"));
        if (!maybe_merge_idx.has_value()) {
          return errors::Internal("Type attribute T of Merge node ",
                                  merge_node.name(),
                                  " not found in graph view");
        }
        int merge_idx = maybe_merge_idx.value();
        merge_idxs.push_back(merge_idx);
        any_merge_is_not_allow =
            any_merge_is_not_allow || !allow_set->count(merge_idx);
      }
      const absl::optional<int> maybe_nextiter_idx =
          graph_type_view_.GetNodeIndex(node.name(), TypeAttrId("T"));
      if (!maybe_nextiter_idx.has_value()) {
        return errors::Internal("Type attribute T of NextIteration node ",
                                node.name(), " not found in graph view");
      }
      int nextiter_idx = maybe_nextiter_idx.value();
      if (any_merge_is_not_allow) {
        for (int merge_idx : merge_idxs) {
          if (allow_set->erase(merge_idx)) {
            VLOG(2) << "Painting type T of Merge node "
                    << graph_type_view_.GetNode(merge_idx)->node->name()
                    << " DENY to match the color of its sibling Merge nodes "
                       "with common NextIteration node "
                    << node.name();
          }
        }
        if (allow_set->erase(nextiter_idx)) {
          VLOG(2) << "Painting type T of NextIteration node " << node.name()
                  << " DENY to match the color of its output Merge node(s)";
        }
      } else {
        if (allow_set->insert(nextiter_idx).second) {
          VLOG(2) << "Painting type T of NextIteration node " << node.name()
                  << " ALLOW to match the color of its output Merge node(s)";
        }
      }
    }
  }
  return absl::OkStatus();
}
void AutoMixedPrecisionImpl::ForceColorMatchBetweenTensorListOps(
    const absl::flat_hash_set<const NodeDef*>& tensor_list_nodes,
    absl::flat_hash_set<int>* allow_set,
    absl::flat_hash_set<int>* deny_set) const {
  bool any_deny = false;
  bool any_allow = false;
  std::vector<int> node_type_idxs;
  node_type_idxs.reserve(tensor_list_nodes.size());
  for (const NodeDef* node : tensor_list_nodes) {
    const NodeTypeId& node_type = *GetTensorListFloat32NodeTypeId(*node);
    const absl::optional<int> maybe_node_type_idx =
        graph_type_view_.GetNodeIndex(node_type);
    DCHECK(maybe_node_type_idx.has_value())
        << "Type attribute " << node_type.type_attr.DebugString() << " of node "
        << node->name() << " not found in graph view";
    node_type_idxs.push_back(maybe_node_type_idx.value());
  }
  for (int node_type_idx : node_type_idxs) {
    if (deny_set->count(node_type_idx)) {
      any_deny = true;
      break;
    } else if (allow_set->count(node_type_idx)) {
      any_allow = true;
    }
  }
  if (!any_deny && !any_allow) return;
  for (int node_type_idx : node_type_idxs) {
    const NodeTypeId& node_type = *graph_type_view_.GetNode(node_type_idx);
    VLOG(2) << "Painting type " << node_type.type_attr.DebugString() << " of "
            << node_type.node->op() << " node " << node_type.node->name() << " "
            << (any_deny ? "DENY" : "ALLOW")
            << " because at least one of its siblings is "
            << (any_deny ? "DENY" : "ALLOW");
    if (any_deny) {
      allow_set->erase(node_type_idx);
      deny_set->insert(node_type_idx);
    } else {
      allow_set->insert(node_type_idx);
    }
  }
}
bool AutoMixedPrecisionImpl::NodeImplicitlyReadsNonResourceVariable(
    const NodeDef& node) const {
  if (node.op() == "Identity" || node.op() == "Enter") {
    GraphView::InputPort node_input(&node, 0);
    MutableGraphView::OutputPort prev_output =
        graph_view_.GetRegularFanin(node_input);
    const NodeDef* input = prev_output.node;
    if (input && ((node.op() == "Identity" && (input->op() == "Variable" ||
                                               input->op() == "VariableV2")) ||
                  (node.op() == "Enter" &&
                   NodeImplicitlyReadsNonResourceVariable(*input)))) {
      return true;
    }
  }
  return false;
}
void AutoMixedPrecisionImpl::MakeCastsAllowIfAllOutputsAllow(
    absl::flat_hash_set<int>* allow_set) const {
  int num_nodes_preop = graph_->node_size();
  for (int node_idx = 0; node_idx < num_nodes_preop; ++node_idx) {
    NodeDef* node = graph_->mutable_node(node_idx);
    NodeTypeId node_type(node, TypeAttrId("DstT"));
    if (node->op() != "Cast" || !IsFloat32(node_type)) {
      continue;
    }
    bool all_fanouts_allow = true;
    MutableGraphView::OutputPort src(node, 0);
    const auto& fanout = graph_view_.GetFanout(src);
    for (const MutableGraphView::InputPort& dst : fanout) {
      TypeAttrId dst_type_attr =
          node_type_map_.GetInputTypeAttr(*dst.node, dst.port_id);
      const absl::optional<int> maybe_dst_type_idx =
          graph_type_view_.GetNodeIndex(dst.node->name(), dst_type_attr);
      DCHECK(maybe_dst_type_idx.has_value())
          << "Type attribute " << dst_type_attr.DebugString() << " of node "
          << dst.node->name() << " not found in graph view";
      int dst_type_idx = maybe_dst_type_idx.value();
      bool dst_is_allow = allow_set->count(dst_type_idx);
      if (!dst_is_allow) {
        all_fanouts_allow = false;
        break;
      }
    }
    if (!fanout.empty() && all_fanouts_allow) {
      const absl::optional<int> maybe_node_type_idx =
          graph_type_view_.GetNodeIndex(node_type);
      DCHECK(maybe_node_type_idx.has_value())
          << "Type attribute " << node_type.type_attr.DebugString()
          << " of node " << node_type.node->name()
          << " not found in graph view";
      int node_type_idx = maybe_node_type_idx.value();
      allow_set->insert(node_type_idx);
    }
  }
}
absl::StatusOr<NodeDef*> AutoMixedPrecisionImpl::InsertCastNodeAtFanout(
    const absl::flat_hash_set<int>& allow_set, const bool src_is_allow,
    const CastType& cast_type, MutableGraphView::OutputPort& src) {
  NodeDef* added_cast_node = nullptr;
  auto fanout = graph_view_.GetFanout(src);
  for (const MutableGraphView::InputPort& dst : fanout) {
    TypeAttrId dst_type_attr =
        node_type_map_.GetInputTypeAttr(*dst.node, dst.port_id);
    const absl::optional<int> maybe_dst_type_idx =
        graph_type_view_.GetNodeIndex(dst.node->name(), dst_type_attr);
    if (!maybe_dst_type_idx.has_value()) {
      return errors::Internal("Type attribute ", dst_type_attr.DebugString(),
                              " of ", dst.node->op(), " node ",
                              dst.node->name(), " not found in graph view");
    }
    int dst_type_idx = maybe_dst_type_idx.value();
    bool dst_is_allow = allow_set.count(dst_type_idx);
    bool to_f16 = false;
    bool should_cast = false;
    switch (cast_type) {
      case CastType::AUTO:
        if (src_is_allow != dst_is_allow) {
          to_f16 = dst_is_allow;
          should_cast = true;
        }
        break;
      case CastType::FP16:
        to_f16 = true;
        should_cast = true;
        break;
      case CastType::FP32:
        to_f16 = false;
        should_cast = true;
        break;
      default:
        return errors::Internal("Invalid Cast Type: ",
                                static_cast<int>(cast_type));
    }
    if (!should_cast) continue;
    if (added_cast_node == nullptr) {
      VLOG(1) << "Inserting cast to "
              << (to_f16 ? DataTypeString(target_dtype_) : "DT_FLOAT") << " at "
              << src.node->op() << " " << src.node->name() << ":"
              << src.port_id;
      added_cast_node =
          graph_view_.AddNode(BuildCastNode(src, to_f16, src.node->device()));
      if (to_f16 && !IsConstant(*src.node) && !IsVariable(*src.node) &&
          !NodeImplicitlyReadsNonResourceVariable(*src.node)) {
        ++num_nonvar_casts_to_f16_;
      }
    }
    TF_RETURN_IF_ERROR(graph_view_.UpdateRegularFaninByPort(
        dst.node->name(), dst.port_id, {added_cast_node->name(), 0}));
  }
  return added_cast_node;
}
absl::StatusOr<DataType> AutoMixedPrecisionImpl::GetCastToType(
    const NodeDef* node) const {
  CHECK_EQ(node->op(), "Cast")  
      << "Node " << node->name() << " is not a Cast op";
  return node->attr().at("DstT").type();
}
void AutoMixedPrecisionImpl::CollectOutputPorts(
    const TypeAttrId& type_attr, NodeDef* node,
    std::vector<MutableGraphView::OutputPort>& output_ports) const {
  for (int port_id : node_type_map_.GetOutputPorts(*node, type_attr)) {
    output_ports.emplace_back(node, port_id);
  }
}
Status AutoMixedPrecisionImpl::ChangeTypeAttrsAndAddCasts(
    const absl::flat_hash_set<int>& allow_set) {
  int num_nodes_changed = 0;
  const int num_nodes_preop = graph_->node_size();
  bool emulate_f16 = false;
  if (mode_ == AutoMixedPrecisionMode::CPU) {
    TF_CHECK_OK(
        ReadBoolFromEnvVar("TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_EMULATE_FP16",
                           true, &emulate_f16));
  }
  VLOG(1) << "Setting emulate_f16 = " << emulate_f16;
  for (int node_idx = 0; node_idx < num_nodes_preop; ++node_idx) {
    NodeDef* node = graph_->mutable_node(node_idx);
    for (const TypeAttrId& type_attr : node_type_map_.GetTypeAttrs(*node)) {
      const absl::optional<int> maybe_node_type_idx =
          graph_type_view_.GetNodeIndex(node->name(), type_attr);
      if (!maybe_node_type_idx.has_value()) {
        return errors::Internal("Type attribute ", type_attr.DebugString(),
                                " of ", node->op(), " node ", node->name(),
                                " not found in graph view");
      }
      int node_type_idx = maybe_node_type_idx.value();
      if (!IsFloat32(*graph_type_view_.GetNode(node_type_idx))) continue;
      bool src_is_allow = allow_set.count(node_type_idx);
      std::vector<MutableGraphView::OutputPort> output_ports;
      if (src_is_allow) {
        if (emulate_f16) {
          for (int port_id : node_type_map_.GetInputPorts(*node, type_attr)) {
            VLOG(2) << "Cast to F32 at fanin of node " << node->name() << ":"
                    << port_id;
            MutableGraphView::InputPort dst(node, port_id);
            MutableGraphView::OutputPort src = graph_view_.GetRegularFanin(dst);
            NodeDef* added_cast_node = graph_view_.AddNode(
                BuildCastNode(src, false, src.node->device()));
            VLOG(1) << "Inserting cast to DT_FLOAT at " << src.node->op() << " "
                    << src.node->name() << ":" << src.port_id;
            TF_RETURN_IF_ERROR(graph_view_.UpdateRegularFaninByPort(
                dst.node->name(), dst.port_id, {added_cast_node->name(), 0}));
          }
          for (int port_id : node_type_map_.GetOutputPorts(*node, type_attr)) {
            MutableGraphView::OutputPort src(node, port_id);
            VLOG(2) << "Cast to F16 at fanout of node " << node->name() << ":"
                    << port_id;
            TF_ASSIGN_OR_RETURN(NodeDef * added_cast_node,
                                InsertCastNodeAtFanout(allow_set, src_is_allow,
                                                       CastType::FP16, src));
            if (added_cast_node != nullptr) {
              output_ports.emplace_back(added_cast_node, 0);
            }
          }
        } else {
          VLOG(1) << "Changing type " << type_attr.DebugString() << " of "
                  << node->op() << " node " << node->name() << " to "
                  << DataTypeString(target_dtype_);
          if (!SetDataType(node, type_attr, target_dtype_)) {
            return errors::Internal("Failed to set type attribute");
          }
          ++num_nodes_changed;
          CollectOutputPorts(type_attr, node, output_ports);
        }
      } else {
        CollectOutputPorts(type_attr, node, output_ports);
      }
      for (auto output_port : output_ports) {
        VLOG(2) << "Cast to required data type at fanout of node "
                << output_port.node->name() << ":" << output_port.port_id;
        TF_RETURN_IF_ERROR(InsertCastNodeAtFanout(allow_set, src_is_allow,
                                                  CastType::AUTO, output_port)
                               .status());
      }
    }
  }
  const char* type_str = target_dtype_ == DT_HALF ? "float16" : "bfloat16";
  LOG(INFO) << "Converted " << num_nodes_changed << "/" << num_nodes_preop
            << " nodes to " << type_str << " precision using "
            << num_nonvar_casts_to_f16_ << " cast(s) to " << type_str
            << " (excluding Const and Variable casts)";
  return absl::OkStatus();
}
int GetNumGPUs(const Cluster& cluster) {
  if (ShouldSimulateGpu()) {
    return 1;
  }
  auto devices = cluster.GetDevices();
  int num_gpus = 0;
  for (const auto& device : devices) {
    const DeviceProperties& device_properties = device.second;
    if (device_properties.type() == "GPU" &&
        (ShouldIgnorePerformance() || HasFastFP16Support(device_properties))) {
      num_gpus++;
    }
  }
  return num_gpus;
}
}  
Status AutoMixedPrecision::Optimize(Cluster* cluster, const GrapplerItem& item,
                                    GraphDef* output) {
  if (cluster == nullptr) {
    return errors::InvalidArgument("cluster == nullptr");
  }
#if !defined(INTEL_MKL)
  if (mode_ == AutoMixedPrecisionMode::BF16) {
    return errors::Unimplemented(
        "The auto_mixed_precision_onednn_bfloat16 optimizer cannot be used "
        "since this build of TensorFlow is not compiled with oneDNN support "
        "for bfloat16. "
        "For information on oneDNN builds, see: "
        "https:
        "tensorflow-installation-guide");
  }
#endif  
  *output = item.graph;
  int num_gpus = GetNumGPUs(*cluster);
  if (num_gpus < 1 && mode_ == AutoMixedPrecisionMode::CUDA) {
    VLOG(1) << "No (suitable) GPUs detected, skipping " << name()
            << " graph optimizer";
    return absl::OkStatus();
  }
  if (mode_ == AutoMixedPrecisionMode::FP16_CPU &&
      !IsAMXDataTypeSupportedByOneDNNOnThisCPU(DT_HALF) &&
      !IsAVXConvertSupportedByOneDNNOnThisCPU()) {
    VLOG(1) << "No support for " << name() << " graph optimizer on CPU";
    return absl::OkStatus();
  }
  if (num_gpus >= 1 && mode_ == AutoMixedPrecisionMode::BF16) {
    LOG(WARNING) << "Note: GPUs detected. Using " << name()
                 << " graph optimizer configured for BFloat16 on CPUs";
  }
  AutoMixedPrecisionImpl optimizer(cluster, item.NodesToPreserve(), output,
                                   item.id, mode_);
  if (item.id == "tf_graph") {
    LOG(INFO) << "Running " << name() << " graph optimizer";
  } else {
    VLOG(1) << "Running " << name() << " graph optimizer on " << item.id;
  }
  Status status = optimizer.Optimize();
  if (!status.ok()) {
    *output = item.graph;
    LOG(WARNING) << name() << " graph optimizer FAILED: " << status.ToString();
  }
  return status;
}
}  
}  