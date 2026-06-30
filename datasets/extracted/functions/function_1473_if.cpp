#include "tensorflow/core/grappler/optimizers/scoped_allocator_optimizer.h"
#include "tensorflow/core/common_runtime/scoped_allocator.h"
#include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils/frame.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#define LOG_WARNING_AND_RETURN_IF_ERROR(...)            \
  do {                                                  \
    const ::tensorflow::Status _status = (__VA_ARGS__); \
    if (TF_PREDICT_FALSE(!_status.ok())) {              \
      LOG(WARNING) << "error: " << _status;             \
      return _status;                                   \
    }                                                   \
  } while (0)
namespace tensorflow {
namespace grappler {
namespace {
const char kScopedAllocatorAttrName[] = "_scoped_allocator";
bool HasOpName(const string& node_name, const string& op_name) {
  size_t begin = node_name.rfind('/');
  if (begin == string::npos) {
    begin = 0;
  } else {
    ++begin;
  }
  size_t end = node_name.rfind('_');
  if (end != string::npos) {
    size_t p = end + 1;
    while (p < node_name.size()) {
      if (!isdigit(node_name[p])) {
        end = node_name.size();
        break;
      }
      ++p;
    }
  } else {
    end = node_name.size();
  }
  return node_name.substr(begin, end - begin) == op_name;
}
Status GetOutputDataType(
    const std::vector<OpInfo::TensorProperties>& output_props, int output_index,
    DataType* dtype) {
  int output_props_size = output_props.size();
  if (output_index >= output_props_size) {
    return errors::Internal("Invalid output index ", output_index,
                            " size of output_props ", output_props.size());
  }
  *dtype = output_props[output_index].dtype();
  return absl::OkStatus();
}
Status CheckTypesAndGetShapes(const GraphProperties& graph_properties,
                              const std::vector<NodeDef*>& ops, DataType* type,
                              std::vector<TensorShape>* shapes) {
  VLOG(1) << "CheckTypesAndGetShapes";
  *type = DT_INVALID;
  for (NodeDef* n : ops) {
    AttrSlice n_attrs = AttrSlice(*n);
    DataType dtype;
    LOG_WARNING_AND_RETURN_IF_ERROR(GetNodeAttr(n_attrs, "T", &dtype));
    VLOG(2) << "op " << n->name() << " has type " << dtype << " shapes.size() "
            << shapes->size();
    if (!graph_properties.HasOutputProperties(n->name())) {
      LOG(ERROR) << "Node " << n->DebugString() << " lacks output shape.";
      return errors::Aborted("Node ", n->name(), " lacks output shape.");
    }
    const std::vector<OpInfo::TensorProperties>& prop_list =
        graph_properties.GetOutputProperties(n->name());
    if (prop_list.size() != 1) {
      return errors::Aborted("Node ", n->name(),
                             " does not have exactly one output as expected "
                             "by ScopedAllocatorOptimizer");
    }
    const OpInfo::TensorProperties& props = prop_list[0];
    if (shapes->empty()) {
      *type = props.dtype();
    } else if (*type != props.dtype()) {
      return errors::Aborted("Group ops don't all have same type");
    }
    if (*type != dtype) {
      return errors::Internal(
          "Type mismatch: type in op attr = ", DataTypeString(dtype),
          ", type in output props = ", DataTypeString(*type));
    }
    if (!TensorShape::IsValid(props.shape()) || props.shape().unknown_rank()) {
      return errors::Aborted("Complete shape not known for ", n->name());
    }
    VLOG(2) << "Adding shape " << props.shape().DebugString();
    shapes->push_back(TensorShape(props.shape()));
  }
  return absl::OkStatus();
}
struct InputDesc {
  NodeDef* from_node_def;
  int output_slot;
  NodeDef* to_node_def;
  InputDesc(NodeDef* f, int os, NodeDef* t)
      : from_node_def(f), output_slot(os), to_node_def(t) {}
};
void RemoveNode(NodeDef* nd, GraphDef* graph, NodeMap* node_map) {
  node_map->RemoveNode(nd->name());
  protobuf::RepeatedPtrField<NodeDef>* nodes = graph->mutable_node();
  for (int i = 0; i < nodes->size(); ++i) {
    if (nd->name() == (*nodes)[i].name()) {
      nodes->SwapElements(i, nodes->size() - 1);
      nodes->RemoveLast();
      return;
    }
  }
  LOG(FATAL) << "Failed to find node " << nd->name() << " in graph";
}
Status RemoveEdge(const string& input_edge_name, const string& from_node_name,
                  NodeDef* to_node, NodeMap* node_map) {
  protobuf::RepeatedPtrField<string>* inputs = to_node->mutable_input();
  int edge_index = -1;
  for (edge_index = 0; edge_index < inputs->size(); ++edge_index) {
    VLOG(2) << " consider edge " << (*inputs)[edge_index];
    if ((*inputs)[edge_index] == input_edge_name) {
      break;
    }
  }
  if (edge_index >= inputs->size()) {
    return errors::Internal("Could not find input name ", input_edge_name,
                            " at node ", to_node->name());
  }
  if (node_map) {
    node_map->RemoveOutput(from_node_name, to_node->name());
  }
  inputs->DeleteSubrange(edge_index, 1);
  return absl::OkStatus();
}
Status MaybeRewriteInput(ScopedAllocatorOptimizer* sa_opti,
                         int64_t invocation_count, GraphDef* graph,
                         NodeMap* node_map, const DataType& dtype,
                         NodeDef* input, const string& edge_name,
                         int output_index, NodeDef* op, NodeDef** new_input,
                         int* new_output_index, bool* rewrite) {
  *rewrite = IsConstant(*input) || IsExit(*input) ||
             (sa_opti->repeated_outputs().find(edge_name) !=
              sa_opti->repeated_outputs().end());
  if (!(*rewrite)) {
    *new_input = input;
    *new_output_index = output_index;
    return absl::OkStatus();
  }
  int unique_id;
  LOG_WARNING_AND_RETURN_IF_ERROR(sa_opti->NewIdentityId(&unique_id));
  string identity_name = strings::StrCat("scoped_allocator_identity_",
                                         unique_id, "_", invocation_count);
  NodeDefBuilder identity_builder(identity_name, "Identity");
  identity_builder.Device(op->device());
  identity_builder.Attr("T", dtype);
  identity_builder.Input(
      NodeDefBuilder::NodeOut(input->name(), output_index, dtype));
  NodeDef* identity = graph->add_node();
  LOG_WARNING_AND_RETURN_IF_ERROR(identity_builder.Finalize(identity));
  node_map->AddNode(identity_name, identity);
  node_map->AddOutput(input->name(), identity_name);
  node_map->UpdateInput(op->name(), input->name(), identity_name);
  *op->mutable_input(0) = identity_name;
  *new_input = identity;
  *new_output_index = 0;
  VLOG(1) << "Rewrite input " << edge_name << " op " << op->name()
          << " old output index " << output_index << " with identity "
          << identity_name << " new output index 0";
  return absl::OkStatus();
}
Status GetInputs(ScopedAllocatorOptimizer* sa_opti, int64_t invocation_count,
                 GraphDef* graph, const GraphProperties& graph_properties,
                 NodeMap* node_map, const std::vector<NodeDef*>& ops,
                 DataType dtype, std::vector<InputDesc>* inputs) {
  VLOG(1) << "Getinputs";
  for (NodeDef* n : ops) {
    NodeDef* inode = nullptr;
    int output_index = 0;
    DataType inode_dtype = DT_INVALID;
    VLOG(2) << "for node " << n->name();
    for (const auto& input_name : n->input()) {
      if (!IsControlInput(input_name)) {
        if (inode) {
          return errors::Internal("Found more than one input for node ",
                                  n->name());
        }
        ParseNodeName(input_name, &output_index);
        inode = node_map->GetNode(input_name);
        if (inode == nullptr) {
          return errors::Internal("Did not find node ", input_name);
        }
        VLOG(2) << "inode " << inode->DebugString() << " output_index "
                << output_index;
        bool rewrite;
        LOG_WARNING_AND_RETURN_IF_ERROR(MaybeRewriteInput(
            sa_opti, invocation_count, graph, node_map, dtype, inode,
            input_name, output_index, n, &inode, &output_index, &rewrite));
        if (rewrite) {
          inode_dtype = dtype;
        }
        VLOG(2) << "inode after rewrite " << inode->DebugString()
                << " output_index " << output_index;
      }
    }
    if (inode == nullptr) {
      return errors::Internal("Did not find node");
    }
    if (inode_dtype == DT_INVALID) {
      if (!graph_properties.HasOutputProperties(inode->name())) {
        return errors::Internal("Input node ", inode->name(),
                                " does not have output properties");
      }
      const auto& inode_output_props =
          graph_properties.GetOutputProperties(inode->name());
      LOG_WARNING_AND_RETURN_IF_ERROR(
          GetOutputDataType(inode_output_props, output_index, &inode_dtype));
    }
    if (inode_dtype != dtype) {
      return errors::Aborted("ScopedAllocatorOptimizer expected input type ",
                             dtype, " but found ", inode_dtype);
    }
    inputs->emplace_back(inode, output_index, n);
  }
  return absl::OkStatus();
}
Status GetDataInputs(GraphDef* graph, NodeMap* node_map, NodeDef* op,
                     std::vector<InputDesc>* inputs) {
  VLOG(2) << "GetDataInputs for node " << op->name();
  NodeDef* inode = nullptr;
  int output_index = 0;
  for (const auto& input_name : op->input()) {
    if (IsControlInput(input_name)) {
      continue;
    }
    ParseNodeName(input_name, &output_index);
    inode = nullptr;
    inode = node_map->GetNode(input_name);
    if (inode == nullptr) {
      return errors::Internal("Did not find node ", input_name);
    }
    VLOG(2) << "inode " << inode->DebugString() << " output_index "
            << output_index;
    inputs->emplace_back(inode, output_index, op);
  }
  return absl::OkStatus();
}
void DumpGraphToVLOG(const GraphDef& graph, int log_level) {
  if (VLOG_IS_ON(log_level)) {
    for (const auto& line : str_util::Split(graph.DebugString(), "\n\r")) {
      VLOG(log_level) << line;
    }
  }
}
}  
void ScopedAllocatorOptimizer::ExtendNodeAttr(StringPiece name,
                                              const std::vector<int32>& values,
                                              NodeDef* node_def) {
  if (HasNodeAttr(*node_def, name)) {
    VLOG(2) << "extending";
    AttrValue* existing = &(*node_def->mutable_attr())[string(name)];
    for (int32_t i : values) {
      existing->mutable_list()->add_i(i);
    }
  } else {
    VLOG(2) << "setting new attr value";
    AddNodeAttr(name, values, node_def);
  }
}
class UnaryElementwiseRewriter : public ScopedAllocatorOptimizer::Rewriter {
 public:
  ~UnaryElementwiseRewriter() override {}
  Status CheckUsesAllocatorAttributes(const std::vector<InputDesc>& inputs) {
    for (const InputDesc& nd : inputs) {
      if (IsConstant(*nd.from_node_def)) {
        return errors::Aborted(
            "Abandoning ScopedAllocatorOptimizer because input ",
            nd.from_node_def->name(),
            " is a Const op which does not use AllocatorAttributes");
      }
    }
    return absl::OkStatus();
  }
  Status CheckExistingScopedAllocator(const std::vector<InputDesc>& inputs) {
    for (const InputDesc& nd : inputs) {
      VLOG(2) << "get attrs for " << nd.from_node_def->name();
      AttrSlice n_attrs = AttrSlice(*nd.from_node_def);
      std::vector<int32> scope_ids;
      Status ss = GetNodeAttr(n_attrs, kScopedAllocatorAttrName, &scope_ids);
      if (ss.ok() && scope_ids[0] == nd.output_slot) {
        LOG(INFO) << "Abandoning ScopedAllocatorOptimizer because input "
                  << nd.from_node_def->name() << " output " << scope_ids[0]
                  << " is already assigned to scope_id " << scope_ids[1];
        return errors::Aborted(
            "Abandoning ScopedAllocatorOptimizer because input ",
            nd.from_node_def->name(), " output ", scope_ids[0], " is already ",
            "assigned to scope_id ", scope_ids[1]);
      }
    }
    return absl::OkStatus();
  }
  Status CheckInternalDataDependency(const std::set<string>& op_set,
                                     const std::vector<InputDesc>& inputs) {
    for (const InputDesc& nd : inputs) {
      if (op_set.find(nd.from_node_def->name()) != op_set.end()) {
        if (nd.output_slot != tensorflow::Graph::kControlSlot) {
          return errors::Aborted("Data edge exists between ",
                                 nd.from_node_def->name(),
                                 " and another "
                                 "node in the set");
        }
      }
    }
    return absl::OkStatus();
  }
  void ClearInternalControlInputs(const std::set<string>& op_set,
                                  const std::vector<NodeDef*>& ops,
                                  NodeMap* node_map) {
    for (NodeDef* n : ops) {
      for (const auto& input_name : n->input()) {
        if (IsControlInput(input_name)) {
          int position = 0;
          string input_node_name = ParseNodeName(input_name, &position);
          CHECK_EQ(position, -1);
          if (op_set.find(input_node_name) != op_set.end()) {
            VLOG(1) << "Remove control output from " << input_node_name
                    << " via edge " << input_name << " to " << n->name();
            TF_CHECK_OK(RemoveEdge(input_name, input_node_name, n, node_map));
          }
        }
      }
    }
  }
  Status AnalyzeInputs(ScopedAllocatorOptimizer* sa_opti,
                       int64_t invocation_count, GraphDef* graph,
                       NodeMap* node_map, const std::vector<NodeDef*>& ops,
                       const std::set<string>& op_instance_names,
                       string* device_name, DataType* dtype,
                       std::vector<TensorShape>* input_shapes,
                       std::vector<InputDesc>* inputs, TensorShape* sa_shape) {
    CHECK(graph_properties_);
    LOG_WARNING_AND_RETURN_IF_ERROR(
        CheckTypesAndGetShapes(*graph_properties_, ops, dtype, input_shapes));
    LOG_WARNING_AND_RETURN_IF_ERROR(
        GetInputs(sa_opti, invocation_count, graph, *graph_properties_,
                  sa_opti->node_map(), ops, *dtype, inputs));
    LOG_WARNING_AND_RETURN_IF_ERROR(CheckUsesAllocatorAttributes(*inputs));
    LOG_WARNING_AND_RETURN_IF_ERROR(CheckExistingScopedAllocator(*inputs));
    LOG_WARNING_AND_RETURN_IF_ERROR(
        CheckInternalDataDependency(op_instance_names, *inputs));
    ClearInternalControlInputs(op_instance_names, ops, node_map);
    *device_name = ops[0]->device();
    CHECK(!device_name->empty());
    CHECK(!input_shapes->empty());
    CHECK_EQ(0, Allocator::kAllocatorAlignment % DataTypeSize(*dtype))
        << "ScopedAllocatorOptimizer only applies to types that evenly "
        << "divide kAllocatorAlignment";
    std::vector<ScopedAllocator::Field> sa_fields;
    int64_t num_bytes = ScopedAllocatorMgr::PopulateFields(
        0 , *input_shapes, *dtype, &sa_fields);
    int64_t num_elts = num_bytes / DataTypeSize(*dtype);
    VLOG(2) << "num_bytes " << num_bytes << " num_elts=" << num_elts;
    *sa_shape = TensorShape({num_elts});
    return absl::OkStatus();
  }
  Status TransitiveFanoutWithinFrame(
      GraphDef* graph, NodeMap* node_map,
      const std::vector<const NodeDef*>& source_nodes,
      absl::flat_hash_set<const NodeDef*>* fanout) {
    std::deque<const NodeDef*> queue(source_nodes.begin(), source_nodes.end());
    absl::flat_hash_set<const NodeDef*> visited;
    while (!queue.empty()) {
      const NodeDef* node = queue.front();
      queue.pop_front();
      if (!visited.insert(node).second) {
        continue;
      }
      fanout->insert(node);
      for (const NodeDef* output : node_map->GetOutputs(node->name())) {
        if (!ModifiesFrameInfo(*output)) {
          queue.push_back(output);
        }
        VLOG(2) << "TransitiveFanout parent: " << node->name()
                << " child: " << output->name() << " of type " << output->op();
      }
    }
    return absl::OkStatus();
  }
  Status ConstructScopedAllocatorNode(
      ScopedAllocatorOptimizer* sa_opti, GraphDef* graph, NodeMap* node_map,
      const std::vector<NodeDef*>& ops, const string& device_name,
      DataType dtype, int sa_id, const string& sa_name,
      const std::vector<TensorShape>& input_shapes,
      const std::vector<InputDesc>& inputs, const TensorShape& sa_shape) {
    VLOG(2) << "ConstructScopedAllocatorNode " << sa_name;
    NodeDefBuilder sa_builder(sa_name, "_ScopedAllocator");
    sa_builder.Device(device_name);
    sa_builder.Attr("sa_name", sa_name);
    sa_builder.Attr("T", dtype);
    sa_builder.Attr("id", sa_id);
    sa_builder.Attr("shapes", input_shapes);
    sa_builder.Attr("shape", sa_shape);
    sa_builder.Attr("expected_call_count", static_cast<int64_t>(ops.size()));
    NodeDef* sa_node = graph->add_node();
    LOG_WARNING_AND_RETURN_IF_ERROR(sa_builder.Finalize(sa_node));
    node_map->AddNode(sa_name, sa_node);
    std::vector<const NodeDef*> fanout_sources;
    fanout_sources.reserve(inputs.size());
    for (const auto& input : inputs) {
      fanout_sources.push_back(input.from_node_def);
    }
    absl::flat_hash_set<const NodeDef*> fanout;
    TF_RETURN_IF_ERROR(
        TransitiveFanoutWithinFrame(graph, node_map, fanout_sources, &fanout));
    for (int i = 0, end = inputs.size(); i < end; ++i) {
      auto& nd = inputs[i];
      if (IsArg(*nd.from_node_def)) {
        return errors::Aborted(
            "ScopedAllocatorOptimizer does not work well when the op inputs "
            "are _Arg ops; skipping this optimizer for this function");
      }
      VLOG(2) << "To input " << i << ": " << nd.from_node_def->name()
              << " add control input "
              << "^" << sa_name;
      nd.from_node_def->add_input(strings::StrCat("^", sa_name));
      ScopedAllocatorOptimizer::ExtendNodeAttr(kScopedAllocatorAttrName,
                                               {nd.output_slot, sa_id + 1 + i},
                                               nd.from_node_def);
      node_map->AddOutput(sa_name, nd.from_node_def->name());
    }
    bool added_delay_edge = false;
    for (auto& nd : inputs) {
      std::vector<InputDesc> inputs_to_first;
      LOG_WARNING_AND_RETURN_IF_ERROR(GetDataInputs(
          graph, sa_opti->node_map(), nd.from_node_def, &inputs_to_first));
      for (int i = 0, end = inputs_to_first.size(); i < end; ++i) {
        if (fanout.find(inputs_to_first[i].from_node_def) != fanout.end()) {
          VLOG(2) << "Found node " << inputs_to_first[i].from_node_def->name()
                  << " in the fanout of " << sa_name;
          continue;
        }
        sa_node->add_input(
            strings::StrCat("^", inputs_to_first[i].from_node_def->name()));
        node_map->AddOutput(inputs_to_first[i].from_node_def->name(), sa_name);
        added_delay_edge = true;
        VLOG(2) << "Adding control dependency from "
                << inputs_to_first[i].from_node_def->name() << " to "
                << sa_node->name();
        break;
      }
      if (added_delay_edge) {
        break;
      }
    }
    if (!added_delay_edge) {
      LOG(WARNING) << "Found no node from which a control edge can be added to "
                      "scoped allocator node.  If you run into issues with "
                      "graphs that contain control flow, turn off the "
                      "ScopedAllocatorOptimizer and file a bug.";
    }
    return absl::OkStatus();
  }
  Status BuildSAConcatNode(GraphDef* graph, NodeMap* node_map,
                           const std::vector<NodeDef*>& ops,
                           const std::set<string>& op_instance_names,
                           const string& device_name, DataType dtype, int sa_id,
                           const string& sa_name, const string& sac_name,
                           const TensorShape& sa_shape,
                           std::vector<NodeDefBuilder::NodeOut>* sac_inputs) {
    VLOG(2) << "BuildSAConcatNode " << sac_name;
    absl::flat_hash_map<string, string> sac_ctl_inputs;
    for (int i = 0, end = ops.size(); i < end; ++i) {
      NodeDef* old_op = ops[i];
      for (const string& old_op_input : old_op->input()) {
        int position = 0;
        string input_name = ParseNodeName(old_op_input, &position);
        if (position == -1) {
          if (op_instance_names.find(old_op_input) == op_instance_names.end()) {
            sac_ctl_inputs.emplace(old_op_input, input_name);
          }
        } else {
          if (op_instance_names.find(old_op_input) != op_instance_names.end()) {
            LOG(ERROR) << "Data edge between " << old_op_input << " and "
                       << old_op->name() << " cannot build ScopedAllocator.";
            return errors::Aborted("Data edge between ", old_op_input, " and ",
                                   old_op->name(),
                                   " cannot build ScopedAllocator.");
          }
          sac_inputs->push_back(
              NodeDefBuilder::NodeOut(old_op_input, 0, dtype));
        }
        VLOG(3) << "from op " << i << ": " << old_op->name()
                << " sac_inputs append " << old_op_input;
      }
    }
    NodeDefBuilder sac_builder(sac_name, "_ScopedAllocatorConcat");
    VLOG(2) << "New sac_name " << sac_name << " shape "
            << sa_shape.DebugString();
    sac_builder.Device(device_name);
    sac_builder.Attr("sa_name", sa_name);
    sac_builder.Attr("id", sa_id);
    sac_builder.Attr("T", dtype);
    sac_builder.Attr("shape", sa_shape);
    sac_builder.Attr("N", static_cast<int>(sac_inputs->size()));
    sac_builder.Input(NodeDefBuilder::NodeOut(sa_name, 0, dtype));
    sac_builder.Input(*sac_inputs);
    NodeDef* sac_node = graph->add_node();
    LOG_WARNING_AND_RETURN_IF_ERROR(sac_builder.Finalize(sac_node));
    node_map->AddNode(sac_name, sac_node);
    node_map->AddOutput(sa_name, sac_name);
    for (const auto& ctl_input : sac_ctl_inputs) {
      const auto& ctl_edge = ctl_input.first;
      const auto& input_name = ctl_input.second;
      sac_node->add_input(ctl_edge);
      node_map->AddOutput(input_name, sac_node->name());
    }
    return absl::OkStatus();
  }
  Status BuildReplacementOp(GraphDef* graph, NodeMap* node_map,
                            const std::vector<NodeDef*>& ops,
                            const string& device_name, DataType dtype,
                            const string& op_name, const string& sac_name,
                            const string& sa_op_name) {
    VLOG(2) << "BuildReplacementOp " << sa_op_name;
    NodeDefBuilder op_builder(sa_op_name, op_name);
    op_builder.Device(device_name);
    AttrSlice first_slice(*ops[0]);
    for (auto& it : first_slice) {
      op_builder.Attr(it.first, it.second);
    }
    op_builder.Attr("_forward_input", {0, 0});
    op_builder.Input(sac_name, 0, dtype);
    NodeDef* sa_op_node = graph->add_node();
    LOG_WARNING_AND_RETURN_IF_ERROR(op_builder.Finalize(sa_op_node));
    node_map->AddNode(sa_op_name, sa_op_node);
    node_map->AddOutput(sac_name, sa_op_name);
    return absl::OkStatus();
  }
  Status BuildSplitNode(GraphDef* graph, NodeMap* node_map,
                        const std::vector<NodeDef*>& ops,
                        const std::vector<TensorShape>& input_shapes,
                        const std::vector<NodeDefBuilder::NodeOut>& sac_inputs,
                        const string& device_name, DataType dtype,
                        const string& op_name, int sa_id,
                        const string& sas_name, const string& sa_name,
                        const string& sa_op_name) {
    VLOG(2) << "new ScopedAllocatorSplit " << sas_name;
    NodeDefBuilder sas_builder(sas_name, "_ScopedAllocatorSplit");
    sas_builder.Device(device_name);
    sas_builder.Attr("sa_name", sa_name);
    sas_builder.Attr("id", sa_id);
    sas_builder.Attr("T", dtype);
    sas_builder.Attr("shapes", input_shapes);
    std::vector<NodeDefBuilder::NodeOut> sas_inputs = sac_inputs;
    sas_builder.Attr("N", static_cast<int>(sas_inputs.size()));
    sas_builder.Input(NodeDefBuilder::NodeOut(sa_op_name, 0, dtype));
    sas_builder.Input(sas_inputs);
    NodeDef* sas_node = graph->add_node();
    LOG_WARNING_AND_RETURN_IF_ERROR(sas_builder.Finalize(sas_node));
    node_map->AddNode(sas_name, sas_node);
    node_map->AddOutput(sa_op_name, sas_name);
    for (const auto& input : sas_inputs) {
      node_map->AddOutput(input.node, sas_name);
    }
    return absl::OkStatus();
  }
  Status RewireSubgraph(GraphDef* graph, NodeMap* node_map,
                        const std::vector<NodeDef*>& ops,
                        const std::set<string>& op_instance_names,
                        const string& op_name, const string& sas_name) {
    VLOG(2) << "RewireSubgraph";
    for (int op_idx = 0, idx_limit = ops.size(); op_idx < idx_limit; ++op_idx) {
      NodeDef* old_op = ops[op_idx];
      auto output_nodes = node_map->GetOutputs(old_op->name());
      VLOG(3) << "old_op " << old_op->name() << " had " << output_nodes.size()
              << " outputs.  Moving them to the ScopedAllocatorSplit node.";
      if (VLOG_IS_ON(2)) {
        for (NodeDef* n : output_nodes) {
          VLOG(3) << "    output: " << n->name();
        }
      }
      for (NodeDef* n : output_nodes) {
        VLOG(3) << "really checking old output " << n->name()
                << " for corresponding input.";
        if (op_instance_names.find(n->name()) != op_instance_names.end()) {
          VLOG(3) << "Dropping control output from " << old_op->name() << " to "
                  << n->name();
          Status ignore = RemoveEdge(strings::StrCat("^", old_op->name()),
                                     old_op->name(), n, node_map);
          continue;
        }
        bool found = false;
        VLOG(3) << "about to iterate over " << n->input_size() << " inputs";
        for (int i = 0; i < n->input_size(); ++i) {
          VLOG(3) << "input " << n->input(i);
          int position = 0;
          string input_node = ParseNodeName(n->input(i), &position);
          if (input_node == old_op->name()) {
            found = true;
            VLOG(3) << "match pos=" << position;
            if (position == -1) {
              *n->mutable_input(i) = strings::StrCat("^", sas_name);
            } else {
              CHECK_EQ(0, position)
                  << "name " << n->input(i) << " pos " << position;
              *n->mutable_input(i) = strings::StrCat(sas_name, ":", op_idx);
            }
            node_map->UpdateInput(n->name(), old_op->name(), sas_name);
            VLOG(3) << "breaking on success";
            break;
          } else {
            VLOG(3) << "other input " << n->input(i);
          }
        }
        VLOG(3) << "before HasOp";
        if (!HasOpName(n->name(), op_name)) {
          CHECK(found) << "old_op " << old_op->name() << " node "
                       << " could not find input edge on " << n->DebugString()
                       << " to replace."
                       << " " << op_name << " not in " << n->name();
        }
        VLOG(3) << "bottom of for output_nodes";
      }
      VLOG(3) << "Clearing all inputs of " << old_op->name();
      node_map->RemoveInputs(old_op->name());
      old_op->clear_input();
      node_map->RemoveOutputs(old_op->name());
      VLOG(3) << "after clear: " << old_op->DebugString();
      RemoveNode(old_op, graph, node_map);
    }
    return absl::OkStatus();
  }
  Status Rewrite(ScopedAllocatorOptimizer* sa_opti, int64_t invocation_count,
                 GraphDef* graph, const string& op_name,
                 const std::vector<NodeDef*>& ops, bool* applied) override {
    if (VLOG_IS_ON(1)) {
      VLOG(1) << "Rewrite";
      string op_names;
      for (auto& nd : ops) {
        strings::StrAppend(&op_names, nd->name(), ", ");
      }
      VLOG(1) << "UnaryElementwiseRewriter::Rewrite " << op_name
              << " to: " << op_names;
    }
    NodeMap* node_map = sa_opti->node_map();
    std::set<string> op_instance_names;
    for (auto& nd : ops) {
      op_instance_names.insert(nd->name());
      VLOG(2) << "op_instance_name " << nd->name();
    }
    DataType dtype;
    std::vector<TensorShape> input_shapes;
    std::vector<InputDesc> inputs;
    TensorShape sa_shape;
    string device_name;
    TF_RETURN_IF_ERROR(AnalyzeInputs(
        sa_opti, invocation_count, graph, node_map, ops, op_instance_names,
        &device_name, &dtype, &input_shapes, &inputs, &sa_shape));
    int sa_id = sa_opti->NewScopedAllocatorId(input_shapes.size());
    string sa_name =
        strings::StrCat("scoped_allocator_", sa_id, "_", invocation_count);
    TF_RETURN_IF_ERROR(ConstructScopedAllocatorNode(
        sa_opti, graph, node_map, ops, device_name, dtype, sa_id, sa_name,
        input_shapes, inputs, sa_shape));
    std::vector<NodeDefBuilder::NodeOut> sac_inputs;
    string sac_name = strings::StrCat("scoped_allocator_concat_", sa_id, "_",
                                      invocation_count);
    TF_RETURN_IF_ERROR(BuildSAConcatNode(
        graph, node_map, ops, op_instance_names, device_name, dtype, sa_id,
        sa_name, sac_name, sa_shape, &sac_inputs));
    string sa_op_name = strings::StrCat(sa_name, "_", op_name);
    TF_RETURN_IF_ERROR(BuildReplacementOp(graph, node_map, ops, device_name,
                                          dtype, op_name, sac_name,
                                          sa_op_name));
    string sas_name = strings::StrCat("scoped_allocator_split_", sa_id, "_",
                                      invocation_count);
    TF_RETURN_IF_ERROR(BuildSplitNode(graph, node_map, ops, input_shapes,
                                      sac_inputs, device_name, dtype, op_name,
                                      sa_id, sas_name, sa_name, sa_op_name));
    TF_RETURN_IF_ERROR(RewireSubgraph(graph, node_map, ops, op_instance_names,
                                      op_name, sas_name));
    *applied = true;
    return absl::OkStatus();
  }
};
ScopedAllocatorOptimizer::ScopedAllocatorOptimizer(
    RewriterConfig::Toggle opt_level, const ScopedAllocatorOptions& opts)
    : opt_level_(opt_level) {
  VLOG(1) << "ScopedAllocatorOptimizer::ScopedAllocatorOptimizer";
  Rewriter* r = new UnaryElementwiseRewriter();
  to_delete_.push_back(r);
  if (opts.enable_op_size() == 0) {
    for (const auto& op_name : {"CollectiveReduce"}) {
      op_name_set_.insert(op_name);
      rewriters_[op_name] = r;
    }
  } else {
    for (const auto& op_name : opts.enable_op()) {
      op_name_set_.insert(op_name);
      rewriters_[op_name] = r;
    }
  }
}
Status ScopedAllocatorOptimizer::Optimize(Cluster* ,
                                          const GrapplerItem& item,
                                          GraphDef* optimized_graph) {
  VLOG(3) << "Input graph:";
  DumpGraphToVLOG(item.graph, 3);
  nodes_to_preserve_ = item.NodesToPreserve();
  GraphProperties graph_properties(item);
  const bool assume_valid_feeds = opt_level_ == RewriterConfig::AGGRESSIVE;
  LOG_WARNING_AND_RETURN_IF_ERROR(graph_properties.InferStatically(
      assume_valid_feeds, false,
      false));
  *optimized_graph = item.graph;
  node_map_ = std::make_unique<NodeMap>(optimized_graph);
  LOG_WARNING_AND_RETURN_IF_ERROR(ScopedAllocatorOptimizer::ProcessGraphDef(
      optimized_graph, graph_properties));
  VLOG(1) << "ScopedAllocatorOptimizer::Optimize() done";
  VLOG(3) << "Optimized graph:";
  DumpGraphToVLOG(*optimized_graph, 3);
  return absl::OkStatus();
}
ScopedAllocatorOptimizer::Rewriter* ScopedAllocatorOptimizer::GetRewriter(
    const string& op_name) {
  auto it = rewriters_.find(op_name);
  if (it != rewriters_.end()) {
    return it->second;
  }
  return nullptr;
}
int ScopedAllocatorOptimizer::NewScopedAllocatorId(int num_fields) {
  CHECK_GT(num_fields, 0);
  int id = next_sa_id_;
  next_sa_id_ += (num_fields + 1);
  CHECK_GT(next_sa_id_, 0);
  return id;
}
Status ScopedAllocatorOptimizer::NewIdentityId(int* id) {
  *id = next_identity_id_++;
  if (next_identity_id_ < 0) {
    return errors::Aborted("NewIdentityId overflow");
  }
  return absl::OkStatus();
}
ScopedAllocatorOptimizer::~ScopedAllocatorOptimizer() {
  for (auto ptr : to_delete_) {
    delete ptr;
  }
}
void ScopedAllocatorOptimizer::FindOpOccurrences(GraphDef* graph,
                                                 const OpNameSet& op_names,
                                                 GraphOpOccurrences* occs) {
  VLOG(1) << "FindOpOccurrences ";
  for (const auto& it : op_names) {
    VLOG(1) << "search target " << it;
  }
  for (int ni = 0; ni < graph->node_size(); ++ni) {
    NodeDef* node = graph->mutable_node(ni);
    const string& op_name = node->op();
    if (op_names.find(op_name) != op_names.end()) {
      VLOG(1) << "found " << op_name << " on dev " << node->device();
      (*occs)[node->device()][op_name].push_back(node);
    }
  }
}
namespace {
struct OpNameOrder {
  bool operator()(const NodeDef* a, const NodeDef* b) {
    return a->name() <= b->name();
  }
};
class Tree {
 public:
  Tree(const string& edge, int depth) : edge_(edge), depth_(depth) {}
  ~Tree() {
    for (const auto& it : subtrees_) delete it.second;
  }
  Tree* GetSubTree(const string& edge) {
    auto it = subtrees_.find(edge);
    if (it != subtrees_.end()) {
      return it->second;
    }
    Tree* t = new Tree(edge, depth_ + 1);
    subtrees_[edge] = t;
    return t;
  }
  void InsertNode(NodeDef* n) { nodes_.push_back(n); }
  string edge_;
  int depth_;
  std::vector<NodeDef*> nodes_;
  absl::flat_hash_map<string, Tree*> subtrees_;
};
Status ApplyToAll(Tree* tree, const std::function<Status(Tree*)>& func) {
  Status s;
  for (const auto& it : tree->subtrees_) {
    s = ApplyToAll(it.second, func);
    if (!s.ok()) return s;
  }
  s = func(tree);
  return s;
}
Tree* ComputeScopeTree(const string& op_name,
                       const std::vector<NodeDef*>& node_vec) {
  Tree* root = new Tree("", 0);
  for (NodeDef* n : node_vec) {
    std::vector<string> pieces = str_util::Split(n->name(), "/");
    int depth = pieces.size() - 1;
    Tree* subtree = root;
    for (int i = 0; i < depth; ++i) {
      subtree = subtree->GetSubTree(pieces[i]);
    }
    subtree->InsertNode(n);
  }
  return root;
}
void PartitionByLoopStructure(const FrameView& frame_view,
                              std::vector<NodeDef*> nodes,
                              std::vector<std::vector<NodeDef*>>* loop_groups) {
  absl::flat_hash_map<uint64, std::vector<NodeDef*>> loop_sets;
  for (NodeDef* nd : nodes) {
    uint64 hash = 0;
    const std::vector<int>& loop_ids = frame_view.Frames(*nd);
    for (int id : loop_ids) {
      hash = Hash64Combine(hash, static_cast<uint64>(id));
    }
    loop_sets[hash].push_back(nd);
  }
  for (auto it : loop_sets) {
    loop_groups->push_back(std::move(it.second));
  }
}
void IdentifyRepeatedInputs(const std::vector<NodeDef*>& nodes,
                            absl::flat_hash_set<string>* seen_outputs,
                            absl::flat_hash_set<string>* repeated_outputs) {
  for (NodeDef* node : nodes) {
    for (const auto& input_name : node->input()) {
      if (!seen_outputs->insert(input_name).second) {
        repeated_outputs->insert(input_name);
      }
    }
  }
}
}  
Status ScopedAllocatorOptimizer::ProcessGraphDef(
    GraphDef* graph, const GraphProperties& graph_properties) {
  static std::atomic<int64_t> invocation_counter(1);
  const int64_t invocation_count =
      invocation_counter.fetch_add(1, std::memory_order_seq_cst);
  VLOG(1) << "ProcessGraphDef " << invocation_count;
  Status status;
  GraphOpOccurrences occ;
  FindOpOccurrences(graph, op_name_set_, &occ);
  if (!occ.empty()) {
    FrameView frame_view;
    LOG_WARNING_AND_RETURN_IF_ERROR(frame_view.InferFromGraph(*graph));
    for (auto& dt : occ) {
      VLOG(2) << "Processing device " << dt.first;
      const DevOpOccurrences& dev_occ = dt.second;
      for (auto& it : dev_occ) {
        string op_name = it.first;
        VLOG(1) << "Processing " << op_name << " set size " << it.second.size();
        Rewriter* rewriter = GetRewriter(op_name);
        if (!rewriter) {
          LOG(ERROR) << "Failed to find Rewriter in ScopedAllocatorOptimizer "
                     << "for op_name " << op_name;
          continue;
        }
        rewriter->SetGraphProperties(graph_properties);
        std::unique_ptr<Tree> root(ComputeScopeTree(it.first, it.second));
        absl::flat_hash_set<string> seen_outputs;
        status = ApplyToAll(root.get(), [this, &seen_outputs](Tree* t) {
          IdentifyRepeatedInputs(t->nodes_, &seen_outputs, &repeated_outputs_);
          return absl::OkStatus();
        });
        if (!status.ok()) {
          break;
        }
        status = ApplyToAll(root.get(), [this, rewriter, graph, &frame_view,
                                         &op_name, invocation_count](Tree* t) {
          VLOG(2) << "applied to tree node " << t->edge_ << " at depth "
                  << t->depth_ << " of size " << t->nodes_.size();
          if (t->nodes_.size() > 1) {
            std::vector<std::vector<NodeDef*>> loop_groups;
            PartitionByLoopStructure(frame_view, t->nodes_, &loop_groups);
            for (auto& lg : loop_groups) {
              if (lg.size() > 1) {
                bool applied = false;
                Status s = OrderNodeSet(&lg);
                TF_RETURN_IF_ERROR(s);
                VLOG(1) << "Applying Rewriter for " << op_name;
                s = rewriter->Rewrite(this, invocation_count, graph, op_name,
                                      lg, &applied);
                LOG_WARNING_AND_RETURN_IF_ERROR(s);
              }
            }
          }
          return absl::OkStatus();
        });
        if (!status.ok()) {
          break;
        }
      }
      if (!status.ok()) {
        break;
      }
    }
  }
  VLOG(1) << "ScopedAllocatorOptimizer returning " << status;
  if (!status.ok()) {
    LOG(ERROR) << "ScopedAllocatorOptimizer: " << status;
  }
  return status;
}
namespace {
struct InstanceKeyLess {
  bool operator()(const NodeDef* a, const NodeDef* b) const {
    AttrSlice a_attrs = AttrSlice(*a);
    AttrSlice b_attrs = AttrSlice(*b);
    int32_t a_key = -1;
    int32_t b_key = -1;
    Status s = GetNodeAttr(a_attrs, "instance_key", &a_key);
    CHECK(s.ok());
    s = GetNodeAttr(b_attrs, "instance_key", &b_key);
    CHECK(s.ok());
    return a_key < b_key;
  }
};
struct NameLess {
  bool operator()(const NodeDef* a, const NodeDef* b) const {
    return a->name() < b->name();
  }
};
bool IsCollectiveNode(const NodeDef& n) {
  AttrSlice attrs = AttrSlice(n);
  int key = -1;
  if (!IsCollective(n)) return false;
  Status s = GetNodeAttr(attrs, "instance_key", &key);
  if (s.ok() && key >= 0) {
    return true;
  }
  return false;
}
}  
Status ScopedAllocatorOptimizer::OrderNodeSet(
    std::vector<NodeDef*>* nodes) const {
  if (nodes->size() <= 1) return absl::OkStatus();
  if (IsCollectiveNode(*nodes->at(0))) {
    std::sort(nodes->begin(), nodes->end(), InstanceKeyLess());
  } else {
    std::sort(nodes->begin(), nodes->end(), NameLess());
  }
  return absl::OkStatus();
}
}  
}  
#undef LOG_WARNING_AND_RETURN_IF_ERROR