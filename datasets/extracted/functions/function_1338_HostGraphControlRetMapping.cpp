#include "tensorflow/compiler/jit/extract_outside_compilation_pass.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include "tensorflow/compiler/jit/encapsulate_util.h"
#include "tensorflow/compiler/tf2xla/side_effect_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/dump_graph.h"
namespace tensorflow {
namespace {
std::optional<string> HostGraphControlRetMapping(const Node* n) {
  if (HasNodeAttr(n->def(), kXlaHasHostTransferAttrName)) {
    return n->name();
  }
  return std::nullopt;
}
absl::StatusOr<Node*> AddHostComputeKeyPlaceholder(
    const string& xla_cluster_name, Graph* g) {
  NodeDef key_def;
  NodeDefBuilder builder(absl::StrCat(xla_cluster_name, "_key_placeholder"),
                         "Placeholder");
  builder.Attr("dtype", DT_STRING);
  builder.Attr("shape", PartialTensorShape({2}));
  builder.Attr("_host_compute_call_node", xla_cluster_name);
  Status s = builder.Finalize(&key_def);
  if (!s.ok()) return s;
  Node* n = g->AddNode(key_def, &s);
  if (!s.ok()) return s;
  return n;
}
bool IsKeyPlaceholderNode(const Node& n) {
  return n.type_string() == "Placeholder" &&
         absl::EndsWith(n.name(), "_key_placeholder");
}
std::vector<Node*> GatherNodesWithType(const Graph& g, const string& type) {
  std::vector<Node*> result;
  for (Node* n : g.nodes()) {
    if (n->type_string() == type) {
      result.push_back(n);
    }
  }
  return result;
}
Status GetArgDataTypes(const std::vector<Node*>& arg_nodes,
                       std::vector<DataType>* recv_at_host_dtypes) {
  recv_at_host_dtypes->resize(arg_nodes.size(), DT_INVALID);
  for (auto* n : arg_nodes) {
    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
    DataType dtype;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "T", &dtype));
    (*recv_at_host_dtypes)[index] = dtype;
  }
  for (int i = 0, end = recv_at_host_dtypes->size(); i < end; i++) {
    if ((*recv_at_host_dtypes)[i] == DT_INVALID) {
      return errors::Internal("Cannot get datatype for input ", i);
    }
  }
  return absl::OkStatus();
}
absl::StatusOr<Node*> BuildRecvAtHostNode(
    Graph* g, const string& oc_cluster_name,
    const std::vector<DataType>& recv_at_host_dtypes, Node* key_placeholder) {
  NodeDefBuilder recv_at_host_builder(
      absl::StrCat("outside_compilation_", oc_cluster_name, "_recv"),
      "_XlaRecvAtHost");
  NodeDef recv_at_host_def;
  recv_at_host_builder.Attr("Toutputs", recv_at_host_dtypes);
  AttrValue device_ordinal_value;
  device_ordinal_value.set_placeholder("_device_ordinal");
  recv_at_host_builder.Attr("device_ordinal", device_ordinal_value);
  recv_at_host_builder.Attr(
      "key", absl::StrCat("host_compute_channel_", oc_cluster_name));
  recv_at_host_builder.Attr(kXlaHasHostTransferAttrName, true);
  recv_at_host_builder.Input(key_placeholder->name(), 0, DT_STRING);
  TF_RETURN_IF_ERROR(recv_at_host_builder.Finalize(&recv_at_host_def));
  TF_ASSIGN_OR_RETURN(Node * recv_at_host_node, g->AddNode(recv_at_host_def));
  return recv_at_host_node;
}
absl::StatusOr<Node*> ReplaceArgNodesWithRecvAtHostNode(
    Graph* g, const string& oc_cluster_name,
    std::vector<DataType>* recv_at_host_dtypes, Node* key_placeholder) {
  std::vector<Node*> arg_nodes = GatherNodesWithType(*g, "_Arg");
  TF_RETURN_IF_ERROR(GetArgDataTypes(arg_nodes, recv_at_host_dtypes));
  TF_ASSIGN_OR_RETURN(
      Node * recv_at_host_node,
      BuildRecvAtHostNode(g, oc_cluster_name, *recv_at_host_dtypes,
                          key_placeholder));
  for (auto* n : arg_nodes) {
    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
    std::vector<OutEdgeInfo> out_edge_info;
    out_edge_info.reserve(n->out_edges().size());
    for (auto edge : n->out_edges()) {
      out_edge_info.push_back(
          {edge->dst(), edge->src_output(), edge->dst_input()});
    }
    g->RemoveNode(n);
    for (const OutEdgeInfo& edge : out_edge_info) {
      if (edge.dst_input == Graph::kControlSlot) {
        g->AddControlEdge(recv_at_host_node, edge.dst);
      } else {
        g->AddEdge(recv_at_host_node, index, edge.dst, edge.dst_input);
      }
    }
    for (int i = 0, end = out_edge_info.size(); i < end; i++) {
      const OutEdgeInfo edge = out_edge_info[i];
      if (edge.dst_input == Graph::kControlSlot) {
        continue;
      }
      Node* dst = edge.dst;
      NodeDef new_def = dst->def();
      *new_def.mutable_input(edge.dst_input) =
          absl::StrCat(recv_at_host_node->name(), ":", index);
      TF_ASSIGN_OR_RETURN(Node * dst_replace, ReplaceNode(g, dst, new_def));
      for (int j = i + 1, end = out_edge_info.size(); j < end; j++) {
        if (out_edge_info[j].dst == dst) {
          out_edge_info[j].dst = dst_replace;
        }
      }
    }
  }
  g->AddEdge(key_placeholder, 0, recv_at_host_node, 0);
  return recv_at_host_node;
}
Status GetRetDataTypes(const std::vector<Node*>& ret_nodes,
                       std::vector<DataType>* send_from_host_dtypes) {
  send_from_host_dtypes->resize(ret_nodes.size(), DT_INVALID);
  for (auto* n : ret_nodes) {
    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
    DataType dtype;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "T", &dtype));
    (*send_from_host_dtypes)[index] = dtype;
  }
  for (int i = 0, end = send_from_host_dtypes->size(); i < end; i++) {
    if ((*send_from_host_dtypes)[i] == DT_INVALID) {
      return errors::Internal("Cannot get datatype for output ", i);
    }
  }
  return absl::OkStatus();
}
absl::StatusOr<Node*> BuildSendFromHostNode(
    Graph* g, const string& oc_cluster_name,
    const std::vector<Node*>& ret_nodes,
    const std::vector<DataType>& send_from_host_dtypes, Node* key_placeholder) {
  NodeDefBuilder send_from_host_builder(
      absl::StrCat("outside_compilation_", oc_cluster_name, "_send"),
      "_XlaSendFromHost");
  NodeDef send_from_host_def;
  send_from_host_builder.Attr("Tinputs", send_from_host_dtypes);
  AttrValue device_ordinal_value;
  device_ordinal_value.set_placeholder("_device_ordinal");
  send_from_host_builder.Attr("device_ordinal", device_ordinal_value);
  send_from_host_builder.Attr(
      "key", absl::StrCat("host_compute_channel_", oc_cluster_name));
  send_from_host_builder.Attr(kXlaHasHostTransferAttrName, true);
  std::vector<NodeDefBuilder::NodeOut> inputs(send_from_host_dtypes.size());
  for (auto* n : ret_nodes) {
    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
    const int num_dtypes = send_from_host_dtypes.size();
    if (index < 0 || index >= num_dtypes) {
      return errors::Internal("Invalid _Retval index: ", index);
    }
    for (auto edge : n->in_edges()) {
      inputs[index] =
          NodeDefBuilder::NodeOut{edge->src()->name(), edge->src_output(),
                                  edge->src()->output_type(edge->src_output())};
    }
  }
  send_from_host_builder.Input(inputs);
  send_from_host_builder.Input(key_placeholder->name(), 0, DT_STRING);
  TF_RETURN_IF_ERROR(send_from_host_builder.Finalize(&send_from_host_def));
  TF_ASSIGN_OR_RETURN(Node * send_from_host_node,
                      g->AddNode(send_from_host_def));
  return send_from_host_node;
}
absl::StatusOr<Node*> ReplaceRetNodesWithSendFromHostNode(
    Graph* g, const string& oc_cluster_name,
    std::vector<DataType>* send_from_host_dtypes, Node* key_placeholder) {
  std::vector<Node*> ret_nodes = GatherNodesWithType(*g, "_Retval");
  TF_RETURN_IF_ERROR(GetRetDataTypes(ret_nodes, send_from_host_dtypes));
  TF_ASSIGN_OR_RETURN(
      Node * send_from_host_node,
      BuildSendFromHostNode(g, oc_cluster_name, ret_nodes,
                            *send_from_host_dtypes, key_placeholder));
  for (auto* n : ret_nodes) {
    int index;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
    for (auto edge : n->in_edges()) {
      if (edge->src_output() == Graph::kControlSlot) {
        g->AddControlEdge(edge->src(), send_from_host_node);
      } else {
        g->AddEdge(edge->src(), edge->src_output(), send_from_host_node, index);
      }
    }
    g->RemoveNode(n);
  }
  g->AddEdge(key_placeholder, 0, send_from_host_node,
             send_from_host_dtypes->size());
  return send_from_host_node;
}
std::optional<std::vector<PartialTensorShape>> GetInferredInputShapes(
    int num_inputs, Node* send_from_host_node) {
  std::vector<PartialTensorShape> results(num_inputs);
  for (int i = 0; i < num_inputs; i++) {
    const Edge* e;
    if (!send_from_host_node->input_edge(i, &e).ok()) {
      return std::nullopt;
    }
    std::vector<PartialTensorShape> shapes;
    if (!GetNodeAttr(e->src()->attrs(), kXlaInferredShapesAttrName, &shapes)
             .ok()) {
      return std::nullopt;
    }
    const PartialTensorShape shape = shapes[e->src_output()];
    if (!shape.IsFullyDefined()) {
      return std::nullopt;
    }
    results[e->dst_input()] = shape;
  }
  return results;
}
string host_compute_node_name(const string& original_oc_name) {
  return absl::StrCat("outside_compilation_", original_oc_name,
                      "_host_compute");
}
absl::StatusOr<NodeDef> BuildXlaHostComputeNodeDef(
    const Node* call_node, const std::map<string, int>& host_compute_core,
    const absl::flat_hash_map<string, std::vector<string>>& cluster_deps) {
  string original_oc_name;
  TF_RETURN_IF_ERROR(GetNodeAttr(
      call_node->attrs(), "_outside_compilation_subgraph", &original_oc_name));
  NodeDefBuilder host_compute_builder(host_compute_node_name(original_oc_name),
                                      "XlaHostCompute");
  host_compute_builder.Attr(kXlaOriginalOutsideCompilationNodeName,
                            host_compute_builder.node_name());
  for (const auto& attr : call_node->attrs()) {
    host_compute_builder.Attr(attr.first, attr.second);
  }
  const auto iter = host_compute_core.find(original_oc_name);
  if (iter != host_compute_core.end()) {
    int core = iter->second;
    host_compute_builder.Attr("tpu_core", core);
  }
  std::vector<string> xla_token_input_nodes;
  xla_token_input_nodes.emplace_back(kXlaTokenArgNodeName);
  auto cluster_deps_it = cluster_deps.find(original_oc_name);
  if (cluster_deps_it != cluster_deps.end()) {
    for (const auto& dep : cluster_deps_it->second) {
      xla_token_input_nodes.emplace_back(host_compute_node_name(dep));
    }
  }
  host_compute_builder.Attr(kXlaTokenInputNodesAttrName, xla_token_input_nodes);
  std::vector<DataType> input_dtypes;
  TF_RETURN_IF_ERROR(GetNodeAttr(call_node->attrs(), "Tinputs", &input_dtypes));
  std::vector<NodeDefBuilder::NodeOut> inputs(input_dtypes.size());
  for (auto e : call_node->in_edges()) {
    if (e->IsControlEdge()) {
      continue;
    }
    const int input_dtypes_size = input_dtypes.size();
    if (e->dst_input() < 0 || e->dst_input() >= input_dtypes_size) {
      return errors::Internal("Invalid dst_input: ", e->dst_input());
    }
    inputs[e->dst_input()] = NodeDefBuilder::NodeOut{
        e->src()->name(), e->src_output(), input_dtypes[e->dst_input()]};
  }
  host_compute_builder.Input(inputs);
  NodeDef new_def;
  TF_RETURN_IF_ERROR(host_compute_builder.Finalize(&new_def));
  return new_def;
}
TF_ATTRIBUTE_NOINLINE absl::StatusOr<Node*> ReplaceOutsideCompilationCallNode(
    Graph* g, Node* call_node, const std::map<string, int>& host_compute_core,
    const absl::flat_hash_map<string, std::vector<string>>& cluster_deps) {
  TF_ASSIGN_OR_RETURN(
      NodeDef node_def,
      BuildXlaHostComputeNodeDef(call_node, host_compute_core, cluster_deps));
  TF_ASSIGN_OR_RETURN(Node * host_compute_node,
                      ReplaceNode(g, call_node, node_def));
  VLOG(4) << "Added HostCompute node: " << host_compute_node->DebugString();
  return host_compute_node;
}
Status ResetDeviceOrdinalToPlaceholderValue(Graph* g) {
  AttrValue device_ordinal_value;
  device_ordinal_value.set_placeholder("_device_ordinal");
  for (Node* n : g->nodes()) {
    if (!HasNodeAttr(n->def(), kXlaHasHostTransferAttrName)) {
      continue;
    }
    if (n->type_string() == "_XlaRecvAtHost" ||
        n->type_string() == "_XlaSendFromHost") {
      n->ClearAttr("device_ordinal");
      n->AddAttr("device_ordinal", device_ordinal_value);
    } else if (n->IsIfNode()) {
      for (const string& attr_name :
           std::vector<string>{"then_branch", "else_branch"}) {
        NameAttrList branch_func;
        TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), attr_name, &branch_func));
        (*branch_func.mutable_attr())["_device_ordinal"] = device_ordinal_value;
        n->ClearAttr(attr_name);
        n->AddAttr(attr_name, branch_func);
      }
    } else if (n->IsWhileNode()) {
      for (const string& attr_name : std::vector<string>{"cond", "body"}) {
        NameAttrList branch_func;
        TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), attr_name, &branch_func));
        (*branch_func.mutable_attr())["_device_ordinal"] = device_ordinal_value;
        n->ClearAttr(attr_name);
        n->AddAttr(attr_name, branch_func);
      }
    } else if (HasNodeAttr(n->def(), "_device_ordinal")) {
      n->ClearAttr("_device_ordinal");
      n->AddAttr("_device_ordinal", device_ordinal_value);
    } else {
      return errors::Internal("Unknown node marked with ",
                              kXlaHasHostTransferAttrName, ": ",
                              n->DebugString());
    }
  }
  return absl::OkStatus();
}
bool HasLiftedArgs(const FunctionDef& function_def) {
  return absl::c_any_of(function_def.node_def(), [](const NodeDef& node_def) {
    return (node_def.op() == "Placeholder" &&
            node_def.attr().find(kXlaLiftedArgOutsideCompilationAttrName) !=
                node_def.attr().end());
  });
}
absl::StatusOr<std::vector<std::pair<Node*, Node*>>>
LiftedArgsAndOutsideCompilationNodesInFunctionBody(
    const FunctionBody& function_body,
    const std::unordered_map<string, Node*>& outside_compilation_attr_to_node) {
  std::vector<std::pair<Node*, Node*>>
      lifted_arg_nodes_and_outside_compilation_nodes;
  for (Node* n : function_body.graph->op_nodes()) {
    string oc_cluster;
    if (n->type_string() == "Placeholder" &&
        GetNodeAttr(n->def(), kXlaLiftedArgOutsideCompilationAttrName,
                    &oc_cluster)
            .ok()) {
      TF_RET_CHECK(outside_compilation_attr_to_node.find(oc_cluster) !=
                   outside_compilation_attr_to_node.end());
      lifted_arg_nodes_and_outside_compilation_nodes.emplace_back(
          n, outside_compilation_attr_to_node.at(oc_cluster));
    }
  }
  return lifted_arg_nodes_and_outside_compilation_nodes;
}
absl::StatusOr<std::vector<DataType>> UpdateTypesAttribute(
    const std::vector<std::pair<Node*, Node*>>&
        lifted_arg_nodes_and_outside_compilation_nodes,
    const string& type_attr_name, Node* n) {
  std::vector<DataType> data_types;
  data_types.reserve(lifted_arg_nodes_and_outside_compilation_nodes.size());
  TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), type_attr_name, &data_types));
  for (auto pair : lifted_arg_nodes_and_outside_compilation_nodes) {
    Node* outside_compilation_node = pair.second;
    DataType data_type;
    TF_RET_CHECK(outside_compilation_node->IsIdentity() ||
                 outside_compilation_node->type_string() == "Placeholder");
    if (outside_compilation_node->IsIdentity()) {
      TF_RETURN_IF_ERROR(
          GetNodeAttr(outside_compilation_node->def(), "T", &data_type));
    } else {
      TF_RETURN_IF_ERROR(
          GetNodeAttr(outside_compilation_node->def(), "dtype", &data_type));
    }
    data_types.push_back(data_type);
  }
  n->ClearAttr(type_attr_name);
  n->AddAttr(type_attr_name, data_types);
  return data_types;
}
void AddEdgesFromOutsideCompilationNodes(
    const int original_arg_count, const int arg_to_input_edge_offset,
    const std::vector<DataType>& data_types,
    const std::vector<Node*>& outside_compilation_nodes, Graph* g, Node* n) {
  for (int i = original_arg_count, end = data_types.size(); i < end; i++) {
    Node* outside_compilation_node =
        outside_compilation_nodes[i - original_arg_count];
    g->AddEdge(outside_compilation_node, 0, n, i + arg_to_input_edge_offset);
  }
}
absl::StatusOr<Node*> AddOutsideCompilationInputArgToFunctionBody(
    const FunctionBody& function_body, const int arg_idx,
    const DataType& data_type) {
  NodeDefBuilder arg_builder(absl::StrCat("arg_", arg_idx), "_Arg");
  arg_builder.Attr("T", data_type);
  arg_builder.Attr("index", arg_idx);
  NodeDef arg_def;
  TF_RETURN_IF_ERROR(arg_builder.Finalize(&arg_def));
  TF_ASSIGN_OR_RETURN(Node * arg_node, function_body.graph->AddNode(arg_def));
  return arg_node;
}
Status AddMatchingRetvalNode(const FunctionBody& function_body,
                             const int arg_idx, const DataType& data_type,
                             Node* arg_node) {
  NodeDefBuilder ret_builder(absl::StrCat("ret_", arg_idx), "_Retval");
  ret_builder.Attr("T", data_type);
  ret_builder.Attr("index", arg_idx);
  ret_builder.Input(arg_node->name(), 0, data_type);
  NodeDef ret_def;
  TF_RETURN_IF_ERROR(ret_builder.Finalize(&ret_def));
  TF_ASSIGN_OR_RETURN(Node * ret_node, function_body.graph->AddNode(ret_def));
  function_body.graph->AddEdge(arg_node, 0, ret_node, 0);
  return absl::OkStatus();
}
void ReplaceLiftedArgNodePlaceholderWithArg(
    const FunctionBody& function_body, const int original_arg_count,
    const int arg_idx, const std::vector<Node*>& lifted_arg_nodes,
    Node* arg_node) {
  Node* lifted_arg_node = lifted_arg_nodes[arg_idx - original_arg_count];
  if (!lifted_arg_node) {
    return;
  }
  for (const Edge* e : lifted_arg_node->out_edges()) {
    if (e->IsControlEdge()) {
      function_body.graph->AddControlEdge(arg_node, e->dst());
    } else {
      function_body.graph->AddEdge(arg_node, 0, e->dst(), e->dst_input());
    }
  }
  function_body.graph->RemoveNode(lifted_arg_node);
}
Status AddFunctionWithNewName(const std::string& new_name,
                              const std::string& func_attr_name,
                              const FunctionDef& function_def,
                              NameAttrList* func_attr, Node* callsite_node,
                              FunctionLibraryDefinition* fld) {
  TF_RETURN_IF_ERROR(fld->AddFunctionDef(function_def));
  func_attr->set_name(new_name);
  callsite_node->ClearAttr(func_attr_name);
  callsite_node->AddAttr(func_attr_name, *func_attr);
  return absl::OkStatus();
}
Status PostprocessLiftedArgsForWhile(
    const std::unordered_map<string, Node*>& outside_compilation_attr_to_node,
    Graph* g, Node* n, FunctionLibraryDefinition* fld) {
  TF_RET_CHECK(n->IsWhileNode());
  NameAttrList body_func;
  TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "body", &body_func));
  const FunctionDef* body_function_def = fld->Find(body_func.name());
  TF_RET_CHECK(body_function_def);
  if (!HasLiftedArgs(*body_function_def)) {
    return absl::OkStatus();
  }
  std::unique_ptr<FunctionBody> body_function_body;
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*body_function_def,
                                             AttrSlice(&body_func.attr()), fld,
                                             &body_function_body));
  int original_arg_count = body_function_body->arg_nodes.size();
  TF_ASSIGN_OR_RETURN(
      auto lifted_arg_nodes_and_outside_compilation_nodes,
      LiftedArgsAndOutsideCompilationNodesInFunctionBody(
          *body_function_body, outside_compilation_attr_to_node));
  TF_ASSIGN_OR_RETURN(
      std::vector<DataType> data_types,
      UpdateTypesAttribute(lifted_arg_nodes_and_outside_compilation_nodes, "T",
                           n));
  std::vector<Node*> outside_compilation_nodes;
  outside_compilation_nodes.reserve(
      lifted_arg_nodes_and_outside_compilation_nodes.size());
  std::transform(
      lifted_arg_nodes_and_outside_compilation_nodes.begin(),
      lifted_arg_nodes_and_outside_compilation_nodes.end(),
      std::back_inserter(outside_compilation_nodes),
      [](const std::pair<Node*, Node*>& pair) { return pair.second; });
  AddEdgesFromOutsideCompilationNodes(original_arg_count,
                                      0,
                                      data_types, outside_compilation_nodes, g,
                                      n);
  std::vector<Node*> lifted_arg_nodes;
  lifted_arg_nodes.reserve(
      lifted_arg_nodes_and_outside_compilation_nodes.size());
  std::transform(
      lifted_arg_nodes_and_outside_compilation_nodes.begin(),
      lifted_arg_nodes_and_outside_compilation_nodes.end(),
      std::back_inserter(lifted_arg_nodes),
      [](const std::pair<Node*, Node*>& pair) { return pair.first; });
  for (int i = original_arg_count, end = data_types.size(); i < end; i++) {
    TF_ASSIGN_OR_RETURN(Node * arg_node,
                        AddOutsideCompilationInputArgToFunctionBody(
                            *body_function_body, i, data_types[i]));
    TF_RETURN_IF_ERROR(
        AddMatchingRetvalNode(*body_function_body, i, data_types[i], arg_node));
    ReplaceLiftedArgNodePlaceholderWithArg(
        *body_function_body, original_arg_count, i, lifted_arg_nodes, arg_node);
  }
  const auto new_body_function_name =
      fld->UniqueFunctionName(absl::StrCat(body_func.name(), "_lifted_arg_"));
  FunctionDef rewritten_body_function_def;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(
      *body_function_body->graph, new_body_function_name,
      HostGraphControlRetMapping, &rewritten_body_function_def));
  TF_RETURN_IF_ERROR(AddFunctionWithNewName(new_body_function_name, "body",
                                            rewritten_body_function_def,
                                            &body_func, n, fld));
  NameAttrList cond_func;
  TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "cond", &cond_func));
  const FunctionDef* cond_function_def = fld->Find(cond_func.name());
  TF_RET_CHECK(cond_function_def);
  std::unique_ptr<FunctionBody> cond_function_body;
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*cond_function_def,
                                             AttrSlice(&cond_func.attr()), fld,
                                             &cond_function_body));
  for (int i = original_arg_count, end = data_types.size(); i < end; i++) {
    absl::StatusOr<Node*> arg_node_or =
        AddOutsideCompilationInputArgToFunctionBody(*cond_function_body, i,
                                                    data_types[i]);
    TF_RETURN_IF_ERROR(arg_node_or.status());
  }
  const auto new_cond_function_name =
      fld->UniqueFunctionName(absl::StrCat(cond_func.name(), "_lifted_arg_"));
  FunctionDef rewritten_cond_function_def;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(
      *cond_function_body->graph, new_cond_function_name,
      HostGraphControlRetMapping, &rewritten_cond_function_def));
  TF_RETURN_IF_ERROR(AddFunctionWithNewName(new_cond_function_name, "cond",
                                            rewritten_cond_function_def,
                                            &cond_func, n, fld));
  return absl::OkStatus();
}
Status PostprocessLiftedArgsForIf(
    const std::unordered_map<string, Node*>& outside_compilation_attr_to_node,
    Graph* g, Node* n, FunctionLibraryDefinition* fld) {
  TF_RET_CHECK(n->IsIfNode());
  NameAttrList then_branch_func;
  TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "then_branch", &then_branch_func));
  const FunctionDef* then_branch_function_def =
      fld->Find(then_branch_func.name());
  TF_RET_CHECK(then_branch_function_def);
  NameAttrList else_branch_func;
  TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "else_branch", &else_branch_func));
  const FunctionDef* else_branch_function_def =
      fld->Find(else_branch_func.name());
  TF_RET_CHECK(else_branch_function_def);
  if (!HasLiftedArgs(*then_branch_function_def) &&
      !HasLiftedArgs(*else_branch_function_def)) {
    return absl::OkStatus();
  }
  std::unique_ptr<FunctionBody> then_branch_function_body;
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
      *then_branch_function_def, AttrSlice(&then_branch_func.attr()), fld,
      &then_branch_function_body));
  std::unique_ptr<FunctionBody> else_branch_function_body;
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
      *else_branch_function_def, AttrSlice(&else_branch_func.attr()), fld,
      &else_branch_function_body));
  int original_arg_count = then_branch_function_body->arg_nodes.size();
  TF_ASSIGN_OR_RETURN(
      auto then_branch_lifted_arg_nodes_and_outside_compilation_nodes,
      LiftedArgsAndOutsideCompilationNodesInFunctionBody(
          *then_branch_function_body, outside_compilation_attr_to_node));
  TF_ASSIGN_OR_RETURN(
      auto else_branch_lifted_arg_nodes_and_outside_compilation_nodes,
      LiftedArgsAndOutsideCompilationNodesInFunctionBody(
          *else_branch_function_body, outside_compilation_attr_to_node));
  std::vector<Node*> outside_compilation_nodes;
  std::vector<Node*> then_branch_lifted_arg_nodes;
  outside_compilation_nodes.reserve(
      then_branch_lifted_arg_nodes_and_outside_compilation_nodes.size());
  then_branch_lifted_arg_nodes.reserve(
      then_branch_lifted_arg_nodes_and_outside_compilation_nodes.size());
  for (const auto& pair :
       then_branch_lifted_arg_nodes_and_outside_compilation_nodes) {
    outside_compilation_nodes.push_back(pair.second);
    then_branch_lifted_arg_nodes.push_back(pair.first);
  }
  for (const auto& pair :
       else_branch_lifted_arg_nodes_and_outside_compilation_nodes) {
    if (std::find(outside_compilation_nodes.begin(),
                  outside_compilation_nodes.end(),
                  pair.second) == outside_compilation_nodes.end()) {
      outside_compilation_nodes.push_back(pair.second);
      then_branch_lifted_arg_nodes.push_back(nullptr);
    }
  }
  std::vector<Node*> else_branch_lifted_arg_nodes(
      outside_compilation_nodes.size());
  for (const auto& pair :
       else_branch_lifted_arg_nodes_and_outside_compilation_nodes) {
    auto iter = std::find(outside_compilation_nodes.begin(),
                          outside_compilation_nodes.end(), pair.second);
    TF_RET_CHECK(iter != outside_compilation_nodes.end());
    int index = iter - outside_compilation_nodes.begin();
    else_branch_lifted_arg_nodes[index] = pair.first;
  }
  std::vector<DataType> data_types;
  data_types.reserve(outside_compilation_nodes.size());
  TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "Tin", &data_types));
  for (Node* n : outside_compilation_nodes) {
    data_types.push_back(n->output_type(0));
  }
  n->ClearAttr("Tin");
  n->AddAttr("Tin", data_types);
  AddEdgesFromOutsideCompilationNodes(original_arg_count,
                                      1,
                                      data_types, outside_compilation_nodes, g,
                                      n);
  for (int i = original_arg_count, end = data_types.size(); i < end; ++i) {
    TF_ASSIGN_OR_RETURN(Node * then_branch_arg_node,
                        AddOutsideCompilationInputArgToFunctionBody(
                            *then_branch_function_body, i, data_types[i]));
    ReplaceLiftedArgNodePlaceholderWithArg(
        *then_branch_function_body, original_arg_count, i,
        then_branch_lifted_arg_nodes, then_branch_arg_node);
    TF_ASSIGN_OR_RETURN(Node * else_branch_arg_node,
                        AddOutsideCompilationInputArgToFunctionBody(
                            *else_branch_function_body, i, data_types[i]));
    ReplaceLiftedArgNodePlaceholderWithArg(
        *else_branch_function_body, original_arg_count, i,
        else_branch_lifted_arg_nodes, else_branch_arg_node);
  }
  const auto new_then_function_name = fld->UniqueFunctionName(
      absl::StrCat(then_branch_func.name(), "_lifted_arg_"));
  FunctionDef rewritten_then_branch_function_def;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(
      *then_branch_function_body->graph, new_then_function_name,
      HostGraphControlRetMapping, &rewritten_then_branch_function_def));
  TF_RETURN_IF_ERROR(AddFunctionWithNewName(
      new_then_function_name, "then_branch", rewritten_then_branch_function_def,
      &then_branch_func, n, fld));
  const auto new_else_function_name = fld->UniqueFunctionName(
      absl::StrCat(else_branch_func.name(), "_lifted_arg_"));
  FunctionDef rewritten_else_branch_function_def;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(
      *else_branch_function_body->graph, new_else_function_name,
      HostGraphControlRetMapping, &rewritten_else_branch_function_def));
  TF_RETURN_IF_ERROR(AddFunctionWithNewName(
      new_else_function_name, "else_branch", rewritten_else_branch_function_def,
      &else_branch_func, n, fld));
  return absl::OkStatus();
}
Status PostprocessLiftedArgsForCall(
    const std::unordered_map<string, Node*>& outside_compilation_attr_to_node,
    Graph* g, Node* n, FunctionLibraryDefinition* fld) {
  const FunctionDef* fdef = fld->Find(n->type_string());
  TF_RET_CHECK(fdef);
  if (!HasLiftedArgs(*fdef)) {
    return absl::OkStatus();
  }
  std::unique_ptr<FunctionBody> fbody;
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*fdef, n->attrs(), fld, &fbody));
  int original_arg_count = fbody->arg_nodes.size();
  TF_ASSIGN_OR_RETURN(auto lifted_arg_nodes_and_outside_compilation_nodes,
                      LiftedArgsAndOutsideCompilationNodesInFunctionBody(
                          *fbody, outside_compilation_attr_to_node));
  std::vector<DataType> data_types(n->input_types().begin(),
                                   n->input_types().end());
  for (auto pair : lifted_arg_nodes_and_outside_compilation_nodes) {
    Node* outside_compilation_node = pair.second;
    DataType data_type;
    TF_RET_CHECK(outside_compilation_node->IsIdentity() ||
                 outside_compilation_node->type_string() == "Placeholder");
    if (outside_compilation_node->IsIdentity()) {
      TF_RETURN_IF_ERROR(
          GetNodeAttr(outside_compilation_node->def(), "T", &data_type));
    } else {
      TF_RETURN_IF_ERROR(
          GetNodeAttr(outside_compilation_node->def(), "dtype", &data_type));
    }
    data_types.push_back(data_type);
  }
  std::vector<Node*> lifted_arg_nodes;
  lifted_arg_nodes.reserve(
      lifted_arg_nodes_and_outside_compilation_nodes.size());
  std::transform(
      lifted_arg_nodes_and_outside_compilation_nodes.begin(),
      lifted_arg_nodes_and_outside_compilation_nodes.end(),
      std::back_inserter(lifted_arg_nodes),
      [](const std::pair<Node*, Node*>& pair) { return pair.first; });
  for (int i = original_arg_count, end = data_types.size(); i < end; ++i) {
    TF_ASSIGN_OR_RETURN(
        Node * arg_node,
        AddOutsideCompilationInputArgToFunctionBody(*fbody, i, data_types[i]));
    ReplaceLiftedArgNodePlaceholderWithArg(*fbody, original_arg_count, i,
                                           lifted_arg_nodes, arg_node);
  }
  FunctionDef rewritten_fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(*fbody->graph, n->type_string(),
                                        HostGraphControlRetMapping,
                                        &rewritten_fdef));
  const auto new_function_name =
      fld->UniqueFunctionName(absl::StrCat(n->type_string(), "_lifted_arg_"));
  rewritten_fdef.mutable_signature()->set_name(new_function_name);
  TF_RETURN_IF_ERROR(fld->AddFunctionDef(rewritten_fdef));
  NodeDef node_def = n->def();
  *node_def.mutable_op() = new_function_name;
  for (int i = original_arg_count, end = data_types.size(); i < end; i++) {
    Node* outside_compilation_node =
        lifted_arg_nodes_and_outside_compilation_nodes[i - original_arg_count]
            .second;
    node_def.add_input(absl::StrCat(outside_compilation_node->name(), ":", 0));
  }
  TF_ASSIGN_OR_RETURN(n, ReplaceNode(g, n, node_def));
  std::vector<Node*> outside_compilation_nodes;
  outside_compilation_nodes.reserve(
      lifted_arg_nodes_and_outside_compilation_nodes.size());
  std::transform(
      lifted_arg_nodes_and_outside_compilation_nodes.begin(),
      lifted_arg_nodes_and_outside_compilation_nodes.end(),
      std::back_inserter(outside_compilation_nodes),
      [](const std::pair<Node*, Node*>& pair) { return pair.second; });
  AddEdgesFromOutsideCompilationNodes(original_arg_count,
                                      0,
                                      data_types, outside_compilation_nodes, g,
                                      n);
  return absl::OkStatus();
}
absl::StatusOr<std::unordered_map<string, Node*>> OutsideCompilationAttrToNode(
    const Graph& g) {
  std::unordered_map<string, Node*> outside_compilation_attr_to_node;
  for (Node* n : g.op_nodes()) {
    bool is_lifted_arg;
    string outside_compilation_attr;
    if (TryGetNodeAttr(n->def(), kXlaIsLiftedArgAttrName, &is_lifted_arg) &&
        TryGetNodeAttr(n->def(), "_xla_outside_compilation",
                       &outside_compilation_attr)) {
      TF_RET_CHECK(is_lifted_arg);
      TF_RET_CHECK(n->IsIdentity() || n->type_string() == "Placeholder");
      outside_compilation_attr_to_node[outside_compilation_attr] = n;
    }
  }
  return outside_compilation_attr_to_node;
}
Status PostprocessLiftedArgs(Graph* g, FunctionLibraryDefinition* fld) {
  TF_ASSIGN_OR_RETURN(auto outside_compilation_attr_to_node,
                      OutsideCompilationAttrToNode(*g));
  std::vector<Node*> call_nodes;
  for (Node* n : g->op_nodes()) {
    if (!HasNodeAttr(n->def(), kXlaHasHostTransferAttrName)) {
      continue;
    }
    if (n->IsWhileNode()) {
      TF_RETURN_IF_ERROR(PostprocessLiftedArgsForWhile(
          outside_compilation_attr_to_node, g, n, fld));
    }
    if (n->IsIfNode()) {
      TF_RETURN_IF_ERROR(PostprocessLiftedArgsForIf(
          outside_compilation_attr_to_node, g, n, fld));
    }
    if (fld->Contains(n->type_string())) {
      call_nodes.push_back(n);
    }
  }
  for (Node* n : call_nodes) {
    TF_RETURN_IF_ERROR(PostprocessLiftedArgsForCall(
        outside_compilation_attr_to_node, g, n, fld));
  }
  return absl::OkStatus();
}
Status ConstructHostGraph(
    const string& xla_cluster_name, const string& outside_compilation_attr_name,
    const std::vector<string>& outside_compilation_host_graphs,
    FunctionLibraryDefinition* fld, std::unique_ptr<Graph>* host_graph) {
  host_graph->reset(new Graph(fld));
  NodeDefBuilder sequencer_builder(absl::StrCat(xla_cluster_name, "_sequencer"),
                                   "NoOp");
  sequencer_builder.Attr("_xla_host_transfer_sequencer", xla_cluster_name);
  NodeDef sequencer_def;
  TF_RETURN_IF_ERROR(sequencer_builder.Finalize(&sequencer_def));
  TF_ASSIGN_OR_RETURN(Node * sequencer, (*host_graph)->AddNode(sequencer_def));
  TF_ASSIGN_OR_RETURN(
      Node * key_placeholder,
      AddHostComputeKeyPlaceholder(xla_cluster_name, host_graph->get()));
  for (const string& host_func : outside_compilation_host_graphs) {
    VLOG(4) << "Expanding host graph " << host_func;
    AttrValue device_ordinal_attr;
    device_ordinal_attr.set_i(0);
    protobuf::Map<string, AttrValue> attrs;
    attrs["_device_ordinal"] = device_ordinal_attr;
    std::unique_ptr<FunctionBody> host_fbody;
    const FunctionDef* host_fdef = fld->Find(host_func);
    TF_RET_CHECK(host_fdef);
    TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*host_fdef, AttrSlice(&attrs),
                                               fld, &host_fbody));
    FixupSourceAndSinkEdges(host_fbody->graph);
    std::map<const Node*, Node*> node_map;
    node_map[host_fbody->graph->source_node()] = (*host_graph)->source_node();
    node_map[host_fbody->graph->sink_node()] = (*host_graph)->sink_node();
    Status s;
    ReverseDFS(
        *host_fbody->graph, nullptr,
        [&](const Node* n) {
          if (!s.ok()) {
            return;
          }
          Node* copy;
          if (node_map.find(n) != node_map.end()) {
            copy = node_map.at(n);
          } else if (IsKeyPlaceholderNode(*n)) {
            copy = key_placeholder;
            node_map[n] = copy;
          } else {
            NodeDef copy_def = n->def();
            copy_def.clear_device();
            copy = (*host_graph)->AddNode(copy_def, &s);
            if (!s.ok()) {
              return;
            }
            node_map[n] = copy;
          }
          for (auto e : n->in_edges()) {
            if (node_map.find(e->src()) == node_map.end()) {
              s = errors::Internal("Cannot find node image for ",
                                   e->src()->DebugString());
              return;
            }
            (*host_graph)
                ->AddEdge(node_map[e->src()], e->src_output(), copy,
                          e->dst_input());
          }
          if (HasNodeAttr(copy->def(), kXlaHasHostTransferAttrName)) {
            (*host_graph)->AddControlEdge(copy, sequencer);
          }
        },
        NodeComparatorID());
    if (!s.ok()) {
      return s;
    }
  }
  TF_RETURN_IF_ERROR(ResetDeviceOrdinalToPlaceholderValue(host_graph->get()));
  if (!sequencer->in_edges().empty()) {
    (*host_graph)->AddControlEdge(sequencer, (*host_graph)->sink_node());
  }
  PruneForReverseReachability(
      host_graph->get(),
      std::unordered_set<const Node*>{(*host_graph)->sink_node()});
  TF_RETURN_IF_ERROR(PostprocessEdgesBetweenOutsideCompilations(
      host_graph->get(), outside_compilation_attr_name));
  TF_RETURN_IF_ERROR(PostprocessLiftedArgs(host_graph->get(), fld));
  if (VLOG_IS_ON(4)) {
    DumpGraphToFile(absl::StrCat("extract_outside_compilation_host_graph_for_",
                                 xla_cluster_name),
                    **host_graph, fld);
  }
  return absl::OkStatus();
}
Status ExpandHostGraphIntoMainGraph(Graph* main_graph,
                                    FunctionLibraryDefinition* fld,
                                    const string& host_graph_func_name,
                                    Node* xla_computation_node,
                                    Node* pivot_node) {
  AttrValue device_ordinal_attr;
  device_ordinal_attr.set_i(0);
  protobuf::Map<string, AttrValue> attrs;
  attrs["_device_ordinal"] = device_ordinal_attr;
  std::unique_ptr<FunctionBody> fbody;
  const FunctionDef* host_graph_func = fld->Find(host_graph_func_name);
  TF_RET_CHECK(host_graph_func);
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*host_graph_func,
                                             AttrSlice(&attrs), fld, &fbody));
  Graph* host_graph = fbody->graph;
  FixupSourceAndSinkEdges(host_graph);
  std::map<const Node*, Node*> node_map;
  if (pivot_node) {
    node_map[host_graph->source_node()] = pivot_node;
  } else {
    node_map[host_graph->source_node()] = main_graph->source_node();
  }
  node_map[host_graph->sink_node()] = main_graph->sink_node();
  Status s = absl::OkStatus();
  auto copy_node_fn = [&](const Node* n) {
    if (!s.ok()) {
      return;
    }
    Node* copy;
    if (node_map.find(n) != node_map.end()) {
      copy = node_map.at(n);
    } else {
      NodeDef copy_def = n->def();
      copy = main_graph->AddNode(copy_def, &s);
      if (!s.ok()) {
        return;
      }
      node_map[n] = copy;
    }
    for (auto e : n->in_edges()) {
      if (node_map.find(e->src()) == node_map.end()) {
        s = errors::Internal("Cannot find node image for ",
                             e->src()->DebugString());
        return;
      }
      main_graph->AddEdge(node_map[e->src()], e->src_output(), copy,
                          e->dst_input());
    }
    if (copy->type_string() == "NoOp" &&
        HasNodeAttr(copy->def(), "_xla_host_transfer_sequencer")) {
      main_graph->AddControlEdge(copy, xla_computation_node);
    }
  };
  ReverseDFS(*host_graph, nullptr, copy_node_fn, NodeComparatorID());
  return s;
}
Status RewriteShapeInferenceGraph(const string& shape_inference_graph_name,
                                  Graph* host_graph, Node* pivot_node,
                                  FunctionLibraryDefinition* fld) {
  AttrValue device_ordinal_attr;
  device_ordinal_attr.set_i(0);
  protobuf::Map<string, AttrValue> attrs;
  attrs["_device_ordinal"] = device_ordinal_attr;
  std::unique_ptr<FunctionBody> fbody;
  const FunctionDef* shape_inference_graph =
      fld->Find(shape_inference_graph_name);
  TF_RET_CHECK(shape_inference_graph);
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*shape_inference_graph,
                                             AttrSlice(&attrs), fld, &fbody));
  Graph* g = fbody->graph;
  Node* send_from_host = nullptr;
  for (Node* n : g->nodes()) {
    if (n->type_string() == "_XlaSendFromHost") {
      send_from_host = n;
      break;
    }
  }
  if (!send_from_host) {
    return errors::Internal("Shape inference graph ",
                            shape_inference_graph_name,
                            " does not have _XlaSendFromHost node.");
  }
  Node* send_node_in_host_graph = nullptr;
  for (Node* n : host_graph->nodes()) {
    if (n->name() == send_from_host->name()) {
      send_node_in_host_graph = n;
      break;
    }
  }
  if (send_node_in_host_graph) {
    std::vector<Node*> nodes;
    nodes.reserve(g->num_op_nodes());
    for (Node* n : g->op_nodes()) {
      nodes.push_back(n);
    }
    for (Node* n : nodes) {
      g->RemoveNode(n);
    }
    Node* start_node = pivot_node ? pivot_node : host_graph->source_node();
    struct Visit {
      Node* n;
      bool is_exiting;
    };
    std::vector<Visit> stack{{send_node_in_host_graph, false}};
    std::map<Node*, Node*> node_map;
    node_map[host_graph->source_node()] = g->source_node();
    while (!stack.empty()) {
      Visit& curr = stack.back();
      if (curr.is_exiting) {
        if (node_map.find(curr.n) == node_map.end()) {
          Node* copy = g->CopyNode(curr.n);
          if (curr.n != start_node) {
            for (const Edge* e : curr.n->in_edges()) {
              auto node_iter = node_map.find(e->src());
              if (node_iter == node_map.end()) {
                return errors::Internal("Cannot find node image for ",
                                        e->src()->DebugString());
              }
              g->AddEdge(node_iter->second, e->src_output(), copy,
                         e->dst_input());
            }
          }
          node_map[curr.n] = copy;
        }
        stack.pop_back();
      } else {
        curr.is_exiting = true;
        if (curr.n != start_node) {
          for (const Edge* e : curr.n->in_edges()) {
            if (node_map.find(e->src()) != node_map.end()) {
              continue;
            }
            stack.push_back({e->src(), false});
          }
        }
      }
    }
    send_from_host = node_map[send_node_in_host_graph];
  } else {
  }
  for (auto e : g->edges()) {
    if (e->IsControlEdge()) {
      g->RemoveEdge(e);
    }
  }
  PruneForReverseReachability(g,
                              std::unordered_set<const Node*>{send_from_host});
  if (VLOG_IS_ON(4)) {
    DumpGraphToFile(shape_inference_graph_name, *g, fld);
  }
  FunctionDef fdef_replace;
  TF_RETURN_IF_ERROR(
      GraphToFunctionDef(*g, shape_inference_graph_name, &fdef_replace));
  TF_RETURN_IF_ERROR(
      fld->ReplaceFunction(shape_inference_graph_name, fdef_replace));
  return absl::OkStatus();
}
void SetMaximalSharding(NodeDefBuilder& node_builder) {
  xla::OpSharding sharding;
  sharding.set_type(xla::OpSharding::MAXIMAL);
  sharding.add_tile_assignment_dimensions(1);
  sharding.add_tile_assignment_devices(0);
  node_builder.Attr("_XlaSharding", sharding.SerializeAsString());
}
TF_ATTRIBUTE_NOINLINE absl::StatusOr<Node*> BuildSendIfPredNode(
    const string& name, const string& host_transfer_key, Node* pred_node,
    Graph* g) {
  NodeDefBuilder send_pred_builder(name, "XlaSendToHost");
  send_pred_builder.Attr("Tinput", DT_BOOL);
  send_pred_builder.Attr("key", absl::StrCat(host_transfer_key, "_dtoh_0"));
  send_pred_builder.Attr(kXlaTokenInputNodesAttrName,
                         std::vector<string>{kXlaTokenArgNodeName});
  send_pred_builder.Attr(kXlaOriginalOutsideCompilationNodeName, name);
  SetMaximalSharding(send_pred_builder);
  send_pred_builder.Input(pred_node->name(), 0, DT_BOOL);
  NodeDef send_pred_def;
  TF_RETURN_IF_ERROR(send_pred_builder.Finalize(&send_pred_def));
  TF_ASSIGN_OR_RETURN(Node * send_pred_node, g->AddNode(send_pred_def));
  g->AddEdge(pred_node, 0, send_pred_node, 0);
  return send_pred_node;
}
Status ReplaceKeyPlaceholderWithArgNode(const string& xla_cluster_name,
                                        const string& func_name,
                                        FunctionLibraryDefinition* fld) {
  AttrValue device_ordinal_attr;
  device_ordinal_attr.set_i(0);
  protobuf::Map<string, AttrValue> attrs;
  attrs["_device_ordinal"] = device_ordinal_attr;
  std::unique_ptr<FunctionBody> fbody;
  const FunctionDef* func = fld->Find(func_name);
  TF_RETURN_IF_ERROR(
      FunctionDefToBodyHelper(*func, AttrSlice(&attrs), fld, &fbody));
  Graph* g = fbody->graph;
  Node* key_placeholder = nullptr;
  for (Node* n : g->nodes()) {
    if (IsKeyPlaceholderNode(*n)) {
      key_placeholder = n;
      break;
    }
  }
  if (!key_placeholder) {
    TF_ASSIGN_OR_RETURN(key_placeholder,
                        AddHostComputeKeyPlaceholder(xla_cluster_name, g));
  }
  NodeDefBuilder arg_builder("key_arg", FunctionLibraryDefinition::kArgOp);
  arg_builder.Attr("T", DT_STRING);
  arg_builder.Attr("index", 0);
  NodeDef arg_def;
  TF_RETURN_IF_ERROR(arg_builder.Finalize(&arg_def));
  TF_RETURN_IF_ERROR(ReplaceNode(g, key_placeholder, arg_def).status());
  TF_RETURN_IF_ERROR(ResetDeviceOrdinalToPlaceholderValue(g));
  FunctionDef replace_fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(
      *g, func_name, HostGraphControlRetMapping, &replace_fdef));
  TF_RETURN_IF_ERROR(fld->ReplaceFunction(func_name, replace_fdef));
  return absl::OkStatus();
}
TF_ATTRIBUTE_NOINLINE Status BuildHostGraphForIfNode(
    const string& xla_cluster_attr_name,
    const string& outside_compilation_attr_name, const string& xla_cluster_name,
    const string& if_node_name, const string& host_transfer_key,
    const string& host_graph_func_name, FunctionLibraryDefinition* fld,
    const string& then_branch_host_func_name,
    const string& else_branch_host_func_name) {
  Graph host_graph(fld);
  string outside_compilation_name = absl::StrCat("oc_if_", if_node_name);
  AttrValue device_ordinal_value;
  device_ordinal_value.set_placeholder("_device_ordinal");
  TF_ASSIGN_OR_RETURN(
      Node * key_placeholder,
      AddHostComputeKeyPlaceholder(xla_cluster_name, &host_graph));
  NodeDefBuilder recv_pred_builder(
      absl::StrCat("recv_oc_if_pred_", if_node_name), "_XlaRecvAtHost");
  recv_pred_builder.Attr("Toutputs", std::vector<DataType>{DT_BOOL});
  recv_pred_builder.Attr("key", host_transfer_key);
  recv_pred_builder.Attr("device_ordinal", device_ordinal_value);
  recv_pred_builder.Attr(xla_cluster_attr_name, xla_cluster_name);
  recv_pred_builder.Attr(outside_compilation_attr_name,
                         outside_compilation_name);
  recv_pred_builder.Attr(kXlaHasHostTransferAttrName, true);
  recv_pred_builder.Input(key_placeholder->name(), 0, DT_STRING);
  NodeDef recv_pred_def;
  TF_RETURN_IF_ERROR(recv_pred_builder.Finalize(&recv_pred_def));
  TF_ASSIGN_OR_RETURN(Node * recv_pred_node, host_graph.AddNode(recv_pred_def));
  host_graph.AddEdge(key_placeholder, 0, recv_pred_node, 0);
  TF_RETURN_IF_ERROR(ReplaceKeyPlaceholderWithArgNode(
      xla_cluster_name, then_branch_host_func_name, fld));
  TF_RETURN_IF_ERROR(ReplaceKeyPlaceholderWithArgNode(
      xla_cluster_name, else_branch_host_func_name, fld));
  NodeDefBuilder if_builder(absl::StrCat("oc_if_", if_node_name), "If");
  if_builder.Attr("Tcond", DT_BOOL);
  if_builder.Attr("Tin", std::vector<DataType>{DT_STRING});
  if_builder.Attr("Tout", std::vector<DataType>{});
  NameAttrList host_then_branch, host_else_branch;
  host_then_branch.set_name(then_branch_host_func_name);
  (*host_then_branch.mutable_attr())["_device_ordinal"] = device_ordinal_value;
  host_else_branch.set_name(else_branch_host_func_name);
  (*host_else_branch.mutable_attr())["_device_ordinal"] = device_ordinal_value;
  if_builder.Attr("then_branch", host_then_branch);
  if_builder.Attr("else_branch", host_else_branch);
  if_builder.Attr(kXlaHasHostTransferAttrName, true);
  if_builder.Attr(xla_cluster_attr_name, xla_cluster_name);
  if_builder.Attr(outside_compilation_attr_name, outside_compilation_name);
  if_builder.Input(recv_pred_node->name(), 0, DT_BOOL);
  std::vector<NodeDefBuilder::NodeOut> if_inputs{
      {key_placeholder->name(), 0, DT_STRING}};
  if_builder.Input(if_inputs);
  NodeDef if_def;
  TF_RETURN_IF_ERROR(if_builder.Finalize(&if_def));
  TF_ASSIGN_OR_RETURN(Node * if_node, host_graph.AddNode(if_def));
  host_graph.AddEdge(recv_pred_node, 0, if_node, 0);
  host_graph.AddEdge(key_placeholder, 0, if_node, 1);
  FunctionDef oc_host_graph_fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(host_graph, host_graph_func_name,
                                        &oc_host_graph_fdef));
  if (fld->Find(host_graph_func_name)) {
    TF_RETURN_IF_ERROR(
        fld->ReplaceFunction(host_graph_func_name, oc_host_graph_fdef));
  } else {
    TF_RETURN_IF_ERROR(fld->AddFunctionDef(oc_host_graph_fdef));
  }
  return absl::OkStatus();
}
TF_ATTRIBUTE_NOINLINE Status AddSendLoopPredToLoopCond(
    const string& cond_xla_func_name, const string& host_transfer_key,
    NameAttrList* loop_cond_func, FunctionLibraryDefinition* fld,
    Node* while_node) {
  std::unique_ptr<FunctionBody> fbody;
  const FunctionDef* loop_cond_fdef = fld->Find(loop_cond_func->name());
  TF_RET_CHECK(loop_cond_fdef);
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
      *loop_cond_fdef, AttrSlice(&loop_cond_func->attr()), fld, &fbody));
  Graph* g = fbody->graph;
  Node* ret_node = nullptr;
  for (Node* n : g->nodes()) {
    if (n->type_string() == "_Retval") {
      if (ret_node) {
        return errors::Internal("Multiple return node for loop cond function ",
                                loop_cond_func->name(), ": ",
                                ret_node->DebugString(), " and ",
                                n->DebugString());
      } else {
        ret_node = n;
      }
    }
  }
  if (!ret_node) {
    return errors::Internal("No _Retval node for loop cond function ",
                            loop_cond_func->name());
  }
  Node* loop_cond;
  TF_RETURN_IF_ERROR(ret_node->input_node(0, &loop_cond));
  NodeDefBuilder send_loop_cond_builder(
      absl::StrCat("send_oc_while_cond_", while_node->name()), "XlaSendToHost");
  send_loop_cond_builder.Attr("Tinput", DT_BOOL);
  send_loop_cond_builder.Attr("key",
                              absl::StrCat(host_transfer_key, "_dtoh_0"));
  send_loop_cond_builder.Attr(kXlaTokenInputNodesAttrName,
                              std::vector<string>{kXlaTokenArgNodeName});
  send_loop_cond_builder.Attr(kXlaOriginalOutsideCompilationNodeName,
                              send_loop_cond_builder.node_name());
  SetMaximalSharding(send_loop_cond_builder);
  send_loop_cond_builder.Input(loop_cond->name(), 0, DT_BOOL);
  NodeDef send_loop_cond_def;
  TF_RETURN_IF_ERROR(send_loop_cond_builder.Finalize(&send_loop_cond_def));
  TF_ASSIGN_OR_RETURN(Node * send_loop_cond_node,
                      g->AddNode(send_loop_cond_def));
  g->AddEdge(loop_cond, 0, send_loop_cond_node, 0);
  FunctionDef replace_fdef;
  if (loop_cond_func->name() == cond_xla_func_name) {
    TF_RETURN_IF_ERROR(
        GraphToFunctionDef(*g, loop_cond_func->name(), &replace_fdef));
    TF_RETURN_IF_ERROR(
        fld->ReplaceFunction(loop_cond_func->name(), replace_fdef));
  } else {
    const auto new_name = fld->UniqueFunctionName(
        absl::StrCat(loop_cond_func->name(), "_send_pred_added_"));
    TF_RETURN_IF_ERROR(GraphToFunctionDef(*g, new_name, &replace_fdef));
    TF_RETURN_IF_ERROR(fld->AddFunctionDef(replace_fdef));
    loop_cond_func->set_name(new_name);
    while_node->ClearAttr("cond");
    while_node->AddAttr("cond", *loop_cond_func);
  }
  return absl::OkStatus();
}
Status RewriteHostWhileLoopCond(
    const string& cond_host_func_name, const string& while_node_name,
    const string& host_transfer_key, const string& xla_cluster_attr_name,
    const string& xla_cluster_name, const string& outside_compilation_attr_name,
    const string& outside_compilation_name, FunctionLibraryDefinition* fld) {
  TF_RETURN_IF_ERROR(ReplaceKeyPlaceholderWithArgNode(
      xla_cluster_name, cond_host_func_name, fld));
  AttrValue device_ordinal_temp_value;
  device_ordinal_temp_value.set_i(0);
  protobuf::Map<string, AttrValue> attrs;
  attrs["_device_ordinal"] = device_ordinal_temp_value;
  std::unique_ptr<FunctionBody> cond_fbody;
  const FunctionDef* cond_host_func = fld->Find(cond_host_func_name);
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*cond_host_func, AttrSlice(&attrs),
                                             fld, &cond_fbody));
  Graph* cond_graph = cond_fbody->graph;
  Node* key_arg = nullptr;
  for (Node* n : cond_graph->nodes()) {
    if (n->type_string() == "_Arg") {
      key_arg = n;
    }
  }
  if (!key_arg) {
    return errors::Internal(
        "No _Arg node found for host compute key in function ",
        cond_host_func_name);
  }
  NodeDefBuilder recv_pred_builder(
      absl::StrCat("recv_oc_while_cond_", while_node_name), "_XlaRecvAtHost");
  recv_pred_builder.Attr("Toutputs", std::vector<DataType>{DT_BOOL});
  recv_pred_builder.Attr("key", host_transfer_key);
  AttrValue device_ordinal_value;
  device_ordinal_value.set_placeholder("_device_ordinal");
  recv_pred_builder.Attr("device_ordinal", device_ordinal_value);
  recv_pred_builder.Attr(xla_cluster_attr_name, xla_cluster_name);
  recv_pred_builder.Attr(outside_compilation_attr_name,
                         outside_compilation_name);
  recv_pred_builder.Attr(kXlaHasHostTransferAttrName, true);
  recv_pred_builder.Input(key_arg->name(), 0, DT_STRING);
  NodeDef recv_pred_def;
  TF_RETURN_IF_ERROR(recv_pred_builder.Finalize(&recv_pred_def));
  TF_ASSIGN_OR_RETURN(Node * recv_pred_node,
                      cond_graph->AddNode(recv_pred_def));
  cond_graph->AddEdge(key_arg, 0, recv_pred_node, 0);
  NodeDefBuilder ret_builder(
      absl::StrCat("recv_oc_while_cond_ret_", while_node_name), "_Retval");
  ret_builder.Attr("T", DT_BOOL);
  ret_builder.Attr("index", 0);
  ret_builder.Input(recv_pred_node->name(), 0, DT_BOOL);
  NodeDef ret_def;
  TF_RETURN_IF_ERROR(ret_builder.Finalize(&ret_def));
  TF_ASSIGN_OR_RETURN(Node * ret_node, cond_graph->AddNode(ret_def));
  cond_graph->AddEdge(recv_pred_node, 0, ret_node, 0);
  TF_RETURN_IF_ERROR(ResetDeviceOrdinalToPlaceholderValue(cond_graph));
  FunctionDef cond_replace_fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(*cond_graph, cond_host_func_name,
                                        HostGraphControlRetMapping,
                                        &cond_replace_fdef));
  TF_RETURN_IF_ERROR(
      fld->ReplaceFunction(cond_host_func_name, cond_replace_fdef));
  return absl::OkStatus();
}
Status RewriteHostWhileLoopBody(
    const string& body_host_func_name, const string& while_node_name,
    const string& host_transfer_key, const string& xla_cluster_attr_name,
    const string& xla_cluster_name, const string& outside_compilation_attr_name,
    const string& outside_compilation_name, FunctionLibraryDefinition* fld) {
  TF_RETURN_IF_ERROR(ReplaceKeyPlaceholderWithArgNode(
      xla_cluster_name, body_host_func_name, fld));
  AttrValue device_ordinal_temp_value;
  device_ordinal_temp_value.set_i(0);
  protobuf::Map<string, AttrValue> attrs;
  attrs["_device_ordinal"] = device_ordinal_temp_value;
  std::unique_ptr<FunctionBody> body_fbody;
  const FunctionDef* body_host_func = fld->Find(body_host_func_name);
  TF_RET_CHECK(body_host_func);
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*body_host_func, AttrSlice(&attrs),
                                             fld, &body_fbody));
  Graph* body_graph = body_fbody->graph;
  Node* key_arg = nullptr;
  for (Node* n : body_graph->nodes()) {
    if (n->type_string() == "_Arg") {
      key_arg = n;
    }
  }
  if (!key_arg) {
    return errors::Internal(
        "No _Arg node found for host compute key in function ",
        body_host_func_name);
  }
  NodeDefBuilder ret_builder(
      absl::StrCat("recv_oc_while_body_ret_", while_node_name), "_Retval");
  ret_builder.Attr("T", DT_STRING);
  ret_builder.Attr("index", 0);
  ret_builder.Input(key_arg->name(), 0, DT_STRING);
  NodeDef ret_def;
  TF_RETURN_IF_ERROR(ret_builder.Finalize(&ret_def));
  TF_ASSIGN_OR_RETURN(Node * ret_node, body_graph->AddNode(ret_def));
  body_graph->AddEdge(key_arg, 0, ret_node, 0);
  TF_RETURN_IF_ERROR(ResetDeviceOrdinalToPlaceholderValue(body_graph));
  FunctionDef body_replace_fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(*body_graph, body_host_func_name,
                                        HostGraphControlRetMapping,
                                        &body_replace_fdef));
  TF_RETURN_IF_ERROR(
      fld->ReplaceFunction(body_host_func_name, body_replace_fdef));
  return absl::OkStatus();
}
TF_ATTRIBUTE_NOINLINE Status BuildHostGraphForWhileNode(
    const string& xla_cluster_attr_name,
    const string& outside_compilation_attr_name, const string& xla_cluster_name,
    const string& while_node_name, const string& host_transfer_key,
    const string& host_graph_func_name, FunctionLibraryDefinition* fld,
    const string& cond_host_func_name, const string& body_host_func_name) {
  Graph host_graph(fld);
  string outside_compilation_name = absl::StrCat("oc_while_", while_node_name);
  TF_ASSIGN_OR_RETURN(
      Node * key_placeholder,
      AddHostComputeKeyPlaceholder(xla_cluster_name, &host_graph));
  TF_RETURN_IF_ERROR(RewriteHostWhileLoopCond(
      cond_host_func_name, while_node_name, host_transfer_key,
      xla_cluster_attr_name, xla_cluster_name, outside_compilation_attr_name,
      outside_compilation_name, fld));
  TF_RETURN_IF_ERROR(RewriteHostWhileLoopBody(
      body_host_func_name, while_node_name, host_transfer_key,
      xla_cluster_attr_name, xla_cluster_name, outside_compilation_attr_name,
      outside_compilation_name, fld));
  NodeDefBuilder while_builder(absl::StrCat("oc_while_", while_node_name),
                               "While");
  while_builder.Attr("T", std::vector<DataType>{DT_STRING});
  NameAttrList func;
  AttrValue device_ordinal_value;
  device_ordinal_value.set_placeholder("_device_ordinal");
  (*func.mutable_attr())["_device_ordinal"] = device_ordinal_value;
  func.set_name(cond_host_func_name);
  while_builder.Attr("cond", func);
  func.set_name(body_host_func_name);
  while_builder.Attr("body", func);
  while_builder.Attr(kXlaHasHostTransferAttrName, true);
  while_builder.Attr(xla_cluster_attr_name, xla_cluster_name);
  while_builder.Attr(outside_compilation_attr_name, outside_compilation_name);
  while_builder.Attr("parallel_iterations", 1);
  std::vector<NodeDefBuilder::NodeOut> while_inputs{
      {key_placeholder->name(), 0, DT_STRING}};
  while_builder.Input(while_inputs);
  NodeDef while_def;
  TF_RETURN_IF_ERROR(while_builder.Finalize(&while_def));
  TF_ASSIGN_OR_RETURN(Node * while_node, host_graph.AddNode(while_def));
  host_graph.AddEdge(key_placeholder, 0, while_node, 0);
  FunctionDef oc_host_graph_fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(host_graph, host_graph_func_name,
                                        &oc_host_graph_fdef));
  if (fld->Find(host_graph_func_name)) {
    TF_RETURN_IF_ERROR(
        fld->ReplaceFunction(host_graph_func_name, oc_host_graph_fdef));
  } else {
    TF_RETURN_IF_ERROR(fld->AddFunctionDef(oc_host_graph_fdef));
  }
  return absl::OkStatus();
}
Status BuildHostGraphForFuncCallNode(
    const string& xla_cluster_attr_name, const string& xla_cluster_name,
    const string& outside_compilation_attr_name,
    const string& func_call_node_name, const string& func_call_host_func_name,
    const string& host_graph_func_name, FunctionLibraryDefinition* fld) {
  Graph host_graph(fld);
  AttrValue device_ordinal_value;
  device_ordinal_value.set_placeholder("_device_ordinal");
  TF_ASSIGN_OR_RETURN(
      Node * key_placeholder,
      AddHostComputeKeyPlaceholder(xla_cluster_name, &host_graph));
  TF_RETURN_IF_ERROR(ReplaceKeyPlaceholderWithArgNode(
      xla_cluster_name, func_call_host_func_name, fld));
  NodeDefBuilder call_builder(absl::StrCat("oc_call_", func_call_node_name),
                              func_call_host_func_name, fld);
  call_builder.Input(key_placeholder->name(), 0, DT_STRING);
  call_builder.Attr("_device_ordinal", device_ordinal_value);
  call_builder.Attr(kXlaHasHostTransferAttrName, true);
  call_builder.Attr(xla_cluster_attr_name, xla_cluster_name);
  call_builder.Attr(outside_compilation_attr_name, call_builder.node_name());
  NodeDef call_def;
  TF_RETURN_IF_ERROR(call_builder.Finalize(&call_def));
  TF_ASSIGN_OR_RETURN(Node * call_node, host_graph.AddNode(call_def));
  host_graph.AddEdge(key_placeholder, 0, call_node, 0);
  FunctionDef oc_host_graph_fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(host_graph, host_graph_func_name,
                                        HostGraphControlRetMapping,
                                        &oc_host_graph_fdef));
  if (fld->Find(host_graph_func_name)) {
    TF_RETURN_IF_ERROR(
        fld->ReplaceFunction(host_graph_func_name, oc_host_graph_fdef));
  } else {
    TF_RETURN_IF_ERROR(fld->AddFunctionDef(oc_host_graph_fdef));
  }
  return absl::OkStatus();
}
TF_ATTRIBUTE_NOINLINE Status ExtractOutsideCompilationForFuncCallNode(
    const string& xla_cluster_attr_name,
    const string& outside_compilation_attr_name, const string& xla_cluster_name,
    const std::map<string, int>& host_compute_core, Graph* g, Node* n,
    FunctionLibraryRuntime* flr, FunctionLibraryDefinition* fld,
    std::vector<string>* host_graphs,
    std::vector<string>* shape_inference_graphs,
    bool* has_outside_compilation) {
  bool func_has_outside_compilation = false;
  NameAttrList func;
  if (fld->Contains(n->type_string())) {
    func.set_name(n->type_string());
    typedef protobuf::Map<string, AttrValue> AttrMap;
    *func.mutable_attr() = AttrMap(n->attrs().begin(), n->attrs().end());
  } else if (n->IsPartitionedCall()) {
    TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "f", &func));
  } else {
    TF_RET_CHECK(n->type_string() == FunctionLibraryDefinition::kGradientOp);
    func.set_name(FunctionLibraryDefinition::kGradientOp);
    *func.mutable_attr() = n->def().attr();
  }
  string canonical_func_name;
  if (func.name() == FunctionLibraryDefinition::kGradientOp) {
    NameAttrList forward_func;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "f", &forward_func));
    canonical_func_name = absl::StrCat("gradient_", forward_func.name());
  } else {
    canonical_func_name = func.name();
  }
  string new_func_name = absl::StrCat(canonical_func_name, "_oc");
  string host_func_name =
      absl::StrCat("oc_func_call_host_", canonical_func_name);
  TF_RETURN_IF_ERROR(ExtractOutsideCompilationForFunction(
      xla_cluster_attr_name, outside_compilation_attr_name, xla_cluster_name,
      func, new_func_name, host_func_name, host_compute_core, flr, fld,
      shape_inference_graphs, &func_has_outside_compilation));
  if (!func_has_outside_compilation) {
    return absl::OkStatus();
  }
  *has_outside_compilation = true;
  auto replace_builder =
      std::make_unique<NodeDefBuilder>(n->name(), new_func_name, fld);
  std::vector<NodeDefBuilder::NodeOut> inputs(n->num_inputs());
  for (const Edge* e : n->in_edges()) {
    if (e->IsControlEdge()) {
      continue;
    }
    const bool input_size_check =
        e->dst_input() < static_cast<int>(inputs.size());
    TF_RET_CHECK(e->dst_input() >= 0 && input_size_check);
    inputs[e->dst_input()] =
        NodeDefBuilder::NodeOut{e->src()->name(), e->src_output(),
                                e->src()->output_type(e->src_output())};
  }
  for (const auto& input : inputs) {
    replace_builder->Input(input);
  }
  for (const auto& attr : n->attrs()) {
    replace_builder->Attr(attr.first, attr.second);
  }
  auto replace_def = std::make_unique<NodeDef>();
  TF_RETURN_IF_ERROR(replace_builder->Finalize(replace_def.get()));
  TF_ASSIGN_OR_RETURN(Node * replace, ReplaceNode(g, n, *replace_def));
  replace->AddAttr(kXlaTokenInputNodesAttrName,
                   std::vector<string>{kXlaTokenArgNodeName});
  replace->AddAttr(kXlaOriginalOutsideCompilationNodeName, replace->name());
  string oc_host_graph_name =
      absl::StrCat("oc_func_host_graph_", replace->name());
  TF_RETURN_IF_ERROR(BuildHostGraphForFuncCallNode(
      xla_cluster_attr_name, xla_cluster_name, outside_compilation_attr_name,
      replace->name(), host_func_name, oc_host_graph_name, fld));
  host_graphs->push_back(oc_host_graph_name);
  return absl::OkStatus();
}
Status ExtractOutsideCompilationForIfNode(
    const string& xla_cluster_attr_name,
    const string& outside_compilation_attr_name, const string& xla_cluster_name,
    const std::map<string, int>& host_compute_core, Graph* g, Node* n,
    FunctionLibraryRuntime* flr, FunctionLibraryDefinition* fld,
    std::vector<string>* host_graphs,
    std::vector<string>* shape_inference_graphs,
    bool* has_outside_compilation) {
  NameAttrList then_branch, else_branch;
  TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "then_branch", &then_branch));
  TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "else_branch", &else_branch));
  bool then_branch_has_outside_compilation = false;
  bool else_branch_has_outside_compilation = false;
  string then_branch_host_func_name =
             absl::StrCat("oc_then_branch_host_if_", then_branch.name()),
         else_branch_host_func_name =
             absl::StrCat("oc_else_branch_host_if_", else_branch.name());
  string then_branch_xla_func_name = absl::StrCat(then_branch.name(), "_oc"),
         else_branch_xla_func_name = absl::StrCat(else_branch.name(), "_oc");
  TF_RETURN_IF_ERROR(ExtractOutsideCompilationForFunction(
      xla_cluster_attr_name, outside_compilation_attr_name, xla_cluster_name,
      then_branch, then_branch_xla_func_name, then_branch_host_func_name,
      host_compute_core, flr, fld, shape_inference_graphs,
      &then_branch_has_outside_compilation));
  TF_RETURN_IF_ERROR(ExtractOutsideCompilationForFunction(
      xla_cluster_attr_name, outside_compilation_attr_name, xla_cluster_name,
      else_branch, else_branch_xla_func_name, else_branch_host_func_name,
      host_compute_core, flr, fld, shape_inference_graphs,
      &else_branch_has_outside_compilation));
  if (!then_branch_has_outside_compilation &&
      !else_branch_has_outside_compilation) {
    return absl::OkStatus();
  }
  *has_outside_compilation = true;
  if (then_branch_has_outside_compilation) {
    then_branch.set_name(then_branch_xla_func_name);
    n->ClearAttr("then_branch");
    n->AddAttr("then_branch", then_branch);
  }
  if (else_branch_has_outside_compilation) {
    else_branch.set_name(else_branch_xla_func_name);
    n->ClearAttr("else_branch");
    n->AddAttr("else_branch", else_branch);
  }
  n->AddAttr(kXlaOriginalOutsideCompilationNodeName, n->name());
  string host_transfer_key = absl::StrCat("oc_if_pred_", n->name());
  Node* pred_node;
  TF_RETURN_IF_ERROR(n->input_node(0, &pred_node));
  TF_ASSIGN_OR_RETURN(
      Node * send_pred_node,
      BuildSendIfPredNode(absl::StrCat("send_oc_if_pred_", n->name()),
                          host_transfer_key, pred_node, g));
  n->AddAttr(kXlaTokenInputNodesAttrName,
             std::vector<string>{send_pred_node->name()});
  g->AddControlEdge(send_pred_node, n);
  if (!then_branch_has_outside_compilation) {
    std::unique_ptr<Graph> then_branch_host_graph(new Graph(fld));
    std::vector<string> then_branch_host_graphs;
    TF_RETURN_IF_ERROR(ConstructHostGraph(
        xla_cluster_name, outside_compilation_attr_name,
        then_branch_host_graphs, fld, &then_branch_host_graph));
    FunctionDef then_branch_host_fdef;
    TF_RETURN_IF_ERROR(GraphToFunctionDef(*then_branch_host_graph,
                                          then_branch_host_func_name,
                                          &then_branch_host_fdef));
    if (fld->Find(then_branch_host_func_name)) {
      TF_RETURN_IF_ERROR(fld->ReplaceFunction(then_branch_host_func_name,
                                              then_branch_host_fdef));
    } else {
      TF_RETURN_IF_ERROR(fld->AddFunctionDef(then_branch_host_fdef));
    }
  }
  if (!else_branch_has_outside_compilation) {
    std::unique_ptr<Graph> else_branch_host_graph(new Graph(fld));
    std::vector<string> else_branch_host_graphs;
    TF_RETURN_IF_ERROR(ConstructHostGraph(
        xla_cluster_name, outside_compilation_attr_name,
        else_branch_host_graphs, fld, &else_branch_host_graph));
    FunctionDef else_branch_host_fdef;
    TF_RETURN_IF_ERROR(GraphToFunctionDef(*else_branch_host_graph,
                                          else_branch_host_func_name,
                                          &else_branch_host_fdef));
    if (fld->Find(else_branch_host_func_name)) {
      TF_RETURN_IF_ERROR(fld->ReplaceFunction(else_branch_host_func_name,
                                              else_branch_host_fdef));
    } else {
      TF_RETURN_IF_ERROR(fld->AddFunctionDef(else_branch_host_fdef));
    }
  }
  string oc_host_graph_name = absl::StrCat("oc_if_host_graph_", n->name());
  TF_RETURN_IF_ERROR(BuildHostGraphForIfNode(
      xla_cluster_attr_name, outside_compilation_attr_name, xla_cluster_name,
      n->name(), host_transfer_key, oc_host_graph_name, fld,
      then_branch_host_func_name, else_branch_host_func_name));
  host_graphs->push_back(oc_host_graph_name);
  return absl::OkStatus();
}
Status ExtractOutsideCompilationForWhileNode(
    const string& xla_cluster_attr_name,
    const string& outside_compilation_attr_name, const string& xla_cluster_name,
    const std::map<string, int>& host_compute_core, Graph* g, Node* n,
    FunctionLibraryRuntime* flr, FunctionLibraryDefinition* fld,
    std::vector<string>* host_graphs,
    std::vector<string>* shape_inference_graphs,
    bool* has_outside_compilation) {
  NameAttrList cond, body;
  TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "cond", &cond));
  TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "body", &body));
  bool cond_has_outside_compilation = false;
  bool body_has_outside_compilation = false;
  string cond_host_func_name = absl::StrCat("oc_cond_host_while_", cond.name()),
         body_host_func_name = absl::StrCat("oc_body_host_while_", body.name());
  string cond_xla_func_name = absl::StrCat(cond.name(), "_oc"),
         body_xla_func_name = absl::StrCat(body.name(), "_oc");
  TF_RETURN_IF_ERROR(ExtractOutsideCompilationForFunction(
      xla_cluster_attr_name, outside_compilation_attr_name, xla_cluster_name,
      cond, cond_xla_func_name, cond_host_func_name, host_compute_core, flr,
      fld, shape_inference_graphs, &cond_has_outside_compilation));
  TF_RETURN_IF_ERROR(ExtractOutsideCompilationForFunction(
      xla_cluster_attr_name, outside_compilation_attr_name, xla_cluster_name,
      body, body_xla_func_name, body_host_func_name, host_compute_core, flr,
      fld, shape_inference_graphs, &body_has_outside_compilation));
  if (!cond_has_outside_compilation && !body_has_outside_compilation) {
    return absl::OkStatus();
  }
  *has_outside_compilation = true;
  if (cond_has_outside_compilation) {
    cond.set_name(cond_xla_func_name);
    n->ClearAttr("cond");
    n->AddAttr("cond", cond);
  }
  if (body_has_outside_compilation) {
    body.set_name(body_xla_func_name);
    n->ClearAttr("body");
    n->AddAttr("body", body);
  }
  n->AddAttr(kXlaOriginalOutsideCompilationNodeName, n->name());
  string host_transfer_key = absl::StrCat("oc_while_pred_", n->name());
  TF_RETURN_IF_ERROR(AddSendLoopPredToLoopCond(
      cond_xla_func_name, host_transfer_key, &cond, fld, n));
  n->AddAttr(kXlaTokenInputNodesAttrName,
             std::vector<string>{kXlaTokenArgNodeName});
  if (!cond_has_outside_compilation) {
    std::unique_ptr<Graph> cond_host_graph(new Graph(fld));
    std::vector<string> host_graphs;
    TF_RETURN_IF_ERROR(ConstructHostGraph(xla_cluster_name,
                                          outside_compilation_attr_name,
                                          host_graphs, fld, &cond_host_graph));
    FunctionDef cond_host_fdef;
    TF_RETURN_IF_ERROR(GraphToFunctionDef(*cond_host_graph, cond_host_func_name,
                                          &cond_host_fdef));
    if (fld->Find(cond_host_func_name)) {
      TF_RETURN_IF_ERROR(
          fld->ReplaceFunction(cond_host_func_name, cond_host_fdef));
    } else {
      TF_RETURN_IF_ERROR(fld->AddFunctionDef(cond_host_fdef));
    }
  }
  if (!body_has_outside_compilation) {
    std::unique_ptr<Graph> body_host_graph(new Graph(fld));
    std::vector<string> host_graphs;
    TF_RETURN_IF_ERROR(ConstructHostGraph(xla_cluster_name,
                                          outside_compilation_attr_name,
                                          host_graphs, fld, &body_host_graph));
    FunctionDef body_host_fdef;
    TF_RETURN_IF_ERROR(GraphToFunctionDef(*body_host_graph, body_host_func_name,
                                          &body_host_fdef));
    if (fld->Find(body_host_func_name)) {
      TF_RETURN_IF_ERROR(
          fld->ReplaceFunction(body_host_func_name, body_host_fdef));
    } else {
      TF_RETURN_IF_ERROR(fld->AddFunctionDef(body_host_fdef));
    }
  }
  string oc_host_graph_name = absl::StrCat("oc_while_host_graph_", n->name());
  TF_RETURN_IF_ERROR(BuildHostGraphForWhileNode(
      xla_cluster_attr_name, outside_compilation_attr_name, xla_cluster_name,
      n->name(), host_transfer_key, oc_host_graph_name, fld,
      cond_host_func_name, body_host_func_name));
  host_graphs->push_back(oc_host_graph_name);
  return absl::OkStatus();
}
Status ExtractOutsideCompilationForNodesWithAssociatedFunctions(
    Graph* g, const string& xla_cluster_attr_name,
    const string& outside_compilation_attr_name, const string& xla_cluster_name,
    const std::map<string, int>& host_compute_core, FunctionLibraryRuntime* flr,
    FunctionLibraryDefinition* fld, std::vector<string>* host_graphs,
    std::vector<string>* shape_inference_graphs,
    bool* has_outside_compilation) {
  std::vector<Node*> if_nodes, while_nodes, func_call_nodes;
  for (Node* n : g->nodes()) {
    if (n->IsIfNode()) {
      if_nodes.push_back(n);
    } else if (n->IsWhileNode()) {
      while_nodes.push_back(n);
    } else if (IsFunctionCall(*fld, *n)) {
      func_call_nodes.push_back(n);
    }
  }
  for (Node* n : func_call_nodes) {
    TF_RETURN_IF_ERROR(ExtractOutsideCompilationForFuncCallNode(
        xla_cluster_attr_name, outside_compilation_attr_name, xla_cluster_name,
        host_compute_core, g, n, flr, fld, host_graphs, shape_inference_graphs,
        has_outside_compilation));
  }
  for (Node* n : if_nodes) {
    TF_RETURN_IF_ERROR(ExtractOutsideCompilationForIfNode(
        xla_cluster_attr_name, outside_compilation_attr_name, xla_cluster_name,
        host_compute_core, g, n, flr, fld, host_graphs, shape_inference_graphs,
        has_outside_compilation));
  }
  for (Node* n : while_nodes) {
    TF_RETURN_IF_ERROR(ExtractOutsideCompilationForWhileNode(
        xla_cluster_attr_name, outside_compilation_attr_name, xla_cluster_name,
        host_compute_core, g, n, flr, fld, host_graphs, shape_inference_graphs,
        has_outside_compilation));
  }
  return absl::OkStatus();
}
Status CopyOutsideCompilationConstNodes(
    Graph* g, const string& outside_compilation_attr_name) {
  for (Node* n : g->op_nodes()) {
    if (!n->IsConstant() ||
        !HasNodeAttr(n->def(), outside_compilation_attr_name)) {
      continue;
    }
    std::vector<const Edge*> out_edges(n->out_edges().begin(),
                                       n->out_edges().end());
    bool has_non_oc_output = false;
    for (const Edge* e : out_edges) {
      if (!e->IsControlEdge() &&
          !HasNodeAttr(e->dst()->def(), outside_compilation_attr_name)) {
        has_non_oc_output = true;
        break;
      }
    }
    if (!has_non_oc_output) {
      continue;
    }
    NodeDef copy_def = n->def();
    copy_def.set_name(g->NewName(n->name()));
    copy_def.mutable_attr()->erase(outside_compilation_attr_name);
    TF_ASSIGN_OR_RETURN(Node * copy_node, g->AddNode(copy_def));
    for (const Edge* e : n->in_edges()) {
      if (e->IsControlEdge()) {
        g->AddControlEdge(e->src(), copy_node);
      }
    }
    for (const Edge* e : out_edges) {
      if (!e->IsControlEdge() &&
          !HasNodeAttr(e->dst()->def(), outside_compilation_attr_name)) {
        Node* dst = e->dst();
        int dst_input = e->dst_input();
        g->RemoveEdge(e);
        g->AddEdge(copy_node, 0, dst, dst_input);
      }
    }
  }
  return absl::OkStatus();
}
}  
Status RewriteOutsideCompilationSubgraphFn::operator()(
    const std::vector<OutputTensor>& arg_source_tensors,
    std::unique_ptr<Graph>* graph, std::vector<int>* input_permutation,
    std::vector<int>* output_permutation, NodeDef* node_def) {
  string old_name = node_def->op();
  string new_name =
      absl::StrCat(xla_cluster_name_, "_", new_function_name_, "_", old_name);
  node_def->set_op(new_name);
  node_def->set_name(new_name);
  FixupSourceAndSinkEdges(graph->get());
  TF_ASSIGN_OR_RETURN(
      Node * key_placeholder,
      AddHostComputeKeyPlaceholder(xla_cluster_name_, graph->get()));
  std::vector<DataType> recv_at_host_dtypes;
  TF_ASSIGN_OR_RETURN(
      Node * recv_at_host_node,
      ReplaceArgNodesWithRecvAtHostNode(graph->get(), new_name,
                                        &recv_at_host_dtypes, key_placeholder));
  std::vector<DataType> send_from_host_dtypes;
  TF_ASSIGN_OR_RETURN(
      Node * send_from_host_node,
      ReplaceRetNodesWithSendFromHostNode(
          graph->get(), new_name, &send_from_host_dtypes, key_placeholder));
  for (Node* n : (*graph)->nodes()) {
    if (IsKeyPlaceholderNode(*n)) {
      continue;
    }
    n->AddAttr(xla_cluster_attr_name_, xla_cluster_name_);
    n->AddAttr(outside_compilation_attr_name_, old_name);
  }
  std::optional<std::vector<PartialTensorShape>> shapes =
      GetInferredInputShapes(send_from_host_dtypes.size(), send_from_host_node);
  for (Node* n : (*graph)->nodes()) {
    n->ClearAttr(kXlaInferredShapesAttrName);
  }
  for (Node* n : (*graph)->nodes()) {
    if (HasNodeAttr(n->def(), kXlaConnectedToXlaComputationAttrName)) {
      (*graph)->AddControlEdge(n, send_from_host_node);
      n->ClearAttr(kXlaConnectedToXlaComputationAttrName);
    }
    if (HasNodeAttr(n->def(), kXlaConnectedFromXlaComputationAttrName)) {
      (*graph)->AddControlEdge(recv_at_host_node, n);
      n->ClearAttr(kXlaConnectedFromXlaComputationAttrName);
    }
  }
  if (send_from_host_node->in_edges().size() > 1) {
    (*graph)->AddControlEdge(send_from_host_node, (*graph)->sink_node());
  }
  PruneForReverseReachability(
      graph->get(), std::unordered_set<const Node*>{(*graph)->sink_node()});
  AddNodeAttr("_outside_compilation_subgraph", old_name, node_def);
  if (shapes) {
    NameAttrList shape_inference_graph;
    AddNodeAttr("shape_inference_graph", shape_inference_graph, node_def);
    AddNodeAttr("shapes", *shapes, node_def);
  } else {
    string shape_inference_func_name =
        absl::StrCat("_outside_compilation_shape_inference_", new_name);
    NameAttrList shape_inference_graph;
    shape_inference_graph.set_name(shape_inference_func_name);
    AddNodeAttr("shape_inference_graph", shape_inference_graph, node_def);
    AddNodeAttr("shapes", std::vector<TensorShapeProto>{}, node_def);
  }
  AddNodeAttr("ancestors", std::vector<string>{}, node_def);
  AddNodeAttr("Tinputs", recv_at_host_dtypes, node_def);
  AddNodeAttr("Toutputs", send_from_host_dtypes, node_def);
  AddNodeAttr("key", absl::StrCat("host_compute_channel_", new_name), node_def);
  return absl::OkStatus();
}
Status ExtractOutsideCompilationForFunction(
    const string& xla_cluster_attr_name,
    const string& outside_compilation_attr_name, const string& xla_cluster_name,
    const NameAttrList& func_name_attrs, const string& new_func_name,
    const string& host_graph_func_name,
    const std::map<string, int>& host_compute_core, FunctionLibraryRuntime* flr,
    FunctionLibraryDefinition* fld, std::vector<string>* shape_inference_graphs,
    bool* has_outside_compilation) {
  const string& func_name = func_name_attrs.name();
  FunctionLibraryRuntime::Handle handle;
  TF_RETURN_IF_ERROR(
      flr->Instantiate(func_name, AttrSlice(&func_name_attrs.attr()), &handle));
  Status ret_status = absl::OkStatus();
  auto cleanup_handle = gtl::MakeCleanup([&]() {
    auto s = flr->ReleaseHandle(handle);
    if (!s.ok()) {
      ret_status.Update(s);
    }
  });
  const FunctionBody* fbody = flr->GetFunctionBody(handle);
  *has_outside_compilation = false;
  for (Node* n : fbody->graph->nodes()) {
    if (HasNodeAttr(n->def(), outside_compilation_attr_name)) {
      *has_outside_compilation = true;
      break;
    }
  }
  if (VLOG_IS_ON(4)) {
    DumpGraphToFile(
        absl::StrCat("extract_outside_compilation_for_func_before_", func_name),
        *fbody->graph, fld);
  }
  std::unique_ptr<Graph> graph_out;
  std::vector<string> outside_compilation_host_graphs;
  std::vector<string> shape_inference_graphs_to_rewrite;
  if (*has_outside_compilation) {
    TF_RETURN_IF_ERROR(CopyOutsideCompilationConstNodes(
        fbody->graph, outside_compilation_attr_name));
    TF_ASSIGN_OR_RETURN(auto cluster_deps,
                        OutsideCompilationClusterDependencies(
                            fbody->graph, outside_compilation_attr_name));
    TF_RETURN_IF_ERROR(PreprocessEdgesBetweenOutsideCompilations(
        fbody->graph, outside_compilation_attr_name));
    auto rewrite_fn = std::make_unique<RewriteOutsideCompilationSubgraphFn>(
        xla_cluster_attr_name, outside_compilation_attr_name, xla_cluster_name,
        new_func_name);
    TF_RETURN_IF_ERROR(EncapsulateSubgraphsInFunctions(
        outside_compilation_attr_name, *fbody->graph, *rewrite_fn,
        true, &graph_out, fld));
    std::vector<Node*> outside_compilation_nodes;
    for (Node* n : graph_out->nodes()) {
      if (HasNodeAttr(n->def(), "_outside_compilation_subgraph")) {
        outside_compilation_nodes.push_back(n);
        outside_compilation_host_graphs.push_back(n->name());
        auto shape_inference_graph = std::make_unique<NameAttrList>();
        TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "shape_inference_graph",
                                       shape_inference_graph.get()));
        if (!shape_inference_graph->name().empty()) {
          shape_inference_graphs->push_back(shape_inference_graph->name());
          shape_inference_graphs_to_rewrite.push_back(
              shape_inference_graph->name());
          const FunctionDef* xla_fdef = fld->Find(n->name());
          if (!xla_fdef) {
            return errors::Internal("Cannot find XLA function ", n->name());
          }
          auto shape_inference_fdef = std::make_unique<FunctionDef>(*xla_fdef);
          shape_inference_fdef->mutable_signature()->set_name(
              shape_inference_graph->name());
          if (fld->Find(shape_inference_graph->name())) {
            TF_RETURN_IF_ERROR(fld->ReplaceFunction(
                shape_inference_graph->name(), *shape_inference_fdef));
          } else {
            TF_RETURN_IF_ERROR(fld->AddFunctionDef(*shape_inference_fdef));
          }
        }
      }
    }
    std::map<string, Node*> host_compute_nodes;
    for (Node* n : outside_compilation_nodes) {
      auto host_compute_node_or = ReplaceOutsideCompilationCallNode(
          graph_out.get(), n, host_compute_core, *cluster_deps);
      TF_RETURN_IF_ERROR(host_compute_node_or.status());
      Node* host_compute_node = host_compute_node_or.value();
      host_compute_nodes[host_compute_node->name()] = host_compute_node;
    }
    for (const auto& iter : host_compute_nodes) {
      Node* host_compute_node = iter.second;
      std::vector<string> token_input_node_names;
      TF_RETURN_IF_ERROR(GetNodeAttr(host_compute_node->def(),
                                     kXlaTokenInputNodesAttrName,
                                     &token_input_node_names));
      for (const string& node_name : token_input_node_names) {
        if (node_name == kXlaTokenArgNodeName) {
          continue;
        }
        auto iter = host_compute_nodes.find(node_name);
        TF_RET_CHECK(iter != host_compute_nodes.end());
        graph_out->AddControlEdge(iter->second, host_compute_node);
      }
    }
  }
  Graph* g = (*has_outside_compilation) ? graph_out.get() : fbody->graph;
  TF_RETURN_IF_ERROR(ExtractOutsideCompilationForNodesWithAssociatedFunctions(
      g, xla_cluster_attr_name, outside_compilation_attr_name, xla_cluster_name,
      host_compute_core, flr, fld, &outside_compilation_host_graphs,
      shape_inference_graphs, has_outside_compilation));
  if (*has_outside_compilation) {
    std::unique_ptr<Graph> host_graph;
    TF_RETURN_IF_ERROR(
        ConstructHostGraph(xla_cluster_name, outside_compilation_attr_name,
                           outside_compilation_host_graphs, fld, &host_graph));
    auto host_graph_fdef = std::make_unique<FunctionDef>();
    TF_RETURN_IF_ERROR(GraphToFunctionDef(*host_graph, host_graph_func_name,
                                          HostGraphControlRetMapping,
                                          host_graph_fdef.get()));
    if (fld->Find(host_graph_func_name)) {
      TF_RETURN_IF_ERROR(
          fld->ReplaceFunction(host_graph_func_name, *host_graph_fdef));
    } else {
      TF_RETURN_IF_ERROR(fld->AddFunctionDef(*host_graph_fdef));
    }
    for (const string& shape_inference_graph :
         shape_inference_graphs_to_rewrite) {
      TF_RETURN_IF_ERROR(
          RewriteShapeInferenceGraph(shape_inference_graph, host_graph.get(),
                                     nullptr, fld));
    }
    for (const string& func : outside_compilation_host_graphs) {
      TF_RETURN_IF_ERROR(fld->RemoveFunction(func));
    }
    auto updated_fdef = std::make_unique<FunctionDef>();
    TF_RETURN_IF_ERROR(
        GraphToFunctionDef(*g, new_func_name, updated_fdef.get()));
    updated_fdef->mutable_signature()->set_is_stateful(true);
    const FunctionDef* original_fdef = fld->Find(func_name);
    if (original_fdef) {
      for (const auto& attr : original_fdef->attr()) {
        (*updated_fdef->mutable_attr())[attr.first] = attr.second;
      }
    }
    if (fld->Find(new_func_name)) {
      TF_RETURN_IF_ERROR(fld->ReplaceFunction(new_func_name, *updated_fdef));
    } else {
      TF_RETURN_IF_ERROR(fld->AddFunctionDef(*updated_fdef));
    }
    if (VLOG_IS_ON(4)) {
      DumpGraphToFile(
          absl::StrCat("extract_outside_compilation_for_func_after_",
                       func_name),
          *g, fld);
    }
  }
  return ret_status;
}
Status ExtractOutsideCompilation(
    const string& xla_cluster_attr_name,
    const string& outside_compilation_attr_name,
    const std::unordered_map<string, XlaClusterInfo>& clusters, Graph* g,
    FunctionLibraryRuntime* flr, FunctionLibraryDefinition* fld,
    bool* modified) {
  if (VLOG_IS_ON(4)) {
    DumpGraphToFile("extract_outside_compilation_before", *g, fld);
  }
  *modified = false;
  auto node_name_index = g->BuildNodeNameIndex();
  for (auto& iter : clusters) {
    string xla_cluster_name = iter.first;
    Node* n = iter.second.node;
    auto const& func_name_attrs = iter.second.func_name_attrs;
    auto const& host_compute_core = iter.second.host_compute_core;
    std::vector<string> shape_inference_graphs;
    bool has_outside_compilation;
    string host_graph_func_name =
        absl::StrCat("oc_host_graph_", xla_cluster_name);
    TF_RETURN_IF_ERROR(ExtractOutsideCompilationForFunction(
        xla_cluster_attr_name, outside_compilation_attr_name, xla_cluster_name,
        func_name_attrs, func_name_attrs.name(), host_graph_func_name,
        host_compute_core, flr, fld, &shape_inference_graphs,
        &has_outside_compilation));
    *modified |= has_outside_compilation;
    if (has_outside_compilation) {
      string pivot_name = absl::StrCat(xla_cluster_name, "/pivot");
      Node* pivot_node = node_name_index[pivot_name];
      TF_RETURN_IF_ERROR(ExpandHostGraphIntoMainGraph(
          g, fld, host_graph_func_name, n, pivot_node));
      TF_RETURN_IF_ERROR(fld->RemoveFunction(host_graph_func_name));
      for (const auto& shape_inference_graph_name : shape_inference_graphs) {
        TF_RETURN_IF_ERROR(RewriteShapeInferenceGraph(
            shape_inference_graph_name, g, pivot_node, fld));
      }
    }
  }
  if (VLOG_IS_ON(4)) {
    DumpGraphToFile("extract_outside_compilation_after", *g, fld);
  }
  return absl::OkStatus();
}
}  