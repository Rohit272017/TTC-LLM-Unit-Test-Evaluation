#include "tensorflow/core/common_runtime/quantize_training.h"
#include <algorithm>
#include <atomic>
#include <set>
#include <unordered_map>
#include <vector>
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/public/session_options.h"
namespace tensorflow {
namespace {
const uint32 kAllowedInputs = 2;
const float kEMADecay = 0.999;
const auto* nodes_to_rewrite =
    new std::unordered_set<string, StringPieceHasher>{"MatMul", "Conv2D"};
struct EdgeToConvert {
  const Edge* edge;
  int32 num_bits;
  bool signed_input;
  bool range_given;
  float input_min;
  float input_max;
  EdgeToConvert(const Edge* e, int32_t bits, bool sign, bool range, float min,
                float max)
      : edge(e),
        num_bits(bits),
        signed_input(sign),
        range_given(range),
        input_min(min),
        input_max(max) {}
};
inline bool IsGradientNode(const Graph* graph, const Node* node) {
  static const string tag = "gradients";
  return (node->name().compare(0, tag.size(), tag) == 0);
}
bool FindType(const Graph* graph, const Node* node, bool* signed_input,
              bool* range_given, float* input_min, float* input_max) {
  const string& src_op = node->type_string();
  if (src_op == "Const" || src_op == "Variable" || src_op == "VariableV2") {
    *signed_input = true;
    *range_given = false;
  } else if (src_op == "Relu") {
    *signed_input = false;
    *range_given = false;
  } else if (src_op == "Relu6") {
    *signed_input = false;
    *range_given = true;
    *input_min = 0;
    *input_max = 6;
  } else if (src_op == "Sigmoid") {
    *signed_input = false;
    *range_given = true;
    *input_min = 0;
    *input_max = 1;
  } else if (src_op == "Tanh") {
    *signed_input = true;
    *range_given = true;
    *input_min = -1;
    *input_max = 1;
  } else if (src_op == "Reshape" || src_op == "ConcatV2") {
    for (const Edge* edge : node->in_edges()) {
      if (edge->src_output() != Graph::kControlSlot && edge->dst_input() == 0) {
        FindType(graph, edge->src(), signed_input, range_given, input_min,
                 input_max);
      }
    }
  } else if (src_op == "Identity" || src_op == "MaxPool" ||
             src_op == "AvgPool" || src_op == "MaxPool3D" ||
             src_op == "AvgPool3D") {
    for (const Edge* edge : node->in_edges()) {
      if (edge->src_output() != Graph::kControlSlot) {
        FindType(graph, edge->src(), signed_input, range_given, input_min,
                 input_max);
      }
    }
  } else {
    *signed_input = true;
    *range_given = false;
    return false;
  }
  return true;
}
Status FindSaveOp(const Graph* graph, Node** save_op,
                  std::vector<const Edge*>* in_edges, bool* found) {
  *found = false;
  for (Node* node : graph->op_nodes()) {
    if (node->type_string() == "SaveV2") {
      if (*found) {
        return errors::InvalidArgument("Input graph has multiple SaveV2 ops.");
      }
      *save_op = node;
      *found = true;
      TF_RETURN_IF_ERROR(node->input_edges(in_edges));
    }
  }
  return absl::OkStatus();
}
Node* FindRestoreAllOp(const Graph* graph, StringPiece save_prefix) {
  for (Node* node : graph->op_nodes()) {
    if (node->name() == strings::StrCat(save_prefix, "/restore_all")) {
      return node;
    }
  }
  return nullptr;
}
StringPiece GetNodeNamePrefix(const Node* node) {
  StringPiece name = node->name();
  return name.substr(0, name.rfind('/'));
}
void FillStringTensor(Tensor* dst, const Tensor& src) {
  auto dst_flat = dst->flat<tstring>();
  auto src_flat = src.flat<tstring>();
  for (int i = 0; i < src.NumElements(); i++) {
    dst_flat(i) = src_flat(i);
  }
}
Status ConnectVariablesToSaveOp(Graph* graph, Node* save_op,
                                const std::vector<const Edge*>& in_edges,
                                const std::vector<Node*>& added_variables) {
  Node* tensor_names_op = in_edges[1]->src();
  Node* shape_and_slices_op = in_edges[2]->src();
  Tensor tensor_names;
  Tensor shape_and_slices;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(tensor_names_op->attrs(), "value", &tensor_names));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(shape_and_slices_op->attrs(), "value", &shape_and_slices));
  int tn_size = tensor_names.NumElements();
  int var_size = added_variables.size();
  NodeBuilder save_op_builder =
      NodeBuilder(save_op->name(), save_op->type_string());
  for (int i = 0; i < 3; i++) {
    save_op_builder = save_op_builder.Input(in_edges[i]->src());
  }
  std::vector<NodeBuilder::NodeOut> var_nodeouts;
  var_nodeouts.reserve(tn_size + var_size);
  for (int i = 3; i < in_edges.size(); i++) {
    var_nodeouts.emplace_back(in_edges[i]->src());
  }
  Tensor new_tensor_names(DT_STRING, TensorShape({tn_size + var_size}));
  Tensor new_shape_and_slices(DT_STRING, TensorShape({tn_size + var_size}));
  FillStringTensor(&new_tensor_names, tensor_names);
  FillStringTensor(&new_shape_and_slices, shape_and_slices);
  for (int i = 0; i < var_size; i++) {
    Node* var = added_variables[i];
    new_tensor_names.flat<tstring>()(tn_size + i) = var->name();
    new_shape_and_slices.flat<tstring>()(tn_size + i) = "";
    var_nodeouts.emplace_back(var);
  }
  save_op_builder = save_op_builder.Input(var_nodeouts);
  tensor_names_op->AddAttr("value", new_tensor_names);
  shape_and_slices_op->AddAttr("value", new_shape_and_slices);
  Node* new_save_op;
  TF_RETURN_IF_ERROR(save_op_builder.Finalize(graph, &new_save_op));
  for (const Edge* edge : save_op->out_edges()) {
    graph->AddControlEdge(new_save_op, edge->dst());
  }
  graph->RemoveNode(save_op);
  return absl::OkStatus();
}
Status AddRestoreVariableSubgraphs(Graph* graph, Node* save_op,
                                   const std::vector<const Edge*>& in_edges,
                                   const std::vector<Node*>& variables) {
  Node* prefix_op = in_edges[0]->src();
  StringPiece name_prefix = GetNodeNamePrefix(save_op);
  Node* restore_all = FindRestoreAllOp(graph, name_prefix);
  if (restore_all == nullptr) {
    return errors::InvalidArgument("graph has SaveOp, but no restore_all NoOp");
  }
  const string restore_op_name = strings::StrCat(name_prefix, "/RestoreV2");
  const string assign_op_name = strings::StrCat(name_prefix, "/Assign");
  for (Node* var : variables) {
    string new_restore_op_name =
        strings::StrCat(graph->NewName(restore_op_name), "_qt");
    string new_assign_op_name =
        strings::StrCat(graph->NewName(assign_op_name), "_qt");
    string tensor_names_op_name =
        strings::StrCat(new_restore_op_name, "/tensor_names");
    string shape_and_slices_op_name =
        strings::StrCat(new_restore_op_name, "/shape_and_slices");
    Node* tensor_names;
    Tensor tensor_names_val(DT_STRING, TensorShape({1}));
    tensor_names_val.flat<tstring>()(0) = var->name();
    TF_RETURN_IF_ERROR(NodeBuilder(tensor_names_op_name, "Const")
                           .Attr("dtype", DT_STRING)
                           .Attr("value", tensor_names_val)
                           .Finalize(graph, &tensor_names));
    Node* shape_and_slices;
    Tensor shape_and_slices_val(DT_STRING, TensorShape({1}));
    shape_and_slices_val.flat<tstring>()(0) = "";
    TF_RETURN_IF_ERROR(NodeBuilder(shape_and_slices_op_name, "Const")
                           .Attr("dtype", DT_STRING)
                           .Attr("value", shape_and_slices_val)
                           .Finalize(graph, &shape_and_slices));
    Node* restore_op;
    TF_RETURN_IF_ERROR(NodeBuilder(new_restore_op_name, "RestoreV2")
                           .Input(prefix_op)
                           .Input(tensor_names)
                           .Input(shape_and_slices)
                           .Attr("dtypes", {DT_FLOAT})
                           .Finalize(graph, &restore_op));
    Node* assign_op;
    TF_RETURN_IF_ERROR(NodeBuilder(new_assign_op_name, "Assign")
                           .Input(var)
                           .Input(restore_op)
                           .Finalize(graph, &assign_op));
    graph->AddControlEdge(assign_op, restore_all);
  }
  return absl::OkStatus();
}
Status AddSaveAndRestore(Graph* graph, const std::vector<Node*>& variables) {
  Node* save_op = nullptr;
  std::vector<const Edge*> in_edges;
  bool found = false;
  TF_RETURN_IF_ERROR(FindSaveOp(graph, &save_op, &in_edges, &found));
  if (found) {
    TF_RETURN_IF_ERROR(
        AddRestoreVariableSubgraphs(graph, save_op, in_edges, variables));
    TF_RETURN_IF_ERROR(
        ConnectVariablesToSaveOp(graph, save_op, in_edges, variables));
  }
  return absl::OkStatus();
}
Status MakeReductionAxes(Graph* graph, string name_prefix, Node* input,
                         Node** output) {
  name_prefix = strings::StrCat(name_prefix, "/ReductionAxes");
  Node* start;
  Tensor zero_tensor(DT_INT32, TensorShape());
  zero_tensor.flat<int32>()(0) = 0;
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/RangeStart"), "Const")
          .Attr("dtype", DT_INT32)
          .Attr("value", zero_tensor)
          .Finalize(graph, &start));
  Node* delta;
  Tensor one_tensor(DT_INT32, TensorShape());
  one_tensor.flat<int32>()(0) = 1;
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/RangeDelta"), "Const")
          .Attr("dtype", DT_INT32)
          .Attr("value", one_tensor)
          .Finalize(graph, &delta));
  Node* rank;
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/InputRank"), "Rank")
          .Input(input)
          .Finalize(graph, &rank));
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/ReductionAxes"), "Range")
          .Input(start)
          .Input(rank)
          .Input(delta)
          .Finalize(graph, output));
  return absl::OkStatus();
}
Status MakeExponentialMovingAverage(Graph* graph, string name_prefix,
                                    const NodeBuilder::NodeOut& input,
                                    Node* decay, Node* update_variable,
                                    Node** assign_value) {
  name_prefix = strings::StrCat(name_prefix, "/EMA");
  Node* one;
  Tensor one_tensor(DT_FLOAT, TensorShape());
  one_tensor.flat<float>()(0) = 1.0;
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/OneConst"), "Const")
          .Attr("dtype", DT_FLOAT)
          .Attr("value", one_tensor)
          .Finalize(graph, &one));
  Node* decay_complement;
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/DecayComplement"), "Sub")
          .Input(one)
          .Input(decay)
          .Finalize(graph, &decay_complement));
  Node* value_diff;
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/ValueDiff"), "Sub")
          .Input(update_variable)
          .Input(input)
          .Finalize(graph, &value_diff));
  Node* update_value;
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/UpdateValue"), "Mul")
          .Input(value_diff)
          .Input(decay_complement)
          .Finalize(graph, &update_value));
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/EMAValue"), "Sub")
          .Input(update_variable)
          .Input(update_value)
          .Finalize(graph, assign_value));
  return absl::OkStatus();
}
Status MakeInitializedEMAVariable(Graph* graph, const string& name, Node* decay,
                                  Node* init_val,
                                  std::vector<Node*>* added_variables,
                                  Node** var) {
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name, "/Variable"), "VariableV2")
          .Attr("shape", TensorShape())
          .Attr("dtype", DT_FLOAT)
          .Finalize(graph, var));
  added_variables->push_back(*var);
  Node* is_initialized;
  TF_RETURN_IF_ERROR(NodeBuilder(strings::StrCat(name, "/IsInitialized"),
                                 "IsVariableInitialized")
                         .Input(*var)
                         .Finalize(graph, &is_initialized));
  Node* switch_node;
  TF_RETURN_IF_ERROR(NodeBuilder(strings::StrCat(name, "/Switch"), "Switch")
                         .Input(init_val)
                         .Input(is_initialized)
                         .Finalize(graph, &switch_node));
  NodeBuilder::NodeOut output_false = NodeBuilder::NodeOut(switch_node, 0);
  NodeBuilder::NodeOut output_true = NodeBuilder::NodeOut(switch_node, 1);
  Node* ema_value;
  TF_RETURN_IF_ERROR(MakeExponentialMovingAverage(graph, name, output_true,
                                                  decay, *var, &ema_value));
  Node* assign_value;
  TF_RETURN_IF_ERROR(NodeBuilder(strings::StrCat(name, "/Merge"), "Merge")
                         .Input({output_false, ema_value})
                         .Finalize(graph, &assign_value));
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name, "/AssignValue"), "Assign")
          .Input(*var)
          .Input(assign_value)
          .Finalize(graph, var));
  return absl::OkStatus();
}
Status MakeEMAMinMaxVars(Graph* graph, const string& name_prefix, Node* input,
                         std::vector<Node*>* added_variables, Node** min_var,
                         Node** max_var) {
  Tensor decay_tensor(DT_FLOAT, TensorShape());
  decay_tensor.flat<float>()(0) = kEMADecay;
  Node* decay;
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat(name_prefix, "/Decay"), "Const")
          .Attr("dtype", DT_FLOAT)
          .Attr("value", decay_tensor)
          .Finalize(graph, &decay));
  Node* reduction_axes;
  TF_RETURN_IF_ERROR(
      MakeReductionAxes(graph, name_prefix, input, &reduction_axes));
  Node* min;
  string min_name = strings::StrCat(name_prefix, "/Min");
  TF_RETURN_IF_ERROR(NodeBuilder(min_name, "Min")
                         .Input(input)
                         .Input(reduction_axes)
                         .Finalize(graph, &min));
  Node* max;
  string max_name = strings::StrCat(name_prefix, "/Max");
  TF_RETURN_IF_ERROR(NodeBuilder(max_name, "Max")
                         .Input(input)
                         .Input(reduction_axes)
                         .Finalize(graph, &max));
  TF_RETURN_IF_ERROR(MakeInitializedEMAVariable(graph, min_name, decay, min,
                                                added_variables, min_var));
  TF_RETURN_IF_ERROR(MakeInitializedEMAVariable(graph, max_name, decay, max,
                                                added_variables, max_var));
  return absl::OkStatus();
}
Status MakeInputMinMax(Graph* graph, const string& name_prefix,
                       const EdgeToConvert& edge,
                       std::vector<Node*>* added_variables, Node** input_min,
                       Node** input_max) {
  if (edge.range_given) {
    Tensor input_min_tensor(DT_FLOAT, TensorShape());
    input_min_tensor.flat<float>()(0) = edge.input_min;
    TF_RETURN_IF_ERROR(
        NodeBuilder(strings::StrCat(name_prefix, "/InputMin"), "Const")
            .Attr("dtype", DT_FLOAT)
            .Attr("value", input_min_tensor)
            .Finalize(graph, input_min));
    Tensor input_max_tensor(DT_FLOAT, TensorShape());
    input_max_tensor.flat<float>()(0) = edge.input_max;
    TF_RETURN_IF_ERROR(
        NodeBuilder(strings::StrCat(name_prefix, "/InputMax"), "Const")
            .Attr("dtype", DT_FLOAT)
            .Attr("value", input_max_tensor)
            .Finalize(graph, input_max));
  } else {
    TF_RETURN_IF_ERROR(MakeEMAMinMaxVars(graph, name_prefix, edge.edge->src(),
                                         added_variables, input_min,
                                         input_max));
  }
  return absl::OkStatus();
}
Status MakeQuantizeOp(Graph* graph, const string& name_prefix,
                      const string& quant_op_type, const EdgeToConvert& edge,
                      std::vector<Node*>* added_variables,
                      Node** convert_node) {
  Node* input_min;
  Node* input_max;
  TF_RETURN_IF_ERROR(MakeInputMinMax(graph, name_prefix, edge, added_variables,
                                     &input_min, &input_max));
  string quant_name = strings::StrCat(name_prefix, "/", quant_op_type);
  if (quant_op_type == "QuantizeAndDequantizeV2") {
    TF_RETURN_IF_ERROR(NodeBuilder(quant_name, quant_op_type)
                           .Input(edge.edge->src())
                           .Input(input_min)
                           .Input(input_max)
                           .Attr("signed_input", edge.signed_input)
                           .Attr("num_bits", edge.num_bits)
                           .Attr("range_given", true)
                           .Finalize(graph, convert_node));
  } else if (quant_op_type == "FakeQuantWithMinMaxVars") {
    TF_RETURN_IF_ERROR(NodeBuilder(quant_name, quant_op_type)
                           .Input(edge.edge->src())
                           .Input(input_min)
                           .Input(input_max)
                           .Attr("num_bits", edge.num_bits)
                           .Finalize(graph, convert_node));
  } else {
    return errors::InvalidArgument("Unknown quant op type: ", quant_op_type);
  }
  return absl::OkStatus();
}
Status ProcessTargetEdges(Graph* graph, const string& quant_op_type,
                          const std::vector<EdgeToConvert>& target_edges) {
  std::unordered_map<string, Node*, StringPieceHasher> name_index;
  std::vector<Node*> added_variables;
  for (const EdgeToConvert edge : target_edges) {
    Node* convert_node;
    string name_prefix = edge.edge->src()->name();
    auto iter = name_index.find(name_prefix);
    if (iter == name_index.end()) {
      TF_RETURN_IF_ERROR(MakeQuantizeOp(graph, name_prefix, quant_op_type, edge,
                                        &added_variables, &convert_node));
      name_index[name_prefix] = convert_node;
    } else {
      convert_node = iter->second;
    }
    graph->AddEdge(convert_node, 0, edge.edge->dst(), edge.edge->dst_input());
    graph->RemoveEdge(edge.edge);
  }
  TF_RETURN_IF_ERROR(AddSaveAndRestore(graph, added_variables));
  return absl::OkStatus();
}
}  
Status DoQuantizeTraining(int32_t num_bits, const string& quant_op_type,
                          Graph* graph) {
  if (graph == nullptr) {
    return errors::InvalidArgument("Cannot accept empty graph pointer.");
  }
  if (num_bits < 1 || num_bits > 63) {
    return errors::OutOfRange("num_bits should be in range [1, 63] but is: ",
                              num_bits);
  }
  int potential_input = 0;
  std::vector<EdgeToConvert> target_edges;
  for (Node* node : graph->nodes()) {
    if (nodes_to_rewrite->find(node->type_string()) !=
            nodes_to_rewrite->end() &&
        !IsGradientNode(graph, node)) {
      for (const Edge* edge : node->in_edges()) {
        if (edge->src_output() == Graph::kControlSlot) {
          continue;
        } else {
          bool signed_input = false;
          bool range_given = false;
          float input_min = 0;
          float input_max = 0;
          bool known_op = FindType(graph, edge->src(), &signed_input,
                                   &range_given, &input_min, &input_max);
          if (!known_op) {
            potential_input++;
            if (potential_input > kAllowedInputs) {
              return errors::Unimplemented(
                  "Found an unknown op: ", edge->src()->name(),
                  " with type: ", edge->src()->type_string(),
                  "; Unknown ops are considered as model input for now and "
                  "only ",
                  kAllowedInputs, " inputs are supported currently.");
            }
          }
          target_edges.emplace_back(EdgeToConvert(
              edge, num_bits, signed_input, range_given, input_min, input_max));
        }
      }
    }
  }
  TF_RETURN_IF_ERROR(ProcessTargetEdges(graph, quant_op_type, target_edges));
  return absl::OkStatus();
}
Status DoQuantizeTrainingOnGraphDef(const GraphDef& input_graphdef,
                                    int32_t num_bits,
                                    const string& quant_op_type,
                                    GraphDef* result_graphdef) {
  Graph graph(OpRegistry::Global());
  GraphConstructorOptions opts;
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, input_graphdef, &graph));
  TF_RETURN_IF_ERROR(DoQuantizeTraining(num_bits, quant_op_type, &graph));
  graph.ToGraphDef(result_graphdef);
  return absl::OkStatus();
}
Status DoQuantizeTrainingOnSerializedGraphDef(const string& input_graph_string,
                                              int32_t num_bits,
                                              const string& quant_op_type,
                                              string* result_graph_string) {
  GraphDef input_graphdef;
  if (!ParseProtoUnlimited(&input_graphdef, input_graph_string)) {
    return errors::InvalidArgument(
        "input_graph_string is not a serialized GraphDef protocol buffer");
  }
  GraphDef output_graphdef;
  TF_RETURN_IF_ERROR(DoQuantizeTrainingOnGraphDef(
      input_graphdef, num_bits, quant_op_type, &output_graphdef));
  if (!output_graphdef.SerializeToString(result_graph_string)) {
    return errors::Internal(
        "quantize training transformation resulted in invalid GraphDef");
  }
  return absl::OkStatus();
}
}  