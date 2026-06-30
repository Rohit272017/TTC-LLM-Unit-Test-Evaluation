#include "tensorflow/core/common_runtime/eval_const_tensor.h"
#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/strcat.h"
namespace tensorflow {
namespace {
using ::tensorflow::shape_inference::InferenceContext;
bool IsRank(const Node& n) { return n.type_string() == "Rank"; }
bool IsSize(const Node& n) { return n.type_string() == "Size"; }
bool IsShape(const Node& n) { return n.type_string() == "Shape"; }
bool IsStridedSlice(const Node& n) { return n.type_string() == "StridedSlice"; }
bool IsPlaceholderWithDefault(const Node& n) {
  return n.type_string() == "PlaceholderWithDefault";
}
bool IsUnstack(const Node& n) { return n.type_string() == "Unpack"; }
bool HasIntAttr(const Node& n, absl::string_view name, int64_t expected) {
  int64_t actual;
  return TryGetNodeAttr(n.def(), name, &actual) && actual == expected;
}
std::optional<int64_t> GetIntConst(const Node& node) {
  const TensorProto* proto;
  Tensor tensor;
  if (node.IsConstant() && TryGetNodeAttr(node.def(), "value", &proto) &&
      (proto->dtype() == DT_INT32 || proto->dtype() == DT_INT64) &&
      TensorShape(proto->tensor_shape()).num_elements() == 1 &&
      tensor.FromProto(*proto)) {
    if (proto->dtype() == DT_INT32) {
      return *static_cast<const int32_t*>(tensor.data());
    } else {
      return *static_cast<const int64_t*>(tensor.data());
    }
  }
  return std::nullopt;
}
std::optional<int64_t> GetSliceIndex(const Node& node, const int node_output) {
  std::optional<int64_t> ix;
  if (IsUnstack(node)) {
    if (HasIntAttr(node, "axis", 0)) {
      ix = node_output;
    }
  } else if (IsStridedSlice(node)) {
    const Edge* edge;
    if (HasIntAttr(node, "begin_mask", 0) && HasIntAttr(node, "end_mask", 0) &&
        HasIntAttr(node, "ellipsis_mask", 0) &&
        HasIntAttr(node, "new_axis_mask", 0) &&
        HasIntAttr(node, "shrink_axis_mask", 1) &&
        node.input_edge(1, &edge).ok()) {
      ix = GetIntConst(*edge->src());
    }
  }
  return ix;
}
absl::StatusOr<std::optional<Tensor>> TryInferFromShapes(
    const Node& node, const int node_output, const ShapeRefiner& refiner) {
  std::optional<Tensor> result;
  if (node.num_inputs() == 0 || node_output >= node.num_outputs()) {
    return result;
  }
  const auto dtype = node.output_type(node_output);
  if (dtype != DT_INT32 && dtype != DT_INT64) {
    return result;
  }
  absl::InlinedVector<int64_t, 8> data;
  std::optional<TensorShape> shape;
  const Edge* edge;
  if (IsShape(node)) {
    InferenceContext* c = refiner.GetContext(&node);
    if (c != nullptr && c->FullyDefined(c->input(0))) {
      const int64_t rank = c->Rank(c->input(0));
      for (int i = 0; i < rank; ++i) {
        data.push_back(c->Value(c->Dim(c->input(0), i)));
      }
      shape.emplace({rank});
    }
  } else if (IsRank(node)) {
    InferenceContext* c = refiner.GetContext(&node);
    if (c != nullptr && c->RankKnown(c->input(0))) {
      data.push_back(c->Rank(c->input(0)));
      shape.emplace();
    }
  } else if (IsSize(node)) {
    InferenceContext* c = refiner.GetContext(&node);
    if (c != nullptr && c->FullyDefined(c->input(0))) {
      int64_t size = 1;
      for (int i = 0, rank = c->Rank(c->input(0)); i < rank; i++) {
        size *= c->Value(c->Dim(c->input(0), i));
      }
      data.push_back(size);
      shape.emplace();
    }
  } else if (node.input_edge(0, &edge).ok() && IsShape(*edge->src())) {
    InferenceContext* c = refiner.GetContext(edge->src());
    if (c != nullptr && c->RankKnown(c->input(0))) {
      const int64_t rank = c->Rank(c->input(0));
      std::optional<int64_t> ix = GetSliceIndex(node, node_output);
      if (ix.has_value() && -rank <= *ix && *ix < rank &&
          c->ValueKnown(c->Dim(c->input(0), *ix))) {
        data.push_back(c->Value(c->Dim(c->input(0), *ix)));
        shape.emplace();
      }
    }
  }
  if (!shape.has_value()) {
    return result;
  }
  if (dtype == DT_INT32) {
    for (const int64_t value : data) {
      if (TF_PREDICT_FALSE(value >= std::numeric_limits<int32_t>::max())) {
        return errors::InvalidArgument("Value is out of int32 range: ", value);
      }
    }
  }
  result.emplace(dtype, *shape);
  if (dtype == DT_INT32) {
    absl::c_copy(data, static_cast<int32_t*>(result->data()));
  } else {
    absl::c_copy(data, static_cast<int64_t*>(result->data()));
  }
  return result;
}
bool IsSupportedForEvaluation(const Node& node) {
  if (node.IsConstant() || node.IsArg()) {
    return true;
  }
  if (node.num_inputs() == 0 || IsPlaceholderWithDefault(node)) {
    return false;
  }
  if (node.op_def().is_stateful()) {
    return false;
  }
  if (node.IsEnter() || node.IsExit() || node.IsMerge()) {
    return false;
  }
  if (node.IsFunctionCall()) {
    return false;
  }
  for (const auto& [name, attr] : node.attrs()) {
    if (attr.has_func() || !attr.list().func().empty()) {
      return false;
    }
  }
  return KernelDefAvailable(DEVICE_CPU, node.def());
}
struct Subgraph {
  Subgraph(const OpRegistryInterface* op_registry, int32_t graph_def_version)
      : graph(op_registry == nullptr ? OpRegistry::Global() : op_registry) {
    VersionDef versions = graph.versions();
    versions.set_producer(graph_def_version);
    graph.set_versions(versions);
  }
  GraphRunner::NamedTensorList inputs;
  Graph graph;
};
using NodeOutput = std::pair<const Node*, int>;
std::string OutputName(const NodeOutput& output) {
  return strings::StrCat(output.first->name(), ":", output.second);
}
absl::StatusOr<std::unique_ptr<Subgraph>> ExtractConstantSubgraph(
    const Node& target_node, const ShapeRefiner& refiner,
    const absl::FunctionRef<std::optional<Tensor>(const Node&, int)> lookup,
    const OpRegistryInterface* op_registry, const int32_t graph_def_version) {
  std::unique_ptr<Subgraph> subgraph;
  if (!target_node.IsEnter() && !IsSupportedForEvaluation(target_node)) {
    return subgraph;
  }
  std::vector<const Edge*> edges;
  for (const Edge* edge : target_node.in_edges()) {
    if (!edge->IsControlEdge()) {
      edges.push_back(edge);
    }
  }
  absl::flat_hash_map<const Node*, Node*> new_by_old_node;
  absl::InlinedVector<const Node*, 8> arg_nodes;
  absl::flat_hash_map<NodeOutput, Tensor> const_inputs;
  for (int edge_ix = 0; edge_ix < edges.size(); ++edge_ix) {
    const Edge& edge = *edges[edge_ix];
    const Node& node = *edge.src();
    const NodeOutput node_output = {&node, edge.src_output()};
    if (new_by_old_node.contains(&node) || const_inputs.contains(node_output)) {
      continue;
    }
    if (node.IsArg()) {
      arg_nodes.push_back(&node);
      continue;
    }
    auto tensor = lookup(node, node_output.second);
    if (!tensor.has_value()) {
      TF_ASSIGN_OR_RETURN(
          tensor, TryInferFromShapes(node, node_output.second, refiner));
    }
    if (tensor.has_value()) {
      const_inputs.emplace(node_output, *std::move(tensor));
    } else if (!IsSupportedForEvaluation(node)) {
      return subgraph;
    } else {
      new_by_old_node.emplace(&node,  nullptr);
      for (const Edge* edge : node.in_edges()) {
        if (!edge->IsControlEdge()) {
          edges.push_back(edge);
        }
      }
    }
  }
  bool all_args_provided = true;
  for (const Node* node : arg_nodes) {
    auto tensor = lookup(*node, 0);
    all_args_provided = all_args_provided && tensor.has_value();
    if (all_args_provided) {
      const_inputs.emplace(NodeOutput{node, 0}, *std::move(tensor));
    }
  }
  if (!all_args_provided) {
    return subgraph;
  }
  subgraph = std::make_unique<Subgraph>(op_registry, graph_def_version);
  auto& inputs = subgraph->inputs;
  inputs.reserve(const_inputs.size());
  for (auto& [node_output, tensor] : const_inputs) {
    if (!new_by_old_node.contains(node_output.first)) {
      inputs.emplace_back(OutputName(node_output), std::move(tensor));
    }
  }
  Graph& graph = subgraph->graph;
  new_by_old_node[&target_node] = graph.CopyNode(&target_node);
  for (const Edge* edge : edges) {
    Node*& src = new_by_old_node[edge->src()];
    if (src == nullptr) {
      src = graph.CopyNode(edge->src());
    }
    Node* dst = new_by_old_node.at(edge->dst());
    graph.AddEdge(src, edge->src_output(), dst, edge->dst_input());
  }
  return subgraph;
}
}  
absl::StatusOr<std::optional<Tensor>> EvaluateConstantTensor(
    const Node& node, const int node_output, const ShapeRefiner& refiner,
    const absl::FunctionRef<std::optional<Tensor>(const Node&, int)> lookup,
    const std::optional<EvaluateConstantTensorRunner> runner) {
  std::optional<Tensor> result;
  if (result = lookup(node, node_output); result.has_value()) {
    return result;
  }
  if (node.IsArg()) {
    return result;
  }
  if (node.IsConstant()) {
    const TensorProto* proto;
    TF_RETURN_IF_ERROR(GetNodeAttr(node.def(), "value", &proto));
    result.emplace();
    if (TF_PREDICT_FALSE(!result->FromProto(*proto))) {
      return errors::InvalidArgument("Unable to evaluate a constant node");
    }
    return result;
  }
  TF_ASSIGN_OR_RETURN(result, TryInferFromShapes(node, node_output, refiner));
  if (result.has_value()) {
    return result;
  }
  if (!runner.has_value()) {
    return result;
  }
  TF_ASSIGN_OR_RETURN(
      const auto subgraph,
      ExtractConstantSubgraph(node, refiner, lookup, runner->op_registry,
                              runner->graph_def_version));
  if (subgraph != nullptr) {
    GraphRunner* graph_runner = runner->graph_runner;
    std::unique_ptr<GraphRunner> tmp_graph_runner;
    if (graph_runner == nullptr) {
      tmp_graph_runner = std::make_unique<GraphRunner>(Env::Default());
      graph_runner = tmp_graph_runner.get();
    }
    FunctionLibraryRuntime* function_library = nullptr;
    std::vector<Tensor> outputs;
    auto status =
        graph_runner->Run(&subgraph->graph, function_library, subgraph->inputs,
                          {OutputName({&node, node_output})}, &outputs);
    if (status.ok()) {
      result = std::move(outputs[0]);
    }
  }
  return result;
}
}  