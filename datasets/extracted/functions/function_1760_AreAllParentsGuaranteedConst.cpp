#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/mark_for_compilation_pass.h"
#include "tensorflow/compiler/jit/shape_inference_helpers.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "xla/service/graphcycles/graphcycles.h"
#include "xla/status_macros.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/dump_graph.h"
namespace tensorflow {
const char* const kXlaCompiledKernelAttr = "_XlaCompiledKernel";
const char* const kXlaNumConstantArgsAttr = "_XlaNumConstantArgs";
const char* const kXlaNumResourceArgsAttr = "_XlaNumResourceArgs";
const char* const kXlaHostTransferSequencerAttr =
    "_xla_host_transfer_sequencer";
const char* const kXlaHasReferenceVarsAttr = "_XlaHasReferenceVars";
namespace {
bool AreAllParentsGuaranteedConst(
    const Node& n,
    const absl::flat_hash_set<const Node*>& runtime_const_nodes) {
  if (n.type_string() == "GuaranteeConst") {
    return true;
  }
  bool all_parents_const = true;
  bool atleast_one_non_control_edge = false;
  for (const Edge* in : n.in_edges()) {
    atleast_one_non_control_edge =
        atleast_one_non_control_edge || !in->IsControlEdge();
    if (!in->IsControlEdge() && runtime_const_nodes.count(in->src()) == 0) {
      all_parents_const = false;
      break;
    }
  }
  return all_parents_const && atleast_one_non_control_edge;
}
void MarkGuaranteedConstants(
    const Graph& graph,
    const std::vector<std::pair<const Node*, Node*>>& src_arg_pairs) {
  absl::flat_hash_set<const Node*> guaranteed_const_nodes;
  std::vector<const Node*> srcs;
  srcs.reserve(src_arg_pairs.size());
  for (const auto& src_arg : src_arg_pairs) {
    srcs.push_back(src_arg.first);
  }
  ReverseDFSFrom(
      graph, srcs, nullptr,
      [&guaranteed_const_nodes](const Node* n) {
        if (AreAllParentsGuaranteedConst(*n, guaranteed_const_nodes)) {
          guaranteed_const_nodes.insert(n);
        }
      });
  for (auto& src_arg : src_arg_pairs) {
    if (guaranteed_const_nodes.count(src_arg.first) != 0) {
      VLOG(1) << "Guaranteed const found: " << src_arg.first->DebugString();
      src_arg.second->AddAttr("_is_guaranteed_constant", true);
    }
  }
}
struct OutputInputTensorPairHasher {
  uint64 operator()(std::pair<OutputTensor, InputTensor> const& s) const {
    return Hash64Combine(OutputTensor::Hash()(s.first),
                         InputTensor::Hash()(s.second));
  }
};
static const char* const kArgOp = "_Arg";
static const char* const kRetValOp = "_Retval";
class Encapsulator {
 public:
  Encapsulator(string group_attribute, Graph const* graph_in)
      : group_attribute_(std::move(group_attribute)), graph_in_(graph_in) {}
  Status SplitIntoSubgraphs(FunctionLibraryDefinition* library);
  Status BuildFunctionDefs(const RewriteSubgraphFn& rewrite_subgraph_fn,
                           bool reuse_existing_functions,
                           FunctionLibraryDefinition* library);
  Status BuildOutputGraph(Graph* graph_out, FunctionLibraryDefinition* library);
 private:
  class Subgraph {
   public:
    Node* MakeNodeImage(const Graph* graph_in, Node* node);
    Graph* GetGraph() const;
    Status BuildFunctionDef(const string& name_in,
                            const RewriteSubgraphFn& rewrite_subgraph_fn,
                            bool reuse_existing_functions,
                            FunctionLibraryDefinition* library);
    Status AddFunctionCallNode(
        const absl::flat_hash_map<const Node*, Node*>& node_images,
        Graph* graph_out);
    Node* GetCallNode() const;
    int GetArgIndexForEdge(const Edge* edge) const;
    int GetResultIndexForEdge(const Edge* edge) const;
    Status RecordArg(const Edge* edge,
                     const absl::flat_hash_map<const Node*, Node*>& node_images,
                     std::vector<std::pair<const Node*, Node*>>* src_arg_pairs);
    Status RecordControlResult(
        const Edge* edge,
        const absl::flat_hash_map<const Node*, Node*>& node_images);
    Status RecordResult(
        const Edge* edge,
        const absl::flat_hash_map<const Node*, Node*>& node_images);
    Status MakeSequencingNode(const string& subgraph_name, Graph* graph_out);
    void ConnectSequencerToCallNode(Graph* graph_out);
    Status ReplaceFunctionDef(FunctionLibraryDefinition* library);
   private:
    std::unique_ptr<Graph> graph_;
    string device_;
    NodeDef call_node_def_;
    string function_def_name_;
    Node* host_compute_key_placeholder_ = nullptr;
    Node* call_node_;
    absl::flat_hash_map<OutputTensor, int, OutputTensor::Hash> args_by_src_;
    absl::flat_hash_map<InputTensor, int, InputTensor::Hash> args_by_dst_;
    std::vector<Node*> args_;
    absl::flat_hash_map<OutputTensor, int, OutputTensor::Hash> results_;
    absl::flat_hash_set<string> control_output_nodes_;
    Node* sequencer_ = nullptr;
  };
  Status GetFunctionNameAttr(Node const* node, string* attr) const;
  Status CopySubgraphEdges(
      const absl::flat_hash_map<const Node*, Node*>& node_images,
      std::vector<std::pair<const Node*, Node*>>* src_arg_pairs);
  Status CopySubgraphNodes(
      absl::flat_hash_map<const Node*, Node*>* node_images);
  Status CopyNodesToOutputGraph(
      Graph* graph_out, absl::flat_hash_map<const Node*, Node*>* node_images);
  Status AddFunctionCallNodes(
      const absl::flat_hash_map<const Node*, Node*>& node_images,
      Graph* graph_out);
  Status FindOutputImageOfEdgeSrc(
      const string& src_func_id, const string& dst_func_id,
      const absl::flat_hash_map<const Node*, Node*>& node_images,
      const Node* original_src_node, Node** src_image);
  int FindOutputSlotOfEdgeSrc(const string& src_func_id,
                              const string& dst_func_id,
                              const Edge* edge);
  Status FindOutputImageOfEdgeDst(
      const string& src_func_id, const string& dst_func_id,
      const absl::flat_hash_map<const Node*, Node*>& node_images,
      const Node* original_dst_node, Node** dst_image);
  int FindOutputSlotOfEdgeDst(const string& src_func_id,
                              const string& dst_func_id,
                              const Edge* edge);
  Status CopyEdgeToOutputGraph(
      const Edge* edge, const string& src_func_id, const string& dst_func_id,
      const absl::flat_hash_map<const Node*, Node*>& node_images,
      Graph* graph_out,
      absl::flat_hash_set<std::pair<OutputTensor, InputTensor>,
                          OutputInputTensorPairHasher>* edges_added);
  Status AddEdgesToOutputGraph(
      const absl::flat_hash_map<const Node*, Node*>& node_images,
      Graph* graph_out);
  Status MakePrunedGraphCopyAndInline(
      const Graph& graph, const std::vector<Node*>& sink_nodes,
      std::unique_ptr<Graph>* pruned_graph,
      absl::flat_hash_map<const Node*, Node*>* node_images,
      FunctionLibraryDefinition* library);
  const string group_attribute_;
  const Graph* graph_in_;
  absl::flat_hash_map<string, Subgraph> subgraphs_;
  Encapsulator(const Encapsulator&) = delete;
  void operator=(const Encapsulator&) = delete;
};
namespace {
void TopologicalClusterSort(
    const absl::flat_hash_set<string>& clusters,
    const absl::flat_hash_set<string>& has_successors,
    const absl::flat_hash_map<string, absl::flat_hash_set<string>>& ancestors,
    std::vector<string>* sorted) {
  sorted->clear();
  struct Work {
    string cluster;
    bool leave;
  };
  std::set<string> visited;
  std::vector<Work> stack;
  for (const auto& cluster : clusters) {
    if (has_successors.find(cluster) == has_successors.end()) {
      stack.push_back({cluster, false});
    }
  }
  while (!stack.empty()) {
    const Work item = stack.back();
    stack.pop_back();
    if (item.leave) {
      sorted->push_back(item.cluster);
      continue;
    }
    if (visited.find(item.cluster) != visited.end()) continue;
    visited.insert(item.cluster);
    stack.push_back({item.cluster, true});
    const auto& iter = ancestors.find(item.cluster);
    if (iter != ancestors.end()) {
      for (const auto& ancestor : iter->second) {
        stack.push_back({ancestor, false});
      }
    }
  }
  CHECK(sorted->size() == clusters.size());
}
}  
Node* Encapsulator::Subgraph::GetCallNode() const { return call_node_; }
int Encapsulator::Subgraph::GetArgIndexForEdge(const Edge* edge) const {
  return args_by_dst_.at(InputTensor(edge->dst(), edge->dst_input()));
}
int Encapsulator::Subgraph::GetResultIndexForEdge(const Edge* edge) const {
  return results_.at(OutputTensor(edge->src(), edge->src_output()));
}
Node* Encapsulator::Subgraph::MakeNodeImage(const Graph* graph_in, Node* node) {
  if (!graph_) {
    graph_.reset(new Graph(graph_in->op_registry()));
    graph_->set_versions(graph_in->versions());
  }
  if (device_.empty()) {
    device_ = node->assigned_device_name().empty()
                  ? node->requested_device()
                  : node->assigned_device_name();
  }
  return graph_->CopyNode(node);
}
Graph* Encapsulator::Subgraph::GetGraph() const { return graph_.get(); }
Status Encapsulator::Subgraph::RecordArg(
    const Edge* edge,
    const absl::flat_hash_map<const Node*, Node*>& node_images,
    std::vector<std::pair<const Node*, Node*>>* src_arg_pairs) {
  Node* src_node = edge->src();
  int src_slot = edge->src_output();
  absl::flat_hash_map<OutputTensor, int, OutputTensor::Hash>::iterator iter;
  bool inserted;
  std::tie(iter, inserted) = args_by_src_.emplace(
      OutputTensor(src_node, src_slot), args_by_src_.size());
  int arg_index = iter->second;
  if (inserted) {
    NodeDef arg_def;
    NodeDefBuilder builder(
        absl::StrCat(src_node->name(), "_", src_slot, "_arg"), kArgOp,
        NodeDebugInfo(src_node->def()));
    DataType dtype = edge->dst()->input_type(edge->dst_input());
    builder.Attr("T", dtype);
    builder.Attr("index", arg_index);
    Status s = builder.Finalize(&arg_def);
    if (!s.ok()) return s;
    TF_ASSIGN_OR_RETURN(Node * arg, graph_->AddNode(arg_def));
    src_arg_pairs->push_back({src_node, arg});
    args_.push_back(arg);
  }
  Node* dst_node = edge->dst();
  Node* dst_image = node_images.at(dst_node);
  int dst_slot = edge->dst_input();
  args_by_dst_[InputTensor(dst_node, dst_slot)] = arg_index;
  graph_->AddEdge(args_[arg_index], 0, dst_image, dst_slot);
  return absl::OkStatus();
}
Status Encapsulator::Subgraph::RecordControlResult(
    const Edge* edge,
    const absl::flat_hash_map<const Node*, Node*>& node_images) {
  Node* src_node = edge->src();
  Node* src_image = node_images.at(src_node);
  control_output_nodes_.insert(src_image->name());
  return absl::OkStatus();
}
Status Encapsulator::Subgraph::RecordResult(
    const Edge* edge,
    const absl::flat_hash_map<const Node*, Node*>& node_images) {
  Node* src_node = edge->src();
  Node* src_image = node_images.at(src_node);
  int src_slot = edge->src_output();
  absl::flat_hash_map<OutputTensor, int, OutputTensor::Hash>::iterator iter;
  bool inserted;
  std::tie(iter, inserted) =
      results_.emplace(OutputTensor(src_node, src_slot), results_.size());
  int ret_index = iter->second;
  if (inserted) {
    NodeDef ret_def;
    NodeDefBuilder builder(
        absl::StrCat(src_node->name(), "_", src_slot, "_retval"), kRetValOp,
        NodeDebugInfo(src_node->def()));
    DataType dtype = src_node->output_type(src_slot);
    builder.Attr("T", dtype);
    builder.Attr("index", ret_index);
    builder.Input(src_image->name(), src_slot, dtype);
    Status s = builder.Finalize(&ret_def);
    if (!s.ok()) return s;
    TF_ASSIGN_OR_RETURN(Node * ret, graph_->AddNode(ret_def));
    graph_->AddEdge(src_image, src_slot, ret, 0);
  }
  return absl::OkStatus();
}
Status Encapsulator::Subgraph::MakeSequencingNode(const string& subgraph_name,
                                                  Graph* graph_out) {
  if (sequencer_ == nullptr) {
    NodeDef seq_def;
    NodeDefBuilder builder(absl::StrCat(subgraph_name, "_sequencer"), "NoOp");
    builder.Attr(kXlaHostTransferSequencerAttr, subgraph_name);
    builder.Device(device_);
    Status s = builder.Finalize(&seq_def);
    if (!s.ok()) return s;
    TF_ASSIGN_OR_RETURN(sequencer_, graph_out->AddNode(seq_def));
  }
  return absl::OkStatus();
}
void Encapsulator::Subgraph::ConnectSequencerToCallNode(Graph* graph_out) {
  if (sequencer_ != nullptr) {
    VLOG(2) << "ConnectSequencerToCallNode";
    graph_out->AddControlEdge(sequencer_, call_node_,
                               true);
  }
}
Status Encapsulator::Subgraph::BuildFunctionDef(
    const string& name_in, const RewriteSubgraphFn& rewrite_subgraph_fn,
    bool reuse_existing_functions, FunctionLibraryDefinition* library) {
  string name = name_in;
  call_node_def_.set_op(name);
  call_node_def_.set_name(name);
  call_node_def_.set_device(device_);
  if (rewrite_subgraph_fn) {
    std::vector<OutputTensor> arg_source_tensors(args_by_src_.size());
    for (const auto& arg : args_by_src_) {
      arg_source_tensors.at(arg.second) = arg.first;
    }
    std::vector<int> input_permutation(args_by_src_.size());
    std::iota(input_permutation.begin(), input_permutation.end(), 0);
    std::vector<int> output_permutation(results_.size());
    std::iota(output_permutation.begin(), output_permutation.end(), 0);
    TF_RETURN_IF_ERROR(
        rewrite_subgraph_fn(arg_source_tensors, &graph_, &input_permutation,
                            &output_permutation, &call_node_def_));
    if (input_permutation.size() != args_by_src_.size()) {
      return errors::InvalidArgument("Input permutation has incorrect size.");
    }
    if (output_permutation.size() != results_.size()) {
      return errors::InvalidArgument("Output permutation has incorrect size.");
    }
    for (auto& arg : args_by_src_) {
      arg.second = input_permutation[arg.second];
    }
    for (auto& arg : args_by_dst_) {
      arg.second = input_permutation[arg.second];
    }
    for (auto& result : results_) {
      result.second = output_permutation[result.second];
    }
    name = call_node_def_.op();
  }
  function_def_name_ = name;
  FunctionDef fdef;
  auto lookup = [this](const Node* node) -> std::optional<string> {
    if (control_output_nodes_.contains(node->name())) {
      return std::make_optional(node->name());
    }
    return std::nullopt;
  };
  std::vector<ControlFlowInfo> dummy;
  TF_RETURN_IF_ERROR(BuildControlFlowInfo(graph_.get(), &dummy));
  TF_RETURN_IF_ERROR(GraphToFunctionDef(*graph_, name, lookup, &fdef));
  if (VLOG_IS_ON(1)) {
    VLOG(2) << "Build function def " << name;
    DumpGraphToFile(absl::StrCat("encapsulate_fdef_graph_", name), *graph_,
                    library);
    DumpFunctionDefToFile(absl::StrCat("encapsulate_fdef_", name), fdef);
  }
  const FunctionDef* original_fdef = library->Find(name);
  if (!reuse_existing_functions || original_fdef == nullptr) {
    TF_RETURN_IF_ERROR(library->AddFunctionDef(fdef));
  } else if (!FunctionDefsEqual(*original_fdef, fdef)) {
    TF_RETURN_IF_ERROR(library->ReplaceFunction(name, fdef));
  }
  return absl::OkStatus();
}
Status Encapsulator::Subgraph::ReplaceFunctionDef(
    FunctionLibraryDefinition* library) {
  const string& name = function_def_name_;
  FunctionDef fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(*graph_, name, &fdef));
  if (VLOG_IS_ON(1)) {
    VLOG(2) << "Replace function def " << name;
    DumpGraphToFile(absl::StrCat("replace_encapsulate_fdef_graph_", name),
                    *graph_, library);
    DumpFunctionDefToFile(absl::StrCat("replace_encapsulate_fdef_", name),
                          fdef);
  }
  TF_RETURN_IF_ERROR(library->ReplaceFunction(name, fdef));
  return absl::OkStatus();
}
Status Encapsulator::Subgraph::AddFunctionCallNode(
    const absl::flat_hash_map<const Node*, Node*>& node_images,
    Graph* graph_out) {
  TF_ASSIGN_OR_RETURN(call_node_, graph_out->AddNode(call_node_def_));
  call_node_->set_assigned_device_name(device_);
  return absl::OkStatus();
}
Status Encapsulator::GetFunctionNameAttr(Node const* node, string* attr) const {
  AttrSlice attrs = node->attrs();
  attr->clear();
  for (const auto& node_attr : attrs) {
    if (node_attr.first == group_attribute_) {
      TF_RETURN_IF_ERROR(AttrValueHasType(node_attr.second, "string"));
      *attr = node_attr.second.s();
      break;
    }
  }
  return absl::OkStatus();
}
bool IsInSubgraph(const string& func_id) { return !func_id.empty(); }
Status Encapsulator::CopySubgraphNodes(
    absl::flat_hash_map<const Node*, Node*>* node_images) {
  for (Node* node : graph_in_->op_nodes()) {
    string func_id;
    TF_RETURN_IF_ERROR(GetFunctionNameAttr(node, &func_id));
    if (!IsInSubgraph(func_id)) continue;
    Subgraph& subgraph = subgraphs_[func_id];
    Node* image = subgraph.MakeNodeImage(graph_in_, node);
    image->ClearAttr(group_attribute_);
    (*node_images)[node] = image;
  }
  return absl::OkStatus();
}
Status Encapsulator::CopySubgraphEdges(
    const absl::flat_hash_map<const Node*, Node*>& node_images,
    std::vector<std::pair<const Node*, Node*>>* src_arg_pairs) {
  for (const Edge* edge : graph_in_->edges()) {
    string src_func_id;
    TF_RETURN_IF_ERROR(GetFunctionNameAttr(edge->src(), &src_func_id));
    string dst_func_id;
    TF_RETURN_IF_ERROR(GetFunctionNameAttr(edge->dst(), &dst_func_id));
    Node* src_image = gtl::FindWithDefault(node_images, edge->src(), nullptr);
    Node* dst_image = gtl::FindWithDefault(node_images, edge->dst(), nullptr);
    if (IsInSubgraph(src_func_id) && IsInSubgraph(dst_func_id) &&
        src_func_id == dst_func_id) {
      Graph* g = subgraphs_[src_func_id].GetGraph();
      if (edge->IsControlEdge()) {
        g->AddControlEdge(src_image, dst_image,
                           true);
      } else {
        g->AddEdge(src_image, edge->src_output(), dst_image, edge->dst_input());
      }
      continue;
    }
    if (IsInSubgraph(src_func_id)) {
      if (!edge->IsControlEdge()) {
        DataType dtype = edge->src()->output_type(edge->src_output());
        if (IsRefType(dtype)) {
          return errors::InvalidArgument(
              "Ref Tensors (e.g., Variables) are not supported as results: "
              "tensor ",
              edge->src()->name(), ":", edge->src_output());
        }
      }
      Subgraph& src_subgraph = subgraphs_[src_func_id];
      if (edge->IsControlEdge()) {
        TF_RETURN_IF_ERROR(src_subgraph.RecordControlResult(edge, node_images));
      } else {
        TF_RETURN_IF_ERROR(src_subgraph.RecordResult(edge, node_images));
      }
    }
    if (IsInSubgraph(dst_func_id)) {
      if (!edge->IsControlEdge()) {
        DataType dtype = edge->dst()->input_type(edge->dst_input());
        if (IsRefType(dtype)) {
          return errors::InvalidArgument(
              "Ref Tensors (e.g., Variables) are not supported as args: "
              "tensor ",
              edge->src()->name(), ":", edge->src_output());
        }
      }
      Subgraph& dst_subgraph = subgraphs_[dst_func_id];
      if (!edge->IsControlEdge()) {
        TF_RETURN_IF_ERROR(
            dst_subgraph.RecordArg(edge, node_images, src_arg_pairs));
      }
    }
  }
  return absl::OkStatus();
}
Status Encapsulator::SplitIntoSubgraphs(FunctionLibraryDefinition* library) {
  Status s;
  absl::flat_hash_map<const Node*, Node*> node_images;
  std::vector<std::pair<const Node*, Node*>> src_arg_pairs;
  TF_RETURN_IF_ERROR(CopySubgraphNodes(&node_images));
  TF_RETURN_IF_ERROR(CopySubgraphEdges(node_images, &src_arg_pairs));
  MarkGuaranteedConstants(*graph_in_, src_arg_pairs);
  for (auto& entry : subgraphs_) {
    Subgraph& subgraph = entry.second;
    FixupSourceAndSinkEdges(subgraph.GetGraph());
  }
  if (VLOG_IS_ON(1)) {
    for (auto& entry : subgraphs_) {
      DumpGraphToFile(
          absl::StrCat("encapsulate_subgraphs_subgraph_", entry.first),
          *entry.second.GetGraph(), library);
    }
  }
  return s;
}
Status Encapsulator::BuildFunctionDefs(
    const RewriteSubgraphFn& rewrite_subgraph_fn, bool reuse_existing_functions,
    FunctionLibraryDefinition* library) {
  for (auto& subgraph_entry : subgraphs_) {
    string name = subgraph_entry.first;
    Subgraph& subgraph = subgraph_entry.second;
    TF_RETURN_IF_ERROR(subgraph.BuildFunctionDef(
        name, rewrite_subgraph_fn, reuse_existing_functions, library));
  }
  return absl::OkStatus();
}
Status Encapsulator::CopyNodesToOutputGraph(
    Graph* graph_out, absl::flat_hash_map<const Node*, Node*>* node_images) {
  for (Node* node : graph_in_->op_nodes()) {
    string func_id;
    TF_RETURN_IF_ERROR(GetFunctionNameAttr(node, &func_id));
    if (IsInSubgraph(func_id)) continue;
    Node* image = graph_out->CopyNode(node);
    (*node_images)[node] = image;
  }
  (*node_images)[graph_in_->source_node()] = graph_out->source_node();
  (*node_images)[graph_in_->sink_node()] = graph_out->sink_node();
  return absl::OkStatus();
}
Status Encapsulator::AddFunctionCallNodes(
    const absl::flat_hash_map<const Node*, Node*>& node_images,
    Graph* graph_out) {
  for (auto& subgraph_entry : subgraphs_) {
    TF_RETURN_IF_ERROR(
        subgraph_entry.second.AddFunctionCallNode(node_images, graph_out));
  }
  return absl::OkStatus();
}
Status Encapsulator::FindOutputImageOfEdgeSrc(
    const string& src_func_id, const string& dst_func_id,
    const absl::flat_hash_map<const Node*, Node*>& node_images,
    const Node* original_src_node, Node** src_image) {
  if (IsInSubgraph(src_func_id)) {
    *src_image = subgraphs_.at(src_func_id).GetCallNode();
  } else {
    *src_image = node_images.at(original_src_node);
  }
  return absl::OkStatus();
}
int Encapsulator::FindOutputSlotOfEdgeSrc(const string& src_func_id,
                                          const string& dst_func_id,
                                          const Edge* edge) {
  if (IsInSubgraph(src_func_id)) {
    const Subgraph& src_subgraph = subgraphs_.at(src_func_id);
    return src_subgraph.GetResultIndexForEdge(edge);
  } else {
    return edge->src_output();
  }
}
Status Encapsulator::FindOutputImageOfEdgeDst(
    const string& src_func_id, const string& dst_func_id,
    const absl::flat_hash_map<const Node*, Node*>& node_images,
    const Node* original_dst_node, Node** dst_image) {
  if (IsInSubgraph(dst_func_id)) {
    *dst_image = subgraphs_.at(dst_func_id).GetCallNode();
  } else {
    *dst_image = node_images.at(original_dst_node);
  }
  return absl::OkStatus();
}
int Encapsulator::FindOutputSlotOfEdgeDst(const string& src_func_id,
                                          const string& dst_func_id,
                                          const Edge* edge) {
  if (IsInSubgraph(dst_func_id)) {
    const Subgraph& dst_subgraph = subgraphs_.at(dst_func_id);
      return dst_subgraph.GetArgIndexForEdge(edge);
  } else {
    return edge->dst_input();
  }
}
Status Encapsulator::CopyEdgeToOutputGraph(
    const Edge* edge, const string& src_func_id, const string& dst_func_id,
    const absl::flat_hash_map<const Node*, Node*>& node_images,
    Graph* graph_out,
    absl::flat_hash_set<std::pair<OutputTensor, InputTensor>,
                        OutputInputTensorPairHasher>* edges_added) {
  Node* src_image;
  TF_RETURN_IF_ERROR(FindOutputImageOfEdgeSrc(
      src_func_id, dst_func_id, node_images, edge->src(), &src_image));
  Node* dst_image;
  TF_RETURN_IF_ERROR(FindOutputImageOfEdgeDst(
      src_func_id, dst_func_id, node_images, edge->dst(), &dst_image));
  if (edge->IsControlEdge()) {
    if (edges_added
            ->emplace(OutputTensor(src_image, -1), InputTensor(dst_image, -1))
            .second) {
      graph_out->AddControlEdge(src_image, dst_image,
                                 true);
    }
    return absl::OkStatus();
  }
  int src_output = FindOutputSlotOfEdgeSrc(src_func_id, dst_func_id, edge);
  int dst_input = FindOutputSlotOfEdgeDst(src_func_id, dst_func_id, edge);
  if (edges_added
          ->emplace(OutputTensor(src_image, src_output),
                    InputTensor(dst_image, dst_input))
          .second) {
    graph_out->AddEdge(src_image, src_output, dst_image, dst_input);
  }
  return absl::OkStatus();
}
Status Encapsulator::AddEdgesToOutputGraph(
    const absl::flat_hash_map<const Node*, Node*>& node_images,
    Graph* graph_out) {
  absl::flat_hash_set<std::pair<OutputTensor, InputTensor>,
                      OutputInputTensorPairHasher>
      edges_added;
  for (const Edge* edge : graph_in_->edges()) {
    string src_func_id;
    TF_RETURN_IF_ERROR(GetFunctionNameAttr(edge->src(), &src_func_id));
    string dst_func_id;
    TF_RETURN_IF_ERROR(GetFunctionNameAttr(edge->dst(), &dst_func_id));
    if (IsInSubgraph(src_func_id) && IsInSubgraph(dst_func_id) &&
        src_func_id == dst_func_id) {
      continue;
    }
    TF_RETURN_IF_ERROR(CopyEdgeToOutputGraph(
        edge, src_func_id, dst_func_id, node_images, graph_out, &edges_added));
  }
  for (auto& subgraph_entry : subgraphs_) {
    Subgraph& subgraph = subgraph_entry.second;
    subgraph.ConnectSequencerToCallNode(graph_out);
  }
  return absl::OkStatus();
}
namespace {
Node* AddDummyShapedNode(const Node* src_node, int src_port,
                         const std::vector<ControlFlowInfo>& control_flow_info,
                         const TensorShapeProto& shape, Graph* graph_out) {
  DataType data_type = src_node->output_type(src_port);
  TensorProto dummy_proto;
  dummy_proto.set_dtype(data_type);
  *dummy_proto.mutable_tensor_shape() = shape;
  GraphDefBuilder::Options options(graph_out, nullptr);
  NodeBuilder node_builder(options.GetNameForOp("KnownShape"), "Const",
                           options.op_registry());
  node_builder.Attr("dtype", data_type).Attr("value", dummy_proto);
  Node* node = options.FinalizeBuilder(&node_builder);
  while (!control_flow_info[src_node->id()].frame_name.empty()) {
    NodeDebugInfo debug_info(*src_node);
    NodeBuilder enter_builder(options.GetNameForOp("Enter"), "Enter",
                              options.op_registry(), &debug_info);
    enter_builder.Attr("frame_name",
                       control_flow_info[src_node->id()].frame_name);
    enter_builder.Attr("is_constant", true);
    enter_builder.Input(node, 0);
    Node* enter_node = options.FinalizeBuilder(&enter_builder);
    node = enter_node;
    src_node = control_flow_info[src_node->id()].parent_frame;
  }
  return node;
}
}  
Status Encapsulator::MakePrunedGraphCopyAndInline(
    const Graph& graph, const std::vector<Node*>& sink_nodes,
    std::unique_ptr<Graph>* pruned_graph,
    absl::flat_hash_map<const Node*, Node*>* node_images,
    FunctionLibraryDefinition* library) {
  pruned_graph->reset(new Graph(library));
  (*pruned_graph)->set_versions(graph.versions());
  ReverseDFSFrom(graph, sink_nodes,
                 nullptr,
                 [&](Node* n) {
                   if (!n->IsSource()) {
                     Node* copied = (*pruned_graph)->CopyNode(n);
                     node_images->emplace(n, copied);
                   }
                 });
  for (auto entry : *node_images) {
    const Node* orig = entry.first;
    Node* image = entry.second;
    for (const Edge* out_edge : orig->out_edges()) {
      auto iter = node_images->find(out_edge->dst());
      if (iter != node_images->end()) {
        (*pruned_graph)
            ->AddEdge(image, out_edge->src_output(), iter->second,
                      out_edge->dst_input());
      }
    }
  }
  std::vector<Node*> function_nodes;
  for (auto node : (*pruned_graph)->nodes()) {
    const OpRegistrationData* op_reg_data;
    TF_RETURN_IF_ERROR(library->LookUp(node->type_string(), &op_reg_data));
    if (op_reg_data->is_function_op) {
      function_nodes.push_back(node);
    }
  }
  for (auto node : function_nodes) {
    VLOG(2) << "Inlining function " << node->name();
    const FunctionDef* fdef = library->Find(node->type_string());
    if (fdef == nullptr) {
      return errors::Internal("Failed to find function ", node->type_string(),
                              " in function library.");
    }
    std::unique_ptr<FunctionBody> fbody;
    TF_RETURN_IF_ERROR(
        FunctionDefToBodyHelper(*fdef, node->attrs(), library, &fbody));
    InlineFunctionBodyOptions inline_opts;
    TF_RETURN_IF_ERROR(InlineFunctionBody(*library, pruned_graph->get(), node,
                                          fbody.get(), inline_opts));
  }
  return absl::OkStatus();
}
Status Encapsulator::BuildOutputGraph(Graph* graph_out,
                                      FunctionLibraryDefinition* library) {
  absl::flat_hash_map<const Node*, Node*> node_images;
  TF_RETURN_IF_ERROR(CopyNodesToOutputGraph(graph_out, &node_images));
  TF_RETURN_IF_ERROR(AddFunctionCallNodes(node_images, graph_out));
  TF_RETURN_IF_ERROR(AddEdgesToOutputGraph(node_images, graph_out));
  return absl::OkStatus();
}
}  
Status EncapsulateSubgraphsInFunctions(
    string group_attribute, const Graph& graph_in,
    const RewriteSubgraphFn& rewrite_subgraph_fn, bool reuse_existing_functions,
    std::unique_ptr<Graph>* graph_out, FunctionLibraryDefinition* library) {
  Encapsulator encapsulator(std::move(group_attribute),
                            &graph_in);
  TF_RETURN_IF_ERROR(encapsulator.SplitIntoSubgraphs(library));
  TF_RETURN_IF_ERROR(encapsulator.BuildFunctionDefs(
      rewrite_subgraph_fn, reuse_existing_functions, library));
  std::unique_ptr<Graph> out(new Graph(library));
  out->set_versions(graph_in.versions());
  TF_RETURN_IF_ERROR(encapsulator.BuildOutputGraph(out.get(), library));
  *graph_out = std::move(out);
  return absl::OkStatus();
}
static Status GetArgTypes(const Graph& graph, DataTypeVector* types) {
  for (Node* n : graph.op_nodes()) {
    if (n->type_string() == kArgOp) {
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
      const int num_types = types->size();
      if (index < 0 || index >= num_types) {
        return errors::InvalidArgument("Invalid argument number");
      }
      (*types)[index] = n->output_type(0);
    }
  }
  return absl::OkStatus();
}
static Status RenumberArguments(Graph* graph,
                                const std::vector<int>& permutation) {
  for (Node* n : graph->op_nodes()) {
    if (n->type_string() == kArgOp) {
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
      const int permutation_size = permutation.size();
      if (index < 0 || index >= permutation_size) {
        return errors::InvalidArgument("Invalid argument number");
      }
      n->AddAttr("index", permutation[index]);
    }
  }
  return absl::OkStatus();
}
Status EncapsulateSubgraphsPass::Run(
    const GraphOptimizationPassOptions& options) {
  VLOG(1) << "EncapsulateSubgraphsPass::Run";
  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("encapsulate_subgraphs_before", **options.graph,
                    options.flib_def);
  }
  for (Node* n : (*options.graph)->nodes()) {
    if (n->type_string() == "TPUExecute" ||
        n->type_string() == "TPUExecuteAndUpdateVariables") {
      return absl::OkStatus();
    }
  }
  std::unique_ptr<Graph> graph_out;
  FunctionLibraryDefinition* const library = options.flib_def;
  SessionOptions session_options;
  auto* device_count = session_options.config.mutable_device_count();
  device_count->insert({"CPU", 1});
  std::vector<std::unique_ptr<Device>> devices;
  DeviceFactory* cpu_factory = DeviceFactory::GetFactory("CPU");
  if (!cpu_factory) {
    return errors::NotFound(
        "CPU Factory not registered. Can't run EncapsulateSubgraphsPass");
  }
  TF_RETURN_IF_ERROR(cpu_factory->CreateDevices(
      session_options, "/job:localhost/replica:0/task:0", &devices));
  if (devices.empty()) {
    return errors::NotFound(
        "Failed to create a CPU device for EncapsulateSubgraphsPass");
  }
  std::unique_ptr<DeviceMgr> device_mgr =
      std::make_unique<StaticDeviceMgr>(std::move(devices));
  const auto* config = &options.session_options->config;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(
      new ProcessFunctionLibraryRuntime(
          device_mgr.get(), options.session_options->env,
          config, TF_GRAPH_DEF_VERSION, library,
          config->graph_options().optimizer_options()));
  FunctionLibraryRuntime* flr =
      pflr->GetFLR("/job:localhost/replica:0/task:0/device:CPU:0");
  if (flr == nullptr) {
    return errors::Internal(
        "Failed to create and retrieve function library runtime to run "
        "constant folding");
  }
  auto rewrite_subgraph =
      [flr](const std::vector<OutputTensor>& arg_source_tensors,
            std::unique_ptr<Graph>* subgraph,
            std::vector<int>* input_permutation,
            std::vector<int>* output_permutation, NodeDef* node) {
        bool disable_constant_folding =
            GetBuildXlaOpsPassFlags()->tf_xla_disable_constant_folding;
        auto cf_consider_fn = [disable_constant_folding](const Node* n) {
          if (disable_constant_folding) return false;
          for (const auto& output_arg : n->op_def().output_arg()) {
            if (output_arg.type() == DT_VARIANT) {
              return false;
            }
          }
          return true;
        };
        GraphOptimizer::Options graph_optimizer_options;
        graph_optimizer_options.cf_consider_fn = cf_consider_fn;
        OptimizeGraph(flr, subgraph, graph_optimizer_options);
        const int num_args = input_permutation->size();
        std::vector<bool> const_args(num_args);
        TF_RETURN_IF_ERROR(
            BackwardsConstAnalysis(**subgraph, &const_args,
                                   nullptr, flr));
        DataTypeVector arg_types(num_args);
        TF_RETURN_IF_ERROR(GetArgTypes(**subgraph, &arg_types));
        const int num_consts =
            std::count(const_args.begin(), const_args.end(), true);
        const int num_resources =
            std::count(arg_types.begin(), arg_types.end(), DT_RESOURCE);
        const int num_nonconsts = num_args - num_resources - num_consts;
        if (num_nonconsts < 0) {
          return errors::Internal("num_nonconsts should be >= 0, was ",
                                  num_nonconsts);
        }
        int const_pos = 0;
        int arg_pos = num_consts;
        int resource_pos = num_consts + num_nonconsts;
        for (int i = 0; i < num_args; ++i) {
          if (const_args[i]) {
            if (arg_types[i] == DT_RESOURCE) {
              return errors::Internal(
                  "Resource arguments cannot be constant (argument ", i, ")");
            }
            (*input_permutation)[i] = const_pos;
            ++const_pos;
          } else if (arg_types[i] == DT_RESOURCE) {
            (*input_permutation)[i] = resource_pos;
            ++resource_pos;
          } else {
            (*input_permutation)[i] = arg_pos;
            ++arg_pos;
          }
        }
        TF_RETURN_IF_ERROR(
            RenumberArguments(subgraph->get(), *input_permutation));
        AddNodeAttr(kXlaCompiledKernelAttr, true, node);
        AddNodeAttr(kXlaNumConstantArgsAttr, num_consts, node);
        AddNodeAttr(kXlaNumResourceArgsAttr, num_resources, node);
        return absl::OkStatus();
      };
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      EncapsulateSubgraphsInFunctions(
          kXlaClusterAttr, **options.graph, rewrite_subgraph,
          false, &graph_out, library),
      "EncapsulateSubgraphsPass failed");
  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("encapsulate_subgraphs_after", *graph_out,
                    options.flib_def);
  }
  *options.graph = std::move(graph_out);
  TF_ASSIGN_OR_RETURN(absl::flat_hash_set<Node*> ref_related_nodes,
                      GetNodesRelatedToRefVariables(**options.graph, flr));
  for (Node* node : (*options.graph)->nodes()) {
    bool has_ref_vars = ref_related_nodes.contains(node);
    node->AddAttr(kXlaHasReferenceVarsAttr, has_ref_vars);
    VLOG(3) << "Has ref vars = " << has_ref_vars
            << ", node: " << node->def().DebugString();
  }
  return absl::OkStatus();
}
bool IsXlaCompiledKernel(const Node& node) {
  bool is_compiled = false;
  bool has_compilation_attr =
      TryGetNodeAttr(node.attrs(), kXlaCompiledKernelAttr, &is_compiled) &&
      is_compiled;
  return has_compilation_attr ? is_compiled : false;
}
}  