#include "tensorflow/compiler/jit/partially_decluster_pass.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/jit/device_util.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/public/version.h"
namespace tensorflow {
namespace {
bool NotBackedge(const Edge& edge) { return !edge.src()->IsNextIteration(); }
namespace reduce_device_to_host_copies {
Status FindNodesToDecluster(const Graph& graph,
                            absl::flat_hash_set<Node*>* result,
                            absl::Span<Node* const> post_order) {
  MemoryTypeVector input_mtypes, output_mtypes;
  for (Node* n : post_order) {
    std::optional<absl::string_view> from_cluster = GetXlaClusterForNode(*n);
    if (!from_cluster) {
      continue;
    }
    if (IsShapeConsumerOp(*n)) {
      continue;
    }
    if (HasResourceInputOrOutput(*n)) {
      continue;
    }
    DeviceType device_type("");
    TF_RETURN_IF_ERROR(
        DeviceNameToDeviceType(n->assigned_device_name(), &device_type));
    TF_RETURN_IF_ERROR(MemoryTypesForNode(graph.op_registry(), device_type,
                                          n->def(), &input_mtypes,
                                          &output_mtypes));
    for (const Edge* e : n->out_edges()) {
      Node* dst = e->dst();
      if (e->IsControlEdge()) {
        continue;
      }
      bool edge_incurs_extra_device_to_host_copy;
      if (output_mtypes[e->src_output()] == DEVICE_MEMORY) {
        edge_incurs_extra_device_to_host_copy = false;
      } else {
        MemoryTypeVector dst_input_mtypes, dst_output_mtypes;
        DeviceType dst_device_type("");
        TF_RETURN_IF_ERROR(DeviceNameToDeviceType(dst->assigned_device_name(),
                                                  &dst_device_type));
        TF_RETURN_IF_ERROR(MemoryTypesForNode(graph.op_registry(), device_type,
                                              dst->def(), &dst_input_mtypes,
                                              &dst_output_mtypes));
        edge_incurs_extra_device_to_host_copy =
            dst_input_mtypes[e->dst_input()] == HOST_MEMORY;
      }
      if (!edge_incurs_extra_device_to_host_copy) {
        continue;
      }
      std::optional<absl::string_view> dst_cluster =
          result->count(dst) ? std::nullopt : GetXlaClusterForNode(*dst);
      if (from_cluster != dst_cluster) {
        CHECK(result->insert(n).second);
        break;
      }
    }
  }
  return absl::OkStatus();
}
Status PartiallyDeclusterNode(Graph* graph, Node* n) {
  absl::string_view cluster_name = *GetXlaClusterForNode(*n);
  absl::InlinedVector<const Edge*, 6> out_edges_to_clone;
  for (const Edge* out_edge : n->out_edges()) {
    if (out_edge->IsControlEdge()) {
      continue;
    }
    Node* dst = out_edge->dst();
    std::optional<absl::string_view> dst_cluster_name =
        GetXlaClusterForNode(*dst);
    if (dst_cluster_name != cluster_name) {
      out_edges_to_clone.push_back(out_edge);
    }
  }
  CHECK(!out_edges_to_clone.empty()) << n->DebugString();
  NodeDef ndef = n->def();
  ndef.set_name(absl::StrCat(n->name(), "/declustered"));
  MergeDebugInfo(NodeDebugInfo(n->def()), &ndef);
  RemoveFromXlaCluster(&ndef);
  TF_ASSIGN_OR_RETURN(Node * cloned_node, graph->AddNode(ndef));
  cloned_node->set_assigned_device_name(n->assigned_device_name());
  for (const Edge* in_edge : n->in_edges()) {
    graph->AddEdge(in_edge->src(), in_edge->src_output(), cloned_node,
                   in_edge->dst_input());
  }
  for (const Edge* out_edge_to_clone : out_edges_to_clone) {
    graph->AddEdge(cloned_node, out_edge_to_clone->src_output(),
                   out_edge_to_clone->dst(), out_edge_to_clone->dst_input());
    graph->RemoveEdge(out_edge_to_clone);
  }
  if (n->out_edges().empty()) {
    graph->RemoveNode(n);
  }
  return absl::OkStatus();
}
Status PartiallyDeclusterGraph(Graph* graph) {
  std::vector<Node*> post_order;
  GetPostOrder(*graph, &post_order, NodeComparatorName(),
               NotBackedge);
  absl::flat_hash_set<Node*> nodes_to_partially_decluster;
  TF_RETURN_IF_ERROR(
      FindNodesToDecluster(*graph, &nodes_to_partially_decluster, post_order));
  if (VLOG_IS_ON(3)) {
    for (Node* n : post_order) {
      if (nodes_to_partially_decluster.count(n)) {
        VLOG(3) << n->DebugString();
      }
    }
  }
  for (Node* n : post_order) {
    if (nodes_to_partially_decluster.count(n)) {
      TF_RETURN_IF_ERROR(PartiallyDeclusterNode(graph, n));
    }
  }
  post_order.clear();
  GetPostOrder(*graph, &post_order, NodeComparatorName(),
               NotBackedge);
  nodes_to_partially_decluster.clear();
  TF_RETURN_IF_ERROR(
      FindNodesToDecluster(*graph, &nodes_to_partially_decluster, post_order));
  CHECK(nodes_to_partially_decluster.empty());
  return absl::OkStatus();
}
}  
namespace reduce_recompilation {
bool IsIntraClusterEdge(const Edge& edge) {
  std::optional<absl::string_view> src_cluster_name =
      GetXlaClusterForNode(*edge.src());
  std::optional<absl::string_view> dst_cluster_name =
      GetXlaClusterForNode(*edge.dst());
  return src_cluster_name.has_value() && src_cluster_name == dst_cluster_name;
}
bool IsMustCompileDevice(const DeviceType& device_type) {
  const XlaOpRegistry::DeviceRegistration* registration;
  if (XlaOpRegistry::GetCompilationDevice(device_type.type(), &registration)) {
    return registration->autoclustering_policy ==
           XlaOpRegistry::AutoclusteringPolicy::kAlways;
  }
  return false;
}
Status MustCompileNode(const Node* n, bool* must_compile) {
  DeviceType device_type("");
  TF_RETURN_IF_ERROR(
      DeviceNameToDeviceType(n->assigned_device_name(), &device_type));
  if (IsMustCompileDevice(device_type)) {
    *must_compile = true;
    return absl::OkStatus();
  }
  *must_compile = !FindKernelDef(device_type, n->def(), nullptr, nullptr).ok();
  return absl::OkStatus();
}
Status PartiallyDeclusterGraph(Graph* graph,
                               const FunctionLibraryDefinition* flib_def,
                               Env* env) {
  std::vector<bool> compile_time_const_nodes(graph->num_node_ids());
  OptimizerOptions opts;
  auto pflr = std::make_unique<ProcessFunctionLibraryRuntime>(
      nullptr, env, nullptr, TF_GRAPH_DEF_VERSION, flib_def, opts);
  FunctionLibraryRuntime* lib_runtime =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);
  TF_RETURN_IF_ERROR(BackwardsConstAnalysis(*graph, nullptr,
                                            &compile_time_const_nodes,
                                            lib_runtime, IsIntraClusterEdge));
  std::vector<Node*> rpo;
  GetReversePostOrder(*graph, &rpo, NodeComparatorName(),
                      NotBackedge);
  for (Node* n : rpo) {
    if (!compile_time_const_nodes[n->id()]) {
      continue;
    }
    absl::string_view cluster_name = *GetXlaClusterForNode(*n);
    bool node_on_cluster_edge =
        absl::c_all_of(n->in_edges(), [&](const Edge* e) {
          std::optional<absl::string_view> incoming_cluster =
              GetXlaClusterForNode(*e->src());
          return !incoming_cluster || *incoming_cluster != cluster_name;
        });
    if (node_on_cluster_edge) {
      bool must_compile_node;
      TF_RETURN_IF_ERROR(MustCompileNode(n, &must_compile_node));
      if (!must_compile_node) {
        if (n->IsConstant()) {
          for (auto it : n->in_edges()) {
            if (!it->src()->assigned_device_name().empty() &&
                it->src()->assigned_device_name() !=
                    n->assigned_device_name()) {
              VLOG(3) << "Declustering Const with cross-device control input "
                      << n->name();
              RemoveFromXlaCluster(n);
              break;
            }
          }
        } else {
          VLOG(3) << "Declustering must-be-constant node " << n->name();
          RemoveFromXlaCluster(n);
        }
      }
    }
  }
  return absl::OkStatus();
}
}  
namespace decluster_root_shape_consumers {
Status PartiallyDeclusterGraph(Graph* graph) {
  std::vector<Node*> reverse_post_order;
  GetReversePostOrder(*graph, &reverse_post_order,
                      NodeComparatorName(),
                      NotBackedge);
  for (Node* n : reverse_post_order) {
    if (!IsShapeConsumerOp(*n)) {
      continue;
    }
    std::optional<absl::string_view> cluster = GetXlaClusterForNode(*n);
    if (!cluster.has_value()) {
      continue;
    }
    auto input_belongs_to_same_cluster = [&](const Edge* e) {
      return cluster == GetXlaClusterForNode(*e->src());
    };
    if (absl::c_any_of(n->in_edges(), input_belongs_to_same_cluster)) {
      continue;
    }
    VLOG(2) << "Declustering " << n->name()
            << " because it is a root shape consumer";
    RemoveFromXlaCluster(n);
  }
  return absl::OkStatus();
}
}  
}  
Status PartiallyDeclusterPass::Run(
    const GraphOptimizationPassOptions& options) {
  Graph* graph = options.graph->get();
  TF_RETURN_IF_ERROR(
      reduce_device_to_host_copies::PartiallyDeclusterGraph(graph));
  if (options.flib_def == nullptr) {
    return errors::InvalidArgument(
        "GraphOptimizationPassOptions::flib_def must be set for "
        "PartiallyDeclusterPass.");
  }
  if (options.session_options == nullptr ||
      options.session_options->env == nullptr) {
    return errors::InvalidArgument(
        "GraphOptimizationPassOptions::session_options::env must be set for "
        "PartiallyDeclusterPass.");
  }
  TF_RETURN_IF_ERROR(reduce_recompilation::PartiallyDeclusterGraph(
      graph, options.flib_def, options.session_options->env));
  TF_RETURN_IF_ERROR(
      decluster_root_shape_consumers::PartiallyDeclusterGraph(graph));
  return absl::OkStatus();
}
}  