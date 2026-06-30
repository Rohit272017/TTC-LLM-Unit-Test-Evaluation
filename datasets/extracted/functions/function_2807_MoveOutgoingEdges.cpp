#include "tensorflow/compiler/jit/build_xla_ops_pass.h"
#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/control_flow_ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/logging_ops.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/device_util.h"
#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/tf2xla/cc/ops/xla_jit_ops.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/status_macros.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/dump_graph.h"
namespace tensorflow {
namespace {
struct DebuggingOpts {
  bool print_outputs;
  bool check_input_numerics;
  bool check_output_numerics;
};
void MoveOutgoingEdges(Graph* g, Node* old_node, Node* new_node) {
  std::vector<const Edge*> out_edges(old_node->out_edges().begin(),
                                     old_node->out_edges().end());
  for (const Edge* edge : out_edges) {
    g->AddEdge(new_node, edge->src_output(), edge->dst(), edge->dst_input());
    g->RemoveEdge(edge);
  }
}
Output ControlToData(const Scope& scope, Node* control) {
  Output data = ops::Const(scope.WithOpName("ctrl_as_data"),
                           Tensor(DT_INT32, TensorShape({0})));
  scope.graph()->AddControlEdge(control, data.node());
  return Output(data.node());
}
Operation DataToControl(const Scope& scope, Output data) {
  return Operation(
      ops::Identity(scope.WithOpName("data_as_ctrl"), data).node());
}
void MergeOutgoingDataEdges(const Scope& s, Node* old_node, Node* new_node,
                            absl::string_view cluster_name,
                            const DebuggingOpts& debugging_opts) {
  if (!s.status().ok()) {
    return;
  }
  std::vector<Output> merged_outputs(old_node->num_outputs(), Output(nullptr));
  std::vector<const Edge*> data_edges;
  absl::c_copy_if(old_node->out_edges(), std::back_inserter(data_edges),
                  [](const Edge* e) { return !e->IsControlEdge(); });
  for (const Edge* e : data_edges) {
    int oidx = e->src_output();
    Output merged_output = merged_outputs[oidx];
    if (merged_output.node() == nullptr) {
      Output new_output(new_node, oidx);
      if (debugging_opts.print_outputs) {
        string cpu_device = "/job:localhost/replica:0/task:0/device:CPU:0";
        ops::Print print_op(s.WithOpName("print_", oidx)
                                .WithDevice(cpu_device)
                                .WithAssignedDevice(cpu_device),
                            new_output, {new_output},
                            ops::Print::Attrs{}
                                .Message(absl::StrCat("output ", oidx, " from ",
                                                      old_node->name(), " is "))
                                .FirstN(1000)
                                .Summarize(-1));
        new_output = print_op;
      }
      if (debugging_opts.check_output_numerics &&
          DataTypeIsFloating(new_output.type())) {
        ops::CheckNumerics check_numerics_op(
            s.WithOpName("check_output_", oidx)
                .WithDevice(new_node->requested_device())
                .WithAssignedDevice(new_node->assigned_device_name()),
            new_output,
            absl::StrCat("CheckNumerics failed for output ", oidx, "(",
                         new_output.name(), ") from cluster ", cluster_name));
        new_output = check_numerics_op;
      }
      ops::_XlaMerge xla_merge_op(s.WithOpName("merge_oidx_", oidx),
                                  Output(old_node, oidx), new_output);
      merged_output = merged_outputs[oidx] = xla_merge_op.output;
    }
    Node* dst = e->dst();
    int dst_idx = e->dst_input();
    s.graph()->RemoveEdge(e);
    s.graph()->AddEdge(merged_output.node(), merged_output.index(), dst,
                       dst_idx);
  }
}
void MergeOutgoingControlEdges(const Scope& s, Node* old_node, Node* new_node) {
  if (!s.status().ok()) {
    return;
  }
  std::vector<const Edge*> ctrl_edges;
  absl::c_copy_if(old_node->out_edges(), std::back_inserter(ctrl_edges),
                  [](const Edge* e) { return e->IsControlEdge(); });
  if (ctrl_edges.empty()) {
    return;
  }
  if (ctrl_edges.size() == 1 && ctrl_edges.front()->dst()->IsSink()) {
    s.graph()->AddControlEdge(new_node, s.graph()->sink_node());
    return;
  }
  Output old_ctrl_as_data = ControlToData(s, old_node);
  Output new_ctrl_as_data = ControlToData(s, new_node);
  ops::Merge ctrl_merge_as_data(s.WithOpName("ctrl_merge"),
                                {old_ctrl_as_data, new_ctrl_as_data});
  Operation ctrl_merge = DataToControl(s, ctrl_merge_as_data.output);
  for (const Edge* e : ctrl_edges) {
    s.graph()->AddControlEdge(ctrl_merge.node(), e->dst());
    s.graph()->RemoveControlEdge(e);
  }
}
struct XlaClusterInfo {
  std::vector<Output> constant_inputs;
  std::vector<Output> non_constant_inputs;
  std::vector<Output> resource_inputs;
  NameAttrList function;
};
Output IncomingEdgeAsOutput(const Edge* e) {
  return Output(e->src(), e->src_output());
}
Status GetXlaClusterInfo(Node* n, XlaClusterInfo* result) {
  int num_constant_inputs, num_resource_inputs;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(n->attrs(), kXlaNumConstantArgsAttr, &num_constant_inputs));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(n->attrs(), kXlaNumResourceArgsAttr, &num_resource_inputs));
  if (num_constant_inputs < 0 || num_resource_inputs < 0 ||
      num_constant_inputs + num_resource_inputs > n->num_inputs()) {
    return errors::InvalidArgument(
        "Invalid number of constant/resource arguments to XLA kernel.");
  }
  int num_non_constant_inputs =
      n->num_inputs() - num_constant_inputs - num_resource_inputs;
  std::vector<const Edge*> input_edges_vector;
  TF_RETURN_IF_ERROR(n->input_edges(&input_edges_vector));
  absl::Span<const Edge*> input_edges(input_edges_vector);
  absl::c_transform(input_edges.subspan(0, num_constant_inputs),
                    std::back_inserter(result->constant_inputs),
                    IncomingEdgeAsOutput);
  absl::c_transform(
      input_edges.subspan(num_constant_inputs, num_non_constant_inputs),
      std::back_inserter(result->non_constant_inputs), IncomingEdgeAsOutput);
  absl::c_transform(
      input_edges.subspan(num_constant_inputs + num_non_constant_inputs,
                          num_resource_inputs),
      std::back_inserter(result->resource_inputs), IncomingEdgeAsOutput);
  result->function.set_name(n->type_string());
  *result->function.mutable_attr() = n->def().attr();
  return absl::OkStatus();
}
Status CopyIncomingControlEdges(Graph* g, Node* from, Node* to) {
  for (const Edge* e : from->in_edges()) {
    if (e->IsControlEdge()) {
      g->AddControlEdge(e->src(), to);
    }
  }
  return absl::OkStatus();
}
void RemoveAllIncomingControlEdges(Graph* g, Node* n) {
  std::vector<const Edge*> incoming_ctrl_edges;
  absl::c_copy_if(n->in_edges(), std::back_inserter(incoming_ctrl_edges),
                  [](const Edge* e) { return e->IsControlEdge(); });
  for (const Edge* e : incoming_ctrl_edges) {
    g->RemoveControlEdge(e);
  }
}
Status DeviceRequiresCompilation(const jit::DeviceInfoCache& device_info_cache,
                                 jit::DeviceId device, bool* result) {
  const XlaOpRegistry::DeviceRegistration* registration =
      device_info_cache.GetCompilationDevice(device);
  *result = registration->autoclustering_policy ==
            XlaOpRegistry::AutoclusteringPolicy::kAlways;
  return absl::OkStatus();
}
absl::StatusOr<Node*> ReplaceFunctionCallWithPartitionedCall(
    const GraphOptimizationPassOptions& options,
    const FunctionLibraryDefinition& flib_def, Node* n, Graph* g,
    const NameAttrList& func, const Scope& root) {
  string config_string = options.session_options->config.SerializeAsString();
  int input_count = absl::c_count_if(
      n->in_edges(), [](const Edge* e) { return !e->IsControlEdge(); });
  std::vector<Output> args(input_count);
  for (const Edge* e : n->in_edges()) {
    if (!e->IsControlEdge()) {
      args[e->dst_input()] = Output(e->src(), e->src_output());
    }
  }
  ops::StatefulPartitionedCall call(
      root.WithOpName("stateful_partitioned_call"), args, n->output_types(),
      func, ops::StatefulPartitionedCall::Attrs{}.ConfigProto(config_string));
  for (const Edge* e : n->in_edges()) {
    if (e->IsControlEdge()) {
      g->AddControlEdge(e->src(), call.operation.node());
    }
  }
  std::vector<const Edge*> edges_to_delete;
  for (const Edge* e : n->out_edges()) {
    edges_to_delete.push_back(e);
    if (e->IsControlEdge()) {
      g->AddControlEdge(call.operation.node(), e->dst());
    } else {
      g->AddEdge(call.operation.node(), e->src_output(), e->dst(),
                 e->dst_input());
    }
  }
  for (const Edge* e : edges_to_delete) {
    g->RemoveEdge(e);
  }
  g->RemoveNode(n);
  return call.operation.node();
}
absl::StatusOr<jit::DeviceId> InferDeviceForCluster(
    jit::DeviceInfoCache* device_info_cache, Node* n,
    const string& function_name, const FunctionLibraryDefinition& flib_def) {
  const FunctionDef* func_def = flib_def.Find(function_name);
  TF_RET_CHECK(func_def) << "Could not find " << function_name;
  jit::DeviceSet device_set;
  for (const NodeDef& ndef : func_def->node_def()) {
    VLOG(3) << ndef.DebugString();
    if (!ndef.device().empty()) {
      TF_ASSIGN_OR_RETURN(jit::DeviceId device_id,
                          device_info_cache->GetIdFor(ndef.device()));
      device_set.Insert(device_id);
    }
  }
  if (!n->assigned_device_name().empty()) {
    TF_ASSIGN_OR_RETURN(jit::DeviceId device_id,
                        device_info_cache->GetIdFor(n->assigned_device_name()));
    device_set.Insert(device_id);
  }
  TF_ASSIGN_OR_RETURN(jit::DeviceId result,
                      PickDeviceForXla(*device_info_cache, device_set,
                                       true));
  VLOG(2) << "For " << function_name << " PickDeviceForXla("
          << device_info_cache->DebugString(device_set) << ") -> "
          << device_info_cache->GetNameFor(result);
  return result;
}
std::vector<Output> GetXlaRunArgs(const Scope& s,
                                  const XlaClusterInfo& cluster_info,
                                  const DebuggingOpts& debugging_opts) {
  std::vector<Output> xla_run_args;
  xla_run_args.reserve(cluster_info.non_constant_inputs.size() +
                       cluster_info.resource_inputs.size());
  int input_idx = 0;
  for (const Output& o : cluster_info.non_constant_inputs) {
    if (debugging_opts.check_input_numerics && DataTypeIsFloating(o.type())) {
      ops::CheckNumerics check_numerics_op(
          s.WithOpName("check_input_", input_idx), o,
          absl::StrCat("CheckNumerics failed for input ", input_idx, "(",
                       o.name(), ") into ", cluster_info.function.name()));
      xla_run_args.push_back(check_numerics_op);
    } else {
      xla_run_args.push_back(o);
    }
    input_idx++;
  }
  absl::c_copy(cluster_info.resource_inputs, std::back_inserter(xla_run_args));
  return xla_run_args;
}
absl::StatusOr<MemoryTypeVector> GetOutputMemoryTypes(const Scope& root,
                                                      Node* n) {
  MemoryTypeVector input_mtypes, output_mtypes;
  DeviceType device_type("");
  TF_RETURN_IF_ERROR(
      DeviceNameToDeviceType(n->assigned_device_name(), &device_type));
  TF_RETURN_IF_ERROR(MemoryTypesForNode(root.graph()->op_registry(),
                                        device_type, n->def(), &input_mtypes,
                                        &output_mtypes));
  return output_mtypes;
}
Status PredicateInt32Inputs(const Scope& root, Node* n,
                            Operation predicate_as_control) {
  std::vector<Output> int32_inputs;
  std::vector<int> int32_inputs_input_idxs;
  for (const Edge* e : n->in_edges()) {
    if (e->IsControlEdge()) {
      continue;
    }
    if (e->src()->output_type(e->src_output()) == DT_INT32) {
      TF_ASSIGN_OR_RETURN(MemoryTypeVector source_output_mem_types,
                          GetOutputMemoryTypes(root, e->src()));
      if (source_output_mem_types[e->src_output()] == DEVICE_MEMORY) {
        int32_inputs.push_back(Output(e->src(), e->src_output()));
        int32_inputs_input_idxs.push_back(e->dst_input());
      }
    }
  }
  if (int32_inputs.empty()) {
    return absl::OkStatus();
  }
  ops::IdentityN identity_n(root.WithOpName("int32_id_n"), int32_inputs);
  root.graph()->AddControlEdge(predicate_as_control.node(),
                               identity_n.operation.node());
  for (int i = 0, end = int32_inputs.size(); i < end; i++) {
    TF_RETURN_IF_ERROR(root.graph()->UpdateEdge(identity_n[i].node(), i, n,
                                                int32_inputs_input_idxs[i]));
  }
  return absl::OkStatus();
}
Status ReplaceNodeWithXlaCompileAndXlaRun(
    jit::DeviceInfoCache* device_info_cache,
    const GraphOptimizationPassOptions& options,
    const FunctionLibraryDefinition& flib_def, bool lazy_compilation_enabled,
    const DebuggingOpts& debugging_opts, Graph* g, Node* n) {
  XlaClusterInfo cluster_info;
  TF_RETURN_IF_ERROR(GetXlaClusterInfo(n, &cluster_info));
  TF_ASSIGN_OR_RETURN(
      jit::DeviceId device,
      InferDeviceForCluster(device_info_cache, n, cluster_info.function.name(),
                            flib_def));
  bool requires_compilation;
  TF_RETURN_IF_ERROR(DeviceRequiresCompilation(*device_info_cache, device,
                                               &requires_compilation));
  if (!lazy_compilation_enabled) {
    requires_compilation = true;
  }
  string device_name_str = string(device_info_cache->GetNameFor(device));
  Status status;
  Scope root = NewInternalScope(g, &status, nullptr)
                   .NewSubScope(n->name())
                   .WithDevice(n->requested_device())
                   .WithAssignedDevice(device_name_str);
  ops::_XlaCompile xla_compile(root.WithOpName("xla_compile"),
                               cluster_info.constant_inputs,
                               cluster_info.non_constant_inputs,
                               cluster_info.resource_inputs,
                               requires_compilation,
                               cluster_info.function);
  bool has_ref_attr;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(n->attrs(), kXlaHasReferenceVarsAttr, &has_ref_attr));
  xla_compile.operation.node()->AddAttr(kXlaHasReferenceVarsAttr, has_ref_attr);
  TF_RETURN_IF_ERROR(
      CopyIncomingControlEdges(g, n, xla_compile.key.node()));
  std::vector<Output> xla_run_args =
      GetXlaRunArgs(root, cluster_info, debugging_opts);
  if (requires_compilation) {
    ops::_XlaRun xla_run(root.WithOpName("xla_run"), xla_run_args,
                         xla_compile.key, n->output_types());
    MoveOutgoingEdges(g, n,
                      xla_run.operation.node());
    g->RemoveNode(n);
  } else {
    ops::Switch s(root.WithOpName("predicated_compilation_key"),
                  xla_compile.key, xla_compile.compilation_successful);
    Output predicated_compilation_key = s.output_true;
    Output inverse_predicated_compilation_key = s.output_false;
    ops::_XlaRun xla_run(root.WithOpName("xla_run"), xla_run_args,
                         predicated_compilation_key, n->output_types());
    MergeOutgoingControlEdges(root, n,
                              xla_run.operation.node());
    MergeOutgoingDataEdges(root, n,
                           xla_run.operation.node(),
                           cluster_info.function.name(), debugging_opts);
    TF_RETURN_IF_ERROR(root.status());
    RemoveAllIncomingControlEdges(g, n);
    Operation inverse_predicate_as_control =
        DataToControl(root, inverse_predicated_compilation_key);
    g->AddControlEdge(inverse_predicate_as_control.node(), n);
    n->ClearAttr(kXlaCompiledKernelAttr);
    TF_ASSIGN_OR_RETURN(Node* const pco, ReplaceFunctionCallWithPartitionedCall(
                                             options, flib_def, n, g,
                                             cluster_info.function, root));
    TF_RETURN_IF_ERROR(
        PredicateInt32Inputs(root, pco, inverse_predicate_as_control));
  }
  return absl::OkStatus();
}
}  
Status BuildXlaOpsPass::Run(const GraphOptimizationPassOptions& options) {
  Graph* graph = options.graph->get();
  std::vector<Node*> xla_compiled_kernels;
  absl::c_copy_if(graph->op_nodes(), std::back_inserter(xla_compiled_kernels),
                  [](const Node* n) {
                    if (n->IsSend() || n->IsRecv() || n->IsControlFlow()) {
                      return false;
                    }
                    return IsXlaCompiledKernel(*n);
                  });
  bool lazy_compilation_enabled =
      enable_lazy_compilation_
          ? *enable_lazy_compilation_
          : GetBuildXlaOpsPassFlags()->tf_xla_enable_lazy_compilation;
  jit::DeviceInfoCache device_info_cache;
  const BuildXlaOpsPassFlags& flags = *GetBuildXlaOpsPassFlags();
  DebuggingOpts debugging_opts;
  debugging_opts.print_outputs = flags.tf_xla_print_cluster_outputs;
  debugging_opts.check_input_numerics =
      flags.tf_xla_check_cluster_input_numerics;
  debugging_opts.check_output_numerics =
      flags.tf_xla_check_cluster_output_numerics;
  VLOG(1) << "print_outputs = " << debugging_opts.print_outputs;
  VLOG(1) << "check_input_numerics = " << debugging_opts.check_input_numerics;
  VLOG(1) << "check_output_numerics = " << debugging_opts.check_output_numerics;
  for (Node* n : xla_compiled_kernels) {
    TF_RETURN_IF_ERROR(ReplaceNodeWithXlaCompileAndXlaRun(
        &device_info_cache, options, *options.flib_def,
        lazy_compilation_enabled, debugging_opts, graph, n));
  }
  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("build_xla_ops", *graph, options.flib_def);
  }
  return absl::OkStatus();
}
}  