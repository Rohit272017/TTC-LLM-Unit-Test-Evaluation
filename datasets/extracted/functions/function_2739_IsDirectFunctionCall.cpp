#include "tensorflow/core/grappler/optimizers/function_optimizer.h"
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/lower_case_op.h"
#include "tensorflow/core/common_runtime/lower_functional_ops.h"
#include "tensorflow/core/common_runtime/lower_if_op.h"
#include "tensorflow/core/common_runtime/lower_while_op.h"
#include "tensorflow/core/common_runtime/placer.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/lib/gtl/map_util.h"
namespace tensorflow {
namespace grappler {
namespace {
constexpr const char* const kFuncAttr = FunctionLibraryDefinition::kFuncAttr;
constexpr const char* const kNoSpecializeAttr = "_nospecialize";
constexpr const char* const kGrapplerSpecializedFuncAttr =
    "_GrapplerSpecializedFunc";
bool IsDirectFunctionCall(const FunctionDef& func, const NodeDef& func_node) {
  return func_node.op() == func.signature().name();
}
bool IsIndirectFunctionCall(const FunctionDef& func, const NodeDef& func_node) {
  if (!IsPartitionedCall(func_node) && !IsStatefulPartitionedCall(func_node)) {
    return false;
  }
  auto* func_attr = AttrSlice(func_node).Find(kFuncAttr);
  return func_attr != nullptr && func_attr->has_func() &&
         func_attr->func().name() == func.signature().name();
}
AttrSlice FunctionInstantiationAttributes(const FunctionDef& func,
                                          const NodeDef& func_node) {
  if (IsDirectFunctionCall(func, func_node)) {
    return AttrSlice(func_node);
  } else if (IsIndirectFunctionCall(func, func_node)) {
    auto* func_attr = AttrSlice(func_node).Find(kFuncAttr);
    return AttrSlice(&func_attr->func().attr());
  } else {
    LOG(WARNING) << "Can't resolve function instantiation attributes: "
                 << SummarizeNodeDef(func_node);
    return AttrSlice();
  }
}
class FakeDevice : public Device {
 public:
  FakeDevice(Env* env, const string& device) : Device(env, attr(device)) {}
  explicit FakeDevice(const string& device) : FakeDevice(nullptr, device) {}
  Status Sync() override { return absl::OkStatus(); }
 private:
  static DeviceAttributes attr(const string& device) {
    DeviceNameUtils::ParsedName parsed_name;
    bool parsed = DeviceNameUtils::ParseFullName(device, &parsed_name);
    DCHECK(parsed) << "Failed to parse full device name: " << device;
    DeviceAttributes attr;
    attr.set_name(device);
    attr.set_device_type(parsed_name.type);
    return attr;
  }
};
bool MarkedNoSpecialize(const FunctionDef& fdef) {
  const auto attr = AttrSlice(&fdef.attr());
  bool nospecialize = false;
  return TryGetNodeAttr(attr, kNoSpecializeAttr, &nospecialize) && nospecialize;
}
struct FunctionSpecializationSignature {
  using InputPort = int;
  using OutputPort = int;
  string func_name;
  bool is_in_fetch_set;
  absl::flat_hash_set<OutputPort> active_outputs;
  absl::flat_hash_map<string, DataType> type_parameters;
  absl::flat_hash_map<string, AttrValue> body_parameters;
  absl::flat_hash_map<InputPort, string> const_inputs;
  bool operator==(const FunctionSpecializationSignature& other) const {
    bool equals = func_name == other.func_name &&
                  is_in_fetch_set == other.is_in_fetch_set &&
                  active_outputs == other.active_outputs &&
                  type_parameters == other.type_parameters &&
                  const_inputs == other.const_inputs;
    if (!equals) return false;
    if (body_parameters.size() != other.body_parameters.size()) return false;
    for (const auto& lhs : body_parameters) {
      auto it = other.body_parameters.find(lhs.first);
      if (it == other.body_parameters.end()) return false;
      if (!AreAttrValuesEqual(lhs.second, (*it).second,
                              true)) {
        return false;
      }
    }
    return true;
  }
  template <typename H>
  friend H AbslHashValue(H h, const FunctionSpecializationSignature& s) {
    H base = H::combine(std::move(h), s.func_name, s.is_in_fetch_set);
    std::vector<uint64> hashes;
    hashes.reserve(s.active_outputs.size()         
                   + s.type_parameters.size() * 2  
                   + s.body_parameters.size() * 2  
                   + s.const_inputs.size() * 2);
    absl::c_transform(s.active_outputs, std::back_inserter(hashes),
                      hash<OutputPort>());
    using TypeParam = std::pair<const string, DataType>;
    absl::c_for_each(s.type_parameters, [&hashes](const TypeParam& type_param) {
      AttrValue attr_value;
      attr_value.set_type(type_param.second);
      hashes.push_back(Hash64(type_param.first));
      hashes.push_back(AttrValueHash(attr_value));
    });
    using BodyParam = std::pair<const string, AttrValue>;
    absl::c_for_each(s.body_parameters, [&hashes](const BodyParam& body_param) {
      hashes.push_back(Hash64(body_param.first));
      hashes.push_back(FastAttrValueHash(body_param.second));
    });
    using ConstInput = std::pair<const InputPort, string>;
    absl::c_for_each(s.const_inputs, [&hashes](const ConstInput& const_input) {
      hashes.push_back(hash<InputPort>()(const_input.first));
      hashes.push_back(Hash64(const_input.second));
    });
    absl::c_sort(hashes);
    return H::combine_contiguous(std::move(base), hashes.data(), hashes.size());
  }
};
struct FunctionSpecialization {
  string specialized_func_name;
  bool is_in_fetch_set;
  absl::flat_hash_set<string> const_inputs;
  absl::flat_hash_set<string> control_deps;
  absl::flat_hash_set<int> active_outputs;
  std::vector<std::pair<int, int>> output_mapping;
};
class FunctionOptimizerContext {
 public:
  explicit FunctionOptimizerContext(const GrapplerItem& item,
                                    RewriterConfig::Toggle opt_level,
                                    const GraphDef& graph)
      : item_(&item),
        opt_level_(opt_level),
        function_library_(OpRegistry::Global(), graph.library()),
        truly_const_nodes_(InferTrulyConstNodes(item, graph)),
        graph_view_(&graph) {}
  const GrapplerItem& item() const { return *item_; }
  const int graph_version() const { return item_->graph.versions().producer(); }
  RewriterConfig::Toggle opt_level() const { return opt_level_; }
  const FunctionLibraryDefinition& function_library() const {
    return function_library_;
  }
  FunctionLibraryDefinition& function_library() { return function_library_; }
  const absl::flat_hash_map<SafeTensorId, SafeTensorId, SafeTensorId::Hasher>&
  tensor_mapping() const {
    return tensor_mapping_;
  }
  const GraphView& graph_view() const { return graph_view_; }
  bool IsFeedNode(const string& node_name) const {
    return absl::c_any_of(
        item_->feed, [&](const std::pair<std::string, Tensor>& feed) {
          return ParseTensorName(feed.first).node() == node_name;
        });
  }
  bool IsFetchNode(const string& node_name) const {
    return absl::c_any_of(item_->fetch, [&](const string& fetch) {
      return ParseTensorName(fetch).node() == node_name;
    });
  }
  bool IsTrulyConst(const string& name) const {
    return TrulyConstNode(name) != nullptr;
  }
  const NodeDef* TrulyConstNode(const string& name) const {
    return gtl::FindWithDefault(truly_const_nodes_, name, nullptr);
  }
  const FunctionSpecialization* FindFunctionSpecialization(
      const FunctionSpecializationSignature& sig) const {
    return gtl::FindOrNull(specialized_functions_, sig);
  }
  void AddSpecializedFunction(const FunctionSpecializationSignature& sig,
                              const FunctionSpecialization& specialized_func) {
    specialized_functions_.emplace(sig, specialized_func);
  }
  void AddTensorMapping(const SafeTensorId& from, const SafeTensorId& to) {
    DCHECK(from.index() != Graph::kControlSlot)
        << "Tensor mapping must be from regular tensor";
    DCHECK(to.index() != Graph::kControlSlot)
        << "Tensor mapping must be to regular tensor";
    auto inserted = tensor_mapping_.insert({from, to});
    DCHECK(inserted.second)
        << "Failed to insert duplicated tensor mapping: "
        << "from=" << from.ToString() << " to=" << to.ToString();
  }
  void AddTensorMapping(const string& func_node,
                        const FunctionSpecialization& specialized_func) {
    for (const auto& pair : specialized_func.output_mapping) {
      int from_idx = pair.first;
      int to_idx = pair.second;
      if (from_idx != to_idx) {
        SafeTensorId from_tensor(func_node, from_idx);
        SafeTensorId to_tensor(func_node, to_idx);
        AddTensorMapping(from_tensor, to_tensor);
      }
    }
  }
 private:
  static absl::flat_hash_map<string, const NodeDef*> InferTrulyConstNodes(
      const GrapplerItem& item, const GraphDef& graph) {
    absl::flat_hash_set<absl::string_view> feed_nodes;
    for (const auto& feed : item.feed) {
      feed_nodes.insert(feed.first);
    }
    absl::flat_hash_map<string, const NodeDef*> const_nodes;
    for (const NodeDef& node : graph.node()) {
      if (IsConstant(node) && !feed_nodes.contains(node.name())) {
        const_nodes[node.name()] = &node;
      }
    }
    return const_nodes;
  }
  const GrapplerItem* item_;  
  RewriterConfig::Toggle opt_level_;
  FunctionLibraryDefinition function_library_;
  absl::flat_hash_map<string, const NodeDef*> truly_const_nodes_;
  absl::flat_hash_map<FunctionSpecializationSignature,
                      const FunctionSpecialization>
      specialized_functions_;
  absl::flat_hash_map<SafeTensorId, SafeTensorId, SafeTensorId::Hasher>
      tensor_mapping_;
  GraphView graph_view_;
  FunctionOptimizerContext(const FunctionOptimizerContext&) = delete;
  void operator=(const FunctionOptimizerContext&) = delete;
};
const FunctionDef* FindFunctionCall(const FunctionOptimizerContext& ctx,
                                    const NodeDef& node) {
  if (IsPartitionedCall(node) || IsStatefulPartitionedCall(node)) {
    const AttrValue* func_attr = AttrSlice(node).Find("f");
    return (func_attr != nullptr && func_attr->has_func())
               ? ctx.function_library().Find(func_attr->func().name())
               : nullptr;
  }
  return ctx.function_library().Find(node.op());
}
absl::flat_hash_set<int> GetActiveOutputs(const NodeDef& node,
                                          const FunctionOptimizerContext& ctx,
                                          int size_hint = 0) {
  absl::flat_hash_set<int> active_outputs;
  active_outputs.reserve(static_cast<size_t>(size_hint));
  const auto node_fanout_edges =
      ctx.graph_view().GetFanoutEdges(node, false);
  for (const GraphView::Edge& edge : node_fanout_edges) {
    active_outputs.insert(edge.src.port_id);
  }
  for (const string& fetch : ctx.item().fetch) {
    TensorId fetch_tensor = ParseTensorName(fetch);
    if (fetch_tensor.node() == node.name()) {
      active_outputs.insert(fetch_tensor.index());
    }
  }
  return active_outputs;
}
bool HasTrulyConstInputs(const NodeDef& node,
                         const FunctionOptimizerContext& ctx) {
  const auto is_truly_const = [&ctx](const string& input) {
    return ctx.IsTrulyConst(NodeName(input));
  };
  return absl::c_any_of(node.input(), is_truly_const);
}
bool HasUnusedOutputs(const NodeDef& func_node, const FunctionDef& func,
                      const FunctionOptimizerContext& ctx) {
  int num_outputs = func.signature().output_arg_size();
  const absl::flat_hash_set<int> active_outputs =
      GetActiveOutputs(func_node, ctx,  num_outputs);
  int active_outputs_size = active_outputs.size();
  return active_outputs_size != num_outputs;
}
FunctionDefLibrary PruneFunctionLibrary(const FunctionLibraryDefinition& flib,
                                        const GraphDef& optimized_graph) {
  FunctionLibraryDefinition pruned_flib =
      flib.ReachableDefinitions(optimized_graph);
  int pruned_functions = static_cast<int>(pruned_flib.num_functions()) -
                         static_cast<int>(flib.num_functions());
  VLOG(3) << "Pruned function library: " << pruned_flib.num_functions()
          << " functions (" << pruned_functions << ")";
  return pruned_flib.ToProto();
}
Status PushDownConstInputs(const NodeDef& func_node,
                           const FunctionOptimizerContext& ctx,
                           GrapplerFunctionItem* item,
                           absl::flat_hash_set<string>* const_inputs,
                           absl::flat_hash_set<string>* control_deps) {
  const auto record_control_deps = [&](const NodeDef* const_input) {
    for (int i = const_input->input_size() - 1; i >= 0; --i) {
      const string& input = const_input->input(i);
      if (IsControlInput(input))
        control_deps->insert(input);
      else
        break;
    }
  };
  for (int i = func_node.input_size() - 1; i >= 0; --i) {
    const string& input = func_node.input(i);
    if (IsControlInput(input)) continue;
    const string node_name = NodeName(input);
    if (ctx.IsTrulyConst(node_name)) {
      VLOG(3) << "Push const into function body: input=" << input;
      const auto* const_input = CHECK_NOTNULL(ctx.TrulyConstNode(node_name));
      const_inputs->insert(input);
      record_control_deps(const_input);
      TF_RETURN_IF_ERROR(ReplaceInputWithConst(*const_input, i, item));
    }
  }
  return absl::OkStatus();
}
void RemovePushedDownConstInputs(const FunctionSpecialization& specialization,
                                 NodeDef* specialized_func_node) {
  if (specialization.const_inputs.empty()) return;
  std::vector<string> keep_inputs;
  const auto& inputs = specialized_func_node->input();
  absl::c_copy_if(inputs, std::back_inserter(keep_inputs),
                  [&](const string& input) {
                    return !specialization.const_inputs.contains(input);
                  });
  specialized_func_node->clear_input();
  for (const auto& keep : keep_inputs) specialized_func_node->add_input(keep);
  if (!specialization.control_deps.empty()) {
    absl::flat_hash_set<string> existing_control_deps;
    for (const string& input : keep_inputs) {
      existing_control_deps.insert(AsControlDependency(NodeName(input)));
    }
    for (const string& ctrl : specialization.control_deps) {
      if (!existing_control_deps.contains(ctrl)) {
        VLOG(3) << "Forward control dependency: input=" << ctrl;
        specialized_func_node->add_input(ctrl);
      }
    }
  }
}
void RemovePushedDownConstInputTypes(
    const FunctionSpecialization& specialization, const NodeDef& func_node,
    NodeDef* specialized_func_node) {
  if (specialization.const_inputs.empty()) return;
  const AttrValue* tin = AttrSlice(func_node).Find("Tin");
  if (tin == nullptr || !tin->has_list()) return;
  auto* attr = specialized_func_node->mutable_attr();
  (*attr)["Tin"].mutable_list()->clear_type();
  for (int i = 0; i < func_node.input_size(); ++i) {
    const string& input = func_node.input(i);
    if (IsControlInput(input)) break;
    if (!specialization.const_inputs.contains(input)) {
      DataType dt = tin->list().type(i);
      (*attr)["Tin"].mutable_list()->add_type(dt);
    }
  }
}
void RemoveUnusedOutputsTypes(const FunctionSpecialization& specialization,
                              const NodeDef& func_node,
                              NodeDef* specialized_func_node) {
  const AttrValue* tout = AttrSlice(func_node).Find("Tout");
  if (tout == nullptr || !tout->has_list()) return;
  int specialization_active_outputs_size = specialization.active_outputs.size();
  if (specialization_active_outputs_size == tout->list().type_size()) return;
  auto* attr = specialized_func_node->mutable_attr();
  (*attr)["Tout"].mutable_list()->clear_type();
  for (int i = 0; i < tout->list().type_size(); ++i) {
    if (specialization.active_outputs.contains(i)) {
      DataType dt = tout->list().type(i);
      (*attr)["Tout"].mutable_list()->add_type(dt);
    }
  }
}
Status UpdateSpecializedFunctionCallSite(const FunctionDef& func,
                                         const NodeDef& func_node,
                                         const string& specialized_func_name,
                                         NodeDef* specialized_func_node) {
  if (IsDirectFunctionCall(func, func_node)) {
    specialized_func_node->set_op(specialized_func_name);
  } else if (IsIndirectFunctionCall(func, func_node)) {
    auto* attr = specialized_func_node->mutable_attr();
    (*attr)[kFuncAttr].mutable_func()->set_name(specialized_func_name);
  } else {
    return absl::InvalidArgumentError("Unknown function call site");
  }
  return absl::OkStatus();
}
Status UpdateSpecializedFunctionNode(
    const FunctionDef& func, const NodeDef& func_node,
    const FunctionSpecialization& specialization,
    NodeDef* specialized_func_node) {
  bool is_indirect_call = IsIndirectFunctionCall(func, func_node);
  TF_RETURN_IF_ERROR(UpdateSpecializedFunctionCallSite(
      func, func_node, specialization.specialized_func_name,
      specialized_func_node));
  RemovePushedDownConstInputs(specialization, specialized_func_node);
  if (is_indirect_call) {
    RemovePushedDownConstInputTypes(specialization, func_node,
                                    specialized_func_node);
  }
  if (is_indirect_call && !specialization.is_in_fetch_set) {
    RemoveUnusedOutputsTypes(specialization, func_node, specialized_func_node);
  }
  specialized_func_node->mutable_attr()->erase("_gradient_op_type");
  return absl::OkStatus();
}
Status InitializeFunctionSpecializationSignature(
    const NodeDef& func_node, const FunctionDef& func,
    const AttrSlice& func_instantiation_attr,
    const FunctionOptimizerContext& ctx, FunctionSpecializationSignature* sig) {
  DCHECK(sig->const_inputs.empty());
  DCHECK(sig->active_outputs.empty());
  sig->func_name = func.signature().name();
  sig->is_in_fetch_set = ctx.IsFetchNode(func_node.name());
  sig->active_outputs = GetActiveOutputs(func_node, ctx);
  TF_RETURN_IF_ERROR(InstantiationTypeParameters(func, func_instantiation_attr,
                                                 &sig->type_parameters));
  TF_RETURN_IF_ERROR(InstantiationBodyParameters(func, func_instantiation_attr,
                                                 &sig->body_parameters));
  for (int i = 0; i < func_node.input_size(); ++i) {
    const string& input = func_node.input(i);
    if (IsControlInput(input)) break;
    if (ctx.IsTrulyConst(input)) {
      sig->const_inputs.emplace(i, input);
    }
  }
  return absl::OkStatus();
}
string SpecializedFunctionName(const FunctionOptimizerContext& ctx,
                               const FunctionDef& func,
                               const NodeDef& func_node) {
  return absl::Substitute(
      "$0_specialized_for_$1_at_$2", func.signature().name(),
      absl::StrReplaceAll(func_node.name(), {{"/", "_"}}), ctx.item().id);
}
Status SpecializeFunction(const NodeDef& func_node, const FunctionDef& func,
                          FunctionOptimizerContext* ctx,
                          GraphDef* optimized_graph) {
  VLOG(2) << "Specialize function call: " << SummarizeNodeDef(func_node);
  const AttrSlice func_instantiation_attr =
      FunctionInstantiationAttributes(func, func_node);
  FunctionSpecializationSignature signature;
  TF_RETURN_IF_ERROR(InitializeFunctionSpecializationSignature(
      func_node, func, func_instantiation_attr, *ctx, &signature));
  const FunctionSpecialization* already_specialized =
      ctx->FindFunctionSpecialization(signature);
  if (already_specialized) {
    VLOG(2) << "Function was already specialized in identical context: "
               "specialized_name="
            << already_specialized->specialized_func_name;
    NodeDef* specialized_func_node = optimized_graph->add_node();
    *specialized_func_node = func_node;
    TF_RETURN_IF_ERROR(UpdateSpecializedFunctionNode(
        func, func_node, *already_specialized, specialized_func_node));
    ctx->AddTensorMapping(specialized_func_node->name(), *already_specialized);
    return absl::OkStatus();
  }
  const auto& flib = ctx->function_library();
  GrapplerFunctionItem item;
  TF_RETURN_IF_ERROR(MakeGrapplerFunctionItem(
      func, func_instantiation_attr, flib, ctx->graph_version(), &item));
  absl::flat_hash_set<string> const_inputs;
  absl::flat_hash_set<string> control_deps;
  TF_RETURN_IF_ERROR(PushDownConstInputs(func_node, *ctx, &item, &const_inputs,
                                         &control_deps));
  std::vector<std::pair<int, int>> output_mapping;
  if (!signature.is_in_fetch_set) {
    int num_func_outputs = item.output_size();
    absl::flat_hash_set<int> remove;
    for (int i = 0; i < num_func_outputs; ++i) {
      if (!signature.active_outputs.count(i)) remove.insert(i);
    }
    TF_RETURN_IF_ERROR(RemoveFunctionOutputs(remove, &item, &output_mapping));
  }
  FunctionDef specialized_func;
  TF_RETURN_IF_ERROR(MakeFunctionDef(item, flib, &specialized_func));
  const string specialized_func_name =
      SpecializedFunctionName(*ctx, func, func_node);
  if (flib.Contains(specialized_func_name)) {
    return absl::InternalError("Created duplicate function specialization");
  }
  specialized_func.mutable_signature()->set_name(specialized_func_name);
  auto* specialized_attr = specialized_func.mutable_attr();
  (*specialized_attr)[kGrapplerSpecializedFuncAttr].set_b(true);
  TF_RETURN_IF_ERROR(ctx->function_library().AddFunctionDef(specialized_func));
  NodeDef* specialized_func_node = optimized_graph->add_node();
  *specialized_func_node = func_node;
  FunctionSpecialization func_specialization = {
      specialized_func_name, signature.is_in_fetch_set, const_inputs,
      control_deps,          signature.active_outputs,  output_mapping};
  TF_RETURN_IF_ERROR(UpdateSpecializedFunctionNode(
      func, func_node, func_specialization, specialized_func_node));
  ctx->AddSpecializedFunction(signature, func_specialization);
  ctx->AddTensorMapping(specialized_func_node->name(), func_specialization);
  return absl::OkStatus();
}
constexpr const char* const kLowerUsingSwitchMergeAttr =
    LowerFunctionalOpsPass::kLowerUsingSwitchMergeAttr;
constexpr const char* const kLowerAsMultiDeviceFunctionAttr =
    LowerFunctionalOpsPass::kLowerAsMultiDeviceFunctionAttr;
using KeepCallerNode = InlineFunctionBodyOptions::KeepCallerNode;
using OutputControlSource = InlineFunctionBodyOptions::OutputControlSource;
bool CheckBoolAttr(const Node* n, absl::string_view attr_name) {
  bool match;
  bool found = TryGetNodeAttr(n->attrs(), attr_name, &match);
  return found && match;
}
bool CheckStringAttr(const Node* n, absl::string_view attr_name) {
  const string& value = GetNodeAttrString(n->attrs(), attr_name);
  return !value.empty();
}
bool LowerUsingSwitchMergeIsOn(const Node* n) {
  return CheckBoolAttr(n, kLowerUsingSwitchMergeAttr);
}
bool LowerAsMultiDeviceFunctionIsOn(const Node* n) {
  return CheckBoolAttr(n, kLowerAsMultiDeviceFunctionAttr);
}
bool MarkedForXlaCompilation(const NodeDef& n) {
  auto is_enabled = [&](std::string attr_name) -> bool {
    auto it = n.attr().find(attr_name);
    return it != n.attr().end() && (!it->second.s().empty() || it->second.b());
  };
  return is_enabled("_xla_compile_id") || is_enabled("_tpu_replicate") ||
         is_enabled(kXlaMustCompileAttr);
}
const bool IsExemptFromSideEffectsExecutionValidation(const string& op) {
  static const auto* exemption = new absl::flat_hash_set<string>(
      {
       "CollectiveGather", "CollectiveReduce", "CollectiveBcastSend",
       "CollectiveBcastRecv", "CollectiveBcastSendV2", "CollectiveBcastRecvV2",
       "NcclAllReduce", "Send", "Recv", "CollectiveAssignGroupsV2",
       "CollectiveInitializeCommunicator",
       "RandomUniform", "RandomUniformInt", "RandomStandardNormal",
       "ParameterizedTruncatedNormal", "TruncatedNormal", "RandomShuffle",
       "Multinomial", "RandomGamma", "RandomGammaGrad", "RandomPoisson",
       "RandomPoissonV2",
       "ReadVariableOp",
       "CudnnRNN", "CudnnRNNBackprop", "CudnnRNNV2", "CudnnRNNV3",
       "CudnnRNNBackpropV2", "CudnnRNNBackpropV3",
       "EnqueueTPUEmbeddingSparseBatch", "EnqueueTPUEmbeddingIntegerBatch",
       "EnqueueTPUEmbeddingSparseTensorBatch",
       "EnqueueTPUEmbeddingRaggedTensorBatch",
       "EnqueueTPUEmbeddingArbitraryTensorBatch",
       "DynamicEnqueueTPUEmbeddingArbitraryTensorBatch",
       "SaveV2", "RestoreV2",
       "InfeedEnqueue", "InfeedEnqueueTuple"});
  return exemption->contains(op);
}
Status ValidateSideEffectsExecution(
    const FunctionBody& fbody, OutputControlSource output_control_source,
    bool has_outgoing_control_edges,
    bool validate_outgoing_control_edge = true) {
  std::vector<const Node*> fbody_side_effects;
  absl::c_copy_if(
      fbody.graph->nodes(), std::back_inserter(fbody_side_effects),
      [](const Node* n) {
        return n->op_def().is_stateful() && !n->IsArg() && !n->IsRetval() &&
               !IsExemptFromSideEffectsExecutionValidation(n->type_string());
      });
  if (!fbody_side_effects.empty() && !has_outgoing_control_edges) {
    const string error_message =
        "Can't guarantee execution of function side-effects after inlining. "
        "Function call node has no outgoing control edges.";
    if (validate_outgoing_control_edge) {
      return absl::InternalError(error_message);
    } else {
      VLOG(3) << error_message;
    }
  }
  absl::flat_hash_set<const Node*> control_sources;
  if (output_control_source == OutputControlSource::kDataOutputs) {
    control_sources = {fbody.ret_nodes.begin(), fbody.ret_nodes.end()};
  } else if (output_control_source == OutputControlSource::kControlOutputs) {
    control_sources = {fbody.control_ret_nodes.begin(),
                       fbody.control_ret_nodes.end()};
  }
  for (const Node* side_effect : fbody_side_effects) {
    VLOG(4) << "Check that node " << side_effect->name()
            << " will execute after inlining.";
    bool will_execute = false;
    const auto is_control_source = [&](const Node* n) -> void {
      const auto it = control_sources.find(n);
      if (it != control_sources.end()) {
        VLOG(4) << "Found a path to control source: " << side_effect->name()
                << " ---> " << (*it)->name();
        will_execute = true;
      }
    };
    DFSFrom(*fbody.graph, {side_effect}, is_control_source,
            {}, NodeComparatorName{});
    if (!will_execute) {
      return absl::InternalError(absl::StrCat(
          "Can't guarantee execution of a side-effectful node, that is not "
          "reachable from function control source. Function body node: ",
          SummarizeNode(*side_effect)));
    }
  }
  return absl::OkStatus();
}
Status ValidateNoDeadOutputs(const FunctionLibraryDefinition& flib_def,
                             const FunctionBody& fbody) {
  absl::flat_hash_set<const Node*> output_nodes = {fbody.ret_nodes.begin(),
                                                   fbody.ret_nodes.end()};
  std::vector<const Node*> dead_tensor_sources;
  for (const Node* n : fbody.graph->nodes()) {
    if (n->IsSwitch()) {
      VLOG(4) << "Add dead tensors source. Switch node: " << n->name();
      dead_tensor_sources.push_back(n);
      continue;
    }
    const FunctionDef* fdef = flib_def.Find(n->type_string());
    if (fdef != nullptr) {
      std::unique_ptr<FunctionBody> nested_fbody;
      NameAttrList func;
      TF_RETURN_IF_ERROR(NameAndAttrsFromFunctionCall(n->def(), &func));
      TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*fdef, AttrSlice(&func.attr()),
                                                 &flib_def, &nested_fbody));
      if (!ValidateNoDeadOutputs(flib_def, *nested_fbody).ok()) {
        VLOG(4) << "Add dead tensors source. Function call: " << func.name()
                << " node=" << n->name();
        dead_tensor_sources.push_back(n);
      }
    }
  }
  for (const Node* dead_tensor_source : dead_tensor_sources) {
    bool has_dead_output = false;
    const auto is_output_node = [&](const Node* n) -> void {
      const auto it = output_nodes.find(n);
      if (it != output_nodes.end()) {
        VLOG(4) << "Found a path to output node from dead tensor source: "
                << dead_tensor_source->name() << " ---> " << (*it)->name();
        has_dead_output = true;
      }
    };
    const auto stop_traversal = [&has_dead_output](const Edge& edge) -> bool {
      return !edge.src()->IsMerge() || has_dead_output;
    };
    DFSFrom(*fbody.graph, {dead_tensor_source}, is_output_node,
            {}, NodeComparatorName{},
            stop_traversal);
    if (has_dead_output) {
      return absl::InternalError(absl::StrCat(
          "Can't inline a function with dead outputs. Dead tensor source: ",
          SummarizeNode(*dead_tensor_source)));
    }
  }
  return absl::OkStatus();
}
Status MakeFunctionBodyForInlining(const Node& node,
                                   const FunctionLibraryDefinition& flib_def,
                                   std::unique_ptr<FunctionBody>* fbody) {
  VLOG(3) << "Make function body for inlining: " << SummarizeNode(node);
  const auto find_fdef = [&flib_def, &node](
                             const string& name,
                             const FunctionDef** fdef) -> Status {
    if ((*fdef = flib_def.Find(name)) == nullptr) {
      return absl::InternalError(absl::StrCat(
          "Was not able to find a function definition (name=", name,
          ") for a function call: ", SummarizeNode(node)));
    }
    return absl::OkStatus();
  };
  if (node.type_string() == FunctionLibraryDefinition::kGradientOp) {
    NameAttrList func;
    TF_RETURN_IF_ERROR(GetNodeAttr(node.attrs(), kFuncAttr, &func));
    const string grad = flib_def.FindGradient(func.name());
    if (!grad.empty()) {
      const FunctionDef* grad_fdef;
      TF_RETURN_IF_ERROR(find_fdef(grad, &grad_fdef));
      VLOG(4) << "Instantiate a custom SymbolicGradient: gradient=" << grad
              << " (function=" << func.name() << ")";
      TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
          *grad_fdef, AttrSlice(&func.attr()), &flib_def, fbody));
    } else if (flib_def.Find(func.name()) == nullptr) {
      gradient::Creator creator;
      TF_RETURN_IF_ERROR(gradient::GetOpGradientCreator(func.name(), &creator));
      if (creator == nullptr) {
        return absl::InvalidArgumentError(
            absl::StrCat("No gradient is defined for ", func.name()));
      }
      FunctionDef grad_fdef;
      TF_RETURN_IF_ERROR(creator(AttrSlice(&func.attr()), &grad_fdef));
      VLOG(4) << "Instantiate a SymbolicGradient for a primitive op: "
              << func.name();
      TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(
          grad_fdef, AttrSlice(&func.attr()), &flib_def, fbody));
    } else {
      const FunctionDef* fdef;
      TF_RETURN_IF_ERROR(find_fdef(func.name(), &fdef));
      VLOG(4) << "Instantiate a SymbolicGradient for a function: "
              << func.name();
      TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*fdef, AttrSlice(&func.attr()),
                                                 &flib_def, fbody));
      *fbody = SymbolicGradient(**fbody);
    }
  } else {
    NameAttrList func;
    TF_RETURN_IF_ERROR(NameAndAttrsFromFunctionCall(node.def(), &func));
    const FunctionDef* fdef;
    TF_RETURN_IF_ERROR(find_fdef(func.name(), &fdef));
    VLOG(4) << "Instantiate a function call: function=" << func.name();
    TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*fdef, AttrSlice(&func.attr()),
                                               &flib_def, fbody));
  }
  return absl::OkStatus();
}
void AddStrictInputSemantics(Node* caller, Graph* g) {
  absl::flat_hash_set<const Node*> existing_control_sources;
  for (const Edge* edge : caller->in_edges()) {
    if (edge->IsControlEdge()) {
      existing_control_sources.insert(edge->src());
    }
  }
  const bool has_incoming_control_edges = !existing_control_sources.empty();
  const bool has_resource_input =
      absl::c_any_of(caller->input_types(),
                     [](const DataType dtype) { return dtype == DT_RESOURCE; });
  const bool has_constant_enter_input =
      absl::c_any_of(caller->in_edges(), [](const Edge* edge) {
        Node* src = edge->src();
        return src->IsEnter() && CheckBoolAttr(src, "is_constant");
      });
  const bool requires_strict_semantics =
      (!has_incoming_control_edges && has_resource_input) ||  
      (has_constant_enter_input);                             
  if (!requires_strict_semantics) return;
  std::set<const Node*> data_inputs;
  for (const Edge* edge : caller->in_edges()) {
    if (!edge->IsControlEdge() &&
        !existing_control_sources.contains(edge->src())) {
      data_inputs.insert(edge->src());
    }
  }
  VLOG(3) << "Add control edges from all data inputs to enforce strict "
             "semantics with regard to function inputs";
  const auto is_placeholder = [](const Node* node) -> bool {
    return node->type_string() == "Placeholder";
  };
  for (const Node* node : data_inputs) {
    if (is_placeholder(node)) continue;
    g->AddControlEdge(g->FindNodeId(node->id()), caller,
                      true);
  }
}
void AddFrameForwardingControlEdge(const std::vector<ControlFlowInfo>& info,
                                   Node* caller, Graph* g) {
  int info_size = info.size();
  if (caller->id() >= info_size) return;
  const Node* frame = info[caller->id()].frame;
  const bool is_in_while_loop = frame->id() != Graph::kSourceId;
  if (!is_in_while_loop) return;
  const bool has_incoming_control_edges =
      absl::c_any_of(caller->in_edges(),
                     [](const Edge* edge) { return edge->IsControlEdge(); });
  if (has_incoming_control_edges) return;
  VLOG(3) << "Add a frame forwarding control edge: from=" << frame->name()
          << " to=" << caller->name();
  Node* enter = g->FindNodeId(frame->id());
  bool is_constant_enter = enter->attrs().Find("is_constant")->b();
  if (is_constant_enter) {
    g->AddControlEdge(enter, caller);
  } else {
    auto it = absl::c_find_if(enter->out_edges(), [](const Edge* e) {
      return !e->IsControlEdge() && e->dst()->IsMerge();
    });
    if (it != enter->out_edges().end()) {
      g->AddControlEdge((*it)->dst(), caller);
    } else {
      LOG(WARNING) << "Enter[is_constant=false] node: " << enter->name()
                   << " does not have an outgoing edge to a Merge.";
    }
  }
}
Status InlineFunctionCalls(const GrapplerItem& item,
                           const RewriterConfig::Toggle opt_level,
                           const bool lower_control_flow,
                           GraphDef* output_graph) {
  bool is_aggressive = opt_level == RewriterConfig::AGGRESSIVE;
  VLOG(2) << "Inline function calls: grappler_item_id=" << item.id
          << " (aggressive_mode=" << is_aggressive << ")";
  FunctionLibraryDefinition flib_def =
      FunctionLibraryDefinition(OpRegistry::Global(), item.graph.library());
  std::unique_ptr<Graph> graph = std::make_unique<Graph>(flib_def);
  GraphConstructorOptions graph_constructor_options;
  graph_constructor_options.allow_internal_ops = true;
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(graph_constructor_options,
                                            item.graph, graph.get()));
  using NodeNames = absl::flat_hash_set<absl::string_view>;
  NodeNames fetch_nodes;
  fetch_nodes.reserve(item.fetch.size());
  for (const string& fetch : item.fetch) {
    fetch_nodes.insert(ParseTensorName(fetch).node());
  }
  NodeNames keep_nodes(item.keep_ops.begin(), item.keep_ops.end());
  if (item.save_op.size() > 0) {
    keep_nodes.insert(item.save_op);
  }
  if (item.restore_op.size() > 0) {
    keep_nodes.insert(item.restore_op);
  }
  std::vector<string> inlined_function_names;
  NodeNames feed_nodes;
  feed_nodes.reserve(item.feed.size());
  for (const std::pair<std::string, Tensor>& feed : item.feed) {
    feed_nodes.insert(ParseTensorName(feed.first).node());
  }
  std::vector<ControlFlowInfo> control_flow_info;
  TF_RETURN_IF_ERROR(BuildControlFlowInfo(graph.get(), &control_flow_info));
  for (int i = 2; i < graph->num_node_ids(); ++i) {
    Node* n = graph->FindNodeId(i);
    if (n == nullptr) continue;  
    if (lower_control_flow && LowerUsingSwitchMergeIsOn(n)) {
      VLOG(2) << "Lower functional control flow op: " << SummarizeNode(*n);
      AddStrictInputSemantics(n, graph.get());
      AddFrameForwardingControlEdge(control_flow_info, n, graph.get());
      if (n->IsIfNode()) {
        TF_RETURN_IF_ERROR(RewriteIfNode(n, graph.get(), false));
      } else if (n->IsCaseNode()) {
        TF_RETURN_IF_ERROR(RewriteCaseNode(n, graph.get(), false));
      } else if (n->IsWhileNode()) {
        TF_RETURN_IF_ERROR(RewriteWhileNode(n, graph.get(), &flib_def, false));
      }
      continue;
    }
    if (!IsFunctionCall(flib_def, *n)) continue;
    if (MarkedForXlaCompilation(n->def())) continue;
    if (feed_nodes.contains(n->name())) continue;
    if (n->name() == item.restore_op || n->name() == item.save_op) continue;
    std::unique_ptr<FunctionBody> fbody;
    TF_RETURN_IF_ERROR(MakeFunctionBodyForInlining(*n, flib_def, &fbody));
    InlineFunctionBodyOptions inline_options;
    inline_options.ignore_noinline = is_aggressive;
    bool force_inline_as_multi_device = LowerAsMultiDeviceFunctionIsOn(n);
    if (n->IsPartitionedCall() || force_inline_as_multi_device) {
      inline_options.output_control_src = OutputControlSource::kControlOutputs;
      inline_options.inlined_function_body_placer =
          InlinedFunctionBodyPlacer::MultiDevice();
    } else {
      inline_options.output_control_src = OutputControlSource::kDataOutputs;
      inline_options.inlined_function_body_placer =
          InlinedFunctionBodyPlacer::SingleDevice();
    }
    if (fetch_nodes.contains(n->name())) {
      inline_options.keep_caller_node = KeepCallerNode::kFetchable;
    } else if (keep_nodes.contains(n->name())) {
      inline_options.keep_caller_node = KeepCallerNode::kTargetable;
    } else {
      inline_options.keep_caller_node = KeepCallerNode::kDoNotKeep;
    }
    Status can_inline_function_call =
        ValidateInlining(n, fbody.get(), inline_options);
    if (can_inline_function_call.ok()) {
      bool has_outgoing_control_edges = absl::c_any_of(
          n->out_edges(),
          [](const Edge* edge) { return edge->IsControlEdge(); });
      can_inline_function_call = ValidateSideEffectsExecution(
          *fbody, inline_options.output_control_src,
          has_outgoing_control_edges);
      if (!can_inline_function_call.ok() &&
          (is_aggressive || force_inline_as_multi_device)) {
        VLOG(2) << "Ignore error: " << can_inline_function_call.message();
        can_inline_function_call = absl::OkStatus();
      }
    }
    if (can_inline_function_call.ok()) {
      can_inline_function_call = ValidateNoDeadOutputs(flib_def, *fbody);
    }
    if (can_inline_function_call.ok()) {
      VLOG(2) << "Inline function call node: " << n->name();
      AddStrictInputSemantics(n, graph.get());
      AddFrameForwardingControlEdge(control_flow_info, n, graph.get());
      TF_RETURN_IF_ERROR(InlineFunctionBody(flib_def, graph.get(), n,
                                            fbody.get(), inline_options));
      inlined_function_names.push_back(
          fbody->record->fdef().signature().name());
    } else {
      VLOG(2) << "Failed to inline function call node: "
              << can_inline_function_call.message();
    }
  }
  VLOG(4) << "Inlined " << inlined_function_names.size()
          << " function calls: " << absl::StrJoin(inlined_function_names, ", ");
  if (inlined_function_names.empty()) {
    VLOG(3) << "Not placing graph after function inlining"
            << " (did not inline any of the function calls).";
  } else if (item.devices().empty()) {
    VLOG(3) << "Not placing graph after function inlining"
            << " (device set is empty)";
  } else {
    VLOG(3) << "Run placer for the graph after function inlining. "
            << "Devices: [" << absl::StrJoin(item.devices(), ", ") << "]";
    DeviceSet device_set;                               
    std::vector<std::unique_ptr<Device>> fake_devices;  
    for (const string& name : item.devices()) {
      auto device = std::make_unique<FakeDevice>(name);
      device_set.AddDevice(device.get());
      fake_devices.push_back(std::move(device));
    }
    Placer placer(graph.get(), item.id, &flib_def, &device_set);
    TF_RETURN_IF_ERROR(placer.Run());
  }
  graph->ToGraphDef(output_graph);
  return absl::OkStatus();
}
void RestoreTensorMapping(const FunctionOptimizerContext& ctx,
                          GraphDef* optimized_graph) {
  if (ctx.tensor_mapping().empty()) return;
  for (NodeDef& node : *optimized_graph->mutable_node()) {
    for (int idx = 0; idx < node.input_size(); ++idx) {
      TensorId input_tensor = ParseTensorName(node.input(idx));
      if (input_tensor.index() == Graph::kControlSlot) break;
      auto mapping = ctx.tensor_mapping().find(input_tensor);
      if (mapping != ctx.tensor_mapping().end()) {
        node.set_input(idx, TensorIdToString(mapping->second));
      }
    }
  }
}
}  
Status FunctionOptimizer::RunFunctionOptimizerPass(
    const GrapplerItem& item, GraphDef* optimized_graph) const {
  VLOG(3) << "Run function optimizer pass: grappler_item_id=" << item.id;
  GraphDef graph_after_inlining;
  TF_RETURN_IF_ERROR(InlineFunctionCalls(item, opt_level_, lower_control_flow_,
                                         &graph_after_inlining));
  FunctionOptimizerContext ctx(item, opt_level_, graph_after_inlining);
  for (const NodeDef& node : graph_after_inlining.node()) {
    const int num_nodes_before = optimized_graph->node_size();
    const auto is_graph_modified = [&]() {
      int num_nodes = optimized_graph->node_size();
      DCHECK_GE(num_nodes, num_nodes_before) << "Nodes should not be removed";
      return num_nodes > num_nodes_before;
    };
    const auto copy_node = [&]() { *optimized_graph->add_node() = node; };
    const FunctionDef* func = FindFunctionCall(ctx, node);
    if (func == nullptr) {
      copy_node();
      continue;
    }
    const string& func_name = func->signature().name();
    const bool specialization_worthy = IsParametrized(*func) ||
                                       HasTrulyConstInputs(node, ctx) ||
                                       HasUnusedOutputs(node, *func, ctx);
    const string grad_func = ctx.function_library().FindGradient(func_name);
    const bool no_specialize =
        !grad_func.empty() || ctx.IsFeedNode(node.name()) ||
        MarkedNoSpecialize(*func) || MarkedForXlaCompilation(node);
    if (specialization_worthy && !no_specialize) {
      Status status = SpecializeFunction(node, *func, &ctx, optimized_graph);
      if (!status.ok() && is_graph_modified()) {
        return status;
      } else if (!status.ok() && !is_graph_modified()) {
        VLOG(3) << "Skip specialization error: " << status.message();
        copy_node();
      }
      continue;
    } else {
      VLOG(2) << "Skip function specialization: " << func->signature().name();
      copy_node();
    }
  }
  RestoreTensorMapping(ctx, optimized_graph);
  *optimized_graph->mutable_versions() = item.graph.versions();
  *optimized_graph->mutable_library() =
      PruneFunctionLibrary(ctx.function_library(), *optimized_graph);
  return absl::OkStatus();
}
Status FunctionOptimizer::Optimize(Cluster*, const GrapplerItem& item,
                                   GraphDef* optimized_graph) {
  if (item.graph.library().function_size() == 0) {
    return absl::AbortedError("Nothing to do.");
  }
  TF_RETURN_IF_ERROR(RunFunctionOptimizerPass(item, optimized_graph));
  return absl::OkStatus();
}
}  
}  