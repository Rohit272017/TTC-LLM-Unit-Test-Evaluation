#include "tensorflow/core/common_runtime/inline_function_utils.h"
#include <deque>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/function_utils.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/optimizer_cse.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/util/device_name_utils.h"
namespace tensorflow {
 constexpr const char* const
    LowerFunctionalOpsConstants::kLowerUsingSwitchMergeAttr;
 constexpr const char* const
    LowerFunctionalOpsConstants::kLowerAsMultiDeviceFunctionAttr;
namespace {
static constexpr const char* const kArgOp = FunctionLibraryDefinition::kArgOp;
static constexpr const char* const kDeviceArgOp =
    FunctionLibraryDefinition::kDeviceArgOp;
static constexpr const char* const kRetOp = FunctionLibraryDefinition::kRetOp;
static constexpr const char* const kDeviceRetOp =
    FunctionLibraryDefinition::kDeviceRetOp;
static constexpr const char* const kGradientOp =
    FunctionLibraryDefinition::kGradientOp;
static constexpr const char* const kNodeLabel = "Func";
static constexpr const char* const kFuncAttr =
    FunctionLibraryDefinition::kFuncAttr;
struct Endpoint {
  Node* node;
  int index;
  string name() const {
    if (index == 0) {
      return node->name();
    } else {
      return strings::StrCat(node->name(), ":", index);
    }
  }
  DataType dtype() const { return node->output_type(index); }
};
struct EndpointHash {
  uint64 operator()(const Endpoint& x) const {
    return Hash64(reinterpret_cast<const char*>(&x.node), sizeof(Node*),
                  x.index);
  }
};
struct EndpointEq {
  bool operator()(const Endpoint& x, const Endpoint& y) const {
    return (x.node == y.node) && (x.index == y.index);
  }
};
static Node* AddNoOp(StringPiece name, Graph* g) {
  NodeDef ndef;
  ndef.set_name(g->NewName(absl::StrCat(kNodeLabel, "/", name)));
  ndef.set_op("NoOp");
  Status s;
  Node* ret = g->AddNode(ndef, &s);
  TF_CHECK_OK(s);
  return ret;
}
static Node* AddIdentity(StringPiece name, Graph* g, Endpoint input) {
  DCHECK_LT(0, input.dtype());
  NodeDef ndef;
  ndef.set_name(g->NewName(absl::StrCat(kNodeLabel, "/", name)));
  ndef.set_op("Identity");
  ndef.add_input(input.name());
  AddNodeAttr("T", BaseType(input.dtype()), &ndef);
  Status s;
  Node* ret = g->AddNode(ndef, &s);
  TF_CHECK_OK(s);
  g->AddEdge(input.node, input.index, ret, 0);
  return ret;
}
std::vector<string> InputDevices(const Node& caller) {
  std::vector<string> input_devices(caller.in_edges().size());
  std::vector<string> input_tensors(caller.in_edges().size());
  for (const Edge* edge : caller.in_edges()) {
    if (edge->IsControlEdge()) continue;
    const string& input_device = edge->src()->has_assigned_device_name()
                                     ? edge->src()->assigned_device_name()
                                     : edge->src()->requested_device();
    input_devices[edge->dst_input()] = input_device;
    input_tensors[edge->dst_input()] =
        absl::StrCat(edge->src()->name(), ":", edge->src_output());
  }
  if (VLOG_IS_ON(4)) {
    VLOG(4) << "Function instantiation input devices:";
    for (int i = 0; i < input_devices.size(); ++i) {
      if (input_tensors[i].empty()) continue;  
      VLOG(4) << "    [index " << i << "]"
              << " device: " << input_devices[i]
              << " (input: " << input_tensors[i] << ")";
    }
  }
  return input_devices;
}
class DefaultFunctionBodyPlacer : public InlinedFunctionBodyPlacer {
 public:
  explicit DefaultFunctionBodyPlacer(const Node& caller)
      : input_devices_(InputDevices(caller)) {}
  absl::optional<string> InputNodeDevice(int input_index) const override {
    return input_devices_[input_index];
  }
  absl::optional<string> OutputNodeDevice(int output_index) const override {
    return absl::nullopt;
  }
  bool ColocateInputOutputIdentities() const override { return false; }
  absl::optional<string> ControlNodeDevice() const override {
    return absl::nullopt;
  }
  absl::optional<string> BodyNodeDevice(const NodeDef& ndef) const override {
    return absl::nullopt;
  }
 private:
  const std::vector<string> input_devices_;
};
class SingleDeviceFunctionBodyPlacer : public InlinedFunctionBodyPlacer {
 public:
  explicit SingleDeviceFunctionBodyPlacer(const Node& caller)
      : caller_device_(caller.def().device()) {}
  absl::optional<string> InputNodeDevice(int input_index) const override {
    return caller_device_;
  }
  absl::optional<string> OutputNodeDevice(int output_index) const override {
    return caller_device_;
  }
  bool ColocateInputOutputIdentities() const override { return false; }
  absl::optional<string> ControlNodeDevice() const override {
    return caller_device_;
  }
  absl::optional<string> BodyNodeDevice(const NodeDef& ndef) const override {
    return caller_device_;
  }
 private:
  const string caller_device_;
};
class MultiDeviceFunctionBodyPlacer : public InlinedFunctionBodyPlacer {
 public:
  explicit MultiDeviceFunctionBodyPlacer(const Node& caller)
      : caller_device_(caller.def().device()),
        input_devices_(InputDevices(caller)) {
    has_parsed_caller_device_ =
        DeviceNameUtils::ParseFullName(caller_device_, &caller_parsed_device_);
  }
  absl::optional<string> InputNodeDevice(int input_index) const override {
    return input_devices_[input_index];
  }
  absl::optional<string> OutputNodeDevice(int output_index) const override {
    return absl::nullopt;
  }
  bool ColocateInputOutputIdentities() const override { return true; }
  absl::optional<string> ControlNodeDevice() const override {
    return caller_device_;
  }
  absl::optional<string> BodyNodeDevice(const NodeDef& ndef) const override {
    if (ndef.device().empty()) return caller_device_;
    if (!has_parsed_caller_device_) return ndef.device();
    DeviceNameUtils::ParsedName ndef_parsed_device;
    if (!DeviceNameUtils::ParseFullName(ndef.device(), &ndef_parsed_device))
      return ndef.device();
    DeviceNameUtils::MergeUnsetDevNames(&ndef_parsed_device,
                                        caller_parsed_device_);
    return DeviceNameUtils::ParsedNameToString(ndef_parsed_device);
  }
 private:
  string caller_device_;
  bool has_parsed_caller_device_;
  DeviceNameUtils::ParsedName caller_parsed_device_;
  std::vector<string> input_devices_;
};
}  
std::unique_ptr<InlinedFunctionBodyPlacer>
InlinedFunctionBodyPlacer::DefaultPlacer(const Graph& graph,
                                         const Node& caller) {
  VLOG(3) << "Create default placer for inlined function body.";
  return std::make_unique<DefaultFunctionBodyPlacer>(caller);
}
std::unique_ptr<InlinedFunctionBodyPlacer>
InlinedFunctionBodyPlacer::SingleDevicePlacer(const Graph& graph,
                                              const Node& caller) {
  VLOG(3) << "Create single device placer for inlined function body.";
  return std::make_unique<SingleDeviceFunctionBodyPlacer>(caller);
}
std::unique_ptr<InlinedFunctionBodyPlacer>
InlinedFunctionBodyPlacer::MultiDevicePlacer(const Graph& graph,
                                             const Node& caller) {
  VLOG(3) << "Create multi device placer for inlined function body.";
  return std::make_unique<MultiDeviceFunctionBodyPlacer>(caller);
}
namespace {
Status ValidateNoInline(const FunctionBody* fbody) {
  const auto attr = AttrSlice(&fbody->record->fdef().attr());
  bool noinline = false;
  if (TryGetNodeAttr(attr, kNoInlineAttr, &noinline) && noinline) {
    return errors::InvalidArgument(
        "Can't inline function marked with '_noinline'");
  }
  return absl::OkStatus();
}
using OutputControlSrc = InlineFunctionBodyOptions::OutputControlSource;
void PropagateDebugInfoToNode(const string& func,
                              const std::vector<const Node*>& nodes,
                              NodeDef* target) {
  if (nodes.empty() || target->has_experimental_debug_info()) {
    return;
  }
  for (const Node* node : nodes) {
    const auto& node_def = node->def();
    if (node_def.has_experimental_debug_info()) {
      target->mutable_experimental_debug_info()->MergeFrom(
          node_def.experimental_debug_info());
    } else {
      target->mutable_experimental_debug_info()->add_original_node_names(
          node_def.name());
      target->mutable_experimental_debug_info()->add_original_func_names(func);
    }
  }
}
}  
string InlineFunctionBodyOptions::DebugString() const {
  const auto true_false = [](bool b) { return b ? "true" : "false"; };
  const auto keep_caller_node_str = [this]() -> string {
    switch (keep_caller_node) {
      case KeepCallerNode::kDoNotKeep:
        return "DoNotKeep";
      case KeepCallerNode::kFetchable:
        return "Fetchable";
      case KeepCallerNode::kTargetable:
        return "Targetable";
    }
  };
  return absl::StrCat(
      "disable_inlining=", true_false(disable_inlining),
      ", ignore_noinline=", true_false(ignore_noinline),
      ", inline_impl_selection_group_functions=",
      true_false(inline_impl_selection_group_functions),
      ", keep_caller_node=", keep_caller_node_str(), ", output_control_src=",
      output_control_src == OutputControlSrc::kDataOutputs ? "DataOutputs"
                                                           : "ControlOutputs",
      ", inlined_function_body_placer=", inlined_function_body_placer.name,
      ", uniquify_frame_names=", true_false(uniquify_frame_names));
}
Status ValidateInlining(const Node* node, const FunctionBody* fbody,
                        const InlineFunctionBodyOptions& options) {
  const auto num_node_inputs = static_cast<size_t>(node->num_inputs());
  const auto num_node_outputs = static_cast<size_t>(node->num_outputs());
  if (num_node_inputs != fbody->arg_types.size() ||
      num_node_inputs != fbody->arg_nodes.size()) {
    return errors::InvalidArgument(
        "Node inputs do not match function arguments: inputs=", num_node_inputs,
        " arg_types=", fbody->arg_types.size(),
        " arg_nodes=", fbody->arg_nodes.size());
  }
  if (num_node_outputs != fbody->ret_types.size() ||
      num_node_outputs != fbody->ret_nodes.size()) {
    return errors::InvalidArgument(
        "Node outputs do not match function returns: outputs=",
        num_node_outputs, " ret_types=", fbody->ret_types.size(),
        " ret_nodes=", fbody->ret_nodes.size());
  }
  for (int i = 0; i < node->num_inputs(); ++i) {
    if (node->input_type(i) != fbody->arg_types[i]) {
      return errors::InvalidArgument(
          "Node input type doesn't match function argument type: ",
          node->input_type(i), " != ", fbody->arg_types[i], " @ index=", i);
    }
  }
  for (int i = 0; i < node->num_outputs(); ++i) {
    if (node->output_type(i) != fbody->ret_types[i]) {
      return errors::InvalidArgument(
          "Node output type doesn't match function return type: ",
          node->output_type(i), " != ", fbody->ret_types[i], " @ index=", i);
    }
  }
  if (options.disable_inlining) {
    return errors::InvalidArgument(
        "Function inlining explicitly disabled by 'options.disable_inlining'");
  }
  if (!options.inline_impl_selection_group_functions) {
    bool is_impl_selection_group_function =
        fbody->record->fdef().attr().find("api_implements") !=
        fbody->record->fdef().attr().end();
    if (is_impl_selection_group_function) {
      return errors::InvalidArgument(
          "Inlining of implementation selection group function ",
          fbody->record->fdef().signature().name(),
          " is disabled by options.inline_impl_selection_group_functions");
    }
  }
  if (!options.ignore_noinline) {
    TF_RETURN_IF_ERROR(ValidateNoInline(fbody));
  }
  return absl::OkStatus();
}
Status InlineFunctionBody(const FunctionLibraryDefinition& flib_def, Graph* g,
                          Node* caller, const FunctionBody* fbody,
                          const InlineFunctionBodyOptions& options) {
  VLOG(3) << "Inline function call: " << SummarizeNode(*caller) << " ["
          << options.DebugString() << "]";
  VLOG(4) << "Inlining function: "
          << fbody->record->fdef().DebugString();  
  VLOG(4) << "Current graphdef: " << g->ToGraphDefDebug().DebugString();
  VLOG(4) << "Caller: " << caller->DebugString();
  Status validation = ValidateInlining(caller, fbody, options);
  if (!validation.ok()) {
    return errors::Internal("Inlining mismatch: ", validation.message());
  }
  const std::unique_ptr<InlinedFunctionBodyPlacer> placer =
      options.inlined_function_body_placer.get(*g, *caller);
  static constexpr bool kDoNotCheckDuplicates = true;
  const auto no_op = [&](StringPiece name) -> Node* {
    Node* node = AddNoOp(absl::StrCat(caller->name(), "/", name), g);
    const absl::optional<string> device = placer->ControlNodeDevice();
    if (device.has_value()) node->set_requested_device(*device);
    return node;
  };
  const auto input_identity = [&](StringPiece name, Endpoint input,
                                  int index) -> Node* {
    Node* node = AddIdentity(absl::StrCat(caller->name(), "/", name), g, input);
    const absl::optional<string> device = placer->InputNodeDevice(index);
    if (device.has_value()) node->set_requested_device(*device);
    bool colocate_identity = placer->ColocateInputOutputIdentities();
    if (colocate_identity) {
      node->AddAttr(kColocationAttrName,
                    std::vector<string>{absl::StrCat(kColocationGroupPrefix,
                                                     input.node->name())});
    }
    return node;
  };
  const auto output_identity = [&](StringPiece name, Endpoint input,
                                   int index) -> Node* {
    Node* node = AddIdentity(absl::StrCat(caller->name(), "/", name), g, input);
    const absl::optional<string> device = placer->OutputNodeDevice(index);
    if (device.has_value()) node->set_requested_device(*device);
    bool colocate_identity = placer->ColocateInputOutputIdentities();
    if (colocate_identity) {
      node->AddAttr(kColocationAttrName,
                    std::vector<string>{absl::StrCat(kColocationGroupPrefix,
                                                     input.node->name())});
    }
    return node;
  };
  auto arg_name = [&](auto& args, size_t i) -> absl::string_view {
    if (i < args.size()) {
      return args[i].name();
    } else {
      return "<unknown>";
    }
  };
  std::vector<Endpoint> inputs(caller->num_inputs());
  Node* input_control_node = nullptr;
  for (const Edge* e : caller->in_edges()) {
    if (e->IsControlEdge()) {
      if (input_control_node == nullptr) {
        input_control_node = no_op("input_control_node");
      }
      g->AddControlEdge(e->src(), input_control_node, kDoNotCheckDuplicates);
    } else {
      inputs[e->dst_input()] = {e->src(), e->src_output()};
    }
  }
  if (input_control_node != nullptr) {
    VLOG(3) << "Created input control node: " << input_control_node->name();
  }
  std::vector<Node*> input_nodes;
  std::map<absl::string_view, absl::string_view> input_node_name_map;
  for (std::size_t i = 0; i < fbody->arg_nodes.size(); ++i) {
    if (inputs[i].node == nullptr)
      return errors::Internal("Null node found for input ", i);
    Node* n = input_identity("input", inputs[i], i);
    input_node_name_map[arg_name(fbody->record->fdef().signature().input_arg(),
                                 i)] = n->name();
    input_nodes.push_back(n);
  }
  std::unordered_set<string> fn_nodes;
  for (Node* n : fbody->graph->op_nodes()) {
    fn_nodes.insert(n->name());
  }
  std::vector<Node*> node_map(fbody->graph->num_node_ids());
  for (Node* n : fbody->graph->op_nodes()) {
    NodeDef ndef = n->def();
    const absl::optional<string> device = placer->BodyNodeDevice(ndef);
    if (device.has_value()) ndef.set_device(*device);
    PropagateDebugInfoToNode(fbody->record->fdef().signature().name(), {n},
                             &ndef);
    const string prefix = strings::StrCat(caller->name(), "/");
    TF_RETURN_IF_ERROR(AddPrefixAndSuffixToNode(prefix, "", &ndef,
                                                options.uniquify_frame_names));
    TF_RETURN_IF_ERROR(
        MaybeUpdateColocationConstraintsWithMap(input_node_name_map, &ndef));
    TF_RETURN_IF_ERROR(
        MaybeAddPrefixToColocationConstraints(fn_nodes, prefix, &ndef));
    Status added_node;
    Node* clone = g->AddNode(std::move(ndef), &added_node);
    TF_CHECK_OK(added_node);
    node_map[n->id()] = clone;
    clone->SetStackTrace(n->GetStackTrace());
    if (input_control_node) {
      const auto is_input_edge = [](const Edge* e) -> bool {
        return !e->src()->IsSource();
      };
      const auto is_control_edge = [](const Edge* e) -> bool {
        return !e->src()->IsSource() && e->IsControlEdge();
      };
      const bool forward_execution_frame =
          (absl::c_none_of(n->in_edges(), is_input_edge) ||       
           (n->IsFunctionCall() &&                                
            absl::c_none_of(n->in_edges(), is_control_edge))) &&  
          !n->IsArg();                                            
      if (forward_execution_frame) {
        VLOG(4) << "Add control edge from input control node to: "
                << clone->name();
        g->AddControlEdge(input_control_node, clone, kDoNotCheckDuplicates);
      }
    }
  }
  for (const Edge* e : fbody->graph->edges()) {
    if (e->src()->IsSource() || e->src()->IsSink() || e->dst()->IsSource() ||
        e->dst()->IsSink()) {
      continue;
    }
    Node* src_copy = node_map[e->src()->id()];
    Node* dst_copy = node_map[e->dst()->id()];
    g->AddEdge(src_copy, e->src_output(), dst_copy, e->dst_input());
  }
  VLOG(4) << "Add input Identity nodes for each function argument:";
  for (std::size_t i = 0; i < fbody->arg_nodes.size(); ++i) {
    Node* arg = node_map[fbody->arg_nodes[i]->id()];
    Node* n = input_nodes[i];
    VLOG(4) << "    [index " << i << "] "
            << arg_name(fbody->record->fdef().signature().input_arg(), i)
            << " as " << n->name() << " (input: " << inputs[i].name()
            << ", requested_device: " << n->requested_device() << ")";
    if (input_control_node) {
      g->AddControlEdge(input_control_node, n, kDoNotCheckDuplicates);
    }
    for (const Edge* e : arg->out_edges()) {
      if (e->IsControlEdge()) {
        g->AddControlEdge(n, e->dst(), kDoNotCheckDuplicates);
      } else {
        g->AddEdge(n, 0, e->dst(), e->dst_input());
      }
    }
    node_map[fbody->arg_nodes[i]->id()] = n;
    g->RemoveNode(arg);  
  }
  VLOG(4) << "Add output Identity nodes for each function output argument:";
  std::vector<Node*> outputs(caller->num_outputs());
  for (std::size_t i = 0; i < fbody->ret_nodes.size(); ++i) {
    Node* ret = node_map[fbody->ret_nodes[i]->id()];
    Endpoint data;  
    for (const Edge* e : ret->in_edges()) {
      if (!e->IsControlEdge()) {
        data = {e->src(), e->src_output()};
        break;
      }
    }
    CHECK(data.node != nullptr);
    Node* n = output_identity("output", data, i);
    outputs[i] = n;
    VLOG(4) << "    [index " << i << "] "
            << arg_name(fbody->record->fdef().signature().output_arg(), i)
            << " as " << n->name() << " (ret: " << data.node->name() << ":"
            << data.index << ", requested_device: " << n->requested_device()
            << ")";
    for (const Edge* e : ret->in_edges()) {
      if (e->IsControlEdge()) {
        g->AddControlEdge(e->src(), n, kDoNotCheckDuplicates);
      }
    }
    g->RemoveNode(ret);  
  }
  Node* output_control_node = nullptr;
  const bool has_control_outputs = absl::c_any_of(
      caller->out_edges(), [](const Edge* e) { return e->IsControlEdge(); });
  using KeepCallerNode = InlineFunctionBodyOptions::KeepCallerNode;
  const bool keep_caller_node =
      options.keep_caller_node == KeepCallerNode::kFetchable ||
      options.keep_caller_node == KeepCallerNode::kTargetable;
  if (has_control_outputs || keep_caller_node) {
    output_control_node = no_op("output_control_node");
    VLOG(4) << "Add output control node: " << output_control_node->name();
    if (options.output_control_src == OutputControlSrc::kDataOutputs) {
      for (Node* n : outputs) {
        VLOG(4) << "    [data output] add control edge from: " << n->name();
        g->AddControlEdge(n, output_control_node, kDoNotCheckDuplicates);
      }
    } else {
      for (Node* fbody_node : fbody->control_ret_nodes) {
        Node* n = node_map[fbody_node->id()];
        VLOG(4) << "    [control output] add control edge from: " << n->name();
        g->AddControlEdge(n, output_control_node, kDoNotCheckDuplicates);
      }
    }
  }
  if (output_control_node && output_control_node->in_edges().empty()) {
    if (input_control_node) {
      VLOG(4) << "Add a control edge between input and output control nodes: "
              << input_control_node->name() << " to "
              << output_control_node->name();
      g->AddControlEdge(input_control_node, output_control_node,
                        kDoNotCheckDuplicates);
    } else {
      VLOG(4) << "Function inlining potentially dropped execution frame "
                 "information from outgoing control edges.";
    }
  }
  for (const Edge* e : caller->out_edges()) {
    if (e->IsControlEdge()) {
      g->AddControlEdge(output_control_node, e->dst(), kDoNotCheckDuplicates);
    } else {
      g->AddEdge(outputs[e->src_output()], 0, e->dst(), e->dst_input());
    }
  }
  if (keep_caller_node) {
    std::vector<NodeBuilder::NodeOut> output_tensors;
    absl::c_transform(outputs, std::back_inserter(output_tensors),
                      [](Node* n) { return NodeBuilder::NodeOut(n, 0); });
    Node* caller_substitute_node;
    if (options.keep_caller_node == KeepCallerNode::kTargetable ||
        output_tensors.empty()) {
      TF_CHECK_OK(NodeBuilder(caller->name(), "NoOp")
                      .Device(caller->requested_device())
                      .ControlInput(output_control_node)
                      .Finalize(g, &caller_substitute_node));
    } else if (options.keep_caller_node == KeepCallerNode::kFetchable) {
      TF_CHECK_OK(NodeBuilder(caller->name(), "IdentityN")
                      .Device(caller->requested_device())
                      .Input(output_tensors)
                      .ControlInput(output_control_node)
                      .Finalize(g, &caller_substitute_node));
    }
  }
  VLOG(3) << "Successfully inlined function call node: " << caller->name();
  g->RemoveNode(caller);
  VLOG(4) << "Final graph: " << g->ToGraphDefDebug().DebugString();
  return absl::OkStatus();
}
bool ExpandInlineFunctions(FunctionLibraryRuntime* lib, Graph* graph,
                           const ExpandInlineFunctionsOptions& options) {
  std::vector<std::pair<Node*, const FunctionBody*>> candidates;
  const FunctionLibraryDefinition* fld = lib->GetFunctionLibraryDefinition();
  for (Node* node : graph->nodes()) {
    if (!IsFunctionCall(*lib->GetFunctionLibraryDefinition(), *node)) {
      continue;
    }
    bool noinline;
    if (fld->GetAttr(*node, kNoInlineAttr, &noinline).ok() && noinline) {
      VLOG(3) << "noinline: " << SummarizeNode(*node);
      continue;
    }
    FunctionLibraryRuntime::Handle handle;
    Status s = InstantiateFunctionCall(node->def(), lib, &handle);
    if (!s.ok()) {
      LOG(ERROR) << "Failed to instantiate a function:  " << s.message();
      continue;
    }
    const FunctionBody* fbody = lib->GetFunctionBody(handle);
    CHECK_NOTNULL(fbody);
    candidates.emplace_back(node, fbody);
  }
  bool inlined_any = false;
  for (const auto& p : candidates) {
    Status inlined = InlineFunctionBody(*fld, graph, p.first, p.second,
                                        p.first->IsPartitionedCall()
                                            ? options.multi_device_options
                                            : options.native_options);
    if (inlined.ok()) {
      inlined_any = true;
    } else {
      VLOG(1) << "Failed to inline function call: node=" << p.first->name()
              << " error=" << inlined.message();
    }
  }
  return inlined_any;
}
}  