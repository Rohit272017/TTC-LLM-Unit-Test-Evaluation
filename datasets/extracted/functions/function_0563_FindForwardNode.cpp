#include "tensorflow/core/grappler/optimizers/implementation_selector.h"
#include <string>
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/function_api_info.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/util/device_name_utils.h"
namespace tensorflow {
namespace grappler {
constexpr char kConstOp[] = "Const";
constexpr char kCaseOp[] = "Case";
constexpr char kStatelessCaseOp[] = "StatelessCase";
constexpr char kDeviceIndexOp[] = "DeviceIndex";
string FindForwardNode(utils::MutableNodeView* backward_node) {
  const int last_input_index = backward_node->NumRegularFanins() - 1;
  const utils::MutableFanoutView& input =
      backward_node->GetRegularFanin(last_input_index);
  if (IsIdentity(*input.node_view()->node())) {
    return input.node_view()->node()->input(0);
  } else if (IsPartitionedCall(*input.node_view()->node()) ||
             IsStatefulPartitionedCall(*input.node_view()->node())) {
    return backward_node->node()->input(last_input_index);
  } else {
    return "";
  }
}
void UpdateForwardIdentityNodeDtype(utils::MutableNodeView* forward_node,
                                    const DataTypeVector& dtypes) {
  const auto& fanouts_vector = forward_node->GetRegularFanouts();
  for (int pos = 0, pos_limit = fanouts_vector.size(); pos < pos_limit; ++pos) {
    const auto& fanouts_at_pos = fanouts_vector[pos];
    for (const auto& fanout : fanouts_at_pos) {
      if ("Identity" == fanout.node_view()->GetOp()) {
        (*fanout.node_view()->node()->mutable_attr())["T"].set_type(
            dtypes[pos]);
        VLOG(3) << "Updated DTYPE for Identity node: "
                << fanout.node_view()->node()->DebugString();
      }
    }
  }
}
Status UpdateNodeDef(utils::MutableNodeView* node_view, const string& funcName,
                     const FunctionApiInfo& apiInfo) {
  NodeDef* node_def = node_view->node();
  VLOG(3) << "Node def before swap is: " << node_def->DebugString();
  node_def->mutable_attr()->find("f")->second.mutable_func()->set_name(
      funcName);
  auto tin = node_def->mutable_attr()->find("Tin");
  tin->second.mutable_list()->clear_type();
  for (const auto& tin_dtype : apiInfo.input_arg_dtypes()) {
    tin->second.mutable_list()->add_type(tin_dtype);
  }
  auto tout = node_def->mutable_attr()->find("Tout");
  tout->second.mutable_list()->clear_type();
  for (const auto& tout_dtype : apiInfo.output_arg_dtypes()) {
    tout->second.mutable_list()->add_type(tout_dtype);
  }
  if (apiInfo.function_type() == FunctionApiInfo::BACKWARD) {
    std::vector<std::string> control_deps;
    for (int i = node_def->input_size() - 1; i >= 0; --i) {
      if (!IsControlInput(node_def->input(i))) break;
      control_deps.push_back(node_def->input(i));
      node_def->mutable_input()->RemoveLast();
    }
    const int prev_input_size = node_def->input_size();
    const int diff = prev_input_size - apiInfo.input_arg_dtypes().size();
    if (diff >= 0) {
      for (int i = 0; i < diff; ++i) node_def->mutable_input()->RemoveLast();
    } else {
      const string last_input = FindForwardNode(node_view);
      const std::vector<string> name_index = ::absl::StrSplit(last_input, ':');
      if (name_index.size() != 2) {
        return errors::InvalidArgument(
            "Invalid format of input node name: ", last_input,
            " Expected: {forward_node_name}:{index}");
      }
      const absl::string_view node_name = name_index[0];
      int last_index;
      if (!::absl::SimpleAtoi(name_index[1], &last_index)) {
        return errors::InvalidArgument(
            "The index of input node is expected to be number, got: ",
            name_index[1]);
      }
      for (int i = 1; i <= -diff; ++i)
        node_def->add_input(strings::StrCat(node_name, ":", i + last_index));
    }
    for (std::string& control : control_deps)
      node_def->add_input(std::move(control));
  } else if (apiInfo.function_type() == FunctionApiInfo::FORWARD) {
    UpdateForwardIdentityNodeDtype(node_view, apiInfo.output_arg_dtypes());
  }
  VLOG(3) << "Node def after swap is: " << node_def->DebugString();
  return absl::OkStatus();
}
Status ImplementationSelector::LoadFunctions(const GraphDef& graph) {
  lib_info_ = std::make_unique<FunctionLibraryApiInfo>();
  TF_RETURN_IF_ERROR(lib_info_->Init(graph.library()));
  return absl::OkStatus();
}
Status ImplementationSelector::MaybeOptimizeFunctionCall(
    utils::MutableNodeView* node_view) const {
  NodeDef* node_def = node_view->node();
  std::vector<string> function_attribute_names;
  for (const auto& attr : node_def->attr()) {
    if (attr.second.has_func() &&
        lib_info_->GetApiInfo(attr.second.func().name()) != nullptr) {
      function_attribute_names.emplace_back(attr.first);
    }
  }
  if (function_attribute_names.empty() &&
      lib_info_->GetApiInfo(node_def->op()) == nullptr) {
    return absl::OkStatus();
  }
  DeviceNameUtils::ParsedName parsed_name;
  if (!DeviceNameUtils::ParseFullName(node_def->device(), &parsed_name) ||
      !parsed_name.has_type) {
    return errors::Internal("Could not parse device name:", node_def->device());
  }
  VLOG(2) << "Op " << node_def->name() << " runs on " << node_def->device()
          << " = (" << parsed_name.type << ")";
  for (const auto& attr_name : function_attribute_names) {
    string function_name = node_def->attr().at(attr_name).func().name();
    if (::absl::StrContains(function_name, "_specialized_for_")) continue;
    std::vector<string> equiv_func_names;
    TF_RETURN_IF_ERROR(lib_info_->GetEquivalentImplementations(
        function_name, &equiv_func_names));
    for (const auto& func_name : equiv_func_names) {
      const auto& func_api_info = lib_info_->GetApiInfo(func_name);
      if (func_api_info->preferred_device() == parsed_name.type) {
        VLOG(2) << "Swapping: " << function_name << " TO: " << func_name;
        TF_RETURN_IF_ERROR(UpdateNodeDef(node_view, func_name, *func_api_info));
        break;
      }
    }
  }
  if (lib_info_->GetApiInfo(node_def->op()) != nullptr &&
      !::absl::StrContains(node_def->op(), "_specialized_for_")) {
    std::vector<string> equiv_func_names;
    TF_RETURN_IF_ERROR(lib_info_->GetEquivalentImplementations(
        node_def->op(), &equiv_func_names));
    for (const string& func_name : equiv_func_names) {
      const auto func_api_info = lib_info_->GetApiInfo(func_name);
      if (func_api_info->preferred_device() == parsed_name.type) {
        node_def->set_op(func_name);
        break;
      }
    }
  }
  return absl::OkStatus();
}
Status FindDeviceIndex(const utils::MutableNodeView* device_index_node,
                       const string& device, int* index) {
  DeviceNameUtils::ParsedName parsed_name;
  if (!DeviceNameUtils::ParseFullName(device, &parsed_name) ||
      !parsed_name.has_type) {
    return errors::Internal("Could not parse device name:", device);
  }
  const auto& device_list =
      device_index_node->GetAttr("device_names")->list().s();
  auto it = absl::c_find(device_list, parsed_name.type);
  if (it != device_list.end()) {
    *index = it - device_list.begin();
  } else {
    *index = device_list.size();
  }
  return absl::OkStatus();
}
void RewriteDeviceIndexOp(utils::MutableNodeView* device_index_node,
                          int index) {
  auto node = device_index_node->node();
  node->set_op(kConstOp);
  EraseRegularNodeAttributes(node);
  (*node->mutable_attr())["dtype"].set_type(DT_INT32);
  auto* tensor = (*node->mutable_attr())["value"].mutable_tensor();
  tensor->set_dtype(DT_INT32);
  tensor->add_int_val(index);
  VLOG(2) << "Node after rewriting:" << node->DebugString();
}
Status ImplementationSelector::SelectDeviceIndex(GraphDef* graph) const {
  Status status;
  VLOG(2) << "graph before rewriting device index:" << graph->DebugString();
  utils::MutableGraphView graph_view(graph, &status);
  TF_RETURN_IF_ERROR(status);
  const int num_nodes = graph_view.NumNodes();
  for (int k = 0; k < num_nodes; ++k) {
    auto* node_view = graph_view.GetNode(k);
    if (node_view->GetOp() != kDeviceIndexOp) {
      continue;
    }
    VLOG(2) << "Found a node to rewrite the device index";
    for (const auto& fanouts : node_view->GetRegularFanouts()) {
      for (const auto& fanout : fanouts) {
        if (fanout.node_view()->GetOp() != kCaseOp &&
            fanout.node_view()->GetOp() != kStatelessCaseOp)
          continue;
        int index;
        Status status =
            FindDeviceIndex(node_view, fanout.node_view()->GetDevice(), &index);
        if (status.ok()) {
          RewriteDeviceIndexOp(node_view, index);
        }
      }
    }
  }
  return absl::OkStatus();
}
Status ImplementationSelector::SelectImplementation(GraphDef* graph) const {
  if (!graph->has_library()) {
    VLOG(2) << "Skipping graph since it does not have function def";
    return absl::OkStatus();
  }
  if (lib_info_->empty()) {
    VLOG(2) << "Skipping optimization since lib_info is empty";
    return absl::OkStatus();
  }
  Status status;
  utils::MutableGraphView graph_view(graph, &status);
  TF_RETURN_IF_ERROR(status);
  const int num_nodes = graph_view.NumNodes();
  for (int k = 0; k < num_nodes; ++k) {
    TF_RETURN_IF_ERROR(MaybeOptimizeFunctionCall(graph_view.GetNode(k)));
  }
  return absl::OkStatus();
}
Status ImplementationSelector::Optimize(Cluster* cluster,
                                        const GrapplerItem& item,
                                        GraphDef* optimized_graph) {
  auto status = LoadFunctions(item.graph);
  if (!status.ok()) {
    VLOG(2) << "Skipping optimization due to error while loading function "
            << "libraries: " << status;
    return errors::Aborted("Skipped Optimization");
  }
  *optimized_graph = item.graph;
  status = SelectDeviceIndex(optimized_graph);
  if (!status.ok()) {
    *optimized_graph = item.graph;
    VLOG(2) << "Could not rewrite device index due to error:" << status;
  }
  return SelectImplementation(optimized_graph);
}
}  
}  