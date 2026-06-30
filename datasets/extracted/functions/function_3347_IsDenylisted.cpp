#include "tensorflow/core/grappler/optimizers/pin_to_host_optimizer.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils/symbolic_shapes.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/grappler/utils/tpu.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
namespace tensorflow {
namespace grappler {
namespace internal {
constexpr int64_t kTensorMaxSize = 64;
bool IsDenylisted(const NodeDef& node) {
  return
      IsCollective(node) ||
      IsControlFlow(node) ||
      IsNoOp(node);
}
bool IsTensorSmall(const OpInfo::TensorProperties& prop) {
  if (prop.dtype() == DataType::DT_STRING) {
    return true;
  }
  if (prop.dtype() != DataType::DT_INT32 &&
      prop.dtype() != DataType::DT_INT64 &&
      prop.dtype() != DataType::DT_FLOAT) {
    return false;
  }
  const int64_t size = NumCoefficients(prop.shape());
  if (size < 0 || size > kTensorMaxSize) {
    return false;
  }
  return true;
}
Status TryFindKernelDef(const std::vector<DeviceType>& devices,
                        const NodeDef& node, const KernelDef** kdef) {
  for (const DeviceType& device : devices) {
    const KernelDef* kernel = nullptr;
    Status s = FindKernelDef(device, node, &kernel, nullptr);
    if (s.ok()) {
      if (kdef) {
        *kdef = kernel;
      }
      return absl::OkStatus();
    }
  }
  return errors::NotFound("Could not find KernelDef for op: ", node.op());
}
Status IsNodeOutputPortHostFriendly(const GraphView& graph,
                                    GraphProperties* properties,
                                    const NodeDef& node, int port_id,
                                    bool* is_candidate) {
  *is_candidate = false;
  if (IsDenylisted(node)) {
    return absl::OkStatus();
  }
  if (!properties->has_properties()) {
    TF_RETURN_IF_ERROR(properties->InferStatically(
        false, false,
        false));
  }
  const auto& output_properties = properties->GetOutputProperties(node.name());
  int output_properties_size = output_properties.size();
  if (port_id >= output_properties_size) {
    LOG(WARNING) << "port_id=" << port_id
                 << " but output_properties.size()=" << output_properties.size()
                 << "\n"
                 << node.DebugString();
    return absl::OkStatus();
  }
  if (!IsTensorSmall(output_properties[port_id])) {
    return absl::OkStatus();
  }
  if (IsIdentity(node) || IsIdentityNSingleInput(node)) {
    for (const auto& fanin : graph.GetFanins(node, false)) {
      bool fanin_candidate = false;
      TF_RETURN_IF_ERROR(IsNodeOutputPortHostFriendly(
          graph, properties, *fanin.node, fanin.port_id, &fanin_candidate));
      if (!fanin_candidate) {
        return absl::OkStatus();
      }
    }
    *is_candidate = true;
    return absl::OkStatus();
  }
  if (absl::StrContains(node.device(), DEVICE_CPU)) {
    *is_candidate = true;
    return absl::OkStatus();
  }
  const OpDef* op = nullptr;
  Status s = OpRegistry::Global()->LookUpOpDef(node.op(), &op);
  if (!s.ok()) {
    LOG(WARNING) << "Could not find OpDef for : " << node.op();
    return absl::OkStatus();
  }
  const int output_arg_id = OpOutputPortIdToArgId(node, *op, port_id);
  if (output_arg_id < 0) {
    LOG(WARNING) << "Invalid port: " << port_id << "!\n"
                 << node.DebugString() << "\n"
                 << op->DebugString();
    return absl::OkStatus();
  }
  const KernelDef* kernel = nullptr;
  s = TryFindKernelDef({node.device().c_str(), DEVICE_GPU, DEVICE_CPU}, node,
                       &kernel);
  if (!s.ok()) {
    LOG(INFO) << "Could not find KernelDef for: " << node.op();
    return absl::OkStatus();
  }
  for (const string& host_memory_arg : kernel->host_memory_arg()) {
    if (op->output_arg(output_arg_id).name() == host_memory_arg) {
      *is_candidate = true;
      break;
    }
  }
  return absl::OkStatus();
}
bool IsNodeInputPortHostFriendly(const NodeDef& node, int port_id) {
  if (absl::StrContains(node.device(), DEVICE_CPU)) {
    return true;
  }
  const OpDef* op = nullptr;
  Status s = OpRegistry::Global()->LookUpOpDef(node.op(), &op);
  if (!s.ok()) {
    LOG(WARNING) << "Could not find OpDef for : " << node.op();
    return false;
  }
  const int input_arg_id = OpInputPortIdToArgId(node, *op, port_id);
  const KernelDef* kernel = nullptr;
  s = internal::TryFindKernelDef(
      {node.device().c_str(), DEVICE_GPU, DEVICE_CPU}, node, &kernel);
  if (!s.ok()) {
    LOG(INFO) << "Could not find KernelDef for: " << node.op();
    return false;
  }
  for (const string& host_memory_arg : kernel->host_memory_arg()) {
    if (op->input_arg(input_arg_id).name() == host_memory_arg) {
      return true;
    }
  }
  return false;
}
Status IsNodeHostCandidate(const GraphView& graph, GraphProperties* properties,
                           const NodeDef& node, bool* is_candidate) {
  *is_candidate = false;
  if (absl::StrContains(node.device(), DEVICE_CPU)) {
    *is_candidate = true;
    return absl::OkStatus();
  }
  if (IsDenylisted(node)) {
    return absl::OkStatus();
  }
  Status s = TryFindKernelDef({DEVICE_CPU}, node, nullptr);
  if (!s.ok()) {
    return absl::OkStatus();
  }
  for (const GraphView::OutputPort& fanin :
       graph.GetFanins(node, false)) {
    bool fanin_candidate = false;
    TF_RETURN_IF_ERROR(IsNodeOutputPortHostFriendly(
        graph, properties, *fanin.node, fanin.port_id, &fanin_candidate));
    if (!fanin_candidate) {
      return absl::OkStatus();
    }
  }
  if (!properties->has_properties()) {
    TF_RETURN_IF_ERROR(properties->InferStatically(
        false, false,
        false));
  }
  for (const auto& prop : properties->GetOutputProperties(node.name())) {
    if (!IsTensorSmall(prop)) {
      return absl::OkStatus();
    }
  }
  *is_candidate = true;
  return absl::OkStatus();
}
string TryFindHostDevice(const gtl::FlatSet<string>& devices,
                         bool has_device_cpu, const string& device) {
  if (device.empty() && has_device_cpu) {
    return "/device:CPU:0";
  } else if (absl::StrContains(device, DEVICE_GPU)) {
    for (const auto& device_match :
         {std::pair<string, string>("GPU", "CPU:0"),
          std::pair<string, string>("/device", "/device:CPU:0")}) {
      const string device_host =
          strings::StrCat(device.substr(0, device.rfind(device_match.first)),
                          device_match.second);
      if (devices.find(device_host) != devices.end()) {
        return device_host;
      }
    }
  }
  return "";
}
}  
Status PinToHostOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                    GraphDef* optimized_graph) {
  *optimized_graph = item.graph;
  if (IsLegacyTPUBridgeGraphDef(*optimized_graph)) {
    return absl::OkStatus();
  }
  GraphProperties properties(item);
  GraphView graph(optimized_graph);
  gtl::FlatSet<string> devices;
  if (cluster) {
    const std::vector<string> device_names = cluster->GetDeviceNames();
    devices.insert(device_names.begin(), device_names.end());
  } else {
    devices = {"/device:CPU:0"};
  }
  const bool has_device_cpu = devices.find("/device:CPU:0") != devices.end();
  TF_RETURN_IF_ERROR(TopologicalSort(optimized_graph));
  std::vector<std::pair<NodeDef*, string>> const_nodes;
  for (auto& node : *optimized_graph->mutable_node()) {
    GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();
    bool is_candidate = false;
    TF_RETURN_IF_ERROR(
        internal::IsNodeHostCandidate(graph, &properties, node, &is_candidate));
    if (!is_candidate) {
      continue;
    }
    string device =
        internal::TryFindHostDevice(devices, has_device_cpu, node.device());
    if (!device.empty()) {
      if (IsConstant(node)) {
        const_nodes.emplace_back(&node, node.device());
      }
      VLOG(2) << "Moving node " << node.name() << " to device " << device;
      *node.mutable_device() = std::move(device);
    }
  }
  for (auto& it : const_nodes) {
    GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();
    NodeDef* node = it.first;
    const string& device = it.second;
    for (const GraphView::InputPort& fanout : graph.GetFanouts(*node, false)) {
      if (!internal::IsNodeInputPortHostFriendly(*fanout.node,
                                                 fanout.port_id)) {
        VLOG(2) << "Swapping node " << node->name() << " back to device "
                << device;
        node->set_device(device);
        break;
      }
    }
  }
  return absl::OkStatus();
}
}  
}  