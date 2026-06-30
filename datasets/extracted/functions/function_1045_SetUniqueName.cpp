#include "tensorflow/core/common_runtime/replicate_constants_pass.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/config/flag_defs.h"
#include "tensorflow/core/config/flags.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
namespace tensorflow {
namespace {
constexpr int64_t kMaxSize = 16;
void SetUniqueName(Graph* graph, Node* node) {
  node->set_name(graph->NewName(absl::StrCat(node->name(), "/replicate")));
}
bool HasControlOut(Node* node) {
  auto control_out_it =
      std::find_if(node->out_edges().begin(), node->out_edges().end(),
                   [](const auto& e) { return e->IsControlEdge(); });
  return control_out_it != node->out_edges().end();
}
bool HasCpuDevice(const Node* node) {
  DeviceNameUtils::ParsedName device;
  if (!DeviceNameUtils::ParseFullName(node->assigned_device_name(), &device))
    return false;
  return device.type == "CPU";
}
Status DeviceNameToCpuDeviceNameWithDeviceId(const string& device_name,
                                             string* host_device_name) {
  DeviceNameUtils::ParsedName device;
  if (!DeviceNameUtils::ParseFullName(device_name, &device)) {
    return absl::InternalError(
        absl::StrCat("Could not parse device name ", device_name));
  }
  if (flags::Global().enable_aggressive_constant_replication.value() &&
      device.type == "CPU") {
    *host_device_name = device_name;
  } else {
    device.type = "CPU";
    device.has_type = true;
    device.id = 0;
    device.has_id = true;
    *host_device_name = DeviceNameUtils::ParsedNameToString(device);
  }
  return absl::OkStatus();
}
Status GetDestinationCpuDevice(const Node* dst, std::string* device) {
  if (!dst->has_assigned_device_name())
    return absl::AbortedError(
        absl::StrCat("Node name: ", dst->name(), " has no assigned device."));
  return DeviceNameToCpuDeviceNameWithDeviceId(dst->assigned_device_name(),
                                               device);
}
Status GetSuccessorEdges(
    Node* node,
    absl::btree_map<std::string, std::vector<const Edge*>>& device_to_edges) {
  for (const auto& edge : node->out_edges()) {
    const Node* dst = edge->dst();
    std::string device;
    TF_RETURN_IF_ERROR(GetDestinationCpuDevice(dst, &device));
    if (!device_to_edges.count(device)) device_to_edges.insert({device, {}});
    device_to_edges[device].push_back(edge);
  }
  return absl::OkStatus();
}
void ReplicateToEachDevice(
    Graph* graph, Node* node,
    absl::btree_map<std::string, std::vector<const Edge*>>& device_to_edges) {
  for (const auto& pair : device_to_edges) {
    Node* copy = graph->CopyNode(node);
    SetUniqueName(graph, copy);
    const std::string device = pair.first;
    copy->set_assigned_device_name(device);
    for (const Edge* edge : pair.second) {
      graph->AddEdge(copy, edge->src_output(), edge->dst(), edge->dst_input());
    }
    for (Node* src : node->in_nodes()) {
      graph->AddControlEdge(src, copy, true);
    }
  }
  graph->RemoveNode(node);
}
}  
Status ReplicateConstantsPass::Run(
    const GraphOptimizationPassOptions& options) {
  VLOG(1) << "replicate_constants_pass will replicate constants with "
             "number-of-elements <= "
          << kMaxSize;
  if (options.graph == nullptr) {
    VLOG(1) << "No graph in replicate_constants_pass.";
    return absl::OkStatus();
  }
  Graph* graph = options.graph->get();
  if (VLOG_IS_ON(1)) {
    VLOG(1) << DumpGraphToFile("before_replicate_constants_pass", *graph,
                               options.flib_def);
  }
  int64_t min_skipped = std::numeric_limits<int64_t>::max();
  int64_t max_skipped = std::numeric_limits<int64_t>::min();
  for (Node* node : graph->nodes()) {
    if (!node->IsConstant()) continue;
    if (node->out_edges().size() <= 1) continue;
    if (HasControlOut(node)) continue;
    const TensorProto* value = nullptr;
    TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "value", &value));
    TF_ASSIGN_OR_RETURN(TensorShape shape,
                        TensorShape::BuildTensorShape(value->tensor_shape()));
    if (shape.num_elements() > kMaxSize) {
      min_skipped = std::min(min_skipped, shape.num_elements());
      max_skipped = std::max(max_skipped, shape.num_elements());
      continue;
    }
    if (!node->has_assigned_device_name()) continue;
    if (!HasCpuDevice(node)) continue;
    absl::btree_map<std::string, std::vector<const Edge*>> device_to_edges;
    TF_RETURN_IF_ERROR(GetSuccessorEdges(node, device_to_edges));
    if (device_to_edges.size() <= 1) continue;
    ReplicateToEachDevice(graph, node, device_to_edges);
  }
  if (min_skipped != std::numeric_limits<int64_t>::max()) {
    VLOG(1) << "replicate_constants_pass skipped replicating constants with "
               "number of elements in the range "
            << min_skipped << " to " << max_skipped << ".";
  }
  if (VLOG_IS_ON(1)) {
    VLOG(1) << DumpGraphToFile("after_replicate_constants_pass", *graph,
                               options.flib_def);
  }
  return absl::OkStatus();
}
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 3,
                      ReplicateConstantsPass);
}  