#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/clusters/utils.h"
#include "tensorflow/core/grappler/costs/op_level_cost_estimator.h"
namespace tensorflow {
namespace grappler {
VirtualCluster::VirtualCluster(
    const std::unordered_map<string, DeviceProperties>& devices)
    : VirtualCluster(devices, std::make_unique<OpLevelCostEstimator>(),
                     ReadyNodeManagerFactory("FirstReady")) {}
VirtualCluster::VirtualCluster(
    const std::unordered_map<string, DeviceProperties>& devices,
    std::unique_ptr<OpLevelCostEstimator> node_estimator,
    std::unique_ptr<ReadyNodeManager> node_manager)
    : Cluster(0) {
  devices_ = devices;
  estimator_ = std::make_unique<AnalyticalCostEstimator>(
      this, std::move(node_estimator), std::move(node_manager),
      true, false);
}
VirtualCluster::VirtualCluster(const DeviceSet* device_set)
    : VirtualCluster(std::unordered_map<string, DeviceProperties>()) {
  device_set_ = device_set;
  for (const auto& device : device_set_->devices()) {
    DeviceProperties props = GetDeviceInfo(device->parsed_name());
    if (props.type() == "UNKNOWN") continue;
    auto attrs = device->attributes();
    props.set_memory_size(attrs.memory_limit());
    devices_[device->name()] = props;
  }
}
VirtualCluster::~VirtualCluster() {}
Status VirtualCluster::Provision() { return absl::OkStatus(); }
Status VirtualCluster::Initialize(const GrapplerItem& item) {
  return absl::OkStatus();
}
Status VirtualCluster::Run(const GraphDef& graph,
                           const std::vector<std::pair<string, Tensor>>& feed,
                           const std::vector<string>& fetch,
                           RunMetadata* metadata) {
  GrapplerItem item;
  item.graph = graph;
  item.feed = feed;
  item.fetch = fetch;
  return Run(item, metadata);
}
Status VirtualCluster::Run(const GrapplerItem& item, RunMetadata* metadata) {
  if (metadata) {
    metadata->clear_step_stats();
    metadata->clear_cost_graph();
    metadata->clear_partition_graphs();
  }
  TF_RETURN_IF_ERROR(estimator_->Initialize(item));
  TF_RETURN_IF_ERROR(
      estimator_->PredictCosts(item.graph, metadata, nullptr));
  const std::unordered_map<string, DeviceProperties>& device = GetDevices();
  std::unordered_map<string, int64_t> peak_mem_usage =
      estimator_->GetScheduler()->GetPeakMemoryUsage();
  for (const auto& mem_usage : peak_mem_usage) {
    const string& device_name = mem_usage.first;
    auto it = device.find(device_name);
    if (it == device.end()) {
      continue;
    }
    const DeviceProperties& dev = it->second;
    if (dev.memory_size() <= 0) {
      continue;
    }
    int64_t peak_mem = mem_usage.second;
    if (peak_mem >= dev.memory_size()) {
      return errors::ResourceExhausted(
          "Graph requires ", peak_mem, " bytes of memory on device ",
          device_name, " to run ", " but device only has ", dev.memory_size(),
          " available.");
    }
  }
  return absl::OkStatus();
}
}  
}  