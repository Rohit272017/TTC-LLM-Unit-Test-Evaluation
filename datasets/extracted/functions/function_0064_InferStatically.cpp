#include "tensorflow/core/grappler/costs/graph_memory.h"
#include <deque>
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"  
#include "tensorflow/core/framework/tensor_description.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/utils.h"
namespace tensorflow {
namespace grappler {
Status GraphMemory::InferStatically(
    const std::unordered_map<string, DeviceProperties>& devices) {
  VirtualCluster cluster(devices);
  TF_RETURN_IF_ERROR(cluster.Provision());
  TF_RETURN_IF_ERROR(cluster.Initialize(item_));
  RunMetadata metadata;
  Status s = cluster.Run(item_, &metadata);
  if (!s.ok() && s.code() != error::RESOURCE_EXHAUSTED) {
    return s;
  }
  InferFromTrace(metadata.step_stats());
  return absl::OkStatus();
}
Status GraphMemory::InferDynamically(Cluster* cluster) {
  if (!cluster->DetailedStatsEnabled()) {
    return errors::Unavailable("Detailed stats collection must be enabled");
  }
  TF_RETURN_IF_ERROR(cluster->Initialize(item_));
  RunMetadata metadata;
  TF_RETURN_IF_ERROR(cluster->Run(item_, &metadata));
  InferFromTrace(metadata.step_stats());
  return absl::OkStatus();
}
int64_t GraphMemory::GetWorstCaseMemoryUsage() const {
  int64_t worst_case = -1;
  for (const auto& peak_usage : peak_usage_) {
    worst_case = std::max(worst_case, peak_usage.second.used_memory);
  }
  return worst_case;
}
void GraphMemory::InferMemUsageForNodes(
    const std::vector<const NodeDef*>& nodes, GraphProperties* properties,
    int64_t* worst_case_memory_usage, int64_t* best_case_memory_usage) const {
  *worst_case_memory_usage = 0;
  *best_case_memory_usage = 0;
  for (const auto& node : item_.graph.node()) {
    std::vector<OpInfo::TensorProperties> outputs =
        properties->GetOutputProperties(node.name());
    int64_t node_memory_usage = InferMemUsageForNeighbors(outputs);
    *worst_case_memory_usage += node_memory_usage;
    std::vector<OpInfo::TensorProperties> inputs =
        properties->GetInputProperties(node.name());
    node_memory_usage += InferMemUsageForNeighbors(inputs);
    *best_case_memory_usage =
        std::max(*best_case_memory_usage, node_memory_usage);
  }
}
int64_t GraphMemory::InferMemUsageForNeighbors(
    const std::vector<OpInfo::TensorProperties>& props) const {
  int64_t neighbors_memory_usage = 0;
  for (const auto& prop : props) {
    DataType dtype = prop.dtype();
    int size = DataTypeSize(dtype);
    TensorShapeProto shape = prop.shape();
    if (shape.unknown_rank()) {
      continue;
    }
    for (int i = 0; i < shape.dim_size(); ++i) {
      if (shape.dim(i).size() < 0) {
        shape.mutable_dim(i)->set_size(1);
      }
    }
    int num_elems = TensorShape(shape).num_elements();
    neighbors_memory_usage += num_elems * size;
  }
  return neighbors_memory_usage;
}
static GraphMemory::LiveTensor* FindOrCreateLiveTensor(
    const string& node_name, int output_id,
    std::unordered_map<string, GraphMemory::LiveTensor*>* live_tensors,
    std::deque<GraphMemory::LiveTensor>* device_tensors) {
  string name = strings::StrCat(node_name, ":", output_id);
  GraphMemory::LiveTensor* live;
  auto it = live_tensors->find(name);
  if (it == live_tensors->end()) {
    GraphMemory::LiveTensor temp;
    temp.node = node_name;
    temp.output_id = output_id;
    temp.allocation_time = 0;
    temp.deallocation_time = 0;
    device_tensors->push_front(temp);
    live = &device_tensors->front();
    (*live_tensors)[name] = live;
  } else {
    live = it->second;
  }
  return live;
}
namespace {
struct Event {
  Event(int64_t _timestamp, bool _allocated,
        const GraphMemory::LiveTensor* _tensor)
      : timestamp(_timestamp), allocated(_allocated), tensor(_tensor) {}
  int64_t timestamp;
  bool allocated;
  const GraphMemory::LiveTensor* tensor;
  bool operator<(const Event& other) const {
    return timestamp < other.timestamp;
  }
};
}  
void GraphMemory::InferFromTrace(const StepStats& timeline) {
  std::unordered_map<string, string> node_placement;
  for (const auto& dev_stats : timeline.dev_stats()) {
    for (const auto& node_stats : dev_stats.node_stats()) {
      node_placement[node_stats.node_name()] = dev_stats.device();
    }
  }
  std::unordered_map<string, LiveTensor*> live_tensors;
  std::unordered_map<string, std::deque<LiveTensor>> live_tensors_per_device;
  std::unordered_map<string, const NodeDef*> node_map;
  for (const NodeDef& node : item_.graph.node()) {
    node_map[node.name()] = &node;
  }
  for (const auto& dev_stats : timeline.dev_stats()) {
    const string& device_name = dev_stats.device();
    const bool is_gpu = (device_name.find("GPU:") || device_name.find("gpu:"));
    std::deque<LiveTensor>& device_tensors =
        live_tensors_per_device[dev_stats.device()];
    for (const auto& node_stats : dev_stats.node_stats()) {
      for (int i = 0; i < node_stats.output_size(); ++i) {
        const auto& output = node_stats.output(i);
        LiveTensor* live = FindOrCreateLiveTensor(
            node_stats.node_name(), i, &live_tensors, &device_tensors);
        live->memory_used = output.tensor_description()
                                .allocation_description()
                                .allocated_bytes();
        live->allocation_time =
            Costs::MicroSeconds(node_stats.all_start_micros());
        live->deallocation_time = std::max<Costs::Duration>(
            live->deallocation_time,
            Costs::NanoSeconds(1) +
                Costs::MicroSeconds(node_stats.all_start_micros() +
                                    node_stats.op_end_rel_micros()));
      }
      auto it = node_map.find(node_stats.node_name());
      if (it == node_map.end()) {
        continue;
      }
      const NodeDef* node = it->second;
      std::unordered_set<int> swapped_inputs;
      if (is_gpu) {
        auto it = node->attr().find("_swap_to_host");
        if (it != node->attr().end()) {
          const AttrValue& val = it->second;
          for (int port_id : val.list().i()) {
            swapped_inputs.insert(port_id);
          }
        }
      }
      for (int i = 0; i < node->input_size(); ++i) {
        if (swapped_inputs.find(i) != swapped_inputs.end()) {
          continue;
        }
        const string& input = node->input(i);
        int position;
        string input_node = ParseNodeName(input, &position);
        if (position < 0) {
          continue;
        }
        LiveTensor* live = FindOrCreateLiveTensor(
            input_node, position, &live_tensors,
            &live_tensors_per_device[node_placement[input_node]]);
        live->deallocation_time = std::max<Costs::Duration>(
            live->deallocation_time,
            Costs::NanoSeconds(1) +
                Costs::MicroSeconds(node_stats.all_start_micros() +
                                    node_stats.op_end_rel_micros()));
      }
    }
  }
  for (const auto& live_per_device : live_tensors_per_device) {
    std::vector<Event> events;
    events.reserve(2 * live_per_device.second.size());
    for (const auto& live : live_per_device.second) {
      events.emplace_back(static_cast<int64_t>(live.allocation_time.count()),
                          true, &live);
      events.emplace_back(static_cast<int64_t>(live.deallocation_time.count()),
                          false, &live);
    }
    std::stable_sort(events.begin(), events.end());
    size_t peak = 0;
    std::unordered_set<const LiveTensor*> live_at_peak;
    size_t current = 0;
    std::unordered_set<const LiveTensor*> currently_live;
    int events_size = events.size();
    for (int i = 0; i < events_size; ++i) {
      const auto& event = events[i];
      if (event.allocated) {
        VLOG(1) << "At time " << event.timestamp << " allocated "
                << event.tensor->memory_used << " for tensor "
                << event.tensor->node << ":" << event.tensor->output_id;
        current += event.tensor->memory_used;
        currently_live.insert(event.tensor);
      } else {
        VLOG(1) << "At time " << event.timestamp << " deallocated "
                << event.tensor->memory_used << " for tensor "
                << event.tensor->node << ":" << event.tensor->output_id;
        current -= event.tensor->memory_used;
        currently_live.erase(event.tensor);
      }
      if (i + 1 == events_size || event.timestamp != events[i + 1].timestamp) {
        if (current > peak) {
          peak = current;
          live_at_peak = currently_live;
        }
      }
    }
    MemoryUsage& peak_mem_usage = peak_usage_[live_per_device.first];
    peak_mem_usage.used_memory = peak;
    peak_mem_usage.live_tensors.clear();
    peak_mem_usage.live_tensors.reserve(live_at_peak.size());
    for (const auto& live : live_at_peak) {
      peak_mem_usage.live_tensors.push_back(*live);
    }
  }
}
}  
}  