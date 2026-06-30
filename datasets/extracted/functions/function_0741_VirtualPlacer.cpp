#include "tensorflow/core/grappler/costs/virtual_placer.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/device_name_utils.h"
namespace tensorflow {
namespace grappler {
VirtualPlacer::VirtualPlacer(
    const std::unordered_map<string, DeviceProperties>& devices)
    : devices_(devices),
      default_job_name_lowercase_("localhost") {
  lfqn_map_.reserve(devices_.size());
  for (const auto& kv : devices_) {
    const auto lfqn = to_lfqn_or_empty(kv.first);
    if (lfqn.empty()) {
      LOG(ERROR) << "VirtualPlacer couldn't parse device name from cluster: "
                 << kv.first;
    } else {
      lfqn_map_[lfqn] = kv.first;
    }
  }
  if (devices_.empty()) {
    default_device_name_ = "UNKNOWN";
    DeviceProperties& prop = devices_["UNKNOWN"];
    prop.set_type("UNKNOWN");
  } else if (devices_.size() == 1) {
    default_device_name_ = devices_.begin()->first;
  } else {
    std::map<int, string> cpu_devices;  
    std::map<int, string> gpu_devices;  
    for (const auto& kv : lfqn_map_) {
      const auto& lfqn = kv.first;
      const auto& cluster_device_name = kv.second;
      DeviceNameUtils::ParsedName parsed_name;
      bool parsed = DeviceNameUtils::ParseFullName(lfqn, &parsed_name);
      if (parsed) {
        const auto type = absl::AsciiStrToLower(parsed_name.type);
        if (type == "gpu") {
          gpu_devices[parsed_name.id] = cluster_device_name;
        } else if (type == "cpu") {
          cpu_devices[parsed_name.id] = cluster_device_name;
        }
      }
    }
    if (!gpu_devices.empty()) {
      default_device_name_ = gpu_devices.begin()->second;
    } else if (!cpu_devices.empty()) {
      default_device_name_ = cpu_devices.begin()->second;
    } else {
      default_device_name_ = devices_.begin()->first;  
    }
  }
  VLOG(3) << "default device name: " << default_device_name_;
  std::unordered_set<string> job_names_from_cluster;
  for (const auto& device : lfqn_map_) {
    const auto& lfqn = device.first;
    DeviceNameUtils::ParsedName parsed_name;
    bool parsed = DeviceNameUtils::ParseFullName(lfqn, &parsed_name);
    if (parsed && !parsed_name.job.empty()) {
      job_names_from_cluster.insert(parsed_name.job);
      if (job_names_from_cluster.size() > 1) {
        break;
      }
    }
  }
  if (job_names_from_cluster.size() == 1) {
    auto it = job_names_from_cluster.begin();
    default_job_name_lowercase_ = *it;
  }
  VLOG(3) << "default job name: " << default_job_name_lowercase_;
}
const DeviceProperties& VirtualPlacer::get_device(const NodeDef& node) const {
  string device = get_canonical_device_name(node);
  VLOG(3) << "node.name=" << node.name() << " node.device=" << node.device()
          << " is placed on: " << device;
  auto it = devices_.find(device);
  DCHECK(it != devices_.end());
  return it->second;
}
string VirtualPlacer::get_canonical_device_name(const NodeDef& node) const {
  if (node.device().empty()) {
    return default_device_name_;
  }
  const auto lfqn = to_lfqn_or_empty(node.device());
  if (lfqn.empty()) {
    return default_device_name_;
  }
  const auto it = lfqn_map_.find(lfqn);
  if (it != lfqn_map_.end()) {
    return it->second;
  }
  return default_device_name_;
}
string VirtualPlacer::to_lfqn_or_empty(const string& device_name) const {
  DeviceNameUtils::ParsedName parsed_name;
  const auto lowercase_name = absl::AsciiStrToLower(device_name);
  bool parsed = DeviceNameUtils::ParseFullName(lowercase_name, &parsed_name);
  if (!parsed) {
    parsed = DeviceNameUtils::ParseLocalName(lowercase_name, &parsed_name);
    parsed_name.job = "localhost";
  }
  if (!parsed) {
    if (lowercase_name == "gpu" || lowercase_name == "cpu") {
      parsed_name.job = "localhost";
      parsed_name.type = lowercase_name;
      parsed = true;
    }
  }
  if (!parsed) {
    return {};
  }
  if (parsed_name.job.empty()) {
    parsed_name.job = default_job_name_lowercase_;
  }
  parsed_name.type = absl::AsciiStrToLower(parsed_name.type);
  string lfqn = strings::StrCat(
      "/job:", parsed_name.job, "/replica:", parsed_name.replica,
      "/task:", parsed_name.task, "/device:", parsed_name.type, ":",
      parsed_name.id);
  return lfqn;
}
}  
}  