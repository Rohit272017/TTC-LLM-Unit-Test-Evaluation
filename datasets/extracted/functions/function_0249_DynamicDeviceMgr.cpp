#include <atomic>
#include <iterator>
#include <memory>
#include <vector>
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/device_name_utils.h"
namespace tensorflow {
DynamicDeviceMgr::DynamicDeviceMgr() : cpu_device_(nullptr) {}
DynamicDeviceMgr::DynamicDeviceMgr(
    std::vector<std::unique_ptr<Device>>&& devices)
    : cpu_device_(nullptr) {
  Status status = AddDevices(std::move(devices));
  CHECK(status.ok());  
  mutex_lock l(devices_mu_);
  for (const auto& it : dynamic_devices_) {
    Device* d = it.first;
    if (d->device_type() == DEVICE_CPU && d->parsed_name().id == 0) {
      cpu_device_ = d;
      break;
    }
  }
}
DynamicDeviceMgr::DynamicDeviceMgr(std::unique_ptr<Device>&& device)
    : DynamicDeviceMgr([&device] {
        std::vector<std::unique_ptr<Device>> vector;
        vector.push_back(std::move(device));
        return vector;
      }()) {}
DynamicDeviceMgr::~DynamicDeviceMgr() {
  mutex_lock l(devices_mu_);
  for (const auto& it : dynamic_devices_) {
    it.first->ClearResourceMgr();
  }
}
void DynamicDeviceMgr::ListDeviceAttributes(
    std::vector<DeviceAttributes>* devices) const {
  tf_shared_lock l(devices_mu_);
  devices->reserve(dynamic_devices_.size());
  for (const auto& it : dynamic_devices_) {
    devices->emplace_back(it.first->attributes());
  }
}
std::vector<Device*> DynamicDeviceMgr::ListDevices() const {
  tf_shared_lock l(devices_mu_);
  std::vector<Device*> devices;
  devices.reserve(dynamic_devices_.size());
  for (const auto& it : dynamic_devices_) {
    devices.emplace_back(it.first);
  }
  return devices;
}
string DynamicDeviceMgr::DebugString() const {
  string out;
  tf_shared_lock l(devices_mu_);
  for (const auto& it : dynamic_devices_) {
    strings::StrAppend(&out, it.first->name(), "\n");
  }
  return out;
}
string DynamicDeviceMgr::DeviceMappingString() const {
  string out;
  tf_shared_lock l(devices_mu_);
  for (const auto& it : dynamic_devices_) {
    auto d = it.first;
    if (!d->attributes().physical_device_desc().empty()) {
      strings::StrAppend(&out, d->name(), " -> ",
                         d->attributes().physical_device_desc(), "\n");
    }
  }
  return out;
}
Status DynamicDeviceMgr::LookupDevice(StringPiece name, Device** device) const {
  tf_shared_lock l(devices_mu_);
  auto iter = device_map_.find(string(name));
  if (iter == device_map_.end()) {
    std::vector<StringPiece> device_names;
    device_names.reserve(device_map_.size());
    for (auto&& itr : device_map_) {
      device_names.push_back(itr.first);
    }
    VLOG(1) << "Unknown device: " << name
            << " all devices: " << absl::StrJoin(device_names, ", ");
    return errors::InvalidArgument(name, " unknown device.");
  }
  *device = iter->second;
  return absl::OkStatus();
}
bool DynamicDeviceMgr::ContainsDevice(int64_t device_incarnation) const {
  tf_shared_lock l(devices_mu_);
  return device_incarnation_set_.contains(device_incarnation);
}
void DynamicDeviceMgr::ClearContainers(
    absl::Span<const string> containers) const {
  Status s;
  tf_shared_lock l(devices_mu_);
  for (const auto& it : dynamic_devices_) {
    auto d = it.first;
    if (containers.empty()) {
      s.Update(d->resource_manager()->Cleanup(
          d->resource_manager()->default_container()));
    } else {
      for (const string& c : containers) {
        s.Update(d->resource_manager()->Cleanup(c));
      }
    }
    if (!s.ok()) {
      LOG(WARNING) << s;
    }
  }
}
int DynamicDeviceMgr::NumDeviceType(const string& type) const {
  tf_shared_lock l(devices_mu_);
  auto iter = device_type_counts_.find(type);
  if (iter != device_type_counts_.end()) return iter->second;
  return 0;
}
int DynamicDeviceMgr::NumDevices() const {
  tf_shared_lock l(devices_mu_);
  return dynamic_devices_.size();
}
Status DynamicDeviceMgr::AddDevices(
    std::vector<std::unique_ptr<Device>> devices) {
  mutex_lock l(devices_mu_);
  for (auto& d : devices) {
    if (device_map_.find(d->name()) != device_map_.end()) {
      return errors::InvalidArgument(
          "Trying to add device ", d->name(),
          " to manager but its name conflicts with an existing device.");
    }
    for (const string& name :
         DeviceNameUtils::GetNamesForDeviceMappings(d->parsed_name())) {
      device_map_[name] = d.get();
    }
    for (const string& name :
         DeviceNameUtils::GetLocalNamesForDeviceMappings(d->parsed_name())) {
      device_map_[name] = d.get();
    }
    device_type_counts_[d->device_type()]++;
    device_incarnation_set_.insert(d->attributes().incarnation());
    dynamic_devices_.emplace(d.get(), std::move(d));
  }
  return absl::OkStatus();
}
Status DynamicDeviceMgr::RemoveDevices(const std::vector<Device*>& devices) {
  mutex_lock l(devices_mu_);
  for (const auto& d : devices) {
    if (d == cpu_device_) {
      TF_RETURN_IF_ERROR(
          errors::InvalidArgument("Can not remove HostCPU device ", d->name()));
    }
    const auto it = dynamic_devices_.find(d);
    if (it == dynamic_devices_.end()) {
      return errors::InvalidArgument("Unknown device ", d->name());
    }
  }
  for (const auto& d : devices) {
    for (const string& name :
         DeviceNameUtils::GetNamesForDeviceMappings(d->parsed_name())) {
      device_map_.erase(name);
    }
    for (const string& name :
         DeviceNameUtils::GetLocalNamesForDeviceMappings(d->parsed_name())) {
      device_map_.erase(name);
    }
    device_type_counts_[d->device_type()]--;
    device_incarnation_set_.erase(d->attributes().incarnation());
    auto it = dynamic_devices_.find(d);
    if (it == dynamic_devices_.end()) {
      return errors::InvalidArgument("Unknown device ", d->name());
    }
    CHECK(it != dynamic_devices_.end());  
    stale_devices_.add(std::move(it->second));
    dynamic_devices_.erase(it);
  }
  return absl::OkStatus();
}
Status DynamicDeviceMgr::RemoveDevicesByName(
    const std::vector<string>& device_names) {
  std::vector<Device*> devices_to_remove;
  for (const string& name : device_names) {
    Device* device;
    TF_RETURN_IF_ERROR(LookupDevice(name, &device));
    devices_to_remove.emplace_back(device);
  }
  return RemoveDevices(devices_to_remove);
}
Device* DynamicDeviceMgr::HostCPU() const {
  Device* device = cpu_device_.load(std::memory_order_relaxed);
  if (device != nullptr) return device;
  mutex_lock l(devices_mu_);
  for (const auto& it : dynamic_devices_) {
    Device* d = it.first;
    if (d->device_type() == DEVICE_CPU && d->parsed_name().id == 0) {
      cpu_device_ = d;
      break;
    }
  }
  return cpu_device_.load(std::memory_order_relaxed);
}
}  