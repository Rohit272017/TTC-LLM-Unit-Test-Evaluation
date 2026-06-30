#include "tensorflow/core/distributed_runtime/device_resolver_distributed.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/platform/errors.h"
namespace tensorflow {
DeviceResolverDistributed::DeviceResolverDistributed(const DeviceMgr* dev_mgr) {
  mutex_lock l(mu_);
  for (Device* device : dev_mgr->ListDevices()) {
    attr_table_[device->name()] = device->attributes();
  }
}
Status DeviceResolverDistributed::GetDeviceAttributes(
    const string& device, DeviceAttributes* attributes) {
  mutex_lock l(mu_);
  auto it = attr_table_.find(device);
  if (it == attr_table_.end()) {
    return errors::NotFound(device, " not found");
  }
  *attributes = it->second;
  return absl::OkStatus();
}
Status DeviceResolverDistributed::GetAllDeviceAttributes(
    const string& task, std::vector<DeviceAttributes>* attributes) {
  mutex_lock l(mu_);
  attributes->clear();
  for (const auto& it : attr_table_) {
    const string& device_name = it.first;
    if (DeviceNameUtils::IsSameAddressSpace(task, device_name)) {
      attributes->push_back(it.second);
    }
  }
  if (attributes->empty()) {
    return errors::NotFound(task, " not found in the cache");
  }
  return absl::OkStatus();
}
Status DeviceResolverDistributed::UpdateDeviceAttributes(
    const std::vector<DeviceAttributes>& attributes) {
  mutex_lock l(mu_);
  for (const DeviceAttributes& attr : attributes) {
    auto item = attr_table_.insert({attr.name(), attr});
    auto it = item.first;
    bool success = item.second;
    if (!success && it->second.incarnation() != attr.incarnation()) {
      return errors::FailedPrecondition(
          attr.name(),
          "exists in cache with a different incarnation. "
          "This usually means the remote worker has restarted");
    }
  }
  return absl::OkStatus();
}
}  