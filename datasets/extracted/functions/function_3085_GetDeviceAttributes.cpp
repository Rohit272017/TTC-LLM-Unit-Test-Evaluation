#include "tensorflow/core/common_runtime/device_resolver_local.h"
#include "absl/status/status.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/platform/errors.h"
namespace tensorflow {
Status DeviceResolverLocal::GetDeviceAttributes(const string& device,
                                                DeviceAttributes* attributes) {
  Device* dev;
  Status s = dev_mgr_->LookupDevice(device, &dev);
  if (absl::IsInvalidArgument(s)) {
    return errors::NotFound(device, " not found");
  } else if (!s.ok()) {
    return s;
  }
  *attributes = dev->attributes();
  return absl::OkStatus();
}
Status DeviceResolverLocal::GetAllDeviceAttributes(
    const string& task, std::vector<DeviceAttributes>* attributes) {
  return errors::Internal(
      "GetTaskCached is not supposed to be called in local collectives");
}
Status DeviceResolverLocal::UpdateDeviceAttributes(
    const std::vector<DeviceAttributes>& attributes) {
  return errors::Internal(
      "UpdateDeviceAttributes shouldn't be called with local collectives");
}
}  