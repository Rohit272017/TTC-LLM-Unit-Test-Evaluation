#define EIGEN_USE_THREADS
#include "tensorflow/core/framework/device_base.h"
#include <algorithm>
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/notification.h"
#include "unsupported/Eigen/CXX11/Tensor"  
#include "tensorflow/core/util/work_sharder.h"
namespace tensorflow {
DeviceBase::~DeviceBase() {
  for (auto& temp : eigen_cpu_devices_) {
    delete temp;
  }
  eigen_cpu_devices_.clear();
}
Status DeviceContext::CopyDeviceTensorToCPUSync(const Tensor* device_tensor,
                                                StringPiece tensor_name,
                                                Device* device,
                                                Tensor* cpu_tensor) {
  absl::Notification n;
  Status status;
  CopyDeviceTensorToCPU(device_tensor, tensor_name, device, cpu_tensor,
                        [&](const Status& s) {
                          status = s;
                          n.Notify();
                        });
  n.WaitForNotification();
  return status;
}
Status DeviceContext::CopyCPUTensorToDeviceSync(const Tensor* cpu_tensor,
                                                Device* device,
                                                Tensor* device_tensor) const {
  absl::Notification n;
  Status status;
  CopyCPUTensorToDevice(cpu_tensor, device, device_tensor,
                        [&](const Status& s) {
                          status = s;
                          n.Notify();
                        });
  n.WaitForNotification();
  return status;
}
const DeviceAttributes& DeviceBase::attributes() const {
  LOG(FATAL) << "DeviceBase does not implement attributes()";  
  std::abort();
}
const string& DeviceBase::name() const {
  LOG(FATAL) << "DeviceBase does not implement name()";  
  std::abort();
}
const DeviceNameUtils::ParsedName& DeviceBase::parsed_name() const {
  LOG(FATAL) << "DeviceBase does not implement parsed_name()";  
  std::abort();
}
const std::string& DeviceBase::device_type() const {
  LOG(FATAL) << "DeviceBase does not implement device_type()";  
  std::abort();
}
void DeviceBase::set_eigen_cpu_device(Eigen::ThreadPoolDevice* d) {
  for (int i = 1; i <= d->numThreads(); ++i) {
    eigen_cpu_devices_.push_back(new Eigen::ThreadPoolDevice(
        d->getPool(), i , d->allocator()));
  }
}
const Eigen::ThreadPoolDevice* DeviceBase::eigen_cpu_device() {
  const int parallelism = std::max<int>(
      1,
      std::min<int>(GetPerThreadMaxParallelism(), eigen_cpu_devices_.size()));
  return eigen_cpu_devices_[parallelism - 1];
}
namespace {
absl::flat_hash_set<std::string>* GetSymbolicDeviceList() {
  static absl::flat_hash_set<std::string>* symbolic_device_list =
      new absl::flat_hash_set<std::string>();
  return symbolic_device_list;
}
}  
void AddSymbolicExecutionDevice(const absl::string_view device_name) {
  GetSymbolicDeviceList()->insert(std::string(device_name));
}
bool IsSymbolicExecutionDevice(const absl::string_view device_name) {
  absl::flat_hash_set<std::string>* symbolic_devices = GetSymbolicDeviceList();
  if (symbolic_devices->contains(device_name)) {
    return true;
  } else {
    return false;
  }
}
}  