#include "tensorflow/core/tfrt/ifrt/ifrt_serving_core_selector.h"
#include <cstdint>
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/framework/serving_device_selector.h"
namespace tensorflow {
namespace ifrt_serving {
IfrtServingCoreSelector::IfrtServingCoreSelector(
    tsl::ServingDeviceSelector* device_selector, int num_cores)
    : device_selector_(device_selector), num_cores_(num_cores) {}
tsl::DeviceReservation IfrtServingCoreSelector::ReserveDevice(
    int64_t program_id) {
  absl::MutexLock lock(&mu_);
  int64_t run_count = run_counter_[program_id]++;
  if (run_count < num_cores_) {
    return tsl::DeviceReservation(run_count, nullptr);
  }
  return device_selector_->ReserveDevice(absl::StrCat(program_id));
}
}  
}  