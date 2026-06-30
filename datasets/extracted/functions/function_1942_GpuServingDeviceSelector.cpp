#include "tensorflow/core/common_runtime/gpu/gpu_serving_device_selector.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include "absl/base/attributes.h"
#include "absl/container/fixed_array.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "xla/tsl/framework/serving_device_selector.h"
#include "tensorflow/core/common_runtime/gpu/gpu_scheduling_metrics_storage.h"
namespace tensorflow {
namespace gpu {
constexpr int64_t kDefaultEstimateNs = 1;
ABSL_CONST_INIT int64_t (*NowNs)() = +[]() -> int64_t {
  return absl::GetCurrentTimeNanos();
};
using DeviceStates = GpuServingDeviceSelector::DeviceStates;
GpuServingDeviceSelector::GpuServingDeviceSelector(
    const int num_devices,
    std::unique_ptr<ServingDeviceSelector::Policy> device_selector_policy)
    : device_states_(num_devices),
      device_selector_policy_(std::move(device_selector_policy)),
      req_id_counter_(0) {}
tsl::DeviceReservation GpuServingDeviceSelector::ReserveDevice(
    absl::string_view program_fingerprint) {
  absl::MutexLock lock(&mu_);
  DeviceStates device_states;
  device_states.states = absl::Span<const DeviceState>(device_states_);
  auto [it, emplaced] =
      execution_info_.try_emplace(program_fingerprint, ExecutionInfo());
  const int device_index =
      device_selector_policy_->SelectDevice(program_fingerprint, device_states);
  ServingDeviceSelector::EnqueueHelper(
      device_states_.at(device_index), device_index, it->second,
      program_fingerprint, 0, req_id_counter_++,
      1, 0, NowNs());
  return tsl::DeviceReservation(device_index, this);
}
void GpuServingDeviceSelector::FreeDeviceReservation(
    const tsl::DeviceReservation& reservation) {
  Completed(reservation.device_index(), false);
}
void GpuServingDeviceSelector::Enqueue(int32_t index_on_host,
                                       absl::string_view fingerprint) {
  if (fingerprint.empty()) {
    LOG(ERROR) << "Empty fingerprint.";
    return;
  }
  absl::MutexLock lock(&mu_);
  auto [it, emplaced] =
      execution_info_.try_emplace(fingerprint, ExecutionInfo());
  DeviceState& device_state = device_states_.at(index_on_host);
  ServingDeviceSelector::EnqueueHelper(device_state, index_on_host, it->second,
                                       fingerprint,
                                       0, -1,
                                       1,
                                       0, NowNs());
  int64_t total_estimated_time_ns = TotalEstimatedTimeTillIdleNs();
  GpuSchedulingMetricsStorage::GetGlobalStorage().TotalGpuLoadNs().Set(
      total_estimated_time_ns);
}
void GpuServingDeviceSelector::Completed(int32_t index_on_host,
                                         bool had_error) {
  absl::MutexLock lock(&mu_);
  DeviceState& device_state = device_states_.at(index_on_host);
  ServingDeviceSelector::CompletedHelper(device_state, index_on_host, 0,
                                         min_exec_time_, had_error, NowNs());
  int64_t total_estimated_time_ns = TotalEstimatedTimeTillIdleNs();
  GpuSchedulingMetricsStorage::GetGlobalStorage().TotalGpuLoadNs().Set(
      total_estimated_time_ns);
}
int64_t GpuServingDeviceSelector::TotalEstimatedTimeTillIdleNs() {
  int64_t total_gpu_load_ns = 0;
  for (const auto& device_state : device_states_) {
    total_gpu_load_ns += ServingDeviceSelector::EstimateTimeTillIdleNs(
        device_state, 0, min_exec_time_.value_or(kDefaultEstimateNs), NowNs());
  }
  return total_gpu_load_ns;
}
 void GpuServingDeviceSelector::OverwriteNowNsFunctionForTest(
    int64_t (*now_ns)()) {
  NowNs = now_ns;
}
}  
}  