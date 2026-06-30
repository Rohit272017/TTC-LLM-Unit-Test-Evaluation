#include "xla/stream_executor/sycl/sycl_platform.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include "absl/base/call_once.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
namespace stream_executor {
namespace gpu {
SyclPlatform::SyclPlatform() : name_("SYCL") {}
SyclPlatform::~SyclPlatform() {}
Platform::Id SyclPlatform::id() const { return sycl::kSyclPlatformId; }
int SyclPlatform::VisibleDeviceCount() const {
  static const int num_devices = [] {
    if (!GpuDriver::Init().ok()) return -1;
    return GpuDriver::GetDeviceCount();
  }();
  return num_devices;
}
const std::string& SyclPlatform::Name() const { return name_; }
absl::StatusOr<std::unique_ptr<DeviceDescription>>
SyclPlatform::DescriptionForDevice(int ordinal) const {
  return GpuExecutor::CreateDeviceDescription(ordinal);
}
absl::StatusOr<StreamExecutor*> SyclPlatform::ExecutorForDevice(int ordinal) {
  return executor_cache_.GetOrCreate(
      ordinal, [this, ordinal]() { return GetUncachedExecutor(ordinal); });
}
absl::StatusOr<std::unique_ptr<StreamExecutor>>
SyclPlatform::GetUncachedExecutor(int ordinal {
  auto executor = std::make_unique<GpuExecutor>(this, ordinal);
  TF_RETURN_IF_ERROR(executor->Init());
  return std::move(executor);
}
}  
static void InitializeSyclPlatform() {
  TF_CHECK_OK(
      PlatformManager::RegisterPlatform(std::make_unique<gpu::SyclPlatform>()));
}
}  
STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    sycl_platform, stream_executor::InitializeSyclPlatform());