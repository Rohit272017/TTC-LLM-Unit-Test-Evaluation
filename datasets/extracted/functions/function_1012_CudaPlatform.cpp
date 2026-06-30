#include "xla/stream_executor/cuda/cuda_platform.h"
#include <memory>
#include <string>
#include <utility>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/cuda/cuda_executor.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/platform_manager.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
namespace stream_executor {
namespace gpu {
CudaPlatform::CudaPlatform() : name_("CUDA") {}
Platform::Id CudaPlatform::id() const { return cuda::kCudaPlatformId; }
int CudaPlatform::VisibleDeviceCount() const {
  static const int num_devices = [] {
    if (!GpuDriver::Init().ok()) return -1;
    return GpuDriver::GetDeviceCount();
  }();
  return num_devices;
}
const std::string& CudaPlatform::Name() const { return name_; }
absl::StatusOr<std::unique_ptr<DeviceDescription>>
CudaPlatform::DescriptionForDevice(int ordinal) const {
  return CudaExecutor::CreateDeviceDescription(ordinal);
}
absl::StatusOr<StreamExecutor*> CudaPlatform::ExecutorForDevice(int ordinal) {
  return executor_cache_.GetOrCreate(
      ordinal, [this, ordinal]() { return GetUncachedExecutor(ordinal); });
}
absl::StatusOr<StreamExecutor*> CudaPlatform::FindExisting(int ordinal) {
  return executor_cache_.Get(ordinal);
}
absl::StatusOr<std::unique_ptr<StreamExecutor>>
CudaPlatform::GetUncachedExecutor(int ordinal) {
  auto executor = std::make_unique<CudaExecutor>(this, ordinal);
  TF_RETURN_IF_ERROR(executor->Init());
  return std::move(executor);
}
}  
static void InitializeCudaPlatform() {
  TF_CHECK_OK(
      PlatformManager::RegisterPlatform(std::make_unique<gpu::CudaPlatform>()));
}
}  
STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    cuda_platform, stream_executor::InitializeCudaPlatform());