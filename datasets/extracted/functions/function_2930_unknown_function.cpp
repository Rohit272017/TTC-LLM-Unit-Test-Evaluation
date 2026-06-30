#include "xla/stream_executor/cuda/cuda_kernel.h"
#include <cstddef>
#include <cstdint>
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/launch_dim.h"
namespace stream_executor {
namespace gpu {
absl::StatusOr<int32_t> CudaKernel::GetMaxOccupiedBlocksPerCore(
    ThreadDim threads, size_t dynamic_shared_memory_bytes) const {
  int32_t threads_per_block = threads.x * threads.y * threads.z;
  VLOG(3) << "Get kernel block occupancy: " << name()
          << "; threads_per_block: " << threads_per_block
          << "; dynamic_shared_memory_bytes: " << dynamic_shared_memory_bytes;
  return GpuDriver::GetMaxOccupiedBlocksPerCore(
      gpu_executor_->gpu_context(), gpu_function_, threads_per_block,
      dynamic_shared_memory_bytes);
}
}  
}  