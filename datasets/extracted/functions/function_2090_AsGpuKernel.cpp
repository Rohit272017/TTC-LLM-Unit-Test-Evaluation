#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_KERNEL_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_KERNEL_H_
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/kernel.h"
namespace stream_executor::gpu {
class GpuKernel : public Kernel {
 public:
  virtual GpuFunctionHandle gpu_function() const = 0;
};
inline const GpuKernel* AsGpuKernel(const Kernel* kernel) {
  return static_cast<const GpuKernel*>(kernel);
}
inline GpuKernel* AsGpuKernel(Kernel* kernel) {
  return static_cast<GpuKernel*>(kernel);
}
}  
#endif  