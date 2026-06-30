#include "xla/service/gpu/kernels/topk_custom_kernel.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"
#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
#include "xla/service/gpu/kernels/topk_kernel_common.h"
#endif  
namespace xla::gpu::kernel::topk {
#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
namespace {
using KernelArgsPacking = se::MultiKernelLoaderSpec::KernelArgsPacking;
size_t EstimateOptimalNumThreads(size_t n, size_t k, size_t batch_size) {
  constexpr size_t kEstimatedThreadsPerBlock = 512;
  constexpr size_t kMaxKValue = 16;
  size_t simultaneous_threads_per_block =
      kEstimatedThreadsPerBlock * (kMaxKValue / k);
  size_t threads_per_block =
      std::min(simultaneous_threads_per_block, kTopKMaxThreadsPerBlock);
  size_t min_slice = absl::bit_floor(n / absl::bit_ceil(k));
  return std::min(threads_per_block, min_slice);
}
template <typename T>
absl::StatusOr<void*> GetKernel(int n, int k) {
  if (k <= 1) return GetTopKKernelForK<T, 1>(n);
  if (k <= 2) return GetTopKKernelForK<T, 2>(n);
  if (k <= 4) return GetTopKKernelForK<T, 4>(n);
  if (k <= 8) return GetTopKKernelForK<T, 8>(n);
  if (k <= 16) return GetTopKKernelForK<T, 16>(n);
  return absl::UnimplementedError(absl::StrCat("Unsupported K: ", k));
}
template <typename T>
KernelArgsPacking CreateTopKArgsPacking(size_t num_elements, size_t k) {
  using Packed = absl::StatusOr<std::unique_ptr<se::KernelArgsPackedArrayBase>>;
  return [=](const se::Kernel& kernel, const se::KernelArgs& args) -> Packed {
    auto* mem_args = se::Cast<se::KernelArgsDeviceMemoryArray>(&args);
    se::DeviceMemory<T> data(mem_args->device_memory_args()[0]);
    se::DeviceMemory<T> top_elements(mem_args->device_memory_args()[1]);
    se::DeviceMemory<uint32_t> top_indices(mem_args->device_memory_args()[2]);
    return se::PackKernelArgs(args.number_of_shared_bytes(), data, num_elements,
                              top_elements, top_indices, k);
  };
}
template <typename T>
absl::StatusOr<CustomKernel> GetTypedTopK(std::string name, size_t num_elements,
                                          size_t k, size_t batch_size) {
  constexpr size_t kMaxKVSize = sizeof(uint64_t);
  int shmem_size = absl::bit_ceil(k) * kMaxKVSize * GetTopKWaveFrontSize<T>();
  int num_threads = EstimateOptimalNumThreads(num_elements, k, batch_size);
  if (num_threads == 0) {
    return absl::FailedPreconditionError(
        "Invalid kernel parameters. This is likely a bug in the "
        "TopkSpecializer.");
  }
  auto packing = CreateTopKArgsPacking<T>(num_elements, k);
  se::MultiKernelLoaderSpec spec(5, std::move(packing));
  TF_ASSIGN_OR_RETURN(void* kernel_symbol, GetKernel<T>(num_elements, k));
  spec.AddInProcessSymbol(kernel_symbol, name);
  return CustomKernel(std::move(name), std::move(spec),
                      se::BlockDim(batch_size, 1, 1),
                      se::ThreadDim(num_threads, 1, 1), shmem_size);
}
}  
absl::StatusOr<CustomKernel> GetTopKKernel(std::string name,
                                           PrimitiveType dtype,
                                           size_t num_elements, size_t k,
                                           size_t batch_size) {
  switch (dtype) {
    case PrimitiveType::F32:
      return GetTypedTopK<float>(std::move(name), num_elements, k, batch_size);
    case PrimitiveType::BF16:
      return GetTypedTopK<bfloat16>(std::move(name), num_elements, k,
                                    batch_size);
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported GpuTopK data type: ", dtype));
  }
}
#else
absl::StatusOr<CustomKernel> GetTopKKernel(std::string name,
                                           PrimitiveType dtype,
                                           size_t num_elements, size_t k,
                                           size_t batch_size) {
  return absl::InternalError("XLA compiled without CUDA support");
}
#endif  
}  