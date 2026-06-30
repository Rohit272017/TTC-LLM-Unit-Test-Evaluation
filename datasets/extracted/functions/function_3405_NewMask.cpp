#include "tensorflow/core/common_runtime/gpu/gpu_debug_allocator.h"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/device_id.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
#define MASK_WORDS 2
#define MASK_BYTES (MASK_WORDS * sizeof(int64_t))
namespace tensorflow {
namespace {
int64_t* NewMask(int64_t word) {
  int64_t* m = new int64_t[MASK_WORDS];
  for (int i = 0; i < MASK_WORDS; ++i) {
    m[i] = word;
  }
  return m;
}
int64_t* before_mask = NewMask(0xabababababababab);
int64_t* after_mask = NewMask(0xcdcdcdcdcdcdcdcd);
bool CheckMask(se::StreamExecutor* exec, void* ptr, int64_t* mask) {
  se::DeviceMemory<int64_t> gpu_ptr{se::DeviceMemoryBase{ptr, MASK_BYTES}};
  int64_t tmp[MASK_WORDS];
  absl::Status result = exec->SynchronousMemcpyD2H(gpu_ptr, MASK_BYTES, tmp);
  if (!result.ok()) {
    LOG(FATAL) << "Could not copy debug mask, " << result;
  }
  bool ok = true;
  for (int i = 0; i < MASK_WORDS; ++i) {
    ok &= (mask[i] == tmp[i]);
    if (!ok) {
      LOG(ERROR) << "i=" << i
                 << " mask=" << reinterpret_cast<const void*>(mask[i])
                 << " field=" << reinterpret_cast<const void*>(tmp[i]);
    }
  }
  return ok;
}
void InitMask(se::StreamExecutor* exec, void* ptr, int64_t* mask) {
  se::DeviceMemory<int64_t> gpu_ptr{se::DeviceMemoryBase{ptr, MASK_BYTES}};
  absl::Status result = exec->SynchronousMemcpyH2D(mask, MASK_BYTES, &gpu_ptr);
  if (!result.ok()) {
    LOG(FATAL) << "Could not copy debug mask, " << result;
  }
}
}  
GPUDebugAllocator::GPUDebugAllocator(Allocator* allocator,
                                     tsl::PlatformDeviceId platform_device_id)
    : base_allocator_(allocator) {
  stream_exec_ = se::GPUMachineManager()
                     ->ExecutorForDevice(platform_device_id.value())
                     .value();
}
GPUDebugAllocator::~GPUDebugAllocator() { delete base_allocator_; }
void* GPUDebugAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  num_bytes += (2 * MASK_BYTES);
  void* allocated_ptr = base_allocator_->AllocateRaw(alignment, num_bytes);
  if (allocated_ptr == nullptr) return allocated_ptr;
  void* rv = static_cast<char*>(allocated_ptr) + MASK_BYTES;
  InitMask(stream_exec_, allocated_ptr, before_mask);
  size_t req_size = base_allocator_->RequestedSize(allocated_ptr);
  InitMask(stream_exec_,
           static_cast<char*>(allocated_ptr) + req_size - MASK_BYTES,
           after_mask);
  return rv;
}
void GPUDebugAllocator::DeallocateRaw(void* ptr) {
  if (ptr != nullptr) {
    CHECK(CheckHeader(ptr)) << "before_mask has been overwritten";
    CHECK(CheckFooter(ptr)) << "after_mask has been overwritten";
    ptr = static_cast<void*>(static_cast<char*>(ptr) - MASK_BYTES);
  }
  base_allocator_->DeallocateRaw(ptr);
}
bool GPUDebugAllocator::TracksAllocationSizes() const { return true; }
size_t GPUDebugAllocator::RequestedSize(const void* ptr) const {
  auto req_size = base_allocator_->RequestedSize(static_cast<const char*>(ptr) -
                                                 MASK_BYTES);
  return req_size - 2 * MASK_BYTES;
}
size_t GPUDebugAllocator::AllocatedSize(const void* ptr) const {
  return base_allocator_->AllocatedSize(static_cast<const char*>(ptr) -
                                        MASK_BYTES);
}
int64_t GPUDebugAllocator::AllocationId(const void* ptr) const {
  return base_allocator_->AllocationId(static_cast<const char*>(ptr) -
                                       MASK_BYTES);
}
std::optional<tsl::AllocatorStats> GPUDebugAllocator::GetStats() {
  return base_allocator_->GetStats();
}
bool GPUDebugAllocator::ClearStats() { return base_allocator_->ClearStats(); }
bool GPUDebugAllocator::CheckHeader(void* ptr) {
  return CheckMask(stream_exec_, static_cast<char*>(ptr) - MASK_BYTES,
                   before_mask);
}
bool GPUDebugAllocator::CheckFooter(void* ptr) {
  char* original_ptr = static_cast<char*>(ptr) - MASK_BYTES;
  size_t req_size = base_allocator_->RequestedSize(original_ptr);
  return CheckMask(stream_exec_, original_ptr + req_size - MASK_BYTES,
                   after_mask);
}
GPUNanResetAllocator::GPUNanResetAllocator(
    Allocator* allocator, tsl::PlatformDeviceId platform_device_id)
    : base_allocator_(allocator) {
  stream_exec_ = se::GPUMachineManager()
                     ->ExecutorForDevice(platform_device_id.value())
                     .value();
}
GPUNanResetAllocator::~GPUNanResetAllocator() { delete base_allocator_; }
void* GPUNanResetAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  void* allocated_ptr = base_allocator_->AllocateRaw(alignment, num_bytes);
  if (allocated_ptr == nullptr) return allocated_ptr;
  size_t req_size = base_allocator_->RequestedSize(allocated_ptr);
  std::vector<float> nans((req_size + sizeof(float) - 1) / sizeof(float),
                          std::nanf(""));
  se::DeviceMemory<float> nan_ptr{
      se::DeviceMemoryBase{static_cast<float*>(allocated_ptr), req_size}};
  absl::Status result =
      stream_exec_->SynchronousMemcpyH2D(&nans[0], req_size, &nan_ptr);
  if (!result.ok()) {
    LOG(ERROR) << "Could not initialize to NaNs, " << result;
  }
  return allocated_ptr;
}
void GPUNanResetAllocator::DeallocateRaw(void* ptr) {
  if (ptr != nullptr) {
    size_t req_size = base_allocator_->RequestedSize(ptr);
    std::vector<float> nans((req_size + sizeof(float) - 1) / sizeof(float),
                            std::nanf(""));
    se::DeviceMemory<float> nan_ptr{
        se::DeviceMemoryBase{static_cast<float*>(ptr), req_size}};
    absl::Status result =
        stream_exec_->SynchronousMemcpyH2D(&nans[0], req_size, &nan_ptr);
    if (!result.ok()) {
      LOG(ERROR) << "Could not initialize to NaNs, " << result;
    }
  }
  base_allocator_->DeallocateRaw(ptr);
}
size_t GPUNanResetAllocator::RequestedSize(const void* ptr) const {
  return base_allocator_->RequestedSize(ptr);
}
size_t GPUNanResetAllocator::AllocatedSize(const void* ptr) const {
  return base_allocator_->AllocatedSize(ptr);
}
std::optional<tsl::AllocatorStats> GPUNanResetAllocator::GetStats() {
  return base_allocator_->GetStats();
}
bool GPUNanResetAllocator::ClearStats() {
  return base_allocator_->ClearStats();
}
}  