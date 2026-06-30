#include "tensorflow/compiler/tf2tensorrt/utils/trt_allocator.h"
#include "tensorflow/core/platform/logging.h"
#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#endif  
namespace tensorflow {
namespace tensorrt {
void* Align(uint64_t alignment, uint64_t size, void*& ptr, uint64_t& space) {
  QCHECK_GT(alignment, 0ul) << "alignment must be greater than 0.";
  QCHECK_EQ(0, alignment & (alignment - 1)) << "Alignment must be power of 2.";
  QCHECK_GT(size, 0ul) << "size must be greater than 0.";
  QCHECK(ptr) << "ptr must not be nullptr.";
  QCHECK_GT(space, 0ul) << "space must be greater than 0.";
  const uintptr_t ptr_val = reinterpret_cast<uintptr_t>(ptr);
  QCHECK_GE(ptr_val + space, ptr_val) << "Provided space overflows.";
  if (size > space) return nullptr;
  const uintptr_t aligned_ptr_val = ((ptr_val + alignment - 1) & -alignment);
  if (aligned_ptr_val > ptr_val + space - size) return nullptr;
  ptr = reinterpret_cast<void*>(aligned_ptr_val);
  const uintptr_t diff = aligned_ptr_val - ptr_val;
  space -= diff;
  return ptr;
}
}  
}  
#if GOOGLE_CUDA && GOOGLE_TENSORRT
namespace tensorflow {
namespace tensorrt {
void* TRTDeviceAllocator::allocate(uint64_t size, uint64_t alignment,
                                   uint32_t flags) noexcept {
  if (size == 0) return nullptr;
  alignment = 512;
  assert((alignment & (alignment - 1)) == 0);  
  uint64_t total_size = size + alignment;
  AllocationAttributes attributes;
  attributes.retry_on_failure = false;
  void* mem = allocator_->AllocateRaw(alignment, total_size, attributes);
  if (!mem) return nullptr;
  void* alloc_mem = mem;
  QCHECK(Align(alignment, size, mem, total_size));
  mutex_lock lock(mu_);
  if (mem != alloc_mem) {
    QCHECK(mem_map_.insert({mem, alloc_mem}).second);
  }
  VLOG(2) << "Allocated " << total_size << " bytes memory @" << alloc_mem
          << "; aligned to " << size << " bytes @" << mem << " with alignment "
          << alignment;
  return mem;
}
TRTDeviceAllocator::TRTDeviceAllocator(Allocator* allocator)
    : allocator_(allocator) {
  VLOG(1) << "Using " << allocator->Name() << " allocator from TensorFlow";
}
void TRTDeviceAllocator::free(void* memory) noexcept {
  mutex_lock lock(mu_);
  VLOG(2) << "Deallocating @ " << memory;
  if (memory) {
    auto alloc_mem = mem_map_.find(memory);
    if (alloc_mem != mem_map_.end()) {
      memory = alloc_mem->second;
      mem_map_.erase(alloc_mem->first);
    }
    allocator_->DeallocateRaw(memory);
  }
}
}  
}  
#endif  