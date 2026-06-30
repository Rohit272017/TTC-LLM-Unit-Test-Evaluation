#include "xla/cpu_function_runtime.h"
#include "absl/base/dynamic_annotations.h"
namespace xla {
namespace {
void* aligned_malloc(size_t size, int minimum_alignment) {
#if defined(__ANDROID__) || defined(OS_ANDROID) || defined(OS_CYGWIN)
  return memalign(minimum_alignment, size);
#elif defined(_WIN32)
  return _aligned_malloc(size, minimum_alignment);
#else  
  void* ptr = nullptr;
  const int required_alignment = sizeof(void*);
  if (minimum_alignment < required_alignment) return malloc(size);
  if (posix_memalign(&ptr, minimum_alignment, size) != 0)
    return nullptr;
  else
    return ptr;
#endif
}
void aligned_free(void* aligned_memory) {
#if defined(_WIN32)
  _aligned_free(aligned_memory);
#else
  free(aligned_memory);
#endif
}
size_t align_to(size_t n, size_t align) {
  return (((n - 1) / align) + 1) * align;
}
}  
namespace cpu_function_runtime {
size_t AlignedBufferBytes(const BufferInfo* buffer_infos, size_t n,
                          bool allocate_entry_params) {
  size_t total = 0;
  for (size_t i = 0; i < n; ++i) {
    bool should_allocate =
        buffer_infos[i].is_temp_buffer() ||
        (buffer_infos[i].is_entry_parameter() && allocate_entry_params);
    if (should_allocate) {
      total += align_to(buffer_infos[i].size(), Align());
    }
  }
  return total;
}
void* MallocContiguousBuffers(const BufferInfo* buffer_infos, size_t n,
                              bool allocate_entry_params, void** bufs,
                              bool annotate_initialized) {
  const size_t total =
      AlignedBufferBytes(buffer_infos, n, allocate_entry_params);
  void* contiguous = nullptr;
  if (total > 0) {
    contiguous = aligned_malloc(total, Align());
    if (annotate_initialized) {
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(contiguous, total);
    }
  }
  uintptr_t pos = reinterpret_cast<uintptr_t>(contiguous);
  for (size_t i = 0; i < n; ++i) {
    bool should_allocate =
        buffer_infos[i].is_temp_buffer() ||
        (buffer_infos[i].is_entry_parameter() && allocate_entry_params);
    if (should_allocate) {
      bufs[i] = reinterpret_cast<void*>(pos);
      pos += align_to(buffer_infos[i].size(), Align());
    } else {
      bufs[i] = nullptr;
    }
  }
  return contiguous;
}
void FreeContiguous(void* contiguous) {
  if (contiguous != nullptr) {
    aligned_free(contiguous);
  }
}
}  
}  