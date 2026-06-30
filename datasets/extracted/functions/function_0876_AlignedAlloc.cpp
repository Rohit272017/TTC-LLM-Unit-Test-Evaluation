#ifndef AROLLA_UTIL_MEMORY_H_
#define AROLLA_UTIL_MEMORY_H_
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include "absl/log/check.h"
namespace arolla {
struct FreeDeleter {
  void operator()(const void* ptr) const { std::free(const_cast<void*>(ptr)); }
};
using MallocPtr = std::unique_ptr<void, FreeDeleter>;
struct Alignment {
  size_t value;
};
inline MallocPtr AlignedAlloc(Alignment alignment, size_t size) {
  DCHECK(alignment.value > 0 && !(alignment.value & (alignment.value - 1)));
  if (size == 0) {
    size = 1;  
  }
  if (alignment.value <= sizeof(void*)) {
    return MallocPtr(std::malloc(size));
  }
  void* result = nullptr;
  if (posix_memalign(&result, alignment.value, size)) {
    result = nullptr;
  }
  DCHECK(result) << "posix_memalign failed.";
  return MallocPtr(result);
}
inline bool IsAlignedPtr(size_t alignment, const void* ptr) {
  DCHECK(alignment > 0 && !(alignment & (alignment - 1)));
  return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}
inline bool IsAlignedPtr(Alignment alignment, const void* ptr) {
  return IsAlignedPtr(alignment.value, ptr);
}
}  
#endif  