#include "common/memory.h"
#include <cstddef>
#include <cstring>
#include <new>
#include <ostream>
#include "absl/base/no_destructor.h"
#include "absl/log/absl_check.h"
#include "absl/numeric/bits.h"
#include "google/protobuf/arena.h"
namespace cel {
std::ostream& operator<<(std::ostream& out,
                         MemoryManagement memory_management) {
  switch (memory_management) {
    case MemoryManagement::kPooling:
      return out << "POOLING";
    case MemoryManagement::kReferenceCounting:
      return out << "REFERENCE_COUNTING";
  }
}
void* ReferenceCountingMemoryManager::Allocate(size_t size, size_t alignment) {
  ABSL_DCHECK(absl::has_single_bit(alignment))
      << "alignment must be a power of 2: " << alignment;
  if (size == 0) {
    return nullptr;
  }
  if (alignment <= __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
    return ::operator new(size);
  }
  return ::operator new(size, static_cast<std::align_val_t>(alignment));
}
bool ReferenceCountingMemoryManager::Deallocate(void* ptr, size_t size,
                                                size_t alignment) noexcept {
  ABSL_DCHECK(absl::has_single_bit(alignment))
      << "alignment must be a power of 2: " << alignment;
  if (ptr == nullptr) {
    ABSL_DCHECK_EQ(size, 0);
    return false;
  }
  ABSL_DCHECK_GT(size, 0);
  if (alignment <= __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
#if defined(__cpp_sized_deallocation) && __cpp_sized_deallocation >= 201309L
    ::operator delete(ptr, size);
#else
    ::operator delete(ptr);
#endif
  } else {
#if defined(__cpp_sized_deallocation) && __cpp_sized_deallocation >= 201309L
    ::operator delete(ptr, size, static_cast<std::align_val_t>(alignment));
#else
    ::operator delete(ptr, static_cast<std::align_val_t>(alignment));
#endif
  }
  return true;
}
MemoryManager MemoryManager::Unmanaged() {
  static absl::NoDestructor<google::protobuf::Arena> arena;
  return MemoryManager::Pooling(&*arena);
}
}  