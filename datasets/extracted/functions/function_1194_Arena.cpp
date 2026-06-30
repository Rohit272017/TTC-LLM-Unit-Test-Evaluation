#ifndef TENSORSTORE_INTERNAL_ARENA_H_
#define TENSORSTORE_INTERNAL_ARENA_H_
#include <stddef.h>
#include <memory>
#include <new>
#include <utility>
#include "tensorstore/internal/exception_macros.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/util/span.h"
namespace tensorstore {
namespace internal {
class Arena {
 public:
  Arena() : remaining_bytes_(0) {}
  explicit Arena(tensorstore::span<unsigned char> initial_buffer)
      : initial_buffer_(initial_buffer),
        remaining_bytes_(initial_buffer.size()) {}
  template <typename T = unsigned char>
  T* allocate(size_t n, size_t alignment = alignof(T)) {
    size_t num_bytes;
    if (MulOverflow(n, sizeof(T), &num_bytes)) {
      TENSORSTORE_THROW_BAD_ALLOC;
    }
    void* ptr = static_cast<void*>(initial_buffer_.end() - remaining_bytes_);
    if (std::align(alignment, num_bytes, ptr, remaining_bytes_)) {
      remaining_bytes_ -= num_bytes;
    } else {
      ptr = ::operator new(num_bytes, std::align_val_t(alignment));
    }
    return static_cast<T*>(ptr);
  }
  template <typename T>
  void deallocate(T* p, size_t n, size_t alignment = alignof(T)) {
    if (static_cast<void*>(p) >= static_cast<void*>(initial_buffer_.data()) &&
        static_cast<void*>(p + n) <=
            static_cast<void*>(initial_buffer_.data() +
                               initial_buffer_.size())) {
      return;
    }
    ::operator delete(static_cast<void*>(p), n * sizeof(T),
                      std::align_val_t(alignment));
  }
 private:
  tensorstore::span<unsigned char> initial_buffer_;
  size_t remaining_bytes_;
};
template <typename T = unsigned char>
class ArenaAllocator {
 public:
  using value_type = T;
  using pointer = T*;
  using void_pointer = void*;
  using const_void_pointer = const void*;
  using reference = T&;
  using const_pointer = const T*;
  using const_reference = const T&;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  template <typename U>
  struct rebind {
    using other = ArenaAllocator<U>;
  };
  ArenaAllocator(Arena* arena) : arena_(arena) {}
  template <typename U>
  ArenaAllocator(ArenaAllocator<U> other) : arena_(other.arena()) {}
  T* allocate(size_t n) const { return arena_->allocate<T>(n); }
  void deallocate(T* p, size_t n) const { arena_->deallocate(p, n); }
  template <typename... Arg>
  void construct(T* p, Arg&&... arg) {
    new (p) T(std::forward<Arg>(arg)...);
  }
  void destroy(T* p) { p->~T(); }
  Arena* arena() const { return arena_; }
  friend bool operator==(ArenaAllocator a, ArenaAllocator b) {
    return a.arena_ == b.arena_;
  }
  friend bool operator!=(ArenaAllocator a, ArenaAllocator b) {
    return a.arena_ != b.arena_;
  }
  Arena* arena_;
};
}  
}  
#endif  