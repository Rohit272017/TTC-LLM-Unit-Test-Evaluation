#ifndef THIRD_PARTY_CEL_CPP_COMMON_ALLOCATOR_H_
#define THIRD_PARTY_CEL_CPP_COMMON_ALLOCATOR_H_
#include <cstddef>
#include <new>
#include <type_traits>
#include "absl/base/attributes.h"
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/numeric/bits.h"
#include "common/arena.h"
#include "internal/new.h"
#include "google/protobuf/arena.h"
namespace cel {
enum class AllocatorKind {
  kArena = 1,
  kNewDelete = 2,
};
template <typename S>
void AbslStringify(S& sink, AllocatorKind kind) {
  switch (kind) {
    case AllocatorKind::kArena:
      sink.Append("ARENA");
      return;
    case AllocatorKind::kNewDelete:
      sink.Append("NEW_DELETE");
      return;
    default:
      sink.Append("ERROR");
      return;
  }
}
template <typename T = void>
class Allocator;
template <>
class Allocator<void> {
 public:
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;
  Allocator() = delete;
  Allocator(const Allocator&) = default;
  Allocator& operator=(const Allocator&) = delete;
  Allocator(std::nullptr_t) = delete;
  template <typename U, typename = std::enable_if_t<!std::is_void_v<U>>>
  constexpr Allocator(const Allocator<U>& other) noexcept
      : arena_(other.arena()) {}
  constexpr Allocator(absl::Nullable<google::protobuf::Arena*> arena) noexcept
      : arena_(arena) {}
  absl::Nullable<google::protobuf::Arena*> arena() const noexcept { return arena_; }
  ABSL_MUST_USE_RESULT void* allocate_bytes(
      size_type nbytes, size_type alignment = alignof(std::max_align_t)) {
    ABSL_DCHECK(absl::has_single_bit(alignment));
    if (nbytes == 0) {
      return nullptr;
    }
    if (auto* arena = this->arena(); arena != nullptr) {
      return arena->AllocateAligned(nbytes, alignment);
    }
    return internal::AlignedNew(nbytes,
                                static_cast<std::align_val_t>(alignment));
  }
  void deallocate_bytes(
      void* p, size_type nbytes,
      size_type alignment = alignof(std::max_align_t)) noexcept {
    ABSL_DCHECK((p == nullptr && nbytes == 0) || (p != nullptr && nbytes != 0));
    ABSL_DCHECK(absl::has_single_bit(alignment));
    if (arena() == nullptr) {
      internal::SizedAlignedDelete(p, nbytes,
                                   static_cast<std::align_val_t>(alignment));
    }
  }
  template <typename T>
  ABSL_MUST_USE_RESULT T* allocate_object(size_type n = 1) {
    return static_cast<T*>(allocate_bytes(sizeof(T) * n, alignof(T)));
  }
  template <typename T>
  void deallocate_object(T* p, size_type n = 1) {
    deallocate_bytes(p, sizeof(T) * n, alignof(T));
  }
  template <typename T, typename... Args>
  ABSL_MUST_USE_RESULT T* new_object(Args&&... args) {
    T* object = google::protobuf::Arena::Create<std::remove_const_t<T>>(
        arena(), std::forward<Args>(args)...);
    if constexpr (IsArenaConstructible<T>::value) {
      ABSL_DCHECK_EQ(object->GetArena(), arena());
    }
    return object;
  }
  template <typename T>
  void delete_object(T* p) noexcept {
    ABSL_DCHECK(p != nullptr);
    if constexpr (IsArenaConstructible<T>::value) {
      ABSL_DCHECK_EQ(p->GetArena(), arena());
    }
    if (arena() == nullptr) {
      delete p;
    }
  }
  void delete_object(std::nullptr_t) = delete;
 private:
  template <typename U>
  friend class Allocator;
  absl::Nullable<google::protobuf::Arena*> arena_;
};
template <typename T>
class Allocator : public Allocator<void> {
 public:
  static_assert(!std::is_const_v<T>, "T must not be const qualified");
  static_assert(!std::is_volatile_v<T>, "T must not be volatile qualified");
  static_assert(std::is_object_v<T>, "T must be an object type");
  using value_type = T;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using reference = value_type&;
  using const_reference = const value_type&;
  using Allocator<void>::Allocator;
  pointer allocate(size_type n, const void*  = nullptr) {
    return arena() != nullptr
               ? static_cast<pointer>(
                     arena()->AllocateAligned(n * sizeof(T), alignof(T)))
               : static_cast<pointer>(internal::AlignedNew(
                     n * sizeof(T), static_cast<std::align_val_t>(alignof(T))));
  }
  void deallocate(pointer p, size_type n) noexcept {
    if (arena() == nullptr) {
      internal::SizedAlignedDelete(p, n * sizeof(T),
                                   static_cast<std::align_val_t>(alignof(T)));
    }
  }
  template <typename U, typename... Args>
  void construct(U* p, Args&&... args) {
    static_assert(std::is_same_v<T*, U*>);
    static_assert(!IsArenaConstructible<T>::value);
    ::new (static_cast<void*>(p)) T(std::forward<Args>(args)...);
  }
  template <typename U>
  void destroy(U* p) noexcept {
    static_assert(std::is_same_v<T*, U*>);
    static_assert(!IsArenaConstructible<T>::value);
    p->~T();
  }
};
template <typename T, typename U>
inline bool operator==(Allocator<T> lhs, Allocator<U> rhs) noexcept {
  return lhs.arena() == rhs.arena();
}
template <typename T, typename U>
inline bool operator!=(Allocator<T> lhs, Allocator<U> rhs) noexcept {
  return !operator==(lhs, rhs);
}
inline Allocator<> NewDeleteAllocator() noexcept {
  return Allocator<>(static_cast<google::protobuf::Arena*>(nullptr));
}
template <typename T>
inline Allocator<T> NewDeleteAllocatorFor() noexcept {
  static_assert(!std::is_void_v<T>);
  return Allocator<T>(static_cast<google::protobuf::Arena*>(nullptr));
}
inline Allocator<> ArenaAllocator(
    absl::Nonnull<google::protobuf::Arena*> arena) noexcept {
  return Allocator<>(arena);
}
template <typename T>
inline Allocator<T> ArenaAllocatorFor(
    absl::Nonnull<google::protobuf::Arena*> arena) noexcept {
  static_assert(!std::is_void_v<T>);
  return Allocator<T>(arena);
}
}  
#endif  