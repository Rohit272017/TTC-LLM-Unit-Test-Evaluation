#ifndef TENSORSTORE_INTERNAL_INTRUSIVE_PTR_H_
#define TENSORSTORE_INTERNAL_INTRUSIVE_PTR_H_
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include "tensorstore/internal/memory.h"
#include "tensorstore/internal/type_traits.h"
namespace tensorstore {
namespace internal {
template <typename Derived>
class AtomicReferenceCount {
 public:
  AtomicReferenceCount() = default;
  AtomicReferenceCount(size_t initial_ref_count)
      : ref_count_(initial_ref_count) {}
  AtomicReferenceCount(const AtomicReferenceCount&) noexcept {}
  AtomicReferenceCount& operator=(const AtomicReferenceCount&) noexcept {
    return *this;
  }
  uint32_t use_count() const noexcept {
    return ref_count_.load(std::memory_order_acquire);
  }
  template <typename D>
  friend bool IncrementReferenceCountIfNonZero(
      const AtomicReferenceCount<D>& base);
  template <typename D>
  friend bool DecrementReferenceCount(const AtomicReferenceCount<D>& base);
  friend void intrusive_ptr_increment(const AtomicReferenceCount* p) noexcept {
    p->ref_count_.fetch_add(1, std::memory_order_acq_rel);
  }
  friend void intrusive_ptr_decrement(const AtomicReferenceCount* p) noexcept {
    if (DecrementReferenceCount(*p)) {
      delete static_cast<const Derived*>(p);
    }
  }
 private:
  mutable std::atomic<uint32_t> ref_count_{0};
};
template <typename Derived>
inline bool IncrementReferenceCountIfNonZero(
    const AtomicReferenceCount<Derived>& base) {
  uint32_t count = base.ref_count_.load(std::memory_order_relaxed);
  do {
    if (count == 0) return false;
  } while (!base.ref_count_.compare_exchange_weak(count, count + 1,
                                                  std::memory_order_acq_rel));
  return true;
}
template <typename Derived>
inline bool DecrementReferenceCount(const AtomicReferenceCount<Derived>& base) {
  return base.ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1;
}
template <typename T>
bool DecrementReferenceCountIfGreaterThanOne(std::atomic<T>& reference_count) {
  auto count = reference_count.load(std::memory_order_relaxed);
  while (true) {
    if (count == 1) return false;
    if (reference_count.compare_exchange_weak(count, count - 1,
                                              std::memory_order_acq_rel)) {
      return true;
    }
  }
}
struct DefaultIntrusivePtrTraits {
  template <typename U>
  using pointer = U*;
  template <typename Pointer>
  static void increment(Pointer p) noexcept {
    intrusive_ptr_increment(p);
  }
  template <typename Pointer>
  static void decrement(Pointer p) noexcept {
    intrusive_ptr_decrement(p);
  }
};
struct acquire_object_ref_t {
  explicit constexpr acquire_object_ref_t() = default;
};
struct adopt_object_ref_t {
  explicit constexpr adopt_object_ref_t() = default;
};
constexpr acquire_object_ref_t acquire_object_ref{};
constexpr adopt_object_ref_t adopt_object_ref{};
template <typename T, typename R>
class IntrusivePtr;
template <typename T>
struct IsIntrusivePtr : public std::false_type {};
template <typename T, typename R>
struct IsIntrusivePtr<IntrusivePtr<T, R>> : public std::true_type {};
template <typename T, typename R = DefaultIntrusivePtrTraits>
class IntrusivePtr {
 public:
  using element_type = T;
  using traits_type = R;
  using pointer = typename R::template pointer<T>;
  ~IntrusivePtr() {
    if (pointer p = get()) R::decrement(p);
  }
  constexpr IntrusivePtr() noexcept : ptr_(nullptr) {}
  constexpr IntrusivePtr(std::nullptr_t) noexcept : ptr_(nullptr) {}
  explicit IntrusivePtr(pointer p) noexcept : ptr_(p) {
    if (ptr_) R::increment(ptr_);
  }
  explicit IntrusivePtr(pointer p, acquire_object_ref_t) noexcept : ptr_(p) {
    if (ptr_) R::increment(ptr_);
  }
  constexpr explicit IntrusivePtr(pointer p, adopt_object_ref_t) noexcept
      : ptr_(p) {}
  IntrusivePtr(const IntrusivePtr& rhs) noexcept
      : IntrusivePtr(rhs.get(), acquire_object_ref) {}
  IntrusivePtr& operator=(const IntrusivePtr& rhs) noexcept {
    IntrusivePtr(rhs).swap(*this);
    return *this;
  }
  template <typename U,
            std::enable_if_t<std::is_convertible_v<
                typename R::template pointer<U>, pointer>>* = nullptr>
  IntrusivePtr(const IntrusivePtr<U, R>& rhs) noexcept
      : IntrusivePtr(rhs.get(), acquire_object_ref) {}
  template <typename U, typename = std::enable_if_t<std::is_convertible_v<
                            typename R::template pointer<U>, pointer>>>
  IntrusivePtr& operator=(const IntrusivePtr<U, R>& rhs) noexcept {
    IntrusivePtr(rhs).swap(*this);
    return *this;
  }
  constexpr IntrusivePtr(IntrusivePtr&& rhs) noexcept
      : IntrusivePtr(rhs.release(), adopt_object_ref) {}
  constexpr IntrusivePtr& operator=(IntrusivePtr&& rhs) noexcept {
    IntrusivePtr(std::move(rhs)).swap(*this);
    return *this;
  }
  template <typename U,
            std::enable_if_t<std::is_convertible_v<
                typename R::template pointer<U>, pointer>>* = nullptr>
  constexpr IntrusivePtr(IntrusivePtr<U, R>&& rhs) noexcept
      : IntrusivePtr(rhs.release(), adopt_object_ref) {}
  template <typename U, typename = std::enable_if_t<std::is_convertible_v<
                            typename R::template pointer<U>, pointer>>>
  constexpr IntrusivePtr& operator=(IntrusivePtr<U, R>&& rhs) noexcept {
    IntrusivePtr(std::move(rhs)).swap(*this);
    return *this;
  }
  void reset() noexcept { IntrusivePtr().swap(*this); }
  void reset(std::nullptr_t) noexcept { IntrusivePtr().swap(*this); }
  void reset(pointer rhs) { IntrusivePtr(rhs, acquire_object_ref).swap(*this); }
  void reset(pointer rhs, acquire_object_ref_t) {
    IntrusivePtr(rhs, acquire_object_ref).swap(*this);
  }
  void reset(pointer rhs, adopt_object_ref_t) {
    IntrusivePtr(rhs, adopt_object_ref).swap(*this);
  }
  constexpr explicit operator bool() const { return static_cast<bool>(ptr_); }
  constexpr pointer get() const noexcept { return ptr_; }
  constexpr pointer operator->() const {
    pointer ptr = get();
    assert(static_cast<bool>(ptr));
    return ptr;
  }
  constexpr element_type& operator*() const {
    pointer ptr = get();
    assert(static_cast<bool>(ptr));
    return *ptr;
  }
  constexpr pointer release() noexcept {
    pointer ptr = get();
    ptr_ = pointer{};
    return ptr;
  }
  void swap(IntrusivePtr& rhs) noexcept {
    std::swap(ptr_, rhs.ptr_);
  }
  template <typename H>
  friend H AbslHashValue(H h, const IntrusivePtr& x) {
    return H::combine(std::move(h), x.get());
  }
  friend bool operator==(const IntrusivePtr& p, std::nullptr_t) { return !p; }
  friend bool operator!=(const IntrusivePtr& p, std::nullptr_t) {
    return static_cast<bool>(p);
  }
  friend bool operator==(std::nullptr_t, const IntrusivePtr& p) { return !p; }
  friend bool operator!=(std::nullptr_t, const IntrusivePtr& p) {
    return static_cast<bool>(p);
  }
 private:
  pointer ptr_;
};
template <typename T, typename R>
inline T* to_address(const IntrusivePtr<T, R>& p) {
  return to_address(p.get());
}
template <typename T, typename U, typename R>
inline std::enable_if_t<IsEqualityComparable<typename R::template pointer<T>,
                                             typename R::template pointer<U>>,
                        bool>
operator==(const IntrusivePtr<T, R>& x, const IntrusivePtr<U, R>& y) {
  return x.get() == y.get();
}
template <typename T, typename U, typename R>
inline std::enable_if_t<IsEqualityComparable<typename R::template pointer<T>,
                                             typename R::template pointer<U>>,
                        bool>
operator!=(const IntrusivePtr<T, R>& x, const IntrusivePtr<U, R>& y) {
  return x.get() != y.get();
}
template <typename T, typename U, typename R>
inline IntrusivePtr<T, R> static_pointer_cast(IntrusivePtr<U, R> p) {
  return IntrusivePtr<T, R>(static_pointer_cast<T>(p.release()),
                            adopt_object_ref);
}
template <typename T, typename U, typename R>
inline IntrusivePtr<T, R> const_pointer_cast(IntrusivePtr<U, R> p) {
  return IntrusivePtr<T, R>(const_pointer_cast<T>(p.release()),
                            adopt_object_ref);
}
template <typename T, typename U, typename R>
inline IntrusivePtr<T, R> dynamic_pointer_cast(IntrusivePtr<U, R> p) {
  if (auto new_pointer = dynamic_pointer_cast<T>(p.get())) {
    p.release();
    return IntrusivePtr<T, R>(std::move(new_pointer), adopt_object_ref);
  } else {
    return IntrusivePtr<T, R>(std::move(new_pointer), adopt_object_ref);
  }
}
template <typename T, typename Traits>
std::shared_ptr<T> IntrusiveToShared(internal::IntrusivePtr<T, Traits> p) {
  auto* ptr = p.get();
  return std::shared_ptr<T>(
      std::make_shared<internal::IntrusivePtr<T, Traits>>(std::move(p)), ptr);
}
template <typename T, typename R = DefaultIntrusivePtrTraits, typename... Args>
inline IntrusivePtr<T, R> MakeIntrusivePtr(Args&&... args) {
  return IntrusivePtr<T, R>(new T(std::forward<Args>(args)...));
}
}  
}  
#endif  