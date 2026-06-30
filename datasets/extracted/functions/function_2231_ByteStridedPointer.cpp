#ifndef TENSORSTORE_UTIL_BYTE_STRIDED_POINTER_H_
#define TENSORSTORE_UTIL_BYTE_STRIDED_POINTER_H_
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/util/element_traits.h"
namespace tensorstore {
template <typename T>
class ByteStridedPointer {
 public:
  using element_type = T;
  using difference_type = std::ptrdiff_t;
  constexpr static size_t alignment =
      alignof(std::conditional_t<std::is_void_v<T>, char, T>);
  ByteStridedPointer() = default;
  template <
      typename U,
      std::enable_if_t<IsElementTypeImplicitlyConvertible<U, T>>* = nullptr>
  ByteStridedPointer(U* value)
      : value_(reinterpret_cast<std::uintptr_t>(value)) {
    assert(value_ % alignment == 0);
  }
  template <
      typename U,
      std::enable_if_t<IsElementTypeOnlyExplicitlyConvertible<U, T>>* = nullptr>
  explicit ByteStridedPointer(U* value)
      : value_(reinterpret_cast<std::uintptr_t>(value)) {
    assert(value_ % alignment == 0);
  }
  template <
      typename U,
      std::enable_if_t<IsElementTypeImplicitlyConvertible<U, T>>* = nullptr>
  ByteStridedPointer(ByteStridedPointer<U> value)
      : value_(reinterpret_cast<std::uintptr_t>(value.get())) {
    assert(value_ % alignment == 0);
  }
  template <
      typename U,
      std::enable_if_t<IsElementTypeOnlyExplicitlyConvertible<U, T>>* = nullptr>
  explicit ByteStridedPointer(ByteStridedPointer<U> value)
      : value_(reinterpret_cast<std::uintptr_t>(value.get())) {
    assert(value_ % alignment == 0);
  }
  T* get() const {
    assert(value_ % alignment == 0);
    return reinterpret_cast<T*>(value_);
  }
  T* operator->() const { return get(); }
  template <typename U = T>
  U& operator*() const {
    return *static_cast<U*>(get());
  }
  operator T*() const { return get(); }
  template <
      typename U,
      std::enable_if_t<IsElementTypeOnlyExplicitlyConvertible<T, U>>* = nullptr>
  explicit operator U*() const {
    return static_cast<U*>(get());
  }
  template <typename Integer>
  std::enable_if_t<std::is_integral_v<Integer>, ByteStridedPointer&> operator+=(
      Integer byte_offset) {
    value_ = internal::wrap_on_overflow::Add(
        value_, static_cast<std::uintptr_t>(byte_offset));
    assert(value_ % alignment == 0);
    return *this;
  }
  template <typename Integer>
  std::enable_if_t<std::is_integral_v<Integer>, ByteStridedPointer&> operator-=(
      Integer byte_offset) {
    value_ = internal::wrap_on_overflow::Subtract(
        value_, static_cast<std::uintptr_t>(byte_offset));
    assert(value_ % alignment == 0);
    return *this;
  }
  template <typename Integer>
  std::enable_if_t<std::is_integral_v<Integer>, T>& operator[](
      Integer byte_offset) const {
    ByteStridedPointer x = *this;
    x += byte_offset;
    assert(x.value_ % alignment == 0);
    return *x;
  }
  template <typename U>
  friend std::ptrdiff_t operator-(ByteStridedPointer<T> a,
                                  ByteStridedPointer<U> b) {
    return reinterpret_cast<const char*>(a.get()) -
           reinterpret_cast<const char*>(b.get());
  }
  template <typename Integer>
  friend std::enable_if_t<std::is_integral_v<Integer>, ByteStridedPointer<T>>
  operator+(ByteStridedPointer<T> ptr, Integer byte_offset) {
    ptr += static_cast<std::uintptr_t>(byte_offset);
    return ptr;
  }
  template <typename Integer>
  friend inline std::enable_if_t<std::is_integral_v<Integer>,
                                 ByteStridedPointer<T>>
  operator+(Integer byte_offset, ByteStridedPointer<T> ptr) {
    ptr += static_cast<std::uintptr_t>(byte_offset);
    return ptr;
  }
  template <typename Integer>
  friend inline std::enable_if_t<std::is_integral_v<Integer>,
                                 ByteStridedPointer<T>>
  operator-(ByteStridedPointer<T> ptr, Integer byte_offset) {
    ptr -= static_cast<std::uintptr_t>(byte_offset);
    return ptr;
  }
 private:
  std::uintptr_t value_;
};
}  
#endif  