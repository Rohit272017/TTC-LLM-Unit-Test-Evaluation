#ifndef TENSORSTORE_INTERNAL_TAGGED_PTR_H_
#define TENSORSTORE_INTERNAL_TAGGED_PTR_H_
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
namespace tensorstore {
namespace internal {
template <typename T, int TagBits>
class TaggedPtr {
  constexpr static std::uintptr_t kTagMask =
      (static_cast<std::uintptr_t>(1) << TagBits) - 1;
  constexpr static std::uintptr_t kPointerMask = ~kTagMask;
 public:
  using element_type = T;
  template <typename U>
  using rebind = TaggedPtr<U, TagBits>;
  constexpr TaggedPtr() noexcept : value_(0) {}
  constexpr TaggedPtr(std::nullptr_t) noexcept : value_(0) {}
  constexpr TaggedPtr(std::nullptr_t, std::uintptr_t tag) noexcept
      : value_(tag) {
    assert((tag & kPointerMask) == 0);
  }
  template <typename U,
            std::enable_if_t<std::is_convertible_v<U*, T*>>* = nullptr>
  TaggedPtr(U* ptr, std::uintptr_t tag = 0) noexcept {
    assert((reinterpret_cast<std::uintptr_t>(static_cast<T*>(ptr)) &
            kTagMask) == 0 &&
           (tag & kPointerMask) == 0);
    value_ = reinterpret_cast<std::uintptr_t>(static_cast<T*>(ptr)) | tag;
  }
  template <typename U,
            std::enable_if_t<std::is_convertible_v<U*, T*>>* = nullptr>
  TaggedPtr(TaggedPtr<U, TagBits> other) noexcept
      : TaggedPtr(other.get(), other.tag()) {}
  TaggedPtr& operator=(std::nullptr_t) noexcept {
    value_ = 0;
    return *this;
  }
  template <typename U>
  std::enable_if_t<std::is_convertible_v<U*, T*>, TaggedPtr&> operator=(
      U* ptr) noexcept {
    *this = TaggedPtr(ptr);
    return *this;
  }
  explicit operator bool() const noexcept {
    return static_cast<bool>(reinterpret_cast<T*>(value_ & kPointerMask));
  }
  T* get() const noexcept {
    static_assert(alignof(T) >= (1 << TagBits),
                  "Number of TagBits is incompatible with alignment of T.");
    return reinterpret_cast<T*>(value_ & kPointerMask);
  }
  operator T*() const noexcept { return get(); }
  std::uintptr_t tag() const noexcept { return value_ & kTagMask; }
  template <int Bit>
  std::enable_if_t<(Bit >= 0 && Bit < TagBits), bool> tag() const noexcept {
    return static_cast<bool>((value_ >> Bit) & 1);
  }
  template <int Bit>
  std::enable_if_t<(Bit >= 0 && Bit < TagBits), void> set_tag(
      bool value) noexcept {
    constexpr std::uintptr_t mask = (static_cast<std::uintptr_t>(1) << Bit);
    value_ = (value_ & ~mask) | (static_cast<std::uintptr_t>(value) << Bit);
  }
  void set_tag(std::uintptr_t tag) noexcept {
    assert((tag & kPointerMask) == 0);
    value_ = (value_ & kPointerMask) | tag;
  }
  T* operator->() const noexcept {
    T* ptr = get();
    assert(ptr != nullptr);
    return ptr;
  }
  T& operator*() const noexcept {
    T* ptr = get();
    assert(ptr != nullptr);
    return *ptr;
  }
  friend bool operator==(TaggedPtr x, TaggedPtr y) {
    return x.get() == y.get() && x.tag() == y.tag();
  }
  friend bool operator!=(TaggedPtr x, TaggedPtr y) { return !(x == y); }
  template <typename H>
  friend H AbslHashValue(H h, TaggedPtr x) {
    return H::combine(std::move(h), x.value_);
  }
 private:
  std::uintptr_t value_;
};
template <typename T, int TagBits>
inline T* to_address(TaggedPtr<T, TagBits> p) {
  return p.get();
}
template <typename T, typename U, int TagBits>
TaggedPtr<T, TagBits> static_pointer_cast(TaggedPtr<U, TagBits> p) {
  return TaggedPtr<T, TagBits>(static_cast<T*>(p.get()), p.tag());
}
template <typename T, typename U, int TagBits>
TaggedPtr<T, TagBits> const_pointer_cast(TaggedPtr<U, TagBits> p) {
  return TaggedPtr<T, TagBits>(const_cast<T*>(p.get()), p.tag());
}
template <typename T, typename U, int TagBits>
TaggedPtr<T, TagBits> dynamic_pointer_cast(TaggedPtr<U, TagBits> p) {
  return TaggedPtr<T, TagBits>(dynamic_cast<T*>(p.get()), p.tag());
}
}  
}  
#endif  