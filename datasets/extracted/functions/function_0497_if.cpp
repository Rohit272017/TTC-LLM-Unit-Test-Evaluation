#ifndef AROLLA_UTIL_REFCOUNT_PTR_H_
#define AROLLA_UTIL_REFCOUNT_PTR_H_
#include <cstddef>
#include <memory>
#include <ostream>
#include <type_traits>
#include <utility>
#include "absl/base/nullability.h"
#include "arolla/util/refcount.h"
namespace arolla {
template <typename T>
class
        RefcountPtr;
class RefcountedBase {
  mutable Refcount refcount_;
  template <typename T>
  friend class
          RefcountPtr;
};
template <typename T>
class
        RefcountPtr {
 public:
  template <typename... Args>
  static constexpr RefcountPtr<T> Make(Args&&... args) noexcept {
    return RefcountPtr(
        std::make_unique<T>(std::forward<Args>(args)...).release());
  }
  static constexpr RefcountPtr<T> Own(std::unique_ptr<T>&& ptr) noexcept {
    return RefcountPtr(ptr.release());
  }
  static RefcountPtr<T> NewRef(T* ptr) noexcept {
    if (ptr != nullptr) {
      ptr->refcount_.increment();
    }
    return RefcountPtr(ptr);
  }
  constexpr RefcountPtr() noexcept = default;
  constexpr  RefcountPtr(  
      std::nullptr_t) noexcept {}
  RefcountPtr(const RefcountPtr& rhs) noexcept : ptr_(rhs.ptr_) {
    if (ptr_ != nullptr) {
      ptr_->refcount_.increment();
    }
  }
  RefcountPtr(RefcountPtr&& rhs) noexcept : ptr_(rhs.ptr_) {
    rhs.ptr_ = nullptr;
  }
  ~RefcountPtr() noexcept(noexcept(reset())) { reset(); }
  RefcountPtr& operator=(const RefcountPtr& rhs) noexcept(noexcept(reset())) {
    if (ptr_ != rhs.ptr_) {
    const auto tmp = std::move(*this);
      ptr_ = rhs.ptr_;
      if (ptr_ != nullptr) {
        ptr_->refcount_.increment();
      }
    }
    return *this;
  }
  RefcountPtr& operator=(RefcountPtr&& rhs) noexcept {
    T* const tmp = ptr_;  
    ptr_ = rhs.ptr_;      
    rhs.ptr_ = tmp;       
    return *this;
  }
  void reset() noexcept(std::is_nothrow_destructible_v<T>) {
    T* const tmp = ptr_;
    ptr_ = nullptr;
    if (tmp != nullptr && !tmp->refcount_.decrement()) {
      delete tmp;
    }
  }
  constexpr bool operator==(std::nullptr_t) const noexcept {
    return ptr_ == nullptr;
  }
  constexpr bool operator!=(std::nullptr_t) const noexcept {
    return ptr_ != nullptr;
  }
  bool operator==(const RefcountPtr& rhs) const noexcept {
    return ptr_ == rhs.ptr_;
  }
  bool operator!=(const RefcountPtr& rhs) const noexcept {
    return ptr_ != rhs.ptr_;
  }
  constexpr T* get() const noexcept { return ptr_; }
  T& operator*() const noexcept { return *ptr_; }
  T* operator->() const noexcept { return ptr_; }
  friend void swap(RefcountPtr& lhs, RefcountPtr& rhs) noexcept {
    T* const tmp = lhs.ptr_;  
    lhs.ptr_ = rhs.ptr_;      
    rhs.ptr_ = tmp;           
  }
  using absl_nullability_compatible = void;
 private:
  explicit constexpr RefcountPtr(T* ptr) noexcept : ptr_(ptr) {}
  T* ptr_ = nullptr;
};
template <typename T>
std::ostream& operator<<(std::ostream& ostream, const RefcountPtr<T>& ptr) {
  return ostream << static_cast<T*>(ptr.get());
}
template <typename T>
T* GetRawPointer(const RefcountPtr<T>& ptr) {
  return ptr.get();
}
}  
#endif  