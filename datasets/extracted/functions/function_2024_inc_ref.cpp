#ifndef THIRD_PARTY_PY_AROLLA_PY_UTILS_PY_OBJECT_PTR_IMPL_H_
#define THIRD_PARTY_PY_AROLLA_PY_UTILS_PY_OBJECT_PTR_IMPL_H_
#include <Python.h>
#include <cstddef>
#include <utility>
#include "absl/base/attributes.h"
namespace arolla::python::py_object_ptr_impl_internal {
template <typename SelfType, typename Traits>
class BasePyObjectPtr {
  using GILGuardType = typename Traits::GILGuardType;
  using PyObjectType = typename Traits::PyObjectType;
  ABSL_ATTRIBUTE_ALWAYS_INLINE static void inc_ref(PyObjectType* ptr) {
    Traits().inc_ref(ptr);
  }
  ABSL_ATTRIBUTE_ALWAYS_INLINE static void dec_ref(PyObjectType* ptr) {
    Traits().dec_ref(ptr);
  }
 public:
  [[nodiscard]] static SelfType Own(PyObjectType* ptr) {
    SelfType result;
    result.ptr_ = ptr;
    return result;
  }
  [[nodiscard]] static SelfType NewRef(PyObjectType* ptr) {
    SelfType result;
    if (ptr != nullptr) {
      GILGuardType gil_guard;
      result.ptr_ = ptr;
      inc_ref(result.ptr_);
    }
    return result;
  }
  BasePyObjectPtr() = default;
  ~BasePyObjectPtr() { reset(); }
  BasePyObjectPtr(const BasePyObjectPtr& other) {
    if (other.ptr_ != nullptr) {
      GILGuardType gil_guard;
      ptr_ = other.ptr_;
      inc_ref(ptr_);
    }
  }
  BasePyObjectPtr& operator=(const BasePyObjectPtr& other) {
    if (ptr_ != other.ptr_) {
      GILGuardType gil_guard;
      PyObjectType* old_ptr = std::exchange(ptr_, other.ptr_);
      if (ptr_ != nullptr) {
        inc_ref(ptr_);
      }
      if (old_ptr != nullptr) {
        dec_ref(old_ptr);
      }
    }
    return *this;
  }
  BasePyObjectPtr(BasePyObjectPtr&& other) : ptr_(other.release()) {}
  BasePyObjectPtr& operator=(BasePyObjectPtr&& other) {
    PyObjectType* old_ptr = std::exchange(ptr_, other.release());
    if (old_ptr != nullptr) {
      GILGuardType gil_guard;
      dec_ref(old_ptr);
    }
    return *this;
  }
  [[nodiscard]] PyObjectType* get() const { return ptr_; }
  bool operator==(std::nullptr_t) const { return ptr_ == nullptr; }
  bool operator!=(std::nullptr_t) const { return ptr_ != nullptr; }
  [[nodiscard]] PyObjectType* release() { return std::exchange(ptr_, nullptr); }
  void reset() {
    if (PyObjectType* old_ptr = release()) {
      GILGuardType gil_guard;
      dec_ref(old_ptr);
    }
  }
 private:
  PyObjectType* ptr_ = nullptr;
};
}  
#endif  