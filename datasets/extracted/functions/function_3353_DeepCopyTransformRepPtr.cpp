#ifndef TENSORSTORE_INDEX_SPACE_INTERNAL_DEEP_COPY_TRANSFORM_REP_PTR_H_
#define TENSORSTORE_INDEX_SPACE_INTERNAL_DEEP_COPY_TRANSFORM_REP_PTR_H_
#include <utility>
#include "tensorstore/index_space/internal/transform_rep.h"
namespace tensorstore {
namespace internal_index_space {
class DeepCopyTransformRepPtr {
 public:
  DeepCopyTransformRepPtr(std::nullptr_t = nullptr) : ptr_(nullptr) {}
  explicit DeepCopyTransformRepPtr(TransformRep* ptr,
                                   internal::adopt_object_ref_t)
      : ptr_(ptr) {
    assert(ptr == nullptr ||
           (ptr->input_rank_capacity == 0 && ptr->output_rank_capacity == 0) ||
           ptr->reference_count == 1);
  }
  explicit DeepCopyTransformRepPtr(TransformRep* ptr,
                                   internal::acquire_object_ref_t) {
    if (ptr) {
      ptr_ =
          TransformRep::Allocate(ptr->input_rank, ptr->output_rank).release();
      CopyTransformRep(ptr, ptr_);
    } else {
      ptr_ = nullptr;
    }
  }
  DeepCopyTransformRepPtr(DeepCopyTransformRepPtr&& other)
      : ptr_(std::exchange(other.ptr_, nullptr)) {}
  DeepCopyTransformRepPtr(const DeepCopyTransformRepPtr& other)
      : DeepCopyTransformRepPtr(other.ptr_, internal::acquire_object_ref) {}
  DeepCopyTransformRepPtr& operator=(DeepCopyTransformRepPtr&& other) {
    if (ptr_) Free();
    ptr_ = std::exchange(other.ptr_, nullptr);
    return *this;
  }
  DeepCopyTransformRepPtr& operator=(const DeepCopyTransformRepPtr& other) {
    return *this = DeepCopyTransformRepPtr(other.ptr_,
                                           internal::acquire_object_ref);
  }
  DeepCopyTransformRepPtr& operator=(std::nullptr_t) {
    if (ptr_) Free();
    ptr_ = nullptr;
    return *this;
  }
  ~DeepCopyTransformRepPtr() {
    if (ptr_) Free();
  }
  explicit operator bool() const { return static_cast<bool>(ptr_); }
  TransformRep* get() const { return ptr_; }
  TransformRep* operator->() const { return ptr_; }
  TransformRep& operator*() const { return *ptr_; }
  TransformRep* release() { return std::exchange(ptr_, nullptr); }
 private:
  void Free() {
    TransformRep::Ptr<>(ptr_, internal::adopt_object_ref);
  }
  TransformRep* ptr_;
};
}  
}  
#endif  