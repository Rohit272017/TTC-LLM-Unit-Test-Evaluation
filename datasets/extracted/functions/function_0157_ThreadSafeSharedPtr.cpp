#ifndef AROLLA_UTIL_THREAD_SAFE_SHARED_PTR_H_
#define AROLLA_UTIL_THREAD_SAFE_SHARED_PTR_H_
#include <cstddef>
#include <memory>
#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
namespace arolla {
template <typename T>
class ThreadSafeSharedPtr {
 public:
  ThreadSafeSharedPtr() = default;
  explicit ThreadSafeSharedPtr(std::shared_ptr<T> ptr) : ptr_(std::move(ptr)) {}
  ThreadSafeSharedPtr(const ThreadSafeSharedPtr&) = delete;
  ThreadSafeSharedPtr& operator=(const ThreadSafeSharedPtr&) = delete;
  bool operator==(std::nullptr_t) const ABSL_LOCKS_EXCLUDED(mx_) {
    absl::MutexLock guard(&mx_);
    return ptr_ == nullptr;
  }
  bool operator!=(std::nullptr_t) const ABSL_LOCKS_EXCLUDED(mx_) {
    return !(*this == nullptr);
  }
  std::shared_ptr<T> load() const ABSL_LOCKS_EXCLUDED(mx_) {
    absl::MutexLock guard(&mx_);
    return ptr_;
  }
  void store(std::shared_ptr<T> ptr) ABSL_LOCKS_EXCLUDED(mx_) {
    absl::MutexLock guard(&mx_);
    ptr.swap(ptr_);
  }
 private:
  std::shared_ptr<T> ptr_ ABSL_GUARDED_BY(mx_);
  mutable absl::Mutex mx_;
};
}  
#endif  