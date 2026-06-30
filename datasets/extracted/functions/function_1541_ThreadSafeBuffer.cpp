#ifndef TENSORFLOW_CORE_DATA_SERVICE_THREAD_SAFE_BUFFER_H_
#define TENSORFLOW_CORE_DATA_SERVICE_THREAD_SAFE_BUFFER_H_
#include <deque>
#include <utility>
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
namespace tensorflow {
namespace data {
template <class T>
class ThreadSafeBuffer final {
 public:
  explicit ThreadSafeBuffer(size_t buffer_size);
  StatusOr<T> Pop();
  Status Push(StatusOr<T> value);
  void Cancel(Status status);
  bool Empty() const;
 private:
  const size_t buffer_size_;
  mutable mutex mu_;
  condition_variable ready_to_pop_;
  condition_variable ready_to_push_;
  std::deque<StatusOr<T>> results_ TF_GUARDED_BY(mu_);
  Status status_ TF_GUARDED_BY(mu_) = absl::OkStatus();
  ThreadSafeBuffer(const ThreadSafeBuffer&) = delete;
  void operator=(const ThreadSafeBuffer&) = delete;
};
template <class T>
ThreadSafeBuffer<T>::ThreadSafeBuffer(size_t buffer_size)
    : buffer_size_(buffer_size) {
  DCHECK_GT(buffer_size, 0)
      << "ThreadSafeBuffer must have a positive buffer size. Got "
      << buffer_size << ".";
}
template <class T>
bool ThreadSafeBuffer<T>::Empty() const {
  tf_shared_lock l(mu_);
  return results_.empty();
}
template <class T>
StatusOr<T> ThreadSafeBuffer<T>::Pop() {
  mutex_lock l(mu_);
  while (status_.ok() && results_.empty()) {
    ready_to_pop_.wait(l);
  }
  if (!status_.ok()) {
    return status_;
  }
  StatusOr<T> result = std::move(results_.front());
  results_.pop_front();
  ready_to_push_.notify_one();
  return result;
}
template <class T>
Status ThreadSafeBuffer<T>::Push(StatusOr<T> value) {
  mutex_lock l(mu_);
  while (status_.ok() && results_.size() >= buffer_size_) {
    ready_to_push_.wait(l);
  }
  if (!status_.ok()) {
    return status_;
  }
  results_.push_back(std::move(value));
  ready_to_pop_.notify_one();
  return absl::OkStatus();
}
template <class T>
void ThreadSafeBuffer<T>::Cancel(Status status) {
  DCHECK(!status.ok())
      << "Cancelling ThreadSafeBuffer requires a non-OK status. Got " << status;
  mutex_lock l(mu_);
  status_ = std::move(status);
  ready_to_push_.notify_all();
  ready_to_pop_.notify_all();
}
}  
}  
#endif  