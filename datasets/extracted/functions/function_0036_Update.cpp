#include "tensorflow/core/kernels/batching_util/threadsafe_status.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/platform/mutex.h"
namespace tensorflow {
const Status& ThreadSafeStatus::status() const& {
  tf_shared_lock lock(mutex_);
  return status_;
}
Status ThreadSafeStatus::status() && {
  tf_shared_lock lock(mutex_);
  return std::move(status_);
}
void ThreadSafeStatus::Update(const Status& new_status) {
  if (new_status.ok()) {
    return;
  }
  mutex_lock lock(mutex_);
  status_.Update(new_status);
}
void ThreadSafeStatus::Update(Status&& new_status) {
  if (new_status.ok()) {
    return;
  }
  mutex_lock lock(mutex_);
  status_.Update(std::forward<Status>(new_status));
}
}  