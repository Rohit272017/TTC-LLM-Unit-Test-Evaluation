#ifndef TENSORFLOW_CORE_UTIL_REFFED_STATUS_CALLBACK_H_
#define TENSORFLOW_CORE_UTIL_REFFED_STATUS_CALLBACK_H_
#include <utility>
#include "absl/strings/str_cat.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
namespace tensorflow {
class ReffedStatusCallback : public core::RefCounted {
 public:
  explicit ReffedStatusCallback(StatusCallback done) : done_(std::move(done)) {}
  void UpdateStatus(const Status& s) {
    mutex_lock lock(mu_);
    status_group_.Update(s);
  }
  bool ok() {
    tf_shared_lock lock(mu_);
    return status_group_.ok();
  }
  Status status() {
    tf_shared_lock lock(mu_);
    return status_group_.as_summary_status();
  }
  ~ReffedStatusCallback() override { done_(status_group_.as_summary_status()); }
 private:
  StatusCallback done_;
  mutex mu_;
  StatusGroup status_group_ TF_GUARDED_BY(mu_);
};
}  
#endif  