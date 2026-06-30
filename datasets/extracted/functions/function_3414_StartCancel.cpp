#include "xla/tsl/distributed_runtime/call_options.h"
#include <utility>
#include "tsl/platform/mutex.h"
namespace tsl {
CallOptions::CallOptions() = default;
void CallOptions::StartCancel() {
  mutex_lock l(mu_);
  if (cancel_func_ != nullptr) {
    cancel_func_();
  }
}
void CallOptions::SetCancelCallback(CancelFunction cancel_func) {
  mutex_lock l(mu_);
  cancel_func_ = std::move(cancel_func);
}
void CallOptions::ClearCancelCallback() {
  mutex_lock l(mu_);
  cancel_func_ = nullptr;
}
int64_t CallOptions::GetTimeout() {
  mutex_lock l(mu_);
  return timeout_in_ms_;
}
void CallOptions::SetTimeout(int64_t ms) {
  mutex_lock l(mu_);
  timeout_in_ms_ = ms;
}
}  