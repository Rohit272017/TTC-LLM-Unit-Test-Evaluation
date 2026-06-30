#include "quiche/http2/adapter/window_manager.h"
#include <utility>
#include "quiche/common/platform/api/quiche_bug_tracker.h"
#include "quiche/common/platform/api/quiche_logging.h"
namespace http2 {
namespace adapter {
bool DefaultShouldWindowUpdateFn(int64_t limit, int64_t window, int64_t delta) {
  const int64_t kDesiredMinWindow = limit / 2;
  const int64_t kDesiredMinDelta = limit / 3;
  if (delta >= kDesiredMinDelta) {
    return true;
  } else if (window < kDesiredMinWindow) {
    return true;
  }
  return false;
}
WindowManager::WindowManager(int64_t window_size_limit,
                             WindowUpdateListener listener,
                             ShouldWindowUpdateFn should_window_update_fn,
                             bool update_window_on_notify)
    : limit_(window_size_limit),
      window_(window_size_limit),
      buffered_(0),
      listener_(std::move(listener)),
      should_window_update_fn_(std::move(should_window_update_fn)),
      update_window_on_notify_(update_window_on_notify) {
  if (!should_window_update_fn_) {
    should_window_update_fn_ = DefaultShouldWindowUpdateFn;
  }
}
void WindowManager::OnWindowSizeLimitChange(const int64_t new_limit) {
  QUICHE_VLOG(2) << "WindowManager@" << this
                 << " OnWindowSizeLimitChange from old limit of " << limit_
                 << " to new limit of " << new_limit;
  window_ += (new_limit - limit_);
  limit_ = new_limit;
}
void WindowManager::SetWindowSizeLimit(int64_t new_limit) {
  QUICHE_VLOG(2) << "WindowManager@" << this
                 << " SetWindowSizeLimit from old limit of " << limit_
                 << " to new limit of " << new_limit;
  limit_ = new_limit;
  MaybeNotifyListener();
}
bool WindowManager::MarkDataBuffered(int64_t bytes) {
  QUICHE_VLOG(2) << "WindowManager@" << this << " window: " << window_
                 << " bytes: " << bytes;
  if (window_ < bytes) {
    QUICHE_VLOG(2) << "WindowManager@" << this << " window underflow "
                   << "window: " << window_ << " bytes: " << bytes;
    window_ = 0;
  } else {
    window_ -= bytes;
  }
  buffered_ += bytes;
  if (window_ == 0) {
    MaybeNotifyListener();
  }
  return window_ > 0;
}
void WindowManager::MarkDataFlushed(int64_t bytes) {
  QUICHE_VLOG(2) << "WindowManager@" << this << " buffered: " << buffered_
                 << " bytes: " << bytes;
  if (buffered_ < bytes) {
    QUICHE_BUG(bug_2816_1) << "WindowManager@" << this << " buffered underflow "
                           << "buffered_: " << buffered_ << " bytes: " << bytes;
    buffered_ = 0;
  } else {
    buffered_ -= bytes;
  }
  MaybeNotifyListener();
}
void WindowManager::MaybeNotifyListener() {
  const int64_t delta = limit_ - (buffered_ + window_);
  if (should_window_update_fn_(limit_, window_, delta) && delta > 0) {
    QUICHE_VLOG(2) << "WindowManager@" << this
                   << " Informing listener of delta: " << delta;
    listener_(delta);
    if (update_window_on_notify_) {
      window_ += delta;
    }
  }
}
}  
}  