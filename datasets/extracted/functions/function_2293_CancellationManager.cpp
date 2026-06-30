#include "xla/tsl/framework/cancellation.h"
#include <forward_list>
#include "absl/memory/memory.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
namespace tsl {
const CancellationToken CancellationManager::kInvalidToken = -1;
CancellationManager::CancellationManager()
    : is_cancelling_(false),
      is_cancelled_(false),
      next_cancellation_token_(0) {}
CancellationManager::CancellationManager(CancellationManager* parent)
    : is_cancelling_(false), next_cancellation_token_(0), parent_(parent) {
  is_cancelled_ = parent->RegisterChild(this);
}
void CancellationManager::StartCancel() {
  StartCancelWithStatus(absl::OkStatus());
}
void CancellationManager::StartCancelWithStatus(const absl::Status& status) {
  gtl::FlatMap<CancellationToken, CallbackConfiguration> callbacks_to_run;
  std::forward_list<CancellationManager*> children_to_cancel;
  Notification* cancelled_notification = nullptr;
  {
    mutex_lock l(mu_);
    if (is_cancelled_.load(std::memory_order_relaxed) || is_cancelling_) {
      return;
    }
    is_cancelling_ = true;
    if (state_) {
      std::swap(state_->callbacks, callbacks_to_run);
      CancellationManager* child = state_->first_child;
      while (child != nullptr) {
        children_to_cancel.push_front(child);
        child->is_removed_from_parent_ = true;
        child = child->next_sibling_;
      }
      state_->first_child = nullptr;
      cancelled_notification = &state_->cancelled_notification;
    }
  }
  for (auto key_and_value : callbacks_to_run) {
    CallbackConfiguration& config = key_and_value.second;
    if (!status.ok() && config.log_error) {
      LOG(WARNING) << "Cancellation callback \"" << config.name
                   << "\" is triggered due to a "
                   << (StatusGroup::IsDerived(status) ? "derived" : "root")
                   << " error: " << status.ToString();
    }
    config.callback();
  }
  for (CancellationManager* child : children_to_cancel) {
    child->StartCancelWithStatus(status);
  }
  {
    mutex_lock l(mu_);
    is_cancelling_ = false;
    is_cancelled_.store(true, std::memory_order_release);
  }
  if (cancelled_notification) {
    cancelled_notification->Notify();
  }
}
bool CancellationManager::RegisterCallback(CancellationToken token,
                                           CancelCallback callback) {
  return RegisterCallbackConfig(
      token, CallbackConfiguration{callback, "", false});
}
bool CancellationManager::RegisterCallbackWithErrorLogging(
    CancellationToken token, CancelCallback callback,
    absl::string_view callback_name) {
  return RegisterCallbackConfig(
      token, CallbackConfiguration{callback, std::string(callback_name), true});
}
bool CancellationManager::RegisterCallbackConfig(CancellationToken token,
                                                 CallbackConfiguration config) {
  DCHECK_LT(token, next_cancellation_token_) << "Invalid cancellation token";
  mutex_lock l(mu_);
  bool should_register = !is_cancelled_ && !is_cancelling_;
  if (should_register) {
    if (!state_) {
      state_ = absl::make_unique<State>();
    }
    std::swap(state_->callbacks[token], config);
  }
  return should_register;
}
bool CancellationManager::DeregisterCallback(CancellationToken token) {
  mu_.lock();
  if (is_cancelled_) {
    mu_.unlock();
    return false;
  } else if (is_cancelling_) {
    Notification* cancelled_notification =
        state_ ? &state_->cancelled_notification : nullptr;
    mu_.unlock();
    if (cancelled_notification) {
      cancelled_notification->WaitForNotification();
    }
    return false;
  } else {
    if (state_) {
      state_->callbacks.erase(token);
    }
    mu_.unlock();
    return true;
  }
}
bool CancellationManager::RegisterChild(CancellationManager* child) {
  mutex_lock l(mu_);
  if (is_cancelled_.load(std::memory_order_relaxed) || is_cancelling_) {
    child->is_removed_from_parent_ = true;
    return true;
  }
  if (!state_) {
    state_ = absl::make_unique<State>();
  }
  CancellationManager* current_head = state_->first_child;
  state_->first_child = child;
  child->prev_sibling_ = nullptr;
  child->next_sibling_ = current_head;
  if (current_head) {
    current_head->prev_sibling_ = child;
  }
  return false;
}
void CancellationManager::DeregisterChild(CancellationManager* child) {
  DCHECK_EQ(child->parent_, this);
  Notification* cancelled_notification = nullptr;
  {
    mutex_lock l(mu_);
    if (!child->is_removed_from_parent_) {
      DCHECK(state_);
      if (child->prev_sibling_ == nullptr) {
        DCHECK_EQ(state_->first_child, child);
        state_->first_child = child->next_sibling_;
      } else {
        child->prev_sibling_->next_sibling_ = child->next_sibling_;
      }
      if (child->next_sibling_ != nullptr) {
        child->next_sibling_->prev_sibling_ = child->prev_sibling_;
      }
      child->is_removed_from_parent_ = true;
    }
    if (is_cancelling_) {
      cancelled_notification = &state_->cancelled_notification;
    }
  }
  if (cancelled_notification) {
    cancelled_notification->WaitForNotification();
  }
}
bool CancellationManager::TryDeregisterCallback(CancellationToken token) {
  mutex_lock lock(mu_);
  if (is_cancelled_ || is_cancelling_) {
    return false;
  } else {
    if (state_) {
      state_->callbacks.erase(token);
    }
    return true;
  }
}
CancellationManager::~CancellationManager() {
  if (parent_) {
    parent_->DeregisterChild(this);
  }
  if (state_) {
    StartCancel();
  }
}
bool CancellationManager::IsCancelling() {
  mutex_lock lock(mu_);
  return is_cancelling_;
}
absl::Status RegisterCancellationCallback(
    CancellationManager* cancellation_manager, CancelCallback callback,
    std::function<void()>* deregister_fn) {
  if (cancellation_manager) {
    CancellationToken token = cancellation_manager->get_cancellation_token();
    if (!cancellation_manager->RegisterCallback(token, std::move(callback))) {
      return errors::Cancelled("Operation was cancelled");
    }
    *deregister_fn = [cancellation_manager, token]() {
      cancellation_manager->DeregisterCallback(token);
    };
  } else {
    VLOG(1) << "Cancellation manager is not set. Cancellation callback will "
               "not be registered.";
    *deregister_fn = []() {};
  }
  return absl::OkStatus();
}
}  