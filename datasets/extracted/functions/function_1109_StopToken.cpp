#ifndef TENSORSTORE_UTIL_STOP_TOKEN_H_
#define TENSORSTORE_UTIL_STOP_TOKEN_H_
#include <atomic>
#include <cstddef>
#include <type_traits>
#include <utility>
#include "absl/base/attributes.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/util/stop_token_impl.h"
namespace tensorstore {
class StopSource;
template <typename Callback>
class StopCallback;
class StopToken {
 public:
  StopToken() noexcept = default;
  ~StopToken() noexcept = default;
  StopToken(const StopToken&) noexcept = default;
  StopToken(StopToken&&) noexcept = default;
  StopToken& operator=(const StopToken&) noexcept = default;
  StopToken& operator=(StopToken&&) noexcept = default;
  [[nodiscard]] bool stop_possible() const noexcept {
    return state_ != nullptr;
  }
  [[nodiscard]] bool stop_requested() const noexcept {
    return state_ != nullptr && state_->stop_requested();
  }
  friend bool operator==(const StopToken& a, const StopToken& b) {
    return a.state_ == b.state_;
  }
  friend bool operator!=(const StopToken& a, const StopToken& b) {
    return !(a == b);
  }
 private:
  friend class StopSource;
  template <typename Callback>
  friend class StopCallback;
  StopToken(internal::IntrusivePtr<internal_stop_token::StopState> state)
      : state_(std::move(state)) {}
  internal::IntrusivePtr<internal_stop_token::StopState> state_{nullptr};
};
class StopSource {
 public:
  StopSource() noexcept
      : state_(internal::MakeIntrusivePtr<internal_stop_token::StopState>()) {}
  explicit StopSource(std::nullptr_t) noexcept : state_(nullptr) {}
  ~StopSource() noexcept = default;
  StopSource(const StopSource& b) noexcept = default;
  StopSource(StopSource&&) noexcept = default;
  StopSource& operator=(const StopSource& b) noexcept = default;
  StopSource& operator=(StopSource&&) noexcept = default;
  [[nodiscard]] bool stop_possible() const noexcept {
    return state_ != nullptr;
  }
  [[nodiscard]] bool stop_requested() const noexcept {
    return state_ != nullptr && state_->stop_requested();
  }
  bool request_stop() const noexcept {
    if (state_ != nullptr) {
      return state_->RequestStop();
    }
    return false;
  }
  [[nodiscard]] StopToken get_token() const noexcept {
    return StopToken(state_);
  }
 private:
  internal::IntrusivePtr<internal_stop_token::StopState> state_;
};
template <typename Callback>
class StopCallback : private internal_stop_token::StopCallbackBase {
  static_assert(std::is_invocable_v<Callback>);
 public:
  using callback_type = Callback;
  StopCallback(const StopCallback&) = delete;
  StopCallback& operator=(const StopCallback&) = delete;
  StopCallback(StopCallback&&) = delete;
  StopCallback& operator=(StopCallback&&) = delete;
  template <
      typename... Args,
      std::enable_if_t<std::is_constructible_v<Callback, Args...>, int> = 0>
  explicit StopCallback(const StopToken& token, Args&&... args)
      : callback_(std::forward<Args>(args)...) {
    internal_stop_token::StopState* state = token.state_.get();
    if (state) {
      invoker_ = &StopCallback::Invoker;
      state->RegisterImpl(*this);
    }  
  }
  ~StopCallback() {
    internal_stop_token::StopState* state =
        state_.exchange(nullptr, std::memory_order_acq_rel);
    if (state != nullptr) {
      state->UnregisterImpl(*this);
    }
  }
 private:
  static void Invoker(internal_stop_token::StopCallbackBase& self) noexcept {
    static_cast<Callback&&>(static_cast<StopCallback&&>(self).callback_)();
  }
  ABSL_ATTRIBUTE_NO_UNIQUE_ADDRESS Callback callback_;
};
template <typename Callback>
StopCallback(StopToken token, Callback callback) -> StopCallback<Callback>;
}  
#endif