#ifndef AROLLA_QEXPR_EVAL_CONTEXT_H_
#define AROLLA_QEXPR_EVAL_CONTEXT_H_
#include <cstdint>
#include <type_traits>
#include <utility>
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "arolla/memory/frame.h"
#include "arolla/memory/memory_allocation.h"
#include "arolla/memory/raw_buffer_factory.h"
namespace arolla {
class RootEvaluationContext {
 public:
  RootEvaluationContext() : buffer_factory_(GetHeapBufferFactory()) {}
  explicit RootEvaluationContext(const FrameLayout* layout,
                                 RawBufferFactory* buffer_factory = nullptr)
      : alloc_(layout),
        buffer_factory_(buffer_factory == nullptr ? GetHeapBufferFactory()
                                                  : buffer_factory) {}
  RootEvaluationContext(const RootEvaluationContext&) = delete;
  RootEvaluationContext& operator=(const RootEvaluationContext&) = delete;
  RootEvaluationContext(RootEvaluationContext&&) = default;
  RootEvaluationContext& operator=(RootEvaluationContext&&) = default;
  template <typename T>
  T* GetMutable(FrameLayout::Slot<T> slot) {
    return frame().GetMutable(slot);
  }
  template <typename T, typename S = T>
  void Set(FrameLayout::Slot<T> slot, S&& value) {
    return frame().Set(slot, std::forward<S>(value));
  }
  template <typename T>
  const T& Get(FrameLayout::Slot<T> slot) const {
    return frame().Get(slot);
  }
  FramePtr frame() { return alloc_.frame(); }
  ConstFramePtr frame() const { return alloc_.frame(); }
  RawBufferFactory& buffer_factory() const { return *buffer_factory_; }
  bool IsValid() const { return alloc_.IsValid(); }
 private:
  MemoryAllocation alloc_;
  RawBufferFactory* buffer_factory_ = nullptr;  
};
class EvaluationContext {
 public:
  using CheckInterruptFn = absl::AnyInvocable<absl::Status()>;
  EvaluationContext() = default;
  explicit EvaluationContext(RootEvaluationContext& root_ctx)
      : buffer_factory_(root_ctx.buffer_factory()) {}
  explicit EvaluationContext(RawBufferFactory* buffer_factory,
                             CheckInterruptFn* check_interrupt_fn = nullptr)
      : buffer_factory_(*buffer_factory),
        check_interrupt_fn_(check_interrupt_fn) {
    DCHECK(buffer_factory);
  }
  EvaluationContext(const EvaluationContext&) = delete;
  EvaluationContext& operator=(const EvaluationContext&) = delete;
  EvaluationContext(EvaluationContext&&) = delete;
  EvaluationContext& operator=(EvaluationContext&&) = delete;
  const absl::Status& status() const& { return status_; }
  absl::Status&& status() && { return std::move(status_); }
  template <typename StatusLike>
  auto set_status(StatusLike&& status_like) ->
      std::enable_if_t<
          !std::is_lvalue_reference_v<StatusLike> &&
              !std::is_const_v<std::remove_reference_t<StatusLike>>,
          std::void_t<decltype(static_cast<absl::Status>(
              std::forward<StatusLike>(status_like)))>> {
    status_ = static_cast<absl::Status>(std::forward<StatusLike>(status_like));
    signal_received_ = signal_received_ || !status_.ok();
  }
  auto set_status() {
    return [this](auto&& status_like) {
      this->set_status(std::forward<decltype(status_like)>(status_like));
    };
  }
  RawBufferFactory& buffer_factory() { return buffer_factory_; }
  int64_t requested_jump() const { return jump_; }
  void set_requested_jump(int64_t jump) {
    signal_received_ = true;
    jump_ = jump;
  }
  bool signal_received() const { return signal_received_; }
  void ResetSignals() {
    signal_received_ = false;
    jump_ = 0;
    status_ = absl::OkStatus();
  }
  bool has_check_interrupt_fn() const {
    return check_interrupt_fn_ != nullptr;
  }
  void check_interrupt(int64_t ops_evaluated) {
    constexpr int64_t kCheckInterruptPeriod = 128;
    check_interrupt_counter_ += ops_evaluated;
    if (check_interrupt_counter_ >= kCheckInterruptPeriod &&
        check_interrupt_fn_ && status_.ok()) {
      set_status((*check_interrupt_fn_)());
      check_interrupt_counter_ = 0;
    }
  }
 private:
  bool signal_received_ = false;
  int64_t jump_ = 0;
  absl::Status status_;
  RawBufferFactory& buffer_factory_ = *GetHeapBufferFactory();  
  CheckInterruptFn* check_interrupt_fn_ = nullptr;
  int64_t check_interrupt_counter_ = 0;
};
}  
#endif  