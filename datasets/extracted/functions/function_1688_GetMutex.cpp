#include "tensorstore/util/future.h"
#include <stddef.h>
#include <stdint.h>
#include <atomic>
#include <cassert>
#include <thread>  
#include <utility>
#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/base/optimization.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/internal/container/intrusive_linked_list.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/gauge.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/util/future_impl.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
using ::tensorstore::internal_metrics::MetricMetadata;
namespace tensorstore {
namespace internal_future {
namespace {
auto& live_futures = internal_metrics::Gauge<int64_t>::New(
    "/tensorstore/futures/live", MetricMetadata("Live futures"));
auto& future_ready_callbacks = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/futures/ready_callbacks", MetricMetadata("Ready callbacks"));
auto& future_not_needed_callbacks = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/futures/not_needed_callbacks",
    MetricMetadata("Not needed callbacks"));
auto& future_force_callbacks = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/futures/force_callbacks", MetricMetadata("Force callbacks"));
}  
static CallbackListNode unregister_requested;
struct ABSL_CACHELINE_ALIGNED CacheLineAlignedMutex {
  absl::Mutex mutex{absl::kConstInit};
};
constexpr size_t kNumMutexes = 64;
absl::Mutex* GetMutex(FutureStateBase* ptr) {
  ABSL_CONST_INIT static CacheLineAlignedMutex mutexes[kNumMutexes];
  return &mutexes[absl::HashOf(ptr) % kNumMutexes].mutex;
}
using CallbackListAccessor =
    internal::intrusive_linked_list::MemberAccessor<CallbackListNode>;
namespace {
CallbackPointer MakeUnregisteredCallbackPointer(CallbackBase* callback) {
  assert(callback->reference_count_.load(std::memory_order_relaxed) >= 2);
  callback->next = callback->prev = callback;
  callback->reference_count_.fetch_sub(1, std::memory_order_relaxed);
  return CallbackPointer(callback, internal::adopt_object_ref);
}
}  
CallbackPointer FutureStateBase::RegisterReadyCallback(
    ReadyCallbackBase* callback) {
  assert(callback->reference_count_.load(std::memory_order_relaxed) >= 2);
  {
    absl::MutexLock lock(GetMutex(this));
    future_ready_callbacks.Increment();
    if (!this->ready()) {
      InsertBefore(CallbackListAccessor{}, &ready_callbacks_, callback);
      return CallbackPointer(callback, internal::adopt_object_ref);
    }
  }
  callback->OnReady();
  return MakeUnregisteredCallbackPointer(callback);
}
CallbackPointer FutureStateBase::RegisterNotNeededCallback(
    ResultNotNeededCallbackBase* callback) {
  assert(callback->reference_count_.load(std::memory_order_relaxed) >= 2);
  {
    absl::MutexLock lock(GetMutex(this));
    future_not_needed_callbacks.Increment();
    if (result_needed()) {
      InsertBefore(CallbackListAccessor{}, &promise_callbacks_, callback);
      return CallbackPointer(callback, internal::adopt_object_ref);
    }
  }
  callback->OnResultNotNeeded();
  return MakeUnregisteredCallbackPointer(callback);
}
CallbackPointer FutureStateBase::RegisterForceCallback(
    ForceCallbackBase* callback) {
  assert(callback->reference_count_.load(std::memory_order_relaxed) >= 2);
  auto* mutex = GetMutex(this);
  {
    absl::MutexLock lock(mutex);
    future_force_callbacks.Increment();
    const auto state = state_.load(std::memory_order_acquire);
    if ((state & kResultLocked) != 0 || !has_future()) {
      goto destroy_callback;
    }
    if (state & kForcing) {
      goto already_forced;
    }
    InsertBefore(CallbackListAccessor{}, &promise_callbacks_, callback);
    return CallbackPointer(callback, internal::adopt_object_ref);
  }
already_forced:
  callback->OnForced();
  if (callback->callback_type() == CallbackBase::kLinkCallback) {
    absl::MutexLock lock(mutex);
    if (result_needed()) {
      InsertBefore(CallbackListAccessor{}, &promise_callbacks_, callback);
      return CallbackPointer(callback, internal::adopt_object_ref);
    }
  } else {
    return MakeUnregisteredCallbackPointer(callback);
  }
destroy_callback:
  callback->OnUnregistered();
  return MakeUnregisteredCallbackPointer(callback);
}
CallbackBase::~CallbackBase() {}
void CallbackBase::Unregister(bool block) noexcept {
  auto* shared_state = this->shared_state();
  auto* mutex = GetMutex(shared_state);
  {
    absl::MutexLock lock(mutex);
    if (next == this) {
      return;
    }
    if (next == nullptr || next == &unregister_requested) {
      next = &unregister_requested;
      if (!block || running_callback_thread == std::this_thread::get_id()) {
        return;
      }
      const auto is_done = [&] { return this->next != &unregister_requested; };
      mutex->Await(absl::Condition(&is_done));
      return;
    }
    Remove(CallbackListAccessor{}, this);
    next = this;
  }
  this->OnUnregistered();
  CallbackPointerTraits::decrement(this);
}
FutureStateBase::FutureStateBase()
    : state_(kInitial),
      combined_reference_count_(2),
      promise_reference_count_(1),
      future_reference_count_(1) {
  Initialize(CallbackListAccessor{}, &ready_callbacks_);
  Initialize(CallbackListAccessor{}, &promise_callbacks_);
  live_futures.Increment();
}
namespace {
void NoMorePromiseReferences(FutureStateBase* shared_state) {
  if (shared_state->LockResult()) {
    shared_state->MarkResultWrittenAndCommitResult();
  } else {
    shared_state->CommitResult();
  }
  shared_state->ReleaseCombinedReference();
}
template <typename BeforeUnregisterFunc, typename AfterUnregisterFunc>
inline void RunAndReleaseCallbacks(FutureStateBase* shared_state,
                                   CallbackListNode* head,
                                   BeforeUnregisterFunc before_func,
                                   AfterUnregisterFunc after_func) {
  const auto thread_id = std::this_thread::get_id();
  auto* mutex = GetMutex(shared_state);
  CallbackPointer prev_node;
  while (true) {
    CallbackListNode* next_node;
    {
      absl::MutexLock lock(mutex);
      if (prev_node != nullptr) {
        using Id = std::thread::id;
        prev_node->running_callback_thread.~Id();
        prev_node->next = prev_node.get();
      }
      next_node = head->next;
      if (next_node == head) {
        break;
      }
      Remove(CallbackListAccessor{}, next_node);
      next_node->next = nullptr;
      new (&next_node->running_callback_thread) std::thread::id(thread_id);
    }
    if (prev_node) after_func(prev_node.get());
    prev_node.reset(static_cast<CallbackBase*>(next_node),
                    internal::adopt_object_ref);
    before_func(prev_node.get());
  }
  if (prev_node) after_func(prev_node.get());
}
void RunReadyCallbacks(FutureStateBase* shared_state) {
  RunAndReleaseCallbacks(
      shared_state, &shared_state->ready_callbacks_,
      [](CallbackBase* callback) {
        static_cast<ReadyCallbackBase*>(callback)->OnReady();
      },
      [](CallbackBase* callback) {});
}
void DestroyPromiseCallbacks(FutureStateBase* shared_state) {
  RunAndReleaseCallbacks(
      shared_state, &shared_state->promise_callbacks_,
      [](CallbackBase* callback) {
        if (callback->callback_type() ==
            CallbackBase::kResultNotNeededCallback) {
          static_cast<ResultNotNeededCallbackBase*>(callback)
              ->OnResultNotNeeded();
        }
      },
      [](CallbackBase* callback) {
        if (callback->callback_type() !=
            CallbackBase::kResultNotNeededCallback) {
          callback->OnUnregistered();
        }
      });
}
void RunForceCallbacks(FutureStateBase* shared_state) {
  const auto thread_id = std::this_thread::get_id();
  auto* mutex = GetMutex(shared_state);
  CallbackPointer prev_node;
  CallbackListNode temp_head;
  CallbackListNode* const head = &shared_state->promise_callbacks_;
  while (true) {
    CallbackListNode* next_node;
    {
      absl::MutexLock lock(mutex);
      if (prev_node) {
        using Id = std::thread::id;
        if (prev_node->callback_type() == CallbackBase::kLinkCallback) {
          if (prev_node->next == &unregister_requested) {
            prev_node->next = prev_node.get();
            mutex->Unlock();
            static_cast<CallbackBase*>(prev_node.get())->OnUnregistered();
            mutex->Lock();
          } else {
            prev_node->running_callback_thread.~Id();
            InsertBefore(CallbackListAccessor{}, head, prev_node.release());
          }
        } else {
          assert(prev_node->callback_type() == CallbackBase::kForceCallback);
          prev_node->next = prev_node.get();
        }
      } else {
        temp_head.next = head->next;
        temp_head.next->prev = &temp_head;
        temp_head.prev = head->prev;
        temp_head.prev->next = &temp_head;
        head->next = head->prev = head;
        shared_state->state_.fetch_or(FutureStateBase::kForcing);
      }
      while (true) {
        next_node = temp_head.next;
        if (next_node == &temp_head) return;
        Remove(CallbackListAccessor{}, next_node);
        if (static_cast<CallbackBase*>(next_node)->callback_type() ==
            CallbackBase::kResultNotNeededCallback) {
          InsertBefore(CallbackListAccessor{}, head, next_node);
          continue;
        }
        next_node->next = nullptr;
        new (&next_node->running_callback_thread) std::thread::id(thread_id);
        break;
      }
    }
    prev_node.reset(static_cast<CallbackBase*>(next_node),
                    internal::adopt_object_ref);
    static_cast<ForceCallbackBase*>(prev_node.get())->OnForced();
  }
}
void NoMoreFutureReferences(FutureStateBase* shared_state) {
  DestroyPromiseCallbacks(shared_state);
  shared_state->ReleaseCombinedReference();
}
}  
void FutureStateBase::Force() noexcept {
  StateValue prior_state = kInitial;
  if (!state_.compare_exchange_strong(prior_state, kPreparingToForce)) {
    return;
  }
  RunForceCallbacks(this);
  prior_state = state_.fetch_or(kForced);
  if (prior_state & kResultLocked) {
    DestroyPromiseCallbacks(this);
  }
}
void FutureStateBase::ReleaseFutureReference() {
  if (--future_reference_count_ == 0) {
    NoMoreFutureReferences(this);
  }
}
void FutureStateBase::ReleasePromiseReference() {
  if (--promise_reference_count_ == 0) {
    NoMorePromiseReferences(this);
  }
}
void FutureStateBase::ReleaseCombinedReference() {
  if (--combined_reference_count_ == 0) {
    delete this;
  }
}
bool FutureStateBase::AcquireFutureReference() noexcept {
  auto existing = future_reference_count_.load(std::memory_order_relaxed);
  while (true) {
    if (existing == 0) {
      if ((state_.load(std::memory_order_acquire) & kResultLocked) == 0) {
        return false;
      }
      if (future_reference_count_.fetch_add(1, std::memory_order_acq_rel) ==
          0) {
        combined_reference_count_.fetch_add(1, std::memory_order_relaxed);
      }
      return true;
    }
    if (future_reference_count_.compare_exchange_weak(
            existing, existing + 1, std::memory_order_acq_rel)) {
      return true;
    }
  }
}
bool FutureStateBase::LockResult() noexcept {
  const StateValue prior_state = state_.fetch_or(kResultLocked);
  if (prior_state & kResultLocked) return false;
  if ((prior_state & kForced) != 0 || (prior_state & kPreparingToForce) == 0) {
    DestroyPromiseCallbacks(this);
  } else {
  }
  return true;
}
void FutureStateBase::MarkResultWritten() noexcept {
  const StateValue prior_state = state_.fetch_or(kResultWritten);
  assert(prior_state & kResultLocked);
  assert((prior_state & kResultWritten) == 0);
  if (prior_state & kReady) {
    RunReadyCallbacks(this);
  }
}
bool FutureStateBase::CommitResult() noexcept {
  const StateValue prior_state = state_.fetch_or(kReady);
  if (prior_state & kReady) return false;
  if (prior_state & kResultWritten) {
    RunReadyCallbacks(this);
  }
  return true;
}
void FutureStateBase::MarkResultWrittenAndCommitResult() noexcept {
  [[maybe_unused]] const StateValue prior_state =
      state_.fetch_or(kResultWrittenAndReady);
  assert(prior_state & kResultLocked);
  assert((prior_state & kResultWritten) == 0);
  RunReadyCallbacks(this);
}
bool FutureStateBase::WaitFor(absl::Duration duration) noexcept {
  if (ready()) return true;
  Force();
  absl::Mutex* mutex = GetMutex(this);
  bool is_ready = mutex->LockWhenWithTimeout(
      absl::Condition(this, &FutureStateBase::ready), duration);
  mutex->Unlock();
  return is_ready;
}
bool FutureStateBase::WaitUntil(absl::Time deadline) noexcept {
  if (ready()) return true;
  Force();
  absl::Mutex* mutex = GetMutex(this);
  bool is_ready = mutex->LockWhenWithDeadline(
      absl::Condition(this, &FutureStateBase::ready), deadline);
  mutex->Unlock();
  return is_ready;
}
void FutureStateBase::Wait() noexcept {
  if (ready()) return;
  Force();
  absl::Mutex* mutex = GetMutex(this);
  mutex->LockWhen(absl::Condition(this, &FutureStateBase::ready));
  mutex->Unlock();
}
FutureStateBase::~FutureStateBase() {
  assert(promise_callbacks_.next == &promise_callbacks_);
  assert(ready_callbacks_.next == &ready_callbacks_);
  live_futures.Decrement();
}
}  
ReadyFuture<const void> MakeReadyFuture() {
  static absl::NoDestructor<ReadyFuture<const void>> future{
      MakeReadyFuture<void>(MakeResult())};
  return *future;
}
Future<void> WaitAllFuture(tensorstore::span<const AnyFuture> futures) {
  auto& f = futures;
  switch (f.size()) {
    case 0:
      return MakeReadyFuture<void>(absl::OkStatus());
    case 1:
      return PromiseFuturePair<void>::LinkError(absl::OkStatus(), f[0]).future;
    case 2:
      return PromiseFuturePair<void>::LinkError(absl::OkStatus(), f[0], f[1])
          .future;
    case 3:
      return PromiseFuturePair<void>::LinkError(absl::OkStatus(), f[0], f[1],
                                                f[2])
          .future;
    case 4:
      return PromiseFuturePair<void>::LinkError(absl::OkStatus(), f[0], f[1],
                                                f[2], f[3])
          .future;
    case 5:
      return PromiseFuturePair<void>::LinkError(absl::OkStatus(), f[0], f[1],
                                                f[2], f[3], f[4])
          .future;
    case 6:
      return PromiseFuturePair<void>::LinkError(absl::OkStatus(), f[0], f[1],
                                                f[2], f[3], f[4], f[5])
          .future;
    case 7:
      return PromiseFuturePair<void>::LinkError(absl::OkStatus(), f[0], f[1],
                                                f[2], f[3], f[4], f[5], f[6])
          .future;
    default:
      break;
  }
  auto [promise, result] = PromiseFuturePair<void>::LinkError(
      absl::OkStatus(), f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
  f = f.subspan(8);
  while (f.size() > 8) {
    LinkError(promise, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
    f = f.subspan(8);
  }
  switch (f.size()) {
    case 0:
      return std::move(result);
    case 1:
      LinkError(std::move(promise), f[0]);
      return std::move(result);
    case 2:
      LinkError(std::move(promise), f[0], f[1]);
      return std::move(result);
    case 3:
      LinkError(std::move(promise), f[0], f[1], f[2]);
      return std::move(result);
    case 4:
      LinkError(std::move(promise), f[0], f[1], f[2], f[3]);
      return std::move(result);
    case 5:
      LinkError(std::move(promise), f[0], f[1], f[2], f[3], f[4]);
      return std::move(result);
    case 6:
      LinkError(std::move(promise), f[0], f[1], f[2], f[3], f[4], f[5]);
      return std::move(result);
    case 7:
      LinkError(std::move(promise), f[0], f[1], f[2], f[3], f[4], f[5], f[6]);
      return std::move(result);
    case 8:
      LinkError(std::move(promise), f[0], f[1], f[2], f[3], f[4], f[5], f[6],
                f[7]);
      return std::move(result);
  }
  ABSL_UNREACHABLE();  
}
}  