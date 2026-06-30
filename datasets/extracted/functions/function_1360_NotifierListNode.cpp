#include "xla/tsl/concurrency/async_value.h"
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <utility>
#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/platform/logging.h"
namespace tsl {
class NotifierListNode {
 public:
  explicit NotifierListNode(absl::AnyInvocable<void()> notification)
      : next_(nullptr), notification_(std::move(notification)) {}
 private:
  friend class AsyncValue;
  NotifierListNode* next_;
  absl::AnyInvocable<void()> notification_;
};
uint16_t AsyncValue::CreateTypeInfoAndReturnTypeIdImpl(
    const TypeInfo& type_info) {
  size_t type_id = GetTypeInfoTableSingleton()->emplace_back(type_info) + 1;
  DCHECK(type_id < std::numeric_limits<uint16_t>::max())
      << "Too many different AsyncValue types.";
  return type_id;
}
AsyncValue::TypeInfoTable* AsyncValue::GetTypeInfoTableSingleton() {
  constexpr int kInitialCapacity = 64;
  static auto* type_info_table = new TypeInfoTable(kInitialCapacity);
  return type_info_table;
}
std::atomic<size_t> AsyncValue::total_allocated_async_values_;
void AsyncValue::NotifyAvailable(State available_state) {
  DCHECK((kind() == Kind::kConcrete || kind() == Kind::kIndirect))
      << "Should only be used by ConcreteAsyncValue or IndirectAsyncValue";
  DCHECK(available_state == State::kConcrete ||
         available_state == State::kError);
  auto old_value = waiters_and_state_.exchange(
      WaitersAndState(nullptr, available_state), std::memory_order_acq_rel);
  DCHECK(old_value.state() == State::kUnconstructed ||
         old_value.state() == State::kConstructed);
  RunWaiters(old_value.waiter());
}
void AsyncValue::RunWaiters(NotifierListNode* list) {
  while (list) {
    NotifierListNode* node = list;
    node->notification_();
    list = node->next_;
    delete node;
  }
}
void AsyncValue::EnqueueWaiter(absl::AnyInvocable<void()> waiter,
                               WaitersAndState old_value) {
  auto* node = new NotifierListNode(std::move(waiter));
  auto old_state = old_value.state();
  node->next_ = old_value.waiter();
  auto new_value = WaitersAndState(node, old_state);
  while (!waiters_and_state_.compare_exchange_weak(old_value, new_value,
                                                   std::memory_order_acq_rel,
                                                   std::memory_order_acquire)) {
    if (old_value.state() == State::kConcrete ||
        old_value.state() == State::kError) {
      DCHECK(old_value.waiter() == nullptr);
      node->notification_();
      delete node;
      return;
    }
    node->next_ = old_value.waiter();
  }
  DCHECK(old_value.state() == State::kUnconstructed ||
         old_value.state() == State::kConstructed);
}
void AsyncValue::SetError(absl::Status status) {
  DCHECK(!status.ok());
  if (kind() == Kind::kConcrete) {
    GetTypeInfo().set_error(this, std::move(status));
  } else {
    DCHECK(kind() == Kind::kIndirect);
    auto error_av = MakeErrorAsyncValueRef(std::move(status));
    static_cast<IndirectAsyncValue*>(this)->ForwardTo(std::move(error_av));
  }
}
void IndirectAsyncValue::ForwardTo(RCReference<AsyncValue> value) {
  DCHECK(IsUnavailable());
  auto s = value->state();
  if (s == State::kConcrete || s == State::kError) {
    DCHECK(!value_) << "IndirectAsyncValue::ForwardTo is called more than once";
    auto* concrete_value = value.release();
    if (concrete_value->kind() == Kind::kIndirect) {
      auto* indirect_value = static_cast<IndirectAsyncValue*>(concrete_value);
      concrete_value = indirect_value->value_;
      DCHECK(concrete_value != nullptr);
      DCHECK(concrete_value->kind() == Kind::kConcrete);
      concrete_value->AddRef();
      indirect_value->DropRef();
    }
    DCHECK(type_id_ == kUnknownTypeId || type_id_ == concrete_value->type_id_ ||
           concrete_value->IsType<DummyValueForErrorAsyncValue>())
        << "IndirectAsyncValue::ForwardTo value has an unexpected type id";
    value_ = concrete_value;
    type_id_ = concrete_value->type_id_;
    NotifyAvailable(s);
  } else {
    AsyncValue* av = value.get();
    av->AndThen([self = FormRef(this), value = std::move(value)]() mutable {
      self->ForwardTo(std::move(value));
    });
  }
}
void BlockUntilReady(AsyncValue* async_value) {
  if (ABSL_PREDICT_TRUE(async_value->IsAvailable())) return;
  absl::BlockingCounter cnt(1);
  async_value->AndThen([&] { cnt.DecrementCount(); });
  cnt.Wait();
}
void RunWhenReady(absl::Span<AsyncValue* const> values,
                  absl::AnyInvocable<void()> callee) {
  absl::InlinedVector<AsyncValue*, 4> unavailable_values;
  for (auto i : values) {
    if (!i->IsAvailable()) unavailable_values.push_back(i);
  }
  if (unavailable_values.empty()) return callee();
  if (unavailable_values.size() == 1) {
    unavailable_values[0]->AndThen(
        [callee = std::move(callee)]() mutable { callee(); });
    return;
  }
  struct CounterAndCallee {
    std::atomic<size_t> counter;
    absl::AnyInvocable<void()> callee;
  };
  auto* data =
      new CounterAndCallee{{unavailable_values.size()}, std::move(callee)};
  for (auto* val : unavailable_values) {
    val->AndThen([data]() {
      if (data->counter.fetch_sub(1) != 1) return;
      data->callee();
      delete data;
    });
  }
}
void RunWhenReady(absl::Span<RCReference<AsyncValue> const> values,
                  absl::AnyInvocable<void()> callee) {
  absl::InlinedVector<AsyncValue*, 8> pointers;
  pointers.reserve(values.size());
  for (const auto& ref : values) {
    pointers.push_back(ref.get());
  }
  RunWhenReady(pointers, std::move(callee));
}
}  