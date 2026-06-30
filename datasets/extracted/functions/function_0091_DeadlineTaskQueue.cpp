#include "tensorstore/internal/thread/schedule_at.h"
#include <stdint.h>
#include <algorithm>
#include <atomic>
#include <iterator>
#include <memory>
#include <utility>
#include "absl/base/attributes.h"
#include "absl/base/no_destructor.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/compare.h"
#include "tensorstore/internal/container/intrusive_red_black_tree.h"
#include "tensorstore/internal/metrics/gauge.h"
#include "tensorstore/internal/metrics/histogram.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/metrics/value.h"
#include "tensorstore/internal/tagged_ptr.h"
#include "tensorstore/internal/thread/thread.h"
#include "tensorstore/internal/tracing/tracing.h"
#include "tensorstore/util/stop_token.h"
using ::tensorstore::internal_metrics::MetricMetadata;
namespace tensorstore {
namespace internal {
namespace {
using ScheduleAtTask = absl::AnyInvocable<void() &&>;
auto& schedule_at_queued_ops = internal_metrics::Gauge<int64_t>::New(
    "/tensorstore/internal/thread/schedule_at/queued_ops",
    MetricMetadata("Operations in flight on the schedule_at thread"));
auto& schedule_at_next_event = internal_metrics::Value<absl::Time>::New(
    "/tensorstore/internal/thread/schedule_at/next_event",
    MetricMetadata("Time of the next in-flight schedule_at operation"));
auto& schedule_at_insert_histogram_ms =
    internal_metrics::Histogram<internal_metrics::DefaultBucketer>::New(
        "/tensorstore/internal/thread/schedule_at/insert_histogram_ms",
        MetricMetadata("Histogram of schedule_at insert delays (ms)",
                       internal_metrics::Units::kMilliseconds));
class DeadlineTaskQueue;
using TaggedQueuePointer = TaggedPtr<DeadlineTaskQueue, 1>;
struct DeadlineTaskNode;
using DeadlineTaskTree = intrusive_red_black_tree::Tree<DeadlineTaskNode>;
struct DeadlineTaskStopCallback {
  DeadlineTaskNode& node;
  void operator()() const;
};
struct DeadlineTaskNode : public DeadlineTaskTree::NodeBase {
  DeadlineTaskNode(absl::Time deadline, ScheduleAtTask&& task,
                   const StopToken& token)
      : deadline(deadline),
        task(std::move(task)),
        trace_context(internal_tracing::TraceContext::kThread),
        queue(TaggedQueuePointer{}),
        stop_callback(token, DeadlineTaskStopCallback{*this}) {}
  void RunAndDelete();
  absl::Time deadline;
  ScheduleAtTask task;
  ABSL_ATTRIBUTE_NO_UNIQUE_ADDRESS internal_tracing::TraceContext trace_context;
  std::atomic<TaggedQueuePointer> queue;
  StopCallback<DeadlineTaskStopCallback> stop_callback;
};
using RunImmediatelyQueueAccessor =
    intrusive_red_black_tree::LinkedListAccessor<DeadlineTaskNode>;
class DeadlineTaskQueue {
 public:
  explicit DeadlineTaskQueue()
      : run_immediately_queue_(nullptr),
        next_wakeup_(absl::InfinitePast()),
        woken_up_(absl::InfinitePast()),
        thread_({"TensorstoreScheduleAt"}, &DeadlineTaskQueue::Run, this) {}
  ~DeadlineTaskQueue() { ABSL_UNREACHABLE(); }  
  void ScheduleAt(absl::Time target_time, ScheduleAtTask task,
                  const StopToken& stop_token);
  void Run();
 private:
  friend struct DeadlineTaskNode;
  friend struct DeadlineTaskStopCallback;
  void TryRemove(DeadlineTaskNode& node);
  absl::Mutex mutex_;
  absl::CondVar cond_var_;
  DeadlineTaskTree tree_ ABSL_GUARDED_BY(mutex_);
  DeadlineTaskNode* run_immediately_queue_ ABSL_GUARDED_BY(mutex_);
  absl::Time next_wakeup_ ABSL_GUARDED_BY(mutex_);
  absl::Time woken_up_ ABSL_GUARDED_BY(mutex_);
  Thread thread_;
};
void DeadlineTaskQueue::ScheduleAt(absl::Time target_time, ScheduleAtTask task,
                                   const StopToken& stop_token) {
  schedule_at_queued_ops.Increment();
  schedule_at_insert_histogram_ms.Observe(
      absl::ToInt64Milliseconds(target_time - absl::Now()));
  auto node = std::make_unique<DeadlineTaskNode>(target_time, std::move(task),
                                                 stop_token);
  absl::MutexLock l(&mutex_);
  auto tagged_queue_ptr = node->queue.exchange(TaggedQueuePointer(this));
  if (tagged_queue_ptr.tag()) {
    return;
  }
  if (target_time <= woken_up_) {
    RunImmediatelyQueueAccessor{}.SetNext(node.get(), nullptr);
    if (run_immediately_queue_) {
      RunImmediatelyQueueAccessor{}.SetNext(
          RunImmediatelyQueueAccessor{}.GetPrev(run_immediately_queue_),
          node.get());
      RunImmediatelyQueueAccessor{}.SetPrev(run_immediately_queue_, node.get());
    } else {
      run_immediately_queue_ = node.get();
      RunImmediatelyQueueAccessor{}.SetPrev(node.get(), node.get());
    }
    if (next_wakeup_ != absl::InfinitePast()) {
      next_wakeup_ = absl::InfinitePast();
      cond_var_.Signal();
    }
    node.release();
    return;
  }
  tree_.FindOrInsert(
      [&](DeadlineTaskNode& other) {
        return target_time < other.deadline ? absl::weak_ordering::less
                                            : absl::weak_ordering::greater;
      },
      [&] { return node.release(); });
  if (target_time < next_wakeup_) {
    next_wakeup_ = target_time;
    cond_var_.Signal();
  }
}
void DeadlineTaskQueue::Run() {
  while (true) {
    DeadlineTaskTree runnable;
    DeadlineTaskNode* run_immediately = nullptr;
    {
      absl::MutexLock l(&mutex_);
      do {
        run_immediately = std::exchange(run_immediately_queue_, nullptr);
        if (!run_immediately) {
          next_wakeup_ =
              tree_.empty() ? absl::InfiniteFuture() : tree_.begin()->deadline;
          schedule_at_next_event.Set(next_wakeup_);
          cond_var_.WaitWithDeadline(&mutex_, next_wakeup_);
        }
        auto woken_up = woken_up_ = std::max(woken_up_, absl::Now());
        auto split_result = tree_.FindSplit([&](DeadlineTaskNode& node) {
          return node.deadline <= woken_up ? absl::weak_ordering::greater
                                           : absl::weak_ordering::less;
        });
        runnable = std::move(split_result.trees[0]);
        tree_ = std::move(split_result.trees[1]);
      } while (runnable.empty() && !run_immediately);
      next_wakeup_ = absl::InfinitePast();
    }  
    internal_tracing::TraceContext base =
        internal_tracing::TraceContext(internal_tracing::TraceContext::kThread);
    while (run_immediately) {
      auto* next = RunImmediatelyQueueAccessor{}.GetNext(run_immediately);
      run_immediately->RunAndDelete();
      run_immediately = next;
    }
    for (DeadlineTaskTree::iterator it = runnable.begin(), next;
         it != runnable.end(); it = next) {
      next = std::next(it);
      runnable.Remove(*it);
      it->RunAndDelete();
    }
    internal_tracing::SwapCurrentTraceContext(&base);
  }
}
void DeadlineTaskNode::RunAndDelete() {
  schedule_at_queued_ops.Decrement();
  if (queue.load(std::memory_order_relaxed).tag()) {
  } else {
    internal_tracing::SwapCurrentTraceContext(&trace_context);
    std::move(task)();
  }
  delete this;
}
void DeadlineTaskStopCallback::operator()() const {
  auto tagged_queue_ptr = node.queue.exchange(TaggedQueuePointer{nullptr, 1});
  auto* queue_ptr = tagged_queue_ptr.get();
  if (!queue_ptr) {
    return;
  }
  queue_ptr->TryRemove(node);
}
void DeadlineTaskQueue::TryRemove(DeadlineTaskNode& node) {
  {
    absl::MutexLock lock(&mutex_);
    if (node.deadline <= woken_up_) {
      return;
    }
    tree_.Remove(node);
  }
  delete &node;
  schedule_at_queued_ops.Decrement();
}
}  
void ScheduleAt(absl::Time target_time, ScheduleAtTask task,
                const StopToken& stop_token) {
  static absl::NoDestructor<DeadlineTaskQueue> g_queue;
  g_queue->ScheduleAt(std::move(target_time), std::move(task), stop_token);
}
}  
}  