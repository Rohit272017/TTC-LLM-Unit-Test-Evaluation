#include "xla/tsl/profiler/backends/cpu/traceme_recorder.h"
#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <atomic>
#include <deque>
#include <optional>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "xla/tsl/profiler/utils/lock_free_queue.h"
#include "xla/tsl/profiler/utils/per_thread.h"
#include "tsl/platform/env.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/types.h"
namespace tsl {
namespace profiler {
namespace internal {
#ifdef _WIN32
#define DECL_DLL_EXPORT __declspec(dllexport)
#else
#define DECL_DLL_EXPORT
#endif
DECL_DLL_EXPORT std::atomic<int> g_trace_level(
    TraceMeRecorder::kTracingDisabled);
static_assert(ATOMIC_INT_LOCK_FREE == 2, "Assumed atomic<int> was lock free");
}  
namespace {
class SplitEventTracker {
 public:
  void AddStart(TraceMeRecorder::Event&& event) {
    DCHECK(event.IsStart());
    start_events_.emplace(event.ActivityId(), std::move(event));
  }
  void AddEnd(TraceMeRecorder::Event* event) {
    DCHECK(event->IsEnd());
    if (!FindStartAndMerge(event)) {
      end_events_.push_back(event);
    }
  }
  void HandleCrossThreadEvents() {
    for (auto* event : end_events_) {
      FindStartAndMerge(event);
    }
  }
 private:
  bool FindStartAndMerge(TraceMeRecorder::Event* event) {
    auto iter = start_events_.find(event->ActivityId());
    if (iter == start_events_.end()) return false;
    auto& start_event = iter->second;
    event->name = std::move(start_event.name);
    event->start_time = start_event.start_time;
    start_events_.erase(iter);
    return true;
  }
  absl::flat_hash_map<int64_t, TraceMeRecorder::Event> start_events_;
  std::vector<TraceMeRecorder::Event*> end_events_;
};
class ThreadLocalRecorder {
 public:
  ThreadLocalRecorder() {
    auto* env = Env::Default();
    info_.tid = env->GetCurrentThreadId();
    env->GetCurrentThreadName(&info_.name);
  }
  const TraceMeRecorder::ThreadInfo& Info() const { return info_; }
  void Record(TraceMeRecorder::Event&& event) { queue_.Push(std::move(event)); }
  void Clear() { queue_.Clear(); }
  TF_MUST_USE_RESULT std::deque<TraceMeRecorder::Event> Consume(
      SplitEventTracker* split_event_tracker) {
    std::deque<TraceMeRecorder::Event> events;
    std::optional<TraceMeRecorder::Event> event;
    while ((event = queue_.Pop())) {
      if (event->IsStart()) {
        split_event_tracker->AddStart(*std::move(event));
        continue;
      }
      events.push_back(*std::move(event));
      if (events.back().IsEnd()) {
        split_event_tracker->AddEnd(&events.back());
      }
    }
    return events;
  }
 private:
  TraceMeRecorder::ThreadInfo info_;
  LockFreeQueue<TraceMeRecorder::Event> queue_;
};
}  
 void TraceMeRecorder::Clear() {
  auto recorders = PerThread<ThreadLocalRecorder>::StartRecording();
  for (auto& recorder : recorders) {
    recorder->Clear();
  };
}
 TraceMeRecorder::Events TraceMeRecorder::Consume() {
  TraceMeRecorder::Events result;
  SplitEventTracker split_event_tracker;
  auto recorders = PerThread<ThreadLocalRecorder>::StopRecording();
  for (auto& recorder : recorders) {
    auto events = recorder->Consume(&split_event_tracker);
    if (!events.empty()) {
      result.push_back({recorder->Info(), std::move(events)});
    }
  };
  split_event_tracker.HandleCrossThreadEvents();
  return result;
}
 bool TraceMeRecorder::Start(int level) {
  level = std::max(0, level);
  int expected = kTracingDisabled;
  bool started = internal::g_trace_level.compare_exchange_strong(
      expected, level, std::memory_order_acq_rel);
  if (started) {
    Clear();
  }
  return started;
}
 void TraceMeRecorder::Record(Event&& event) {
  PerThread<ThreadLocalRecorder>::Get().Record(std::move(event));
}
 TraceMeRecorder::Events TraceMeRecorder::Stop() {
  TraceMeRecorder::Events events;
  if (internal::g_trace_level.exchange(
          kTracingDisabled, std::memory_order_acq_rel) != kTracingDisabled) {
    events = Consume();
  }
  return events;
}
 int64_t TraceMeRecorder::NewActivityId() {
  static std::atomic<int32> thread_counter(1);  
  const thread_local static int32_t thread_id =
      thread_counter.fetch_add(1, std::memory_order_relaxed);
  thread_local static uint32 per_thread_activity_id = 0;
  return static_cast<int64_t>(thread_id) << 32 | per_thread_activity_id++;
}
}  
}  