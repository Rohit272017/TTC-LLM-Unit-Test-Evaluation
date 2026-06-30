#include "tensorflow/core/common_runtime/device/device_event_mgr.h"
#include <functional>
#include <memory>
#include <utility>
#include "tensorflow/core/platform/stacktrace.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tsl/platform/thread_annotations.h"
namespace tensorflow {
namespace {
static const int kNumThreads = 2;
}  
namespace device_event_mgr {
class ThreadLabel {
 public:
  static const char* GetValue() { return value_; }
  static void SetValue(const char* v) { value_ = v; }
 private:
  static thread_local const char* value_;
};
thread_local const char* ThreadLabel::value_ = "";
void WarnIfInCallback(std::function<void()> f) {
  const char* label = ThreadLabel::GetValue();
  if (label && !strcmp(label, "device_event_mgr")) {
    if (f) {
      f();
    } else {
      LOG(WARNING) << "Executing inside EventMgr callback thread: "
                   << CurrentStackTrace();
    }
  }
}
void InitThreadpoolLabels(thread::ThreadPool* threadpool) {
  static const char* label = "device_event_mgr";
  mutex mu;
  int init_count = 0;
  condition_variable all_initialized;
  int exit_count = 0;
  condition_variable ready_to_exit;
  const int num_threads = threadpool->NumThreads();
  for (int i = 0; i < num_threads; ++i) {
    threadpool->Schedule([num_threads, &mu, &init_count, &all_initialized,
                          &exit_count, &ready_to_exit]() {
      device_event_mgr::ThreadLabel::SetValue(label);
      mutex_lock l(mu);
      ++init_count;
      if (init_count == num_threads) {
        all_initialized.notify_all();
      }
      while (init_count < num_threads) {
        all_initialized.wait(l);
      }
      if (++exit_count == num_threads) {
        ready_to_exit.notify_all();
      }
    });
  }
  {
    mutex_lock l(mu);
    while (exit_count < num_threads) {
      ready_to_exit.wait(l);
    }
  }
}
}  
EventMgr::EventMgr(se::StreamExecutor* se, const GPUOptions& gpu_options)
    : exec_(se),
      polling_active_delay_usecs_(gpu_options.polling_active_delay_usecs()
                                      ? gpu_options.polling_active_delay_usecs()
                                      : 10),
      threadpool_(Env::Default(), "Device_Event_Manager", kNumThreads) {
  device_event_mgr::InitThreadpoolLabels(&threadpool_);
  StartPollingLoop();
}
EventMgr::~EventMgr() {
  StopPollingLoop();
  for (auto& [stream, stream_callbacks] : callbacks_) {
    for (auto& [event, callback] : stream_callbacks) {
      threadpool_.Schedule(std::move(callback));
    }
  }
}
void EventMgr::StartPollingLoop() {
  CHECK(polling_stopped_ == nullptr);
  {
    mutex_lock l(mu_);
    stop_polling_ = false;
  }
  polling_stopped_ = std::make_unique<Notification>();
  threadpool_.Schedule([this]() { PollLoop(); });
}
void EventMgr::StopPollingLoop() {
  if (polling_stopped_) {
    {
      mutex_lock l(mu_);
      stop_polling_ = true;
      events_pending_.notify_all();
    }
    polling_stopped_->WaitForNotification();
    polling_stopped_.reset(nullptr);
  }
}
void EventMgr::PollLoop() {
  ToFreeVector to_free;
  while (true) {
    bool events_still_pending;
    {
      mutex_lock l(mu_);
      if (stop_polling_) {
        break;
      }
      if (callbacks_.empty()) {
        events_pending_.wait(l);
      }
      PollEvents(nullptr, &to_free);  
      events_still_pending = !callbacks_.empty();
    }
    FreeMemory(to_free);
    to_free.clear();
    if (events_still_pending) {
      Env::Default()->SleepForMicroseconds(polling_active_delay_usecs_);
    }
  }
  polling_stopped_->Notify();
}
void EventMgr::EnqueueCallback(se::Stream* stream, std::function<void()> func) {
  VLOG(2) << "EnqueueCallback with one or more callbacks pending on "
          << callbacks_.size() << " streams and " << free_events_.size()
          << " unused event objects.";
  if (free_events_.empty()) {
    free_events_.emplace_back(exec_->CreateEvent().value());
  }
  std::unique_ptr<se::Event> e = std::move(free_events_.back());
  free_events_.pop_back();
  stream->RecordEvent(e.get()).IgnoreError();
  bool was_empty = callbacks_.empty();
  callbacks_[stream].push_back({std::move(e), std::move(func)});
  if (was_empty) {
    events_pending_.notify_all();
  }
}
void EventMgr::PollEvents(se::Stream* stream,
                          absl::InlinedVector<InUse, 4UL>* to_free) {
  VLOG(2) << "PollEvents with one or more callbacks pending on "
          << callbacks_.size() << " streams and " << free_events_.size()
          << " unused event objects.";
  auto poll_events_for_stream_it =
      [&](auto& stream_it) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        auto& stream_callbacks = stream_it->second;
        auto it = stream_callbacks.begin();
        while (it != stream_callbacks.end()) {
          auto& [event, callback] = *it;
          se::Event::Status s = event->PollForStatus();
          bool keep_looping = true;
          switch (s) {
            case se::Event::Status::kUnknown:
            case se::Event::Status::kError:
              LOG(FATAL) << "Unexpected Event status: " << static_cast<int>(s);
              break;
            case se::Event::Status::kPending:
              keep_looping = false;
              break;
            case se::Event::Status::kComplete:
              free_events_.push_back(std::move(event));
              to_free->push_back({nullptr, std::move(callback)});
              ++it;
              break;
          }
          if (!keep_looping) {
            break;
          }
        }
        stream_callbacks.erase(stream_callbacks.begin(), it);
        if (stream_callbacks.empty()) {
          callbacks_.erase(stream_it++);
        } else {
          stream_it++;
        }
      };
  if (stream != nullptr) {
    auto stream_it = callbacks_.find(stream);
    if (stream_it != callbacks_.end()) {
      poll_events_for_stream_it(stream_it);
    }
  } else {
    for (auto stream_it = callbacks_.begin(); stream_it != callbacks_.end();) {
      poll_events_for_stream_it(stream_it);
    }
  }
}
EventMgrFactory* EventMgrFactory::Singleton() {
  static EventMgrFactory* instance = new EventMgrFactory;
  return instance;
}
EventMgr* EventMgrFactory::GetEventMgr(se::StreamExecutor* se,
                                       const GPUOptions& gpu_options) {
  mutex_lock l(mu_);
  auto itr = event_mgr_map_.find(se);
  if (itr == event_mgr_map_.end()) {
    auto event_mgr = new EventMgr(se, gpu_options);
    event_mgr_map_[se] = event_mgr;
    return event_mgr;
  } else {
    return itr->second;
  }
}
}  