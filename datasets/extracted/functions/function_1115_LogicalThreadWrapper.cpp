#include "tensorflow/core/data/unbounded_thread_pool.h"
#include <functional>
#include <memory>
#include <utility>
#include "absl/memory/memory.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/resource.h"
#include "tensorflow/core/platform/unbounded_work_queue.h"
namespace tensorflow {
namespace data {
class UnboundedThreadPool::LogicalThreadWrapper : public Thread {
 public:
  explicit LogicalThreadWrapper(std::shared_ptr<Notification> done)
      : done_(std::move(done)) {}
  ~LogicalThreadWrapper() override {
    done_->WaitForNotification();
  }
 private:
  std::shared_ptr<Notification> done_;
};
class UnboundedThreadPool::LogicalThreadFactory : public ThreadFactory {
 public:
  explicit LogicalThreadFactory(UnboundedThreadPool* pool) : pool_(pool) {}
  std::unique_ptr<Thread> StartThread(const string& name,
                                      std::function<void()> fn) override {
    auto done = std::make_shared<Notification>();
    pool_->ScheduleOnWorkQueue(std::move(fn), done);
    return std::make_unique<LogicalThreadWrapper>(std::move(done));
  }
 private:
  UnboundedThreadPool* const pool_;  
};
std::shared_ptr<ThreadFactory> UnboundedThreadPool::get_thread_factory() {
  return std::make_shared<LogicalThreadFactory>(this);
}
void UnboundedThreadPool::Schedule(std::function<void()> fn) {
  auto tagged_fn = [fn = std::move(fn)]() {
    tensorflow::ResourceTagger tag(kTFDataResourceTag, "ThreadPool");
    fn();
  };
  ScheduleOnWorkQueue(std::move(tagged_fn), nullptr);
}
int UnboundedThreadPool::NumThreads() const { return -1; }
int UnboundedThreadPool::CurrentThreadId() const { return -1; }
namespace {
void WorkQueueFunc(const std::function<void()>& fn,
                   std::shared_ptr<Notification> done) {
  fn();
  if (done) {
    done->Notify();
  }
}
}  
void UnboundedThreadPool::ScheduleOnWorkQueue(
    std::function<void()> fn, std::shared_ptr<Notification> done) {
  unbounded_work_queue_.Schedule(
      std::bind(&WorkQueueFunc, std::move(fn), std::move(done)));
}
}  
}  