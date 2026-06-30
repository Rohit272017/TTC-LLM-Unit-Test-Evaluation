#include "tsl/platform/default/unbounded_work_queue.h"
#include "absl/memory/memory.h"
#include "tsl/platform/env.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/numa.h"
namespace tsl {
UnboundedWorkQueue::UnboundedWorkQueue(Env* env, const string& thread_name,
                                       const ThreadOptions& thread_options)
    : env_(env), thread_name_(thread_name), thread_options_(thread_options) {}
UnboundedWorkQueue::~UnboundedWorkQueue() {
  {
    mutex_lock l(work_queue_mu_);
    cancelled_ = true;
    work_queue_cv_.notify_all();
    if (!work_queue_.empty()) {
      LOG(ERROR) << "UnboundedWorkQueue named \"" << thread_name_ << "\" was "
                 << "deleted with pending work in its queue. This may indicate "
                 << "a potential use-after-free bug.";
    }
  }
  {
    mutex_lock l(thread_pool_mu_);
    thread_pool_.clear();
  }
}
void UnboundedWorkQueue::Schedule(WorkFunction fn) {
  mutex_lock l(work_queue_mu_);
  work_queue_.push_back(std::move(fn));
  work_queue_cv_.notify_one();
  if (work_queue_.size() > num_idle_threads_) {
    Thread* new_thread =
        env_->StartThread({}, thread_name_, [this]() { PooledThreadFunc(); });
    mutex_lock l(thread_pool_mu_);
    thread_pool_.emplace_back(new_thread);
  }
}
void UnboundedWorkQueue::PooledThreadFunc() {
  if (thread_options_.numa_node != tsl::port::kNUMANoAffinity) {
    tsl::port::NUMASetThreadNodeAffinity(thread_options_.numa_node);
  }
  while (true) {
    WorkFunction fn;
    {
      mutex_lock l(work_queue_mu_);
      ++num_idle_threads_;
      while (!cancelled_ && work_queue_.empty()) {
        work_queue_cv_.wait(l);
      }
      if (cancelled_) {
        return;
      }
      fn = std::move(work_queue_.front());
      work_queue_.pop_front();
      --num_idle_threads_;
    }
    fn();
  }
}
}  