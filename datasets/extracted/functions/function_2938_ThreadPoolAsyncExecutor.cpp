#ifndef TENSORFLOW_TSL_PLATFORM_THREADPOOL_ASYNC_EXECUTOR_H_
#define TENSORFLOW_TSL_PLATFORM_THREADPOOL_ASYNC_EXECUTOR_H_
#include <utility>
#include "xla/tsl/concurrency/async_value.h"
#include "tsl/platform/threadpool.h"
namespace tsl::thread {
class ThreadPoolAsyncExecutor : public AsyncValue::Executor {
 public:
  explicit ThreadPoolAsyncExecutor(ThreadPool* thread_pool)
      : thread_pool_(thread_pool) {}
  void Execute(Task task) final {
    auto* task_ptr = new Task(std::move(task));
    thread_pool_->Schedule([task_ptr] {
      (*task_ptr)();
      delete task_ptr;
    });
  }
 private:
  ThreadPool* thread_pool_;
};
}  
#endif  