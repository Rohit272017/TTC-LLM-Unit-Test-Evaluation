#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BASIC_BATCH_SCHEDULER_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BASIC_BATCH_SCHEDULER_H_
#include <stddef.h>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include "tensorflow/core/kernels/batching_util/shared_batch_scheduler.h"
namespace tensorflow {
namespace serving {
template <typename TaskType>
class BasicBatchScheduler : public BatchScheduler<TaskType> {
 public:
  struct Options {
    string thread_pool_name = {"batch_threads"};
    int num_batch_threads = port::MaxParallelism();
    std::shared_ptr<SharedBatchScheduler<TaskType>> shared_batch_scheduler =
        nullptr;
    int max_batch_size = 1000;
    int64_t batch_timeout_micros = 0;
    int max_enqueued_batches = 10;
    bool enable_large_batch_splitting = false;
    std::function<Status(std::unique_ptr<TaskType>* input_task,
                         int first_output_task_size, int input_batch_size_limit,
                         std::vector<std::unique_ptr<TaskType>>* output_tasks)>
        split_input_task_func;
    int max_execution_batch_size = 10;
    Env* env = Env::Default();
  };
  static Status Create(const Options& options,
                       std::function<void(std::unique_ptr<Batch<TaskType>>)>
                           process_batch_callback,
                       std::unique_ptr<BasicBatchScheduler>* scheduler);
  ~BasicBatchScheduler() override = default;
  Status Schedule(std::unique_ptr<TaskType>* task) override;
  size_t NumEnqueuedTasks() const override;
  size_t SchedulingCapacity() const override;
  size_t max_task_size() const override {
    return shared_scheduler_queue_->max_task_size();
  }
 private:
  explicit BasicBatchScheduler(
      std::unique_ptr<BatchScheduler<TaskType>> shared_scheduler_queue);
  std::unique_ptr<BatchScheduler<TaskType>> shared_scheduler_queue_;
  BasicBatchScheduler(const BasicBatchScheduler&) = delete;
  void operator=(const BasicBatchScheduler&) = delete;
};
template <typename TaskType>
Status BasicBatchScheduler<TaskType>::Create(
    const Options& options,
    std::function<void(std::unique_ptr<Batch<TaskType>>)>
        process_batch_callback,
    std::unique_ptr<BasicBatchScheduler>* scheduler) {
  std::shared_ptr<SharedBatchScheduler<TaskType>> shared_scheduler;
  if (options.shared_batch_scheduler == nullptr) {
    typename SharedBatchScheduler<TaskType>::Options shared_scheduler_options;
    shared_scheduler_options.thread_pool_name = options.thread_pool_name;
    shared_scheduler_options.num_batch_threads = options.num_batch_threads;
    shared_scheduler_options.env = options.env;
    TF_RETURN_IF_ERROR(SharedBatchScheduler<TaskType>::Create(
        shared_scheduler_options, &shared_scheduler));
  } else {
    shared_scheduler = options.shared_batch_scheduler;
  }
  typename SharedBatchScheduler<TaskType>::QueueOptions
      shared_scheduler_queue_options;
  shared_scheduler_queue_options.input_batch_size_limit =
      options.max_batch_size;
  shared_scheduler_queue_options.batch_timeout_micros =
      options.batch_timeout_micros;
  shared_scheduler_queue_options.max_enqueued_batches =
      options.max_enqueued_batches;
  shared_scheduler_queue_options.enable_large_batch_splitting =
      options.enable_large_batch_splitting;
  shared_scheduler_queue_options.split_input_task_func =
      options.split_input_task_func;
  shared_scheduler_queue_options.max_execution_batch_size =
      options.max_execution_batch_size;
  std::unique_ptr<BatchScheduler<TaskType>> shared_scheduler_queue;
  TF_RETURN_IF_ERROR(shared_scheduler->AddQueue(shared_scheduler_queue_options,
                                                process_batch_callback,
                                                &shared_scheduler_queue));
  scheduler->reset(
      new BasicBatchScheduler<TaskType>(std::move(shared_scheduler_queue)));
  return absl::OkStatus();
}
template <typename TaskType>
Status BasicBatchScheduler<TaskType>::Schedule(
    std::unique_ptr<TaskType>* task) {
  return shared_scheduler_queue_->Schedule(task);
}
template <typename TaskType>
size_t BasicBatchScheduler<TaskType>::NumEnqueuedTasks() const {
  return shared_scheduler_queue_->NumEnqueuedTasks();
}
template <typename TaskType>
size_t BasicBatchScheduler<TaskType>::SchedulingCapacity() const {
  return shared_scheduler_queue_->SchedulingCapacity();
}
template <typename TaskType>
BasicBatchScheduler<TaskType>::BasicBatchScheduler(
    std::unique_ptr<BatchScheduler<TaskType>> shared_scheduler_queue)
    : shared_scheduler_queue_(std::move(shared_scheduler_queue)) {}
}  
}  
#endif  