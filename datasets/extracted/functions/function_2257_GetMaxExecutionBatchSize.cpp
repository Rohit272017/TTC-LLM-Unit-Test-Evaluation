#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_SHARED_BATCH_SCHEDULER_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_SHARED_BATCH_SCHEDULER_H_
#include <stddef.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <list>
#include <memory>
#include <utility>
#include <variant>
#include <vector>
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "tensorflow/core/kernels/batching_util/batch_input_task.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler_utils.h"
#include "tensorflow/core/kernels/batching_util/batch_stats.h"
#include "tensorflow/core/kernels/batching_util/periodic_function.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/connected_traceme.h"
#include "tensorflow/core/profiler/lib/context_types.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
#include "tsl/platform/criticality.h"
#include "tsl/platform/errors.h"
namespace tensorflow {
namespace serving {
namespace internal {
template <typename TaskType>
class Queue;
}  
}  
}  
namespace tensorflow {
namespace serving {
template <typename TaskType>
class SharedBatchScheduler
    : public std::enable_shared_from_this<SharedBatchScheduler<TaskType>> {
 public:
  using BatchTaskUniquePtr = std::unique_ptr<Batch<TaskType>>;
  using ProcessBatchCallback =
      std::variant<std::function<void(BatchTaskUniquePtr)>,
                   std::function<void(BatchTaskUniquePtr,
                                      std::vector<std::unique_ptr<TaskType>>)>>;
  struct Options {
    string thread_pool_name = {"batch_threads"};
    int num_batch_threads = port::MaxParallelism();
    Env* env = Env::Default();
  };
  static Status Create(
      const Options& options,
      std::shared_ptr<SharedBatchScheduler<TaskType>>* scheduler);
  virtual ~SharedBatchScheduler();
  struct QueueOptions {
    size_t input_batch_size_limit = 1000;
    int64_t batch_timeout_micros = 0;
    size_t max_enqueued_batches = 10;
    bool enable_large_batch_splitting = false;
    std::function<Status(std::unique_ptr<TaskType>* input_task,
                         int first_output_task_size, int input_batch_size_limit,
                         std::vector<std::unique_ptr<TaskType>>* output_tasks)>
        split_input_task_func;
    size_t max_execution_batch_size = 1000;
    std::vector<int32> allowed_batch_sizes;
    bool disable_padding = false;
    string batch_padding_policy = string(kPadUpPolicy);
    ModelBatchStats* model_batch_stats = nullptr;
    bool enable_priority_queue = false;
    struct PriorityQueueOptions {
      size_t max_execution_batch_size = 0;
      int64_t batch_timeout_micros = 0;
      size_t input_batch_size_limit = 0;
      size_t max_enqueued_batches = 0;
      std::vector<int32> allowed_batch_sizes;
    };
    PriorityQueueOptions high_priority_queue_options;
    PriorityQueueOptions low_priority_queue_options;
    MixedPriorityBatchingPolicy mixed_priority_batching_policy =
        MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize;
  };
  virtual Status AddQueue(const QueueOptions& options,
                          ProcessBatchCallback process_batch_callback,
                          std::unique_ptr<BatchScheduler<TaskType>>* queue);
 protected:
  explicit SharedBatchScheduler(const Options& options);
 private:
  void GetNextWorkItem_Locked(internal::Queue<TaskType>** queue_for_batch_out,
                              BatchTaskUniquePtr* batch_to_process_out)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void ThreadLogic();
  Status AddQueueAfterRewritingOptions(
      const QueueOptions& options, ProcessBatchCallback process_batch_callback,
      std::unique_ptr<BatchScheduler<TaskType>>* queue);
  static bool BatchExists(const BatchTaskUniquePtr& batch_to_process);
  const Options options_;
  mutex mu_;
  using QueueList = std::list<std::unique_ptr<internal::Queue<TaskType>>>;
  QueueList queues_ TF_GUARDED_BY(mu_);
  typename QueueList::iterator next_queue_to_schedule_ TF_GUARDED_BY(mu_);
  condition_variable schedulable_batch_cv_;
  std::vector<std::unique_ptr<PeriodicFunction>> batch_threads_;
  SharedBatchScheduler(const SharedBatchScheduler&) = delete;
  void operator=(const SharedBatchScheduler&) = delete;
};
namespace internal {
template <typename TaskType>
class Queue {
 public:
  using ProcessBatchCallbackWithoutPaddingTasks =
      std::function<void(std::unique_ptr<Batch<TaskType>>)>;
  using ProcessBatchCallbackWithPaddingTasks =
      std::function<void(std::unique_ptr<Batch<TaskType>>,
                         std::vector<std::unique_ptr<TaskType>>)>;
  using ProcessBatchCallback =
      std::variant<ProcessBatchCallbackWithoutPaddingTasks,
                   ProcessBatchCallbackWithPaddingTasks>;
  using SchedulableBatchCallback = std::function<void()>;
  using SplitInputTaskIntoSubtasksCallback = std::function<Status(
      std::unique_ptr<TaskType>* input_task, int open_batch_remaining_slot,
      int max_execution_batch_size,
      std::vector<std::unique_ptr<TaskType>>* output_tasks)>;
  Queue(const typename SharedBatchScheduler<TaskType>::QueueOptions& options,
        Env* env, ProcessBatchCallback process_batch_callback,
        SchedulableBatchCallback schedulable_batch_callback);
  ~Queue();
  Status Schedule(std::unique_ptr<TaskType>* task);
  size_t NumEnqueuedTasks() const;
  size_t SchedulingCapacity() const;
  size_t max_task_size() const { return options_.input_batch_size_limit; }
  size_t max_execution_batch_size() const { return max_execution_batch_size_; }
  typename SharedBatchScheduler<TaskType>::BatchTaskUniquePtr ScheduleBatch();
  std::vector<std::unique_ptr<TaskType>> GetLowPriorityTasksForPadding(
      size_t batch_size);
  void ProcessBatch(std::unique_ptr<Batch<TaskType>> batch,
                    std::vector<std::unique_ptr<TaskType>> padding_task);
  bool IsEmpty() const;
  void CloseAndWaitUntilEmpty();
  bool closed() const TF_NO_THREAD_SAFETY_ANALYSIS { return closed_.load(); }
 private:
  static size_t GetMaxExecutionBatchSize(
      const typename SharedBatchScheduler<TaskType>::QueueOptions& options) {
    if (options.enable_large_batch_splitting) {
      return options.max_execution_batch_size;
    } else {
      return options.input_batch_size_limit;
    }
  }
  bool IsEmptyInternal() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  bool IsLowPriorityTask(std::unique_ptr<TaskType>* task);
  Status ScheduleWithoutOrEagerSplitImpl(std::unique_ptr<TaskType>* task)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void StartNewBatch() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  Status SplitInputBatchIntoSubtasks(
      std::unique_ptr<TaskType>* input_task,
      std::vector<std::unique_ptr<TaskType>>* output_tasks)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  bool IsOpenBatchSchedulable() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  std::unique_ptr<Batch<TaskType>> ScheduleLowPriorityBatch()
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  size_t SchedulingCapacityInternal() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  Status ValidateBatchTaskQueueCapacity(TaskType* task) const
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  Status ValidateLowPriorityTaskQueueCapacity(const TaskType& task) const
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  size_t tail_batch_task_size() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  int64 num_enqueued_batches() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  std::deque<std::unique_ptr<Batch<TaskType>>>& GetBatches()
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  const std::deque<std::unique_ptr<Batch<TaskType>>>& GetBatches() const
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  TaskQueue<TaskType>& GetLowPriorityTaskQueue()
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  std::vector<std::unique_ptr<TaskType>> GetLowPriorityTasks(size_t size);
  const typename SharedBatchScheduler<TaskType>::QueueOptions options_;
  Env* env_;
  const size_t max_execution_batch_size_;
  ProcessBatchCallback process_batch_callback_;
  SchedulableBatchCallback schedulable_batch_callback_;
  mutable mutex mu_;
  std::atomic<bool> closed_ TF_GUARDED_BY(mu_){false};
  TaskQueue<TaskType> low_priority_tasks_ TF_GUARDED_BY(mu_);
  std::deque<std::unique_ptr<Batch<TaskType>>> low_priority_batches_
      TF_GUARDED_BY(mu_);
  std::deque<std::unique_ptr<Batch<TaskType>>> high_priority_batches_
      TF_GUARDED_BY(mu_);
  uint64 traceme_context_id_counter_ TF_GUARDED_BY(mu_) = 0;
  uint64 open_batch_start_time_micros_ TF_GUARDED_BY(mu_);
  bool schedulable_batch_ TF_GUARDED_BY(mu_) = false;
  int num_batches_being_processed_ TF_GUARDED_BY(mu_) = 0;
  Notification* empty_notification_ TF_GUARDED_BY(mu_) = nullptr;
  Queue(const Queue&) = delete;
  void operator=(const Queue&) = delete;
};
template <typename TaskType>
class QueueHandle : public BatchScheduler<TaskType> {
 public:
  QueueHandle(std::shared_ptr<SharedBatchScheduler<TaskType>> scheduler,
              Queue<TaskType>* queue);
  ~QueueHandle() override;
  Status Schedule(std::unique_ptr<TaskType>* task) override;
  size_t NumEnqueuedTasks() const override;
  size_t SchedulingCapacity() const override;
  size_t max_task_size() const override { return queue_->max_task_size(); }
 private:
  std::shared_ptr<SharedBatchScheduler<TaskType>> scheduler_;
  Queue<TaskType>* queue_;
  QueueHandle(const QueueHandle&) = delete;
  void operator=(const QueueHandle&) = delete;
};
}  
template <typename TaskType>
Status SharedBatchScheduler<TaskType>::Create(
    const Options& options,
    std::shared_ptr<SharedBatchScheduler<TaskType>>* scheduler) {
  if (options.num_batch_threads < 1) {
    return errors::InvalidArgument("num_batch_threads must be positive; was ",
                                   options.num_batch_threads);
  }
  scheduler->reset(new SharedBatchScheduler<TaskType>(options));
  return absl::OkStatus();
}
template <typename TaskType>
SharedBatchScheduler<TaskType>::~SharedBatchScheduler() {
  for (;;) {
    {
      mutex_lock l(mu_);
      if (queues_.empty()) {
        break;
      }
    }
    const int64_t kSleepTimeMicros = 100;
    options_.env->SleepForMicroseconds(kSleepTimeMicros);
  }
  batch_threads_.clear();
}
template <typename TaskType>
Status SharedBatchScheduler<TaskType>::AddQueue(
    const QueueOptions& options, ProcessBatchCallback process_batch_callback,
    std::unique_ptr<BatchScheduler<TaskType>>* queue) {
  QueueOptions rewrite_options = options;
  if ((!rewrite_options.enable_large_batch_splitting) &&
      rewrite_options.max_enqueued_batches == 0) {
    rewrite_options.max_enqueued_batches = 1;
  }
  return AddQueueAfterRewritingOptions(rewrite_options, process_batch_callback,
                                       queue);
}
template <typename TaskType>
Status SharedBatchScheduler<TaskType>::AddQueueAfterRewritingOptions(
    const QueueOptions& options, ProcessBatchCallback process_batch_callback,
    std::unique_ptr<BatchScheduler<TaskType>>* queue) {
  if (options.input_batch_size_limit == 0) {
    return errors::InvalidArgument(
        "input_batch_size_limit must be positive; was ",
        options.input_batch_size_limit);
  }
  if (options.batch_timeout_micros < 0) {
    return errors::InvalidArgument(
        "batch_timeout_micros must be non-negative; was ",
        options.batch_timeout_micros);
  }
  if (options.max_enqueued_batches == 0) {
    return errors::InvalidArgument(
        "max_enqueued_batches must be positive; was ",
        options.max_enqueued_batches);
  }
  if (options.enable_large_batch_splitting &&
      options.split_input_task_func == nullptr) {
    return errors::InvalidArgument(
        "split_input_task_func must be specified when split_input_task is "
        "true: ",
        options.enable_large_batch_splitting);
  }
  if (options.enable_large_batch_splitting &&
      (options.input_batch_size_limit < options.max_execution_batch_size)) {
    return errors::InvalidArgument(
        "When enable_large_batch_splitting is true, input_batch_size_limit "
        "must be "
        "greater than or equal to max_execution_batch_size.",
        options.enable_large_batch_splitting, options.input_batch_size_limit,
        options.max_execution_batch_size);
  }
  auto schedulable_batch_callback = [this] {
    mutex_lock l(mu_);
    schedulable_batch_cv_.notify_one();
  };
  auto internal_queue =
      std::unique_ptr<internal::Queue<TaskType>>(new internal::Queue<TaskType>(
          options, options_.env, process_batch_callback,
          schedulable_batch_callback));
  auto handle = std::unique_ptr<BatchScheduler<TaskType>>(
      new internal::QueueHandle<TaskType>(this->shared_from_this(),
                                          internal_queue.get()));
  {
    mutex_lock l(mu_);
    queues_.push_back(std::move(internal_queue));
    if (next_queue_to_schedule_ == queues_.end()) {
      next_queue_to_schedule_ = queues_.begin();
    }
  }
  *queue = std::move(handle);
  return absl::OkStatus();
}
template <typename TaskType>
SharedBatchScheduler<TaskType>::SharedBatchScheduler(const Options& options)
    : options_(options), next_queue_to_schedule_(queues_.end()) {
  PeriodicFunction::Options periodic_fn_options;
  periodic_fn_options.thread_name_prefix =
      strings::StrCat(options.thread_pool_name, "_");
  for (int i = 0; i < options.num_batch_threads; ++i) {
    std::unique_ptr<PeriodicFunction> thread(new PeriodicFunction(
        [this] { this->ThreadLogic(); },
        0 , periodic_fn_options));
    batch_threads_.push_back(std::move(thread));
  }
}
template <typename TaskType>
bool SharedBatchScheduler<TaskType>::BatchExists(
    const BatchTaskUniquePtr& batch_to_process) {
  return batch_to_process != nullptr;
}
template <typename TaskType>
void SharedBatchScheduler<TaskType>::GetNextWorkItem_Locked(
    internal::Queue<TaskType>** queue_for_batch_out,
    BatchTaskUniquePtr* batch_to_process_out) {
  BatchTaskUniquePtr batch_to_process;
  internal::Queue<TaskType>* queue_for_batch = nullptr;
  const int num_queues = queues_.size();
  for (int num_queues_tried = 0;
       !BatchExists(batch_to_process) && num_queues_tried < num_queues;
       ++num_queues_tried) {
    DCHECK(next_queue_to_schedule_ != queues_.end());
    const bool queue_closed = (*next_queue_to_schedule_)->closed();
    batch_to_process = (*next_queue_to_schedule_)->ScheduleBatch();
    if (BatchExists(batch_to_process)) {
      queue_for_batch = next_queue_to_schedule_->get();
    }
    if (queue_closed && (*next_queue_to_schedule_)->IsEmpty() &&
        !BatchExists(batch_to_process)) {
      DCHECK_NE(queue_for_batch, next_queue_to_schedule_->get());
      next_queue_to_schedule_ = queues_.erase(next_queue_to_schedule_);
    } else {
      ++next_queue_to_schedule_;
    }
    if (next_queue_to_schedule_ == queues_.end() && !queues_.empty()) {
      next_queue_to_schedule_ = queues_.begin();
    }
  }
  *queue_for_batch_out = queue_for_batch;
  *batch_to_process_out = std::move(batch_to_process);
}
template <typename TaskType>
void SharedBatchScheduler<TaskType>::ThreadLogic() {
  BatchTaskUniquePtr batch_to_process;
  internal::Queue<TaskType>* queue_for_batch = nullptr;
  {
    mutex_lock l(mu_);
    while (true) {
      GetNextWorkItem_Locked(&queue_for_batch, &batch_to_process);
      if (BatchExists(batch_to_process)) break;
      const int64_t kTimeoutMillis =
          1;  
      WaitForMilliseconds(&l, &schedulable_batch_cv_, kTimeoutMillis);
      if (queues_.empty()) return;
    }
  }
  size_t batch_size_to_schedule = batch_to_process->size();
  queue_for_batch->ProcessBatch(
      std::move(batch_to_process),
      queue_for_batch->GetLowPriorityTasksForPadding(batch_size_to_schedule));
}
namespace internal {
template <typename TaskType>
Queue<TaskType>::Queue(
    const typename SharedBatchScheduler<TaskType>::QueueOptions& options,
    Env* env, ProcessBatchCallback process_batch_callback,
    SchedulableBatchCallback schedulable_batch_callback)
    : options_(options),
      env_(env),
      max_execution_batch_size_(GetMaxExecutionBatchSize(options_)),
      process_batch_callback_(process_batch_callback),
      schedulable_batch_callback_(schedulable_batch_callback) {
  traceme_context_id_counter_ = (absl::GetCurrentTimeNanos() & 0xFFFFFFFF)
                                << 32;
  GetBatches().emplace_back(new Batch<TaskType>);
}
template <typename TaskType>
Queue<TaskType>::~Queue() {
  mutex_lock l(mu_);
  DCHECK(IsEmptyInternal());
  GetBatches().back()->Close();
}
template <typename TaskType>
bool Queue<TaskType>::IsLowPriorityTask(std::unique_ptr<TaskType>* task) {
  if (!options_.enable_priority_queue) {
    return false;
  }
  if constexpr (std::is_base_of_v<BatchTask, TaskType>) {
    return ((*task)->criticality() ==
                tsl::criticality::Criticality::kSheddablePlus ||
            (*task)->criticality() ==
                tsl::criticality::Criticality::kSheddable);
  }
  return false;
}
template <typename TaskType>
Status Queue<TaskType>::ScheduleWithoutOrEagerSplitImpl(
    std::unique_ptr<TaskType>* task) {
  TF_RETURN_IF_ERROR(ValidateBatchTaskQueueCapacity((*task).get()));
  std::deque<std::unique_ptr<Batch<TaskType>>>& batches = GetBatches();
  const int64_t open_batch_remaining_slot =
      max_execution_batch_size() - batches.back()->size();
  const int64_t input_task_size = (*task)->size();
  std::vector<std::unique_ptr<TaskType>> output_tasks;
  if (input_task_size <= open_batch_remaining_slot ||
      !options_.enable_large_batch_splitting) {
    output_tasks.push_back(std::move(*task));
  } else {
    TF_RETURN_IF_ERROR(SplitInputBatchIntoSubtasks(task, &output_tasks));
  }
  for (int i = 0; i < output_tasks.size(); ++i) {
    if (batches.back()->size() + output_tasks[i]->size() >
        max_execution_batch_size()) {
      StartNewBatch();
    }
    if (batches.back()->empty()) {
      open_batch_start_time_micros_ = env_->NowMicros();
    }
    tsl::profiler::TraceMeProducer trace_me(
        [&output_tasks, i] {
          return profiler::TraceMeEncode("ScheduleOutputTask",
                                         {{"size", output_tasks[i]->size()}});
        },
        tsl::profiler::ContextType::kSharedBatchScheduler,
        batches.back()->traceme_context_id());
    batches.back()->AddTask(std::move(output_tasks[i]));
  }
  return absl::OkStatus();
}
template <typename TaskType>
Status Queue<TaskType>::Schedule(std::unique_ptr<TaskType>* task) {
  const bool large_batch_splitting = options_.enable_large_batch_splitting;
  tsl::profiler::TraceMe trace_me([task, large_batch_splitting] {
    return profiler::TraceMeEncode(
        large_batch_splitting ? "ScheduleWithEagerSplit"
                              : "ScheduleWithoutSplit",
        {{"batching_input_task_size", (*task)->size()}});
  });
  bool notify_of_schedulable_batch = false;
  {
    mutex_lock l(mu_);
    DCHECK(!closed_);
    if (IsLowPriorityTask(task)) {
      TF_RETURN_IF_ERROR(ValidateLowPriorityTaskQueueCapacity(**task));
      low_priority_tasks_.AddTask(std::move(*task), env_->NowMicros());
    } else {
      TF_RETURN_IF_ERROR(ScheduleWithoutOrEagerSplitImpl(task));
    }
    if (!schedulable_batch_) {
      if (GetBatches().size() > 1 || IsOpenBatchSchedulable()) {
        schedulable_batch_ = true;
        notify_of_schedulable_batch = true;
      }
    }
  }
  if (notify_of_schedulable_batch) {
    schedulable_batch_callback_();
  }
  return absl::OkStatus();
}
template <typename TaskType>
size_t Queue<TaskType>::NumEnqueuedTasks() const {
  size_t num_enqueued_tasks = 0;
  mutex_lock l(mu_);
  for (const auto& batch : GetBatches()) {
    num_enqueued_tasks += batch->num_tasks();
  }
  return num_enqueued_tasks + low_priority_tasks_.num_tasks();
}
template <typename TaskType>
size_t Queue<TaskType>::SchedulingCapacity() const {
  mutex_lock l(mu_);
  return SchedulingCapacityInternal();
}
template <typename TaskType>
size_t Queue<TaskType>::SchedulingCapacityInternal() const {
  const int64 num_new_batches_schedulable =
      static_cast<int64_t>(options_.max_enqueued_batches) -
      this->num_enqueued_batches();
  const int64 execution_batch_size_limit = max_execution_batch_size();
  const int64 open_batch_capacity =
      execution_batch_size_limit - this->tail_batch_task_size();
  return (num_new_batches_schedulable * execution_batch_size_limit) +
         open_batch_capacity;
}
template <typename TaskType>
Status Queue<TaskType>::ValidateBatchTaskQueueCapacity(TaskType* task) const {
  if (task->size() > options_.input_batch_size_limit) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Task size %d is larger than maximum input batch size %d", task->size(),
        options_.input_batch_size_limit));
  }
  if (options_.enable_large_batch_splitting) {
    if (task->size() > SchedulingCapacityInternal()) {
      return errors::Unavailable(
          "The batch scheduling queue to which this task was submitted is "
          "full; task size is ",
          task->size(), " but scheduling capacity is only ",
          SchedulingCapacityInternal(),
          " (num_enqueued_batches=", num_enqueued_batches(),
          ", max_enqueued_batches=", options_.max_enqueued_batches,
          ", open_batch_size=", tail_batch_task_size(),
          ", max_execution_batch_size=", max_execution_batch_size(), ")");
    }
    return absl::OkStatus();
  }
  const std::deque<std::unique_ptr<Batch<TaskType>>>& batches = GetBatches();
  if (batches.back()->size() + task->size() > options_.input_batch_size_limit) {
    if (batches.size() >= options_.max_enqueued_batches) {
      return errors::Unavailable(
          "The batch scheduling queue to which this task was submitted is "
          "full; currently ",
          batches.size(), " batches enqueued and max_enqueued_batches is ",
          options_.max_enqueued_batches);
    }
  }
  return absl::OkStatus();
}
template <typename TaskType>
Status Queue<TaskType>::ValidateLowPriorityTaskQueueCapacity(
    const TaskType& task) const {
  if (task.size() >
      options_.low_priority_queue_options.max_execution_batch_size) {
    return absl::UnavailableError(absl::StrFormat(
        "The low priority task queue to which this task was submitted has "
        "max_execution_batch_size=%d and the task size is %d",
        options_.low_priority_queue_options.max_execution_batch_size,
        task.size()));
  }
  if (low_priority_tasks_.size() + task.size() >
      options_.low_priority_queue_options.max_enqueued_batches *
          options_.low_priority_queue_options.max_execution_batch_size) {
    return absl::UnavailableError(absl::StrFormat(
        "The low priority task queue to which this task was submitted does not "
        "have the capcity to handle this task; currently the low priority "
        "queue has %d tasks enqueued and the submitted task size is %d while "
        "max_enqueued_batches=%d and max_execution_batch_size=%d",
        low_priority_tasks_.size(), task.size(),
        options_.low_priority_queue_options.max_enqueued_batches,
        options_.low_priority_queue_options.max_execution_batch_size));
  }
  return absl::OkStatus();
}
template <typename TaskType>
typename SharedBatchScheduler<TaskType>::BatchTaskUniquePtr
Queue<TaskType>::ScheduleBatch() {
  std::unique_ptr<Batch<TaskType>> batch_to_schedule;
  {
    mutex_lock l(mu_);
    std::deque<std::unique_ptr<Batch<TaskType>>>& batches = GetBatches();
    if (batches.size() == 1 && IsOpenBatchSchedulable()) {
      Batch<TaskType>& old_batch = *batches[0];
      std::vector<std::unique_ptr<TaskType>> trimmed_tasks;
      MaybeBatchDown(
           old_batch,
           options_.allowed_batch_sizes,
           options_.disable_padding,
           options_.batch_padding_policy,
           options_.model_batch_stats,
           trimmed_tasks);
      StartNewBatch();
      Batch<TaskType>& new_batch = *batches[1];
      for (std::unique_ptr<TaskType>& task : trimmed_tasks) {
        new_batch.AddTask(std::move(task));
      }
      if (!new_batch.empty()) {
        double position = static_cast<double>(old_batch.size()) /
                          (old_batch.size() + new_batch.size());
        open_batch_start_time_micros_ +=
            (env_->NowMicros() - open_batch_start_time_micros_) * position;
      }
    }
    if (batches.size() >= 2) {
      batch_to_schedule = std::move(batches.front());
      batches.pop_front();
    }
    if (batch_to_schedule == nullptr) {
      batch_to_schedule = ScheduleLowPriorityBatch();
    }
    if (batch_to_schedule == nullptr) {
      schedulable_batch_ = false;
      return batch_to_schedule;
    }
    ++num_batches_being_processed_;
  }
  return batch_to_schedule;
}
template <typename TaskType>
std::vector<std::unique_ptr<TaskType>> Queue<TaskType>::GetLowPriorityTasks(
    size_t size) {
  std::vector<std::unique_ptr<TaskType>> low_priority_tasks_to_pad;
  if (!options_.enable_priority_queue || size == 0)
    return low_priority_tasks_to_pad;
  {
    mutex_lock l(mu_);
    low_priority_tasks_to_pad = GetLowPriorityTaskQueue().RemoveTask(size);
  }
  return low_priority_tasks_to_pad;
}
template <typename TaskType>
std::vector<std::unique_ptr<TaskType>>
Queue<TaskType>::GetLowPriorityTasksForPadding(size_t batch_size) {
  size_t target_batch_size;
  switch (options_.mixed_priority_batching_policy) {
    case MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize:
      target_batch_size = max_execution_batch_size();
      break;
    case MixedPriorityBatchingPolicy::
        kLowPriorityPaddingWithNextAllowedBatchSize:
      target_batch_size = GetNextAllowedBatchSize(
          batch_size, options_.allowed_batch_sizes, options_.disable_padding);
      break;
    default:
      target_batch_size = 0;
      break;
  }
  if (target_batch_size <= batch_size) {
    return {};
  }
  return GetLowPriorityTasks(target_batch_size - batch_size);
}
template <typename TaskType>
void Queue<TaskType>::ProcessBatch(
    std::unique_ptr<Batch<TaskType>> batch,
    std::vector<std::unique_ptr<TaskType>> padding_task) {
  tsl::profiler::TraceMeConsumer trace_me(
      [&] {
        return profiler::TraceMeEncode(
            "ProcessBatch", {{"batch_size_before_padding", batch->size()},
                             {"_r", 2} });
      },
      tsl::profiler::ContextType::kSharedBatchScheduler,
      batch->traceme_context_id());
  if (std::holds_alternative<ProcessBatchCallbackWithoutPaddingTasks>(
          process_batch_callback_)) {
    std::get<ProcessBatchCallbackWithoutPaddingTasks>(process_batch_callback_)(
        std::move(batch));
  } else {
    std::get<ProcessBatchCallbackWithPaddingTasks>(process_batch_callback_)(
        std::move(batch), std::move(padding_task));
  }
  {
    mutex_lock l(mu_);
    --num_batches_being_processed_;
    if (empty_notification_ != nullptr && IsEmptyInternal()) {
      empty_notification_->Notify();
    }
  }
}
template <typename TaskType>
bool Queue<TaskType>::IsEmpty() const {
  mutex_lock l(mu_);
  return IsEmptyInternal();
}
template <typename TaskType>
void Queue<TaskType>::CloseAndWaitUntilEmpty() {
  Notification empty;
  {
    mutex_lock l(mu_);
    closed_ = true;
    if (IsEmptyInternal()) {
      empty.Notify();
    } else {
      empty_notification_ = &empty;
    }
  }
  empty.WaitForNotification();
}
template <typename TaskType>
bool Queue<TaskType>::IsEmptyInternal() const {
  const std::deque<std::unique_ptr<Batch<TaskType>>>& batches = GetBatches();
  return num_batches_being_processed_ == 0 && batches.size() == 1 &&
         batches.back()->empty() && low_priority_tasks_.empty();
}
template <typename TaskType>
void Queue<TaskType>::StartNewBatch() {
  std::deque<std::unique_ptr<Batch<TaskType>>>& batches = GetBatches();
  batches.back()->Close();
  batches.emplace_back(new Batch<TaskType>(++traceme_context_id_counter_));
}
template <typename TaskType>
Status Queue<TaskType>::SplitInputBatchIntoSubtasks(
    std::unique_ptr<TaskType>* input_task,
    std::vector<std::unique_ptr<TaskType>>* output_tasks) {
  const int open_batch_remaining_slot =
      max_execution_batch_size() - this->tail_batch_task_size();
  return options_.split_input_task_func(
      std::move(input_task), open_batch_remaining_slot,
      max_execution_batch_size(), std::move(output_tasks));
}
template <typename TaskType>
bool Queue<TaskType>::IsOpenBatchSchedulable() const {
  Batch<TaskType>* open_batch = GetBatches().back().get();
  if (open_batch->empty()) {
    return false;
  }
  return closed_ || open_batch->size() >= max_execution_batch_size() ||
         env_->NowMicros() >=
             open_batch_start_time_micros_ + options_.batch_timeout_micros;
}
template <typename TaskType>
std::unique_ptr<Batch<TaskType>> Queue<TaskType>::ScheduleLowPriorityBatch() {
  std::unique_ptr<Batch<TaskType>> batch_to_schedule;
  if (!options_.enable_priority_queue || low_priority_tasks_.empty()) {
    return batch_to_schedule;
  }
  if (env_->NowMicros() <
          *low_priority_tasks_.EarliestTaskStartTime() +
              options_.low_priority_queue_options.batch_timeout_micros &&
      low_priority_tasks_.size() <
          options_.low_priority_queue_options.max_execution_batch_size) {
    return batch_to_schedule;
  }
  if (!GetBatches().empty() && !GetBatches().front()->empty()) {
    return batch_to_schedule;
  }
  batch_to_schedule = std::make_unique<Batch<TaskType>>();
  for (std::unique_ptr<TaskType>& task : low_priority_tasks_.RemoveTask(
           options_.low_priority_queue_options.max_execution_batch_size)) {
    batch_to_schedule->AddTask(std::move(task));
  }
  batch_to_schedule->Close();
  return batch_to_schedule;
}
template <typename TaskType>
size_t Queue<TaskType>::tail_batch_task_size() const {
  return GetBatches().back()->size();
}
template <typename TaskType>
int64 Queue<TaskType>::num_enqueued_batches() const {
  return GetBatches().size();
}
template <typename TaskType>
std::deque<std::unique_ptr<Batch<TaskType>>>& Queue<TaskType>::GetBatches() {
  return high_priority_batches_;
}
template <typename TaskType>
const std::deque<std::unique_ptr<Batch<TaskType>>>&
Queue<TaskType>::GetBatches() const {
  return high_priority_batches_;
}
template <typename TaskType>
TaskQueue<TaskType>& Queue<TaskType>::GetLowPriorityTaskQueue() {
  return low_priority_tasks_;
}
template <typename TaskType>
QueueHandle<TaskType>::QueueHandle(
    std::shared_ptr<SharedBatchScheduler<TaskType>> scheduler,
    Queue<TaskType>* queue)
    : scheduler_(scheduler), queue_(queue) {}
template <typename TaskType>
QueueHandle<TaskType>::~QueueHandle() {
  queue_->CloseAndWaitUntilEmpty();
}
template <typename TaskType>
Status QueueHandle<TaskType>::Schedule(std::unique_ptr<TaskType>* task) {
  return queue_->Schedule(task);
}
template <typename TaskType>
size_t QueueHandle<TaskType>::NumEnqueuedTasks() const {
  return queue_->NumEnqueuedTasks();
}
template <typename TaskType>
size_t QueueHandle<TaskType>::SchedulingCapacity() const {
  return queue_->SchedulingCapacity();
}
}  
}  
}  
#endif  