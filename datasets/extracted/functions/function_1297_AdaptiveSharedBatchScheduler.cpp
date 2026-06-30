#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_ADAPTIVE_SHARED_BATCH_SCHEDULER_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_ADAPTIVE_SHARED_BATCH_SCHEDULER_H_
#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>
#include "absl/types/optional.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/periodic_function.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/connected_traceme.h"
namespace tensorflow {
namespace serving {
namespace internal {
template <typename TaskType>
class ASBSBatch;
template <typename TaskType>
class ASBSQueue;
}  
template <typename TaskType>
class AdaptiveSharedBatchScheduler
    : public std::enable_shared_from_this<
          AdaptiveSharedBatchScheduler<TaskType>> {
 public:
  ~AdaptiveSharedBatchScheduler() {
    if (owned_batch_thread_pool_) {
      delete batch_thread_pool_;
    }
  }
  struct Options {
    string thread_pool_name = {"batch_threads"};
    int64_t num_batch_threads = port::MaxParallelism();
    thread::ThreadPool* thread_pool = nullptr;
    int64_t min_in_flight_batches_limit = 1;
    int64_t full_batch_scheduling_boost_micros = 0;
    Env* env = Env::Default();
    double initial_in_flight_batches_limit = 3;
    int64_t batches_to_average_over = 1000;
    bool fifo_scheduling = false;
  };
  static Status Create(
      const Options& options,
      std::shared_ptr<AdaptiveSharedBatchScheduler<TaskType>>* scheduler);
  struct QueueOptions {
    int max_batch_size = 1000;
    absl::optional<int> max_input_task_size = absl::nullopt;
    absl::optional<int> max_tasks_per_batch = absl::nullopt;
    int max_enqueued_batches = 10;
    int64_t batch_timeout_micros = 0;
    std::function<Status(std::unique_ptr<TaskType>* input_task, int first_size,
                         int max_batch_size,
                         std::vector<std::unique_ptr<TaskType>>* output_tasks)>
        split_input_task_func;
    bool disable_padding = false;
  };
  using BatchProcessor = std::function<void(std::unique_ptr<Batch<TaskType>>)>;
  Status AddQueue(const QueueOptions& options,
                  BatchProcessor process_batch_callback,
                  std::unique_ptr<BatchScheduler<TaskType>>* queue);
  double in_flight_batches_limit() {
    mutex_lock l(mu_);
    return in_flight_batches_limit_;
  }
 private:
  friend class internal::ASBSQueue<TaskType>;
  explicit AdaptiveSharedBatchScheduler(const Options& options);
  void CallbackWrapper(const internal::ASBSBatch<TaskType>* batch,
                       BatchProcessor callback, bool is_express);
  void MaybeScheduleNextBatch() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void MaybeScheduleNextBatchFIFO() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void MaybeScheduleClosedBatches();
  void MaybeScheduleClosedBatchesLocked() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void MaybeScheduleClosedBatchesLockedFIFO() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void MaybeAdjustInflightLimit() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void AddBatch(const internal::ASBSBatch<TaskType>* batch);
  void RemoveQueue(const internal::ASBSQueue<TaskType>* queue);
  Env* GetEnv() const { return options_.env; }
  const Options options_;
  std::vector<const internal::ASBSBatch<TaskType>*> batches_ TF_GUARDED_BY(mu_);
  std::deque<const internal::ASBSBatch<TaskType>*> fifo_batches_
      TF_GUARDED_BY(mu_);
  std::unordered_map<const internal::ASBSQueue<TaskType>*, BatchProcessor>
      queues_and_callbacks_ TF_GUARDED_BY(mu_);
  mutex mu_;
  thread::ThreadPool* batch_thread_pool_;
  bool owned_batch_thread_pool_ = false;
  double in_flight_batches_limit_ TF_GUARDED_BY(mu_);
  int64_t in_flight_batches_ TF_GUARDED_BY(mu_) = 0;
  int64_t in_flight_express_batches_ TF_GUARDED_BY(mu_) = 0;
  std::default_random_engine rand_engine_;
  std::uniform_real_distribution<double> rand_double_;
  int64_t batch_count_ TF_GUARDED_BY(mu_) = 0;
  struct DelayStats {
    int64_t batch_latency_sum = 0;
    double last_avg_latency_ms = 0;
    bool last_latency_decreased = false;
    int step_direction = 1;
  };
  DelayStats batch_delay_stats_ TF_GUARDED_BY(mu_);
  constexpr static double kMaxStepSizeMultiplier = 0.125;  
  constexpr static double kMinStepSizeMultiplier = 0.0078125;  
  double step_size_multiplier_ TF_GUARDED_BY(mu_) = kMaxStepSizeMultiplier;
  AdaptiveSharedBatchScheduler(const AdaptiveSharedBatchScheduler&) = delete;
  void operator=(const AdaptiveSharedBatchScheduler&) = delete;
};
namespace internal {
template <typename TaskType>
class ASBSQueue : public BatchScheduler<TaskType> {
 public:
  using QueueOptions =
      typename AdaptiveSharedBatchScheduler<TaskType>::QueueOptions;
  ASBSQueue(std::shared_ptr<AdaptiveSharedBatchScheduler<TaskType>> scheduler,
            const QueueOptions& options);
  ~ASBSQueue() override;
  Status Schedule(std::unique_ptr<TaskType>* task) override;
  size_t NumEnqueuedTasks() const override;
  size_t SchedulingCapacity() const override;
  void ReleaseBatch(const ASBSBatch<TaskType>* batch);
  size_t max_task_size() const override { return options_.max_batch_size; }
 private:
  size_t SchedulingCapacityLocked() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  static uint64 NewTraceMeContextIdForBatch();
  std::shared_ptr<AdaptiveSharedBatchScheduler<TaskType>> scheduler_;
  const QueueOptions options_;
  ASBSBatch<TaskType>* current_batch_ TF_GUARDED_BY(mu_) = nullptr;
  int64_t num_enqueued_batches_ TF_GUARDED_BY(mu_) = 0;
  int64_t num_enqueued_tasks_ TF_GUARDED_BY(mu_) = 0;
  mutable mutex mu_;
  ASBSQueue(const ASBSQueue&) = delete;
  void operator=(const ASBSQueue&) = delete;
};
template <typename TaskType>
class ASBSBatch : public Batch<TaskType> {
 public:
  ASBSBatch(ASBSQueue<TaskType>* queue, int64_t creation_time_micros,
            int64_t batch_timeout_micros, uint64 traceme_context_id)
      : queue_(queue),
        creation_time_micros_(creation_time_micros),
        schedulable_time_micros_(creation_time_micros + batch_timeout_micros),
        traceme_context_id_(traceme_context_id) {}
  ~ASBSBatch() override {}
  ASBSQueue<TaskType>* queue() const { return queue_; }
  int64_t creation_time_micros() const { return creation_time_micros_; }
  int64_t schedulable_time_micros() const { return schedulable_time_micros_; }
  uint64 traceme_context_id() const { return traceme_context_id_; }
 private:
  ASBSQueue<TaskType>* queue_;
  const int64_t creation_time_micros_;
  const int64_t schedulable_time_micros_;
  const uint64 traceme_context_id_;
  ASBSBatch(const ASBSBatch&) = delete;
  void operator=(const ASBSBatch&) = delete;
};
}  
template <typename TaskType>
constexpr double AdaptiveSharedBatchScheduler<TaskType>::kMaxStepSizeMultiplier;
template <typename TaskType>
constexpr double AdaptiveSharedBatchScheduler<TaskType>::kMinStepSizeMultiplier;
template <typename TaskType>
Status AdaptiveSharedBatchScheduler<TaskType>::Create(
    const Options& options,
    std::shared_ptr<AdaptiveSharedBatchScheduler<TaskType>>* scheduler) {
  if (options.num_batch_threads < 1) {
    return errors::InvalidArgument("num_batch_threads must be positive; was ",
                                   options.num_batch_threads);
  }
  if (options.min_in_flight_batches_limit < 1) {
    return errors::InvalidArgument(
        "min_in_flight_batches_limit must be >= 1; was ",
        options.min_in_flight_batches_limit);
  }
  if (options.min_in_flight_batches_limit > options.num_batch_threads) {
    return errors::InvalidArgument(
        "min_in_flight_batches_limit (", options.min_in_flight_batches_limit,
        ") must be <= num_batch_threads (", options.num_batch_threads, ")");
  }
  if (options.full_batch_scheduling_boost_micros < 0) {
    return errors::InvalidArgument(
        "full_batch_scheduling_boost_micros can't be negative; was ",
        options.full_batch_scheduling_boost_micros);
  }
  if (options.initial_in_flight_batches_limit > options.num_batch_threads) {
    return errors::InvalidArgument(
        "initial_in_flight_batches_limit (",
        options.initial_in_flight_batches_limit,
        ") should not be larger than num_batch_threads (",
        options.num_batch_threads, ")");
  }
  if (options.initial_in_flight_batches_limit <
      options.min_in_flight_batches_limit) {
    return errors::InvalidArgument("initial_in_flight_batches_limit (",
                                   options.initial_in_flight_batches_limit,
                                   "must be >= min_in_flight_batches_limit (",
                                   options.min_in_flight_batches_limit, ")");
  }
  if (options.batches_to_average_over < 1) {
    return errors::InvalidArgument(
        "batches_to_average_over should be "
        "greater than or equal to 1; was ",
        options.batches_to_average_over);
  }
  scheduler->reset(new AdaptiveSharedBatchScheduler<TaskType>(options));
  return absl::OkStatus();
}
template <typename TaskType>
AdaptiveSharedBatchScheduler<TaskType>::AdaptiveSharedBatchScheduler(
    const Options& options)
    : options_(options),
      in_flight_batches_limit_(options.initial_in_flight_batches_limit),
      rand_double_(0.0, 1.0) {
  std::random_device device;
  rand_engine_.seed(device());
  if (options.thread_pool == nullptr) {
    owned_batch_thread_pool_ = true;
    batch_thread_pool_ = new thread::ThreadPool(
        GetEnv(), options.thread_pool_name, options.num_batch_threads);
  } else {
    owned_batch_thread_pool_ = false;
    batch_thread_pool_ = options.thread_pool;
  }
}
template <typename TaskType>
Status AdaptiveSharedBatchScheduler<TaskType>::AddQueue(
    const QueueOptions& options, BatchProcessor process_batch_callback,
    std::unique_ptr<BatchScheduler<TaskType>>* queue) {
  if (options.max_batch_size <= 0) {
    return errors::InvalidArgument("max_batch_size must be positive; was ",
                                   options.max_batch_size);
  }
  if (options.max_enqueued_batches <= 0) {
    return errors::InvalidArgument(
        "max_enqueued_batches must be positive; was ",
        options.max_enqueued_batches);
  }
  if (options.max_input_task_size.has_value()) {
    if (options.max_input_task_size.value() < options.max_batch_size) {
      return errors::InvalidArgument(
          "max_input_task_size must be larger than or equal to max_batch_size;"
          "got max_input_task_size as ",
          options.max_input_task_size.value(), " and max_batch_size as ",
          options.max_batch_size);
    }
  }
  internal::ASBSQueue<TaskType>* asbs_queue_raw;
  queue->reset(asbs_queue_raw = new internal::ASBSQueue<TaskType>(
                   this->shared_from_this(), options));
  mutex_lock l(mu_);
  queues_and_callbacks_[asbs_queue_raw] = process_batch_callback;
  return absl::OkStatus();
}
template <typename TaskType>
void AdaptiveSharedBatchScheduler<TaskType>::AddBatch(
    const internal::ASBSBatch<TaskType>* batch) {
  mutex_lock l(mu_);
  if (options_.fifo_scheduling) {
    fifo_batches_.push_back(batch);
  } else {
    batches_.push_back(batch);
  }
  int64_t delay_micros =
      batch->schedulable_time_micros() - GetEnv()->NowMicros();
  if (delay_micros <= 0) {
    MaybeScheduleNextBatch();
    return;
  }
  GetEnv()->SchedClosureAfter(
      delay_micros, [this, lifetime_preserver = this->shared_from_this()] {
        mutex_lock l(mu_);
        MaybeScheduleNextBatch();
      });
}
template <typename TaskType>
void AdaptiveSharedBatchScheduler<TaskType>::RemoveQueue(
    const internal::ASBSQueue<TaskType>* queue) {
  mutex_lock l(mu_);
  queues_and_callbacks_.erase(queue);
}
template <typename TaskType>
void AdaptiveSharedBatchScheduler<TaskType>::MaybeScheduleNextBatchFIFO() {
  const internal::ASBSBatch<TaskType>* batch = *fifo_batches_.begin();
  if (batch->schedulable_time_micros() > GetEnv()->NowMicros()) {
    return;
  }
  fifo_batches_.pop_front();
  batch->queue()->ReleaseBatch(batch);
  batch_thread_pool_->Schedule(std::bind(
      &AdaptiveSharedBatchScheduler<TaskType>::CallbackWrapper, this, batch,
      queues_and_callbacks_[batch->queue()], false ));
  in_flight_batches_++;
}
template <typename TaskType>
void AdaptiveSharedBatchScheduler<
    TaskType>::MaybeScheduleClosedBatchesLockedFIFO() {
  int available_threads =
      static_cast<int>(options_.num_batch_threads - in_flight_batches_ -
                       in_flight_express_batches_);
  for (auto it = fifo_batches_.begin();
       it != fifo_batches_.end() && available_threads > 0;
       it = fifo_batches_.begin()) {
    if ((*it)->IsClosed()) {
      const internal::ASBSBatch<TaskType>* batch = *it;
      fifo_batches_.pop_front();
      batch->queue()->ReleaseBatch(batch);
      batch_thread_pool_->Schedule(
          std::bind(&AdaptiveSharedBatchScheduler<TaskType>::CallbackWrapper,
                    this, batch, queues_and_callbacks_[batch->queue()], true));
      in_flight_express_batches_++;
      available_threads--;
    } else {
      break;
    }
  }
}
template <typename TaskType>
void AdaptiveSharedBatchScheduler<TaskType>::MaybeScheduleNextBatch() {
  bool batch_empty =
      options_.fifo_scheduling ? fifo_batches_.empty() : batches_.empty();
  if (batch_empty || in_flight_batches_ >= in_flight_batches_limit_) return;
  if (in_flight_batches_limit_ - in_flight_batches_ < 1 &&
      rand_double_(rand_engine_) >
          in_flight_batches_limit_ - in_flight_batches_) {
    return;
  }
  if (options_.fifo_scheduling) {
    MaybeScheduleNextBatchFIFO();
    return;
  }
  auto best_it = batches_.end();
  double best_score = (std::numeric_limits<double>::max)();
  int64_t now_micros = GetEnv()->NowMicros();
  for (auto it = batches_.begin(); it != batches_.end(); it++) {
    if ((*it)->schedulable_time_micros() > now_micros) continue;
    const double score =
        (*it)->creation_time_micros() -
        options_.full_batch_scheduling_boost_micros * (*it)->size() /
            static_cast<double>((*it)->queue()->max_task_size());
    if (best_it == batches_.end() || score < best_score) {
      best_score = score;
      best_it = it;
    }
  }
  if (best_it == batches_.end()) return;
  const internal::ASBSBatch<TaskType>* batch = *best_it;
  batches_.erase(best_it);
  batch->queue()->ReleaseBatch(batch);
  batch_thread_pool_->Schedule(
      std::bind(&AdaptiveSharedBatchScheduler<TaskType>::CallbackWrapper, this,
                batch, queues_and_callbacks_[batch->queue()], false));
  in_flight_batches_++;
}
template <typename TaskType>
void AdaptiveSharedBatchScheduler<TaskType>::MaybeScheduleClosedBatches() {
  mutex_lock l(mu_);
  MaybeScheduleClosedBatchesLocked();
}
template <typename TaskType>
void AdaptiveSharedBatchScheduler<
    TaskType>::MaybeScheduleClosedBatchesLocked() {
  if (options_.fifo_scheduling) {
    MaybeScheduleClosedBatchesLockedFIFO();
    return;
  }
  int available_threads =
      static_cast<int>(options_.num_batch_threads - in_flight_batches_ -
                       in_flight_express_batches_);
  for (auto it = batches_.begin();
       it != batches_.end() && available_threads > 0;) {
    if ((*it)->IsClosed()) {
      const internal::ASBSBatch<TaskType>* batch = *it;
      it = batches_.erase(it);
      batch->queue()->ReleaseBatch(batch);
      batch_thread_pool_->Schedule(
          std::bind(&AdaptiveSharedBatchScheduler<TaskType>::CallbackWrapper,
                    this, batch, queues_and_callbacks_[batch->queue()], true));
      in_flight_express_batches_++;
      available_threads--;
    } else {
      ++it;
    }
  }
}
template <typename TaskType>
void AdaptiveSharedBatchScheduler<TaskType>::CallbackWrapper(
    const internal::ASBSBatch<TaskType>* batch,
    AdaptiveSharedBatchScheduler<TaskType>::BatchProcessor callback,
    bool is_express) {
  tsl::profiler::TraceMeConsumer trace_me(
      [&] {
        return profiler::TraceMeEncode(
            "ProcessBatch", {{"batch_size_before_padding", batch->size()},
                             {"_r", 2} });
      },
      tsl::profiler::ContextType::kAdaptiveSharedBatchScheduler,
      batch->traceme_context_id());
  const int64_t start_time = batch->creation_time_micros();
  callback(std::unique_ptr<Batch<TaskType>>(
      const_cast<internal::ASBSBatch<TaskType>*>(batch)));
  int64_t end_time = GetEnv()->NowMicros();
  mutex_lock l(mu_);
  if (is_express) {
    in_flight_express_batches_--;
    MaybeScheduleClosedBatchesLocked();
    return;
  }
  in_flight_batches_--;
  batch_count_++;
  batch_delay_stats_.batch_latency_sum += end_time - start_time;
  MaybeAdjustInflightLimit();
  MaybeScheduleNextBatch();
}
template <typename TaskType>
void AdaptiveSharedBatchScheduler<TaskType>::MaybeAdjustInflightLimit() {
  if (batch_count_ == options_.batches_to_average_over) {
    double current_avg_latency_ms =
        (batch_delay_stats_.batch_latency_sum / 1000.) / batch_count_;
    bool current_latency_decreased =
        current_avg_latency_ms < batch_delay_stats_.last_avg_latency_ms;
    if (current_latency_decreased) {
      step_size_multiplier_ *=
          (batch_delay_stats_.last_latency_decreased ? 2 : 0.5);
      step_size_multiplier_ =
          std::min(step_size_multiplier_, kMaxStepSizeMultiplier);
      step_size_multiplier_ =
          std::max(step_size_multiplier_, kMinStepSizeMultiplier);
    } else {
      batch_delay_stats_.step_direction = -batch_delay_stats_.step_direction;
    }
    in_flight_batches_limit_ += batch_delay_stats_.step_direction *
                                in_flight_batches_limit_ *
                                step_size_multiplier_;
    in_flight_batches_limit_ =
        std::min(in_flight_batches_limit_,
                 static_cast<double>(options_.num_batch_threads));
    in_flight_batches_limit_ =
        std::max(in_flight_batches_limit_,
                 static_cast<double>(options_.min_in_flight_batches_limit));
    batch_delay_stats_.last_avg_latency_ms = current_avg_latency_ms;
    batch_delay_stats_.last_latency_decreased = current_latency_decreased;
    batch_count_ = 0;
    batch_delay_stats_.batch_latency_sum = 0;
  }
}
namespace internal {
template <typename TaskType>
ASBSQueue<TaskType>::ASBSQueue(
    std::shared_ptr<AdaptiveSharedBatchScheduler<TaskType>> scheduler,
    const QueueOptions& options)
    : scheduler_(scheduler), options_(options) {}
template <typename TaskType>
ASBSQueue<TaskType>::~ASBSQueue() {
  const int kSleepMicros = 1000;
  for (;;) {
    {
      mutex_lock l(mu_);
      if (num_enqueued_batches_ == 0) {
        break;
      }
    }
    scheduler_->GetEnv()->SleepForMicroseconds(kSleepMicros);
  }
  scheduler_->RemoveQueue(this);
}
template <typename TaskType>
Status ASBSQueue<TaskType>::Schedule(std::unique_ptr<TaskType>* task) {
  size_t size = (*task)->size();
  if (options_.split_input_task_func == nullptr &&
      size > options_.max_batch_size) {
    return errors::InvalidArgument("Task size ", size,
                                   " is larger than maximum batch size ",
                                   options_.max_batch_size);
  }
  if (options_.max_input_task_size.has_value() &&
      (size > options_.max_input_task_size.value())) {
    return errors::InvalidArgument("Task size ", size,
                                   " is larger than max input task size ",
                                   options_.max_input_task_size.value());
  }
  std::vector<std::unique_ptr<TaskType>> tasks_to_schedule;
  std::vector<ASBSBatch<TaskType>*> new_batches;
  bool closed_batch = false;
  {
    mutex_lock l(mu_);
    if (size > SchedulingCapacityLocked()) {
      return errors::Unavailable("The batch scheduling queue is full");
    }
    int remaining_batch_size =
        current_batch_ == nullptr
            ? options_.max_batch_size
            : options_.max_batch_size - current_batch_->size();
    if (options_.split_input_task_func == nullptr ||
        size <= remaining_batch_size) {
      tasks_to_schedule.push_back(std::move(*task));
    } else {
      TF_RETURN_IF_ERROR(options_.split_input_task_func(
          task, remaining_batch_size, options_.max_batch_size,
          &tasks_to_schedule));
    }
    for (auto& task : tasks_to_schedule) {
      if (current_batch_ &&
          current_batch_->size() + task->size() > options_.max_batch_size) {
        current_batch_->Close();
        closed_batch = true;
        current_batch_ = nullptr;
      }
      if (!current_batch_) {
        num_enqueued_batches_++;
        current_batch_ = new ASBSBatch<TaskType>(
            this, scheduler_->GetEnv()->NowMicros(),
            options_.batch_timeout_micros, NewTraceMeContextIdForBatch());
        new_batches.push_back(current_batch_);
      }
      tsl::profiler::TraceMeProducer trace_me(
          [task_size = task->size()] {
            return profiler::TraceMeEncode(
                "ASBSQueue::Schedule",
                {{"batching_input_task_size", task_size}});
          },
          tsl::profiler::ContextType::kAdaptiveSharedBatchScheduler,
          this->current_batch_->traceme_context_id());
      current_batch_->AddTask(std::move(task));
      num_enqueued_tasks_++;
      bool reached_max_tasks =
          (options_.max_tasks_per_batch.has_value() &&
           current_batch_->num_tasks() >= options_.max_tasks_per_batch.value());
      if (current_batch_->size() == options_.max_batch_size ||
          reached_max_tasks) {
        current_batch_->Close();
        closed_batch = true;
        current_batch_ = nullptr;
      }
    }
  }
  for (auto* batch : new_batches) {
    scheduler_->AddBatch(batch);
  }
  if (closed_batch) {
    scheduler_->MaybeScheduleClosedBatches();
  }
  return absl::OkStatus();
}
template <typename TaskType>
void ASBSQueue<TaskType>::ReleaseBatch(const ASBSBatch<TaskType>* batch) {
  mutex_lock l(mu_);
  num_enqueued_batches_--;
  num_enqueued_tasks_ -= batch->num_tasks();
  if (batch == current_batch_) {
    current_batch_->Close();
    current_batch_ = nullptr;
  }
}
template <typename TaskType>
size_t ASBSQueue<TaskType>::NumEnqueuedTasks() const {
  mutex_lock l(mu_);
  return num_enqueued_tasks_;
}
template <typename TaskType>
size_t ASBSQueue<TaskType>::SchedulingCapacity() const {
  mutex_lock l(mu_);
  return SchedulingCapacityLocked();
}
template <typename TaskType>
size_t ASBSQueue<TaskType>::SchedulingCapacityLocked() const {
  const int current_batch_capacity =
      current_batch_ ? options_.max_batch_size - current_batch_->size() : 0;
  const int spare_batches =
      options_.max_enqueued_batches - num_enqueued_batches_;
  return spare_batches * options_.max_batch_size + current_batch_capacity;
}
template <typename TaskType>
uint64 ASBSQueue<TaskType>::NewTraceMeContextIdForBatch() {
  static std::atomic<uint64> traceme_context_id(0);
  return traceme_context_id.fetch_add(1, std::memory_order_relaxed);
}
}  
}  
}  
#endif  