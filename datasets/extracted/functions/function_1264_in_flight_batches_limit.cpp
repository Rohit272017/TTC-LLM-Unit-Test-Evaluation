#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_SERIAL_DEVICE_BATCH_SCHEDULER_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_SERIAL_DEVICE_BATCH_SCHEDULER_H_
#include <algorithm>
#include <functional>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
namespace tensorflow {
namespace serving {
namespace internal {
template <typename TaskType>
class SDBSBatch;
template <typename TaskType>
class SDBSQueue;
}  
template <typename TaskType>
class SerialDeviceBatchScheduler : public std::enable_shared_from_this<
                                       SerialDeviceBatchScheduler<TaskType>> {
 public:
  ~SerialDeviceBatchScheduler();
  struct Options {
    string thread_pool_name = {"batch_threads"};
    int64_t num_batch_threads = port::NumSchedulableCPUs();
    int64_t full_batch_scheduling_boost_micros = 0;
    Env* env = Env::Default();
    int64_t initial_in_flight_batches_limit = 3;
    std::function<int64()> get_pending_on_serial_device;
    double target_pending = 2;
    int64_t batches_to_average_over = 1000;
  };
  static Status Create(
      const Options& options,
      std::shared_ptr<SerialDeviceBatchScheduler<TaskType>>* scheduler);
  struct QueueOptions {
    int max_batch_size = 1000;
    int max_enqueued_batches = 10;
  };
  using BatchProcessor = std::function<void(std::unique_ptr<Batch<TaskType>>)>;
  Status AddQueue(const QueueOptions& options,
                  BatchProcessor process_batch_callback,
                  std::unique_ptr<BatchScheduler<TaskType>>* queue);
  double in_flight_batches_limit() {
    mutex_lock l(mu_);
    return in_flight_batches_limit_;
  }
  double recent_low_traffic_ratio() {
    mutex_lock l(mu_);
    return recent_low_traffic_ratio_;
  }
 private:
  friend class internal::SDBSQueue<TaskType>;
  explicit SerialDeviceBatchScheduler(const Options& options);
  void ProcessBatches();
  void AddBatch(const internal::SDBSBatch<TaskType>* batch);
  void RemoveQueue(const internal::SDBSQueue<TaskType>* queue);
  Env* env() const { return options_.env; }
  const Options options_;
  std::vector<const internal::SDBSBatch<TaskType>*> batches_ TF_GUARDED_BY(mu_);
  std::unordered_map<const internal::SDBSQueue<TaskType>*, BatchProcessor>
      queues_and_callbacks_ TF_GUARDED_BY(mu_);
  std::unique_ptr<thread::ThreadPool> batch_thread_pool_;
  int64_t in_flight_batches_limit_ TF_GUARDED_BY(mu_);
  int64_t processing_threads_ TF_GUARDED_BY(mu_) = 0;
  int64_t batch_count_ TF_GUARDED_BY(mu_) = 0;
  int64_t no_batch_count_ TF_GUARDED_BY(mu_) = 0;
  int64_t pending_sum_ = 0;
  int64_t batch_latency_sum_ = 0;
  int64_t batch_period_micros_ = 0;
  double recent_low_traffic_ratio_ = 0;
  mutex mu_;
  SerialDeviceBatchScheduler(const SerialDeviceBatchScheduler&) = delete;
  void operator=(const SerialDeviceBatchScheduler&) = delete;
};
namespace internal {
template <typename TaskType>
class SDBSQueue : public BatchScheduler<TaskType> {
 public:
  using QueueOptions =
      typename SerialDeviceBatchScheduler<TaskType>::QueueOptions;
  SDBSQueue(std::shared_ptr<SerialDeviceBatchScheduler<TaskType>> scheduler,
            const QueueOptions& options);
  ~SDBSQueue() override;
  Status Schedule(std::unique_ptr<TaskType>* task) override;
  size_t NumEnqueuedTasks() const override;
  size_t SchedulingCapacity() const override;
  void ReleaseBatch(const SDBSBatch<TaskType>* batch);
  size_t max_task_size() const override { return options_.max_batch_size; }
 private:
  std::shared_ptr<SerialDeviceBatchScheduler<TaskType>> scheduler_;
  const QueueOptions options_;
  SDBSBatch<TaskType>* current_batch_ TF_GUARDED_BY(mu_) = nullptr;
  int64_t num_enqueued_batches_ TF_GUARDED_BY(mu_) = 0;
  int64_t num_enqueued_tasks_ TF_GUARDED_BY(mu_) = 0;
  mutable mutex mu_;
  SDBSQueue(const SDBSQueue&) = delete;
  void operator=(const SDBSQueue&) = delete;
};
template <typename TaskType>
class SDBSBatch : public Batch<TaskType> {
 public:
  SDBSBatch(SDBSQueue<TaskType>* queue, int64_t creation_time_micros)
      : queue_(queue), creation_time_micros_(creation_time_micros) {}
  ~SDBSBatch() override {}
  SDBSQueue<TaskType>* queue() const { return queue_; }
  int64_t creation_time_micros() const { return creation_time_micros_; }
 private:
  SDBSQueue<TaskType>* queue_;
  const int64_t creation_time_micros_;
  SDBSBatch(const SDBSBatch&) = delete;
  void operator=(const SDBSBatch&) = delete;
};
}  
template <typename TaskType>
Status SerialDeviceBatchScheduler<TaskType>::Create(
    const Options& options,
    std::shared_ptr<SerialDeviceBatchScheduler<TaskType>>* scheduler) {
  if (options.num_batch_threads < 1) {
    return errors::InvalidArgument("num_batch_threads must be positive; was ",
                                   options.num_batch_threads);
  }
  if (options.initial_in_flight_batches_limit < 1) {
    return errors::InvalidArgument(
        "initial_in_flight_batches_limit must be positive; was ",
        options.initial_in_flight_batches_limit);
  }
  if (options.initial_in_flight_batches_limit > options.num_batch_threads) {
    return errors::InvalidArgument(
        "initial_in_flight_batches_limit (",
        options.initial_in_flight_batches_limit,
        ") should not be larger than num_batch_threads (",
        options.num_batch_threads, ")");
  }
  if (options.full_batch_scheduling_boost_micros < 0) {
    return errors::InvalidArgument(
        "full_batch_scheduling_boost_micros can't be negative; was ",
        options.full_batch_scheduling_boost_micros);
  }
  if (options.batches_to_average_over < 1) {
    return errors::InvalidArgument(
        "batches_to_average_over should be "
        "greater than or equal to 1; was ",
        options.batches_to_average_over);
  }
  if (options.target_pending <= 0) {
    return errors::InvalidArgument(
        "target_pending should be larger than zero; was ",
        options.target_pending);
  }
  if (!options.get_pending_on_serial_device) {
    return errors::InvalidArgument(
        "get_pending_on_serial_device must be "
        "specified");
  }
  scheduler->reset(new SerialDeviceBatchScheduler<TaskType>(options));
  return absl::OkStatus();
}
template <typename TaskType>
SerialDeviceBatchScheduler<TaskType>::SerialDeviceBatchScheduler(
    const Options& options)
    : options_(options),
      in_flight_batches_limit_(options.initial_in_flight_batches_limit),
      processing_threads_(options.initial_in_flight_batches_limit) {
  batch_thread_pool_.reset(new thread::ThreadPool(
      env(), options.thread_pool_name, options.num_batch_threads));
  for (int i = 0; i < processing_threads_; i++) {
    batch_thread_pool_->Schedule(
        std::bind(&SerialDeviceBatchScheduler<TaskType>::ProcessBatches, this));
  }
}
template <typename TaskType>
SerialDeviceBatchScheduler<TaskType>::~SerialDeviceBatchScheduler() {
  {
    mutex_lock l(mu_);
    processing_threads_ = 0;
  }
  batch_thread_pool_.reset();
}
template <typename TaskType>
Status SerialDeviceBatchScheduler<TaskType>::AddQueue(
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
  internal::SDBSQueue<TaskType>* SDBS_queue_raw;
  queue->reset(SDBS_queue_raw = new internal::SDBSQueue<TaskType>(
                   this->shared_from_this(), options));
  mutex_lock l(mu_);
  queues_and_callbacks_[SDBS_queue_raw] = process_batch_callback;
  return absl::OkStatus();
}
template <typename TaskType>
void SerialDeviceBatchScheduler<TaskType>::AddBatch(
    const internal::SDBSBatch<TaskType>* batch) {
  mutex_lock l(mu_);
  batches_.push_back(batch);
}
template <typename TaskType>
void SerialDeviceBatchScheduler<TaskType>::RemoveQueue(
    const internal::SDBSQueue<TaskType>* queue) {
  mutex_lock l(mu_);
  queues_and_callbacks_.erase(queue);
}
template <typename TaskType>
void SerialDeviceBatchScheduler<TaskType>::ProcessBatches() {
  const int64_t kIdleThreadSleepTimeMicros = 1000;
  const double kMaxNoBatchRatio = .1;
  const double kLowTrafficMovingAverageFactor = .1;
  for (;;) {
    mu_.lock();
    if (processing_threads_ < 1 ||
        processing_threads_ > in_flight_batches_limit_) {
      processing_threads_--;
      mu_.unlock();
      break;
    }
    if (batches_.empty()) {
      no_batch_count_++;
      int64_t sleep_time = batch_period_micros_ ? batch_period_micros_
                                                : kIdleThreadSleepTimeMicros;
      mu_.unlock();
      env()->SleepForMicroseconds(sleep_time);
      continue;
    }
    auto best_it = batches_.begin();
    double best_score =
        (*best_it)->creation_time_micros() -
        options_.full_batch_scheduling_boost_micros * (*best_it)->size() /
            static_cast<double>((*best_it)->queue()->max_task_size());
    for (auto it = batches_.begin() + 1; it != batches_.end(); it++) {
      const double score =
          (*it)->creation_time_micros() -
          options_.full_batch_scheduling_boost_micros * (*it)->size() /
              static_cast<double>((*it)->queue()->max_task_size());
      if (score < best_score) {
        best_score = score;
        best_it = it;
      }
    }
    const internal::SDBSBatch<TaskType>* batch = *best_it;
    batches_.erase(best_it);
    batch->queue()->ReleaseBatch(batch);
    auto callback = queues_and_callbacks_[batch->queue()];
    mu_.unlock();
    int64_t start_time = env()->NowMicros();
    callback(std::unique_ptr<Batch<TaskType>>(
        const_cast<internal::SDBSBatch<TaskType>*>(batch)));
    int64_t end_time = env()->NowMicros();
    mu_.lock();
    batch_count_++;
    batch_latency_sum_ += end_time - start_time;
    pending_sum_ += options_.get_pending_on_serial_device();
    if (batch_count_ == options_.batches_to_average_over) {
      recent_low_traffic_ratio_ *= (1 - kLowTrafficMovingAverageFactor);
      if (no_batch_count_ < kMaxNoBatchRatio * batch_count_) {
        double avg_pending = pending_sum_ / static_cast<double>(batch_count_);
        batch_period_micros_ =
            batch_latency_sum_ / batch_count_ / in_flight_batches_limit_;
        in_flight_batches_limit_ +=
            std::round(options_.target_pending - avg_pending);
        in_flight_batches_limit_ =
            std::max(in_flight_batches_limit_, int64_t{1});
        in_flight_batches_limit_ =
            std::min(in_flight_batches_limit_, options_.num_batch_threads);
        if (processing_threads_ > 0 &&
            processing_threads_ < in_flight_batches_limit_) {
          int extra_threads = in_flight_batches_limit_ - processing_threads_;
          for (int i = 0; i < extra_threads; i++) {
            batch_thread_pool_->Schedule(std::bind(
                &SerialDeviceBatchScheduler<TaskType>::ProcessBatches, this));
          }
          processing_threads_ = in_flight_batches_limit_;
        }
      } else {
        recent_low_traffic_ratio_ += kLowTrafficMovingAverageFactor;
      }
      batch_count_ = 0;
      no_batch_count_ = 0;
      pending_sum_ = 0;
      batch_latency_sum_ = 0;
    }
    mu_.unlock();
  }
}
namespace internal {
template <typename TaskType>
SDBSQueue<TaskType>::SDBSQueue(
    std::shared_ptr<SerialDeviceBatchScheduler<TaskType>> scheduler,
    const QueueOptions& options)
    : scheduler_(scheduler), options_(options) {}
template <typename TaskType>
SDBSQueue<TaskType>::~SDBSQueue() {
  const int kSleepMicros = 1000;
  for (;;) {
    {
      mutex_lock l(mu_);
      if (num_enqueued_batches_ == 0) {
        break;
      }
    }
    scheduler_->env()->SleepForMicroseconds(kSleepMicros);
  }
  scheduler_->RemoveQueue(this);
}
template <typename TaskType>
Status SDBSQueue<TaskType>::Schedule(std::unique_ptr<TaskType>* task) {
  SDBSBatch<TaskType>* new_batch = nullptr;
  size_t size = (*task)->size();
  if (size > options_.max_batch_size) {
    return errors::InvalidArgument("Task size ", size,
                                   " is larger than maximum batch size ",
                                   options_.max_batch_size);
  }
  {
    mutex_lock l(mu_);
    if (current_batch_ &&
        current_batch_->size() + size > options_.max_batch_size) {
      if (num_enqueued_batches_ >= options_.max_enqueued_batches) {
        return errors::Unavailable("The batch scheduling queue is full");
      }
      current_batch_->Close();
      current_batch_ = nullptr;
    }
    if (!current_batch_) {
      num_enqueued_batches_++;
      current_batch_ = new_batch =
          new SDBSBatch<TaskType>(this, scheduler_->env()->NowMicros());
    }
    current_batch_->AddTask(std::move(*task));
    num_enqueued_tasks_++;
  }
  if (new_batch != nullptr) scheduler_->AddBatch(new_batch);
  return absl::OkStatus();
}
template <typename TaskType>
void SDBSQueue<TaskType>::ReleaseBatch(const SDBSBatch<TaskType>* batch) {
  mutex_lock l(mu_);
  num_enqueued_batches_--;
  num_enqueued_tasks_ -= batch->num_tasks();
  if (batch == current_batch_) {
    current_batch_->Close();
    current_batch_ = nullptr;
  }
}
template <typename TaskType>
size_t SDBSQueue<TaskType>::NumEnqueuedTasks() const {
  mutex_lock l(mu_);
  return num_enqueued_tasks_;
}
template <typename TaskType>
size_t SDBSQueue<TaskType>::SchedulingCapacity() const {
  mutex_lock l(mu_);
  const int current_batch_capacity =
      current_batch_ ? options_.max_batch_size - current_batch_->size() : 0;
  const int spare_batches =
      options_.max_enqueued_batches - num_enqueued_batches_;
  return spare_batches * options_.max_batch_size + current_batch_capacity;
}
}  
}  
}  
#endif  