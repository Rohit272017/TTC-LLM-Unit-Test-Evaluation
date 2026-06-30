#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_STATS_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_STATS_H_
#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include "absl/container/node_hash_map.h"
#include "absl/time/time.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/thread_annotations.h"
namespace tensorflow::serving {
constexpr int64_t kNumBatchThreadsUnknown = -1;
constexpr int64_t kBatchTimeoutMicrosUnknown = -1;
class CostTracker {
 public:
  void Register(absl::Duration cost) {
    DCHECK_GT(cost, absl::ZeroDuration());
    mutex_lock l(mu_);
    sample_count_++;
    sample_sum_ += cost;
  };
  std::optional<absl::Duration> mean() const {
    int64_t count;
    absl::Duration sum;
    {
      mutex_lock l(mu_);
      count = sample_count_;
      sum = sample_sum_;
    }
    if (count == 0) return std::nullopt;
    return sum / count;
  };
 private:
  mutable mutex mu_;
  int64_t sample_count_ TF_GUARDED_BY(mu_) = 0;
  absl::Duration sample_sum_ TF_GUARDED_BY(mu_);
};
class BatchSizeStats {
 public:
  CostTracker& tpu_cost() { return tpu_cost_; };
 private:
  CostTracker tpu_cost_;
};
class ModelBatchStats {
 public:
  BatchSizeStats& batch_size(int32 batch_size) {
    mutex_lock l(mu_);
    return batch_size_stats_by_batch_size_[batch_size];
  }
  void RegisterProcessedSize(int64_t size) {
    cumulative_processed_size_.fetch_add(size, std::memory_order_relaxed);
  }
  int64_t cumulative_processed_size() const {
    return cumulative_processed_size_.load(std::memory_order_relaxed);
  }
  std::vector<int32> BatchSizes() const {
    std::vector<int32> result;
    mutex_lock l(mu_);
    result.reserve(batch_size_stats_by_batch_size_.size());
    for (const auto& [key, value] : batch_size_stats_by_batch_size_) {
      result.push_back(key);
    }
    return result;
  }
  void SetNumBatchThreads(int64_t num_batch_threads) {
    num_batch_threads_.store(num_batch_threads, std::memory_order_relaxed);
  }
  int64_t num_batch_threads() const {
    return num_batch_threads_.load(std::memory_order_relaxed);
  }
  void SetBatchTimeoutMicros(int64_t batch_timeout_micros) {
    batch_timeout_micros_.store(batch_timeout_micros,
                                std::memory_order_relaxed);
  }
  int64_t batch_timeout_micros() const {
    return batch_timeout_micros_.load(std::memory_order_relaxed);
  }
 private:
  mutable mutex mu_;
  absl::node_hash_map<int32, BatchSizeStats> batch_size_stats_by_batch_size_
      TF_GUARDED_BY(mu_);
  std::atomic<int64_t> cumulative_processed_size_ = 0;
  std::atomic<int64_t> num_batch_threads_ = kNumBatchThreadsUnknown;
  std::atomic<int64_t> batch_timeout_micros_ = kBatchTimeoutMicrosUnknown;
};
class BatchStatsRegistry {
 public:
  ModelBatchStats& model(const std::string& model_name,
                         const std::string& op_name) {
    std::tuple key(model_name, op_name);
    mutex_lock l(mu_);
    return model_batch_stats_by_model_and_op_names_[key];
  }
  std::vector<std::tuple<std::string, std::string>> ModelAndOpNames() const {
    std::vector<std::tuple<std::string, std::string>> result;
    mutex_lock l(mu_);
    result.reserve(model_batch_stats_by_model_and_op_names_.size());
    for (const auto& [key, value] : model_batch_stats_by_model_and_op_names_) {
      result.push_back(key);
    }
    return result;
  }
 private:
  mutable mutex mu_;
  absl::node_hash_map<std::tuple<std::string, std::string>, ModelBatchStats>
      model_batch_stats_by_model_and_op_names_ TF_GUARDED_BY(mu_);
};
inline BatchStatsRegistry& GlobalBatchStatsRegistry() {
  static BatchStatsRegistry* instance = new BatchStatsRegistry();
  return *instance;
}
}  
#endif  