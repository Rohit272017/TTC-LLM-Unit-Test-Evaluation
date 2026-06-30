#include "tensorflow/core/common_runtime/request_cost.h"
#include <string>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
namespace tensorflow {
void RequestCost::RecordCost(
    const std::vector<std::pair<absl::string_view, absl::Duration>>& costs) {
  absl::MutexLock lock(&mutex_);
  for (const auto& cost : costs) {
    cost_map_[cost.first] += cost.second;
  }
}
void RequestCost::ScaleCosts(int scale_factor) {
  absl::MutexLock lock(&mutex_);
  for (auto& [cost_type, cost] : cost_map_) {
    cost *= scale_factor;
  }
}
absl::flat_hash_map<std::string, absl::Duration> RequestCost::GetCosts() const {
  absl::MutexLock lock(&mutex_);
  return cost_map_;
}
void RequestCost::RecordMetrics(
    const std::vector<std::pair<absl::string_view, double>>& metrics) {
  absl::MutexLock lock(&mutex_);
  for (const auto& metric : metrics) {
    metric_map_[metric.first] = metric.second;
  }
}
absl::flat_hash_map<std::string, double> RequestCost::GetMetrics() const {
  absl::MutexLock lock(&mutex_);
  return metric_map_;
}
void RequestCost::RecordBatchMetrics(const BatchMetrics& batch_metrics) {
  absl::MutexLock lock(&mutex_);
  batch_metrics_.push_back(batch_metrics);
}
void RequestCost::ScaleBatchCosts(int scale_factor) {
  absl::MutexLock lock(&mutex_);
  for (auto& batch_metrics : batch_metrics_) {
    for (auto& [cost_type, cost] : batch_metrics.batch_costs) {
      batch_metrics.batch_costs[cost_type] *= scale_factor;
    }
  }
}
std::vector<RequestCost::BatchMetrics> RequestCost::GetBatchMetrics() const {
  absl::MutexLock lock(&mutex_);
  return batch_metrics_;
}
}  