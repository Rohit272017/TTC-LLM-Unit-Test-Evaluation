#include "tensorflow/core/data/metric_utils.h"
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>
#include "absl/time/time.h"
#include "tensorflow/core/data/utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
namespace tensorflow {
namespace data {
namespace {
uint64_t safe_sub(uint64_t x, uint64_t y) { return x >= y ? x - y : 0; }
}  
IteratorMetricsCollector::IteratorMetricsCollector(
    const std::string& device_type, const Env& env)
    : device_type_(device_type), env_(env) {}
absl::Time IteratorMetricsCollector::RecordStart() {
  const uint64_t start_time_us = env_.NowMicros();
  if (!ShouldCollectMetrics()) {
    return absl::FromUnixMicros(start_time_us);
  }
  mutex_lock l(mu_);
  if (end_time_us_ == 0) {
    end_time_us_ = start_time_us;
  }
  uint64_t gap_time_us = 0;
  if (num_active_calls_ == 0) {
    first_start_time_us_ = start_time_us;
    gap_time_us = safe_sub(start_time_us, end_time_us_);
  }
  metrics::RecordTFDataIteratorGap(gap_time_us);
  num_active_calls_++;
  return absl::FromUnixMicros(start_time_us);
}
void IteratorMetricsCollector::RecordStop(absl::Time start_time,
                                          const std::vector<Tensor>& output) {
  if (!ShouldCollectMetrics()) {
    return;
  }
  const uint64_t end_time_us = env_.NowMicros();
  const int64_t latency_micros =
      safe_sub(end_time_us, absl::ToUnixMicros(start_time));
  AddLatencySample(latency_micros);
  IncrementThroughput(GetTotalBytes(output));
  mutex_lock l(mu_);
  metrics::RecordTFDataIteratorLifetime(safe_sub(end_time_us, end_time_us_));
  end_time_us_ = std::max(end_time_us_, end_time_us);
  num_active_calls_--;
  if (num_active_calls_ == 0) {
    metrics::RecordTFDataIteratorBusy(
        safe_sub(end_time_us_, first_start_time_us_));
  }
}
bool IteratorMetricsCollector::ShouldCollectMetrics() const {
  return device_type_ == DEVICE_CPU;
}
}  
}  