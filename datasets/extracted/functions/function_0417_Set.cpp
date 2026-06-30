#ifndef XLA_TSL_FRAMEWORK_REAL_TIME_IN_MEMORY_METRIC_H_
#define XLA_TSL_FRAMEWORK_REAL_TIME_IN_MEMORY_METRIC_H_
#include <atomic>
namespace tsl {
template <typename T>
class RealTimeInMemoryMetric {
 public:
  RealTimeInMemoryMetric() : value_(T{0}) {}
  T Get() const { return value_.load(std::memory_order_relaxed); }
  void Set(T new_value) { value_.store(new_value, std::memory_order_relaxed); }
  RealTimeInMemoryMetric(const RealTimeInMemoryMetric&) = delete;
  RealTimeInMemoryMetric& operator=(const RealTimeInMemoryMetric&) = delete;
  RealTimeInMemoryMetric(RealTimeInMemoryMetric&&) = delete;
  RealTimeInMemoryMetric& operator=(RealTimeInMemoryMetric&&) = delete;
  static_assert(std::is_arithmetic_v<T>);
 private:
  std::atomic<T> value_;
};
}  
#endif  