#include "tensorflow/lite/profiling/memory_usage_monitor.h"
#include <memory>
#include <utility>
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/profiling/memory_info.h"
namespace tflite {
namespace profiling {
namespace memory {
constexpr float MemoryUsageMonitor::kInvalidMemUsageMB;
MemoryUsageMonitor::MemoryUsageMonitor(int sampling_interval_ms,
                                       std::unique_ptr<Sampler> sampler)
    : sampler_(std::move(sampler)),
      is_supported_(false),
      sampling_interval_(absl::Milliseconds(sampling_interval_ms)) {
  is_supported_ = (sampler_ != nullptr && sampler_->IsSupported());
  if (!is_supported_) {
    TFLITE_LOG(TFLITE_LOG_INFO,
               "Getting memory usage isn't supported on this platform!\n");
    return;
  }
}
void MemoryUsageMonitor::Start() {
  if (!is_supported_) return;
  if (check_memory_thd_ != nullptr) {
    TFLITE_LOG(TFLITE_LOG_INFO, "Memory monitoring has already started!\n");
    return;
  }
  stop_signal_ = std::make_unique<absl::Notification>();
  check_memory_thd_ = std::make_unique<std::thread>(([this]() {
    while (true) {
      const auto mem_info = sampler_->GetMemoryUsage();
      if (mem_info.mem_footprint_kb > peak_mem_footprint_kb_) {
        peak_mem_footprint_kb_ = mem_info.mem_footprint_kb;
      }
      if (stop_signal_->HasBeenNotified()) break;
      sampler_->SleepFor(sampling_interval_);
    }
  }));
}
void MemoryUsageMonitor::Stop() {
  if (!is_supported_) return;
  if (check_memory_thd_ == nullptr) {
    TFLITE_LOG(TFLITE_LOG_INFO,
               "Memory monitoring hasn't started yet or has stopped!\n");
    return;
  }
  StopInternal();
}
void MemoryUsageMonitor::StopInternal() {
  if (check_memory_thd_ == nullptr) return;
  stop_signal_->Notify();
  if (check_memory_thd_ != nullptr) {
    check_memory_thd_->join();
  }
  stop_signal_.reset(nullptr);
  check_memory_thd_.reset(nullptr);
}
}  
}  
}  