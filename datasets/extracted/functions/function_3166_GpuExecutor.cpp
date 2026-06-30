#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_EXECUTOR_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_EXECUTOR_H_
#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/host_memory_allocation.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_common.h"
namespace stream_executor {
namespace gpu {
class GpuStream;
class GpuExecutor : public StreamExecutorCommon {
 public:
  GpuExecutor(Platform* platform, int device_ordinal)
      : StreamExecutorCommon(platform),
        context_(nullptr),
        device_ordinal_(device_ordinal) {}
  int device_ordinal() const override { return device_ordinal_; };
  virtual void UnloadKernel(const Kernel* kernel) = 0;
  virtual absl::StatusOr<std::unique_ptr<EventBasedTimer>>
  CreateEventBasedTimer(GpuStream* stream, bool use_delay_kernel) = 0;
  virtual absl::Status TrimGraphMemory() = 0;
  Context* gpu_context() const { return context_; }
  absl::StatusOr<std::vector<ApiTrace>> ExtractApiTrace() override {
    absl::MutexLock lock(&logger_mu_);
    return std::move(argument_logs_);
  }
  absl::Status RecordApiTrace(ApiTrace call) override {
    absl::MutexLock lock(&logger_mu_);
    if (std::holds_alternative<GemmCallTrace>(call) &&
        (argument_logging_mode_ & kLogGemm)) {
      argument_logs_.push_back(call);
    }
    return absl::OkStatus();
  }
  bool SetArgumentLoggingMode(uint64_t mode) override {
    absl::MutexLock lock(&logger_mu_);
    argument_logging_mode_ = mode;
    return true;
  }
  uint64_t GetArgumentLoggingMode() const { return argument_logging_mode_; }
 protected:
  void set_context(Context* context) { context_ = context; }
 private:
  Context* context_;
  int device_ordinal_;
  absl::Mutex logger_mu_;
  mutable std::vector<ApiTrace> argument_logs_ ABSL_GUARDED_BY(logger_mu_);
  uint64_t argument_logging_mode_ = 0;
  GpuExecutor(const GpuExecutor&) = delete;
  void operator=(const GpuExecutor&) = delete;
};
inline GpuExecutor* ExtractGpuExecutor(StreamExecutor* stream_exec) {
  return static_cast<GpuExecutor*>(stream_exec);
}
}  
}  
#endif  