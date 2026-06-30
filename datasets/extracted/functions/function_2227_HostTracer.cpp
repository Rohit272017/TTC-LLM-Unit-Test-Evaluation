#include "xla/backends/profiler/cpu/host_tracer.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/tsl/profiler/backends/cpu/host_tracer_utils.h"
#include "xla/tsl/profiler/backends/cpu/threadpool_listener.h"
#include "xla/tsl/profiler/backends/cpu/traceme_recorder.h"
#include "xla/tsl/profiler/utils/time_utils.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/platform/errors.h"
#include "tsl/profiler/lib/profiler_collection.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
namespace xla {
namespace profiler {
namespace {
class HostTracer : public tsl::profiler::ProfilerInterface {
 public:
  explicit HostTracer(int host_trace_level);
  ~HostTracer() override;
  absl::Status Start() override;  
  absl::Status Stop() override;  
  absl::Status CollectData(  
      tensorflow::profiler::XSpace* space) override;
 private:
  const int host_trace_level_;
  bool recording_ = false;
  uint64_t start_timestamp_ns_ = 0;
  tsl::profiler::TraceMeRecorder::Events events_;
};
HostTracer::HostTracer(int host_trace_level)
    : host_trace_level_(host_trace_level) {}
HostTracer::~HostTracer() { Stop().IgnoreError(); }  
absl::Status HostTracer::Start() {  
  if (recording_) {
    return tsl::errors::Internal("TraceMeRecorder already started");
  }
  start_timestamp_ns_ = tsl::profiler::GetCurrentTimeNanos();
  recording_ = tsl::profiler::TraceMeRecorder::Start(host_trace_level_);
  if (!recording_) {
    return tsl::errors::Internal("Failed to start TraceMeRecorder");
  }
  return absl::OkStatus();
}
absl::Status HostTracer::Stop() {  
  if (!recording_) {
    return tsl::errors::Internal("TraceMeRecorder not started");
  }
  events_ = tsl::profiler::TraceMeRecorder::Stop();
  recording_ = false;
  return absl::OkStatus();
}
absl::Status HostTracer::CollectData(  
    tensorflow::profiler::XSpace* space) {
  VLOG(2) << "Collecting data to XSpace from HostTracer.";
  if (recording_) {
    return tsl::errors::Internal("TraceMeRecorder not stopped");
  }
  if (events_.empty()) {
    return absl::OkStatus();
  }
  tensorflow::profiler::XPlane* plane =
      tsl::profiler::FindOrAddMutablePlaneWithName(
          space, tsl::profiler::kHostThreadsPlaneName);
  ConvertCompleteEventsToXPlane(start_timestamp_ns_, std::exchange(events_, {}),
                                plane);
  return absl::OkStatus();
}
}  
std::unique_ptr<tsl::profiler::ProfilerInterface> CreateHostTracer(
    const HostTracerOptions& options) {
  if (options.trace_level == 0) return nullptr;
  std::vector<std::unique_ptr<tsl::profiler::ProfilerInterface>> profilers;
  profilers.push_back(std::make_unique<HostTracer>(options.trace_level));
  profilers.push_back(
      std::make_unique<tsl::profiler::ThreadpoolProfilerInterface>());
  return std::make_unique<tsl::profiler::ProfilerCollection>(
      std::move(profilers));
}
}  
}  