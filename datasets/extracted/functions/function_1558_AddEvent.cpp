#include "tensorflow/lite/profiling/telemetry/profiler.h"
#include <cstdint>
#include "tensorflow/lite/core/api/profiler.h"
namespace tflite::telemetry {
void TelemetryProfiler::AddEvent(const char* tag, EventType event_type,
                                 uint64_t metric, int64_t event_metadata1,
                                 int64_t event_metadata2) {
  switch (event_type) {
    case EventType::TELEMETRY_EVENT:
    case EventType::TELEMETRY_DELEGATE_EVENT: {
      if (event_metadata1 == -1) {
        ReportTelemetryEvent(tag, TelemetryStatusCode(metric));
      } else {
        ReportTelemetryOpEvent(tag, event_metadata1, event_metadata2,
                               TelemetryStatusCode(metric));
      }
      break;
    }
    case EventType::OPERATOR_INVOKE_EVENT:
    case EventType::DELEGATE_OPERATOR_INVOKE_EVENT:
    case EventType::DELEGATE_PROFILED_OPERATOR_INVOKE_EVENT: {
      ReportOpInvokeEvent(tag, metric, event_metadata1, event_metadata2);
      break;
    }
    default:
      return;
  }
}
void TelemetryProfiler::AddEventWithData(const char* tag, EventType event_type,
                                         const void* data) {
  switch (event_type) {
    case EventType::TELEMETRY_REPORT_SETTINGS:
    case EventType::TELEMETRY_DELEGATE_REPORT_SETTINGS: {
      auto* settings = reinterpret_cast<const TfLiteTelemetrySettings*>(data);
      if (settings) {
        ReportSettings(tag, settings);
      }
      break;
    }
    default:
      return;
  }
}
uint32_t TelemetryProfiler::BeginEvent(const char* tag, EventType event_type,
                                       int64_t event_metadata1,
                                       int64_t event_metadata2) {
  switch (event_type) {
    case EventType::OPERATOR_INVOKE_EVENT:
    case EventType::DELEGATE_OPERATOR_INVOKE_EVENT:
    case EventType::DELEGATE_PROFILED_OPERATOR_INVOKE_EVENT: {
      return ReportBeginOpInvokeEvent(tag, event_metadata1, event_metadata2);
    }
    default:
      return UINT32_MAX;
  }
}
void TelemetryProfiler::EndEvent(uint32_t event_handle) {
  if (event_handle == UINT32_MAX) return;
  ReportEndOpInvokeEvent(event_handle);
}
class TfLiteTelemetryProfiler : public TelemetryProfiler {
 public:
  explicit TfLiteTelemetryProfiler(TfLiteTelemetryProfilerStruct* profiler)
      : profiler_(profiler) {}
  void ReportTelemetryEvent(const char* event_name,
                            TelemetryStatusCode status) override;
  void ReportTelemetryOpEvent(const char* event_name, int64_t op_idx,
                              int64_t subgraph_idx,
                              TelemetryStatusCode status) override;
  void ReportSettings(const char* setting_name,
                      const TfLiteTelemetrySettings* settings) override;
  uint32_t ReportBeginOpInvokeEvent(const char* op_name, int64_t op_idx,
                                    int64_t subgraph_idx) override;
  void ReportEndOpInvokeEvent(uint32_t event_handle) override;
  void ReportOpInvokeEvent(const char* op_name, uint64_t elapsed_time,
                           int64_t op_idx, int64_t subgraph_idx) override;
 private:
  TfLiteTelemetryProfilerStruct* profiler_ = nullptr;
};
void TfLiteTelemetryProfiler::ReportTelemetryEvent(const char* event_name,
                                                   TelemetryStatusCode status) {
  profiler_->ReportTelemetryEvent(profiler_, event_name, status.code());
}
void TfLiteTelemetryProfiler::ReportTelemetryOpEvent(
    const char* event_name, int64_t op_idx, int64_t subgraph_idx,
    TelemetryStatusCode status) {
  profiler_->ReportTelemetryOpEvent(profiler_, event_name, op_idx, subgraph_idx,
                                    status.code());
}
void TfLiteTelemetryProfiler::ReportSettings(
    const char* setting_name, const TfLiteTelemetrySettings* settings) {
  profiler_->ReportSettings(profiler_, setting_name, settings);
}
uint32_t TfLiteTelemetryProfiler::ReportBeginOpInvokeEvent(
    const char* op_name, int64_t op_idx, int64_t subgraph_idx) {
  return profiler_->ReportBeginOpInvokeEvent(profiler_, op_name, op_idx,
                                             subgraph_idx);
}
void TfLiteTelemetryProfiler::ReportEndOpInvokeEvent(uint32_t event_handle) {
  profiler_->ReportEndOpInvokeEvent(profiler_, event_handle);
}
void TfLiteTelemetryProfiler::ReportOpInvokeEvent(const char* op_name,
                                                  uint64_t elapsed_time,
                                                  int64_t op_idx,
                                                  int64_t subgraph_idx) {
  profiler_->ReportOpInvokeEvent(profiler_, op_name, elapsed_time, op_idx,
                                 subgraph_idx);
}
TelemetryProfiler* MakeTfLiteTelemetryProfiler(
    TfLiteTelemetryProfilerStruct* profiler) {
  return new TfLiteTelemetryProfiler(profiler);
}
}  