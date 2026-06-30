#include "tensorflow/lite/delegates/telemetry.h"
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/core/c/common.h"
namespace tflite {
namespace delegates {
TfLiteStatus ReportDelegateSettings(TfLiteContext* context,
                                    TfLiteDelegate* delegate,
                                    const TFLiteSettings& settings) {
  auto* profiler = reinterpret_cast<Profiler*>(context->profiler);
  const int64_t event_metadata1 = reinterpret_cast<int64_t>(delegate);
  const int64_t event_metadata2 = reinterpret_cast<int64_t>(&settings);
  TFLITE_ADD_RUNTIME_INSTRUMENTATION_EVENT(profiler, kDelegateSettingsTag,
                                           event_metadata1, event_metadata2);
  return kTfLiteOk;
}
TfLiteStatus ReportDelegateStatus(TfLiteContext* context,
                                  TfLiteDelegate* delegate,
                                  const DelegateStatus& status) {
  auto* profiler = reinterpret_cast<Profiler*>(context->profiler);
  TFLITE_ADD_RUNTIME_INSTRUMENTATION_EVENT(profiler, kDelegateStatusTag,
                                           status.full_status(),
                                           static_cast<int64_t>(kTfLiteOk));
  return kTfLiteOk;
}
}  
}  