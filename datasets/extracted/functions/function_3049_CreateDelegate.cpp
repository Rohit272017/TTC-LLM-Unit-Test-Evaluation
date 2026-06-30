#include "tensorflow/lite/core/acceleration/configuration/c/xnnpack_plugin.h"
#include <memory>
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/core/acceleration/configuration/c/delegate_plugin.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
extern "C" {
static TfLiteDelegate* CreateDelegate(const void* settings) {
  const ::tflite::TFLiteSettings* tflite_settings =
      static_cast<const ::tflite::TFLiteSettings*>(settings);
  auto options(TfLiteXNNPackDelegateOptionsDefault());
  const auto* xnnpack_settings = tflite_settings->xnnpack_settings();
  if (xnnpack_settings) {
    options.num_threads = xnnpack_settings->num_threads();
    if (xnnpack_settings->flags()) {
      options.flags = xnnpack_settings->flags();
    }
    if (xnnpack_settings->weight_cache_file_path()) {
      options.weight_cache_file_path =
          xnnpack_settings->weight_cache_file_path()->c_str();
    }
  }
  return TfLiteXNNPackDelegateCreate(&options);
}
static void DestroyDelegate(TfLiteDelegate* delegate) {
  TfLiteXNNPackDelegateDelete(delegate);
}
static int DelegateErrno(TfLiteDelegate* from_delegate) { return 0; }
static constexpr TfLiteDelegatePlugin kPluginCApi{
    CreateDelegate,
    DestroyDelegate,
    DelegateErrno,
};
const TfLiteDelegatePlugin* TfLiteXnnpackDelegatePluginCApi() {
  return &kPluginCApi;
}
}  