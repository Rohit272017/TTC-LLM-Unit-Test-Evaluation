#include "tensorflow/lite/core/acceleration/configuration/c/gpu_plugin.h"
#include <memory>
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/acceleration/configuration/gpu_plugin.h"
#include "tensorflow/lite/core/acceleration/configuration/c/delegate_plugin.h"
#include "tensorflow/lite/core/c/common.h"
#if TFLITE_SUPPORTS_GPU_DELEGATE
#include "tensorflow/lite/delegates/gpu/delegate.h"
#elif defined(REAL_IPHONE_DEVICE)
#include "tensorflow/lite/delegates/gpu/metal_delegate.h"
#endif
extern "C" {
static TfLiteDelegate* CreateDelegate(const void* settings) {
  const ::tflite::TFLiteSettings* tflite_settings =
      static_cast<const ::tflite::TFLiteSettings*>(settings);
  tflite::delegates::GpuPlugin gpu_plugin(*tflite_settings);
#if TFLITE_SUPPORTS_GPU_DELEGATE
  return TfLiteGpuDelegateV2Create(&gpu_plugin.Options());
#elif defined(REAL_IPHONE_DEVICE)
  return TFLGpuDelegateCreate(&gpu_plugin.Options());
#else
  return nullptr;
#endif
}
static void DestroyDelegate(TfLiteDelegate* delegate) {
#if TFLITE_SUPPORTS_GPU_DELEGATE
  TfLiteGpuDelegateV2Delete(delegate);
#elif defined(REAL_IPHONE_DEVICE)
  TFLGpuDelegateDelete(delegate);
#endif
}
static int DelegateErrno(TfLiteDelegate* from_delegate) { return 0; }
static constexpr TfLiteDelegatePlugin kPluginCApi{
    CreateDelegate,
    DestroyDelegate,
    DelegateErrno,
};
const TfLiteDelegatePlugin* TfLiteGpuDelegatePluginCApi() {
  return &kPluginCApi;
}
}  