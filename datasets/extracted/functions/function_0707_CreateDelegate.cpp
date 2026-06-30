#include "tensorflow/lite/core/acceleration/configuration/c/nnapi_plugin.h"
#include <memory>
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/core/acceleration/configuration/c/delegate_plugin.h"
#include "tensorflow/lite/core/acceleration/configuration/nnapi_plugin.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
extern "C" {
static TfLiteDelegate* CreateDelegate(const void* settings) {
  const ::tflite::TFLiteSettings* tflite_settings =
      static_cast<const ::tflite::TFLiteSettings*>(settings);
  tflite::delegates::NnapiPlugin nnapi_plugin(*tflite_settings);
  auto support_library_handle = nnapi_plugin.GetSupportLibraryHandle();
  if (support_library_handle) {
    auto nnapi_support_library_driver =
        reinterpret_cast<const NnApiSLDriverImplFL5*>(support_library_handle);
    return new tflite::StatefulNnApiDelegate(nnapi_support_library_driver,
                                             nnapi_plugin.Options());
  }
  return new tflite::StatefulNnApiDelegate(nnapi_plugin.Options());
}
static void DestroyDelegate(TfLiteDelegate* delegate) {
  delete static_cast<tflite::StatefulNnApiDelegate*>(delegate);
}
static int DelegateErrno(TfLiteDelegate* from_delegate) {
  auto nnapi_delegate =
      static_cast<tflite::StatefulNnApiDelegate*>(from_delegate);
  return nnapi_delegate->GetNnApiErrno();
}
static constexpr TfLiteDelegatePlugin kPluginCApi{
    CreateDelegate,
    DestroyDelegate,
    DelegateErrno,
};
const TfLiteDelegatePlugin* TfLiteNnapiDelegatePluginCApi() {
  return &kPluginCApi;
}
}  