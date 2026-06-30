#include <memory>
#include <utility>
#include "tensorflow/lite/acceleration/configuration/c/delegate_plugin.h"
#include "tensorflow/lite/acceleration/configuration/c/stable_delegate.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/sample_stable_delegate.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/stable_delegate_interface.h"
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"
namespace {
TfLiteOpaqueDelegate* SampleStableDelegateCreateFunc(
    const void* tflite_settings) {
  auto delegate = std::make_unique<tflite::example::SampleStableDelegate>();
  return tflite::TfLiteOpaqueDelegateFactory::CreateSimpleDelegate(
      std::move(delegate));
}
void SampleStableDelegateDestroyFunc(
    TfLiteOpaqueDelegate* sample_stable_delegate) {
  tflite::TfLiteOpaqueDelegateFactory::DeleteSimpleDelegate(
      sample_stable_delegate);
}
int SampleStableDelegateErrnoFunc(
    TfLiteOpaqueDelegate* sample_stable_delegate) {
  return 0;
}
const TfLiteOpaqueDelegatePlugin sample_stable_delegate_plugin = {
    SampleStableDelegateCreateFunc, SampleStableDelegateDestroyFunc,
    SampleStableDelegateErrnoFunc};
const TfLiteStableDelegate sample_stable_delegate = {
    TFL_STABLE_DELEGATE_ABI_VERSION, tflite::example::kSampleStableDelegateName,
    tflite::example::kSampleStableDelegateVersion,
    &sample_stable_delegate_plugin};
}  
extern "C" const TfLiteStableDelegate TFL_TheStableDelegate =
    sample_stable_delegate;