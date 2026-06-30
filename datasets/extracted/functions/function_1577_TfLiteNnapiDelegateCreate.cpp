#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_c_api.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/nnapi/sl/public/NeuralNetworksSupportLibraryImpl.h"
TfLiteDelegate* TfLiteNnapiDelegateCreate(
    const TfLiteNnapiDelegateOptions* options) {
  tflite::StatefulNnApiDelegate::StatefulNnApiDelegate::Options
      internal_options;
  internal_options.execution_preference =
      static_cast<tflite::StatefulNnApiDelegate::StatefulNnApiDelegate::
                      Options::ExecutionPreference>(
          options->execution_preference);
  internal_options.accelerator_name = options->accelerator_name;
  internal_options.cache_dir = options->cache_dir;
  internal_options.model_token = options->model_token;
  internal_options.disallow_nnapi_cpu = options->disallow_nnapi_cpu;
  internal_options.max_number_delegated_partitions =
      options->max_number_delegated_partitions;
  internal_options.allow_fp16 = options->allow_fp16;
  tflite::StatefulNnApiDelegate* delegate = nullptr;
  if (options->nnapi_support_library_handle) {
    delegate = new tflite::StatefulNnApiDelegate(
        static_cast<NnApiSLDriverImplFL5*>(
            options->nnapi_support_library_handle),
        internal_options);
  } else {
    delegate = new tflite::StatefulNnApiDelegate(internal_options);
  }
  return delegate;
}
TfLiteNnapiDelegateOptions TfLiteNnapiDelegateOptionsDefault() {
  TfLiteNnapiDelegateOptions result = {};
  tflite::StatefulNnApiDelegate::Options options;
  result.execution_preference =
      static_cast<TfLiteNnapiDelegateOptions::ExecutionPreference>(
          options.execution_preference);
  result.accelerator_name = options.accelerator_name;
  result.cache_dir = options.cache_dir;
  result.model_token = options.model_token;
  result.disallow_nnapi_cpu = options.disallow_nnapi_cpu;
  result.max_number_delegated_partitions =
      options.max_number_delegated_partitions;
  result.allow_fp16 = options.allow_fp16;
  result.nnapi_support_library_handle = nullptr;
  return result;
}
void TfLiteNnapiDelegateDelete(TfLiteDelegate* delegate) {
  if (delegate == nullptr) return;
  delete static_cast<tflite::StatefulNnApiDelegate*>(delegate);
}