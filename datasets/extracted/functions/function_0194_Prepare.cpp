#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/variants/list_ops_lib.h"
#include "tensorflow/lite/kernels/variants/tensor_array.h"
namespace tflite {
namespace variants {
namespace ops {
namespace list_length {
namespace {
using ::tflite::variants::TensorArray;
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* list_input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &list_input));
  TF_LITE_ENSURE_TYPES_EQ(context, list_input->type, kTfLiteVariant);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, output->dims->size, 0);
  output->allocation_type = kTfLiteArenaRw;
  return kTfLiteOk;
}
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* list_input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &list_input));
  TF_LITE_ENSURE_EQ(context, list_input->allocation_type, kTfLiteVariantObject);
  const TensorArray* const input_arr =
      reinterpret_cast<TensorArray*>(list_input->data.data);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const int length = input_arr->NumElements();
  output->data.i32[0] = length;
  return kTfLiteOk;
}
}  
}  
TfLiteRegistration* Register_LIST_LENGTH() {
  static TfLiteRegistration r = {nullptr, nullptr, list_length::Prepare,
                                 list_length::Eval};
  return &r;
}
}  
}  
}  