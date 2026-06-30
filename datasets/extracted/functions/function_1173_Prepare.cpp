#include <stdint.h>
#include <string.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace zeros_like {
constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  output->type = input->type;
  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  const int num_elements = NumElements(input);
  switch (input->type) {
    case kTfLiteInt64:
      memset(GetTensorData<int64_t>(output), 0, num_elements * sizeof(int64_t));
      break;
    case kTfLiteInt32:
      memset(GetTensorData<int32_t>(output), 0, num_elements * sizeof(int32_t));
      break;
    case kTfLiteFloat32:
      memset(GetTensorData<float>(output), 0, num_elements * sizeof(float));
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "ZerosLike only currently supports int64, int32, "
                         "and float32, got %d.",
                         input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}
}  
TfLiteRegistration* Register_ZEROS_LIKE() {
  static TfLiteRegistration r = {nullptr, nullptr,
                                 zeros_like::Prepare, zeros_like::Eval};
  return &r;
}
}  
}  
}  