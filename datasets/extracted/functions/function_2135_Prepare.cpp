#include <stdint.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace rank {
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
  output->type = kTfLiteInt32;
  SetTensorToPersistentRo(output);
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(0);
  TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, output, output_size));
  TF_LITE_ENSURE_EQ(context, NumDimensions(output), 0);
  if (output->type == kTfLiteInt32) {
    int32_t* output_data = GetTensorData<int32_t>(output);
    *output_data = NumDimensions(input);
  } else {
    return kTfLiteError;
  }
  return kTfLiteOk;
}
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}
}  
TfLiteRegistration* Register_RANK() {
  static TfLiteRegistration r = {nullptr, nullptr, rank::Prepare, rank::Eval};
  return &r;
}
}  
}  
}  