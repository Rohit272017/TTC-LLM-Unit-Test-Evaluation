#include <stdint.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_util.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace fill {
namespace {
constexpr int kDimsTensor = 0;
constexpr int kValueTensor = 1;
constexpr int kOutputTensor = 0;
template <typename T>
TfLiteStatus ResizeOutputImpl(TfLiteContext* context, const TfLiteTensor* dims,
                              TfLiteTensor* output) {
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(dims->dims->data[0]);
  for (int i = 0; i < output_shape->size; ++i) {
    T data = GetTensorData<T>(dims)[i];
    if (data < 0) {
      TfLiteIntArrayFree(output_shape);
      TF_LITE_KERNEL_LOG(context, "Fill dimensions must be >= 0 got %d",
                         dims->type);
      return kTfLiteError;
    }
    output_shape->data[i] = data;
  }
  return context->ResizeTensor(context, output, output_shape);
}
TfLiteStatus ResizeOutput(TfLiteContext* context, const TfLiteTensor* dims,
                          TfLiteTensor* output) {
  switch (dims->type) {
    case kTfLiteInt32:
      return ResizeOutputImpl<int32_t>(context, dims, output);
    case kTfLiteInt64:
      return ResizeOutputImpl<int64_t>(context, dims, output);
    default:
      TF_LITE_KERNEL_LOG(
          context,
          "Fill only currently supports int32, int64 for input 0, "
          "got %d.",
          dims->type);
      return kTfLiteError;
  }
}
}  
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* dims;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kDimsTensor, &dims));
  const TfLiteTensor* value;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kValueTensor, &value));
  TF_LITE_ENSURE_EQ(context, NumDimensions(dims), 1);
  const auto dtype = dims->type;
  TF_LITE_ENSURE(context, dtype == kTfLiteInt32 || dtype == kTfLiteInt64);
  TF_LITE_ENSURE_EQ(context, NumDimensions(value), 0);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  output->type = value->type;
  TF_LITE_ENSURE_EQ(context, output->params.scale, value->params.scale);
  TF_LITE_ENSURE_EQ(context, output->params.zero_point,
                    value->params.zero_point);
  if (value->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, value->params.zero_point, 0);
  }
  if (IsConstantOrPersistentTensor(dims)) {
    TF_LITE_ENSURE_OK(context, ResizeOutput(context, dims, output));
  } else {
    SetTensorToDynamic(output);
  }
  return kTfLiteOk;
}
TfLiteStatus FillString(const TfLiteTensor* value, TfLiteTensor* output) {
  DynamicBuffer buffer;
  const auto string_ref = GetString(value, 0);
  int n = 1;
  for (int i = 0; i < output->dims->size; ++i) {
    n *= output->dims->data[i];
  }
  for (int i = 0; i < n; ++i) {
    buffer.AddString(string_ref.str, string_ref.len);
  }
  buffer.WriteToTensor(output, nullptr);
  return kTfLiteOk;
}
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* value;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kValueTensor, &value));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  if (IsDynamicTensor(output)) {
    const TfLiteTensor* dims;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kDimsTensor, &dims));
    TF_LITE_ENSURE_OK(context, ResizeOutput(context, dims, output));
  }
#define TF_LITE_FILL(data_type)                                               \
  reference_ops::Fill(GetTensorShape(value), GetTensorData<data_type>(value), \
                      GetTensorShape(output),                                 \
                      GetTensorData<data_type>(output))
  switch (output->type) {
    case kTfLiteInt8:
      TF_LITE_FILL(int8_t);
      break;
    case kTfLiteInt16:
      TF_LITE_FILL(int16_t);
      break;
    case kTfLiteInt32:
      TF_LITE_FILL(int32_t);
      break;
    case kTfLiteInt64:
      TF_LITE_FILL(int64_t);
      break;
    case kTfLiteFloat16:
      TF_LITE_FILL(Eigen::half);
      break;
    case kTfLiteFloat32:
      TF_LITE_FILL(float);
      break;
    case kTfLiteBool:
      TF_LITE_FILL(bool);
      break;
    case kTfLiteString:
      FillString(value, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(
          context,
          "Fill only currently supports int8, int16, int32, int64, float32, "
          "bool, string for input 1, got %d.",
          value->type);
      return kTfLiteError;
  }
#undef TF_LITE_FILL
  return kTfLiteOk;
}
}  
TfLiteRegistration* Register_FILL() {
  static TfLiteRegistration r = {nullptr, nullptr,
                                 fill::Prepare, fill::Eval};
  return &r;
}
}  
}  
}  