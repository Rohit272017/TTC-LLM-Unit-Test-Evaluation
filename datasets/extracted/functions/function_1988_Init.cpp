#include <stddef.h>
#include <stdint.h>
#include <map>
#include <memory>
#include <vector>
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace unique {
void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return nullptr;
}
void Free(TfLiteContext* context, void* buffer) {}
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  static const int kOutputUniqueTensor = 0;
  static const int kOutputIndexTensor = 1;
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 2);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output_unique_tensor;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutputUniqueTensor,
                                           &output_unique_tensor));
  TfLiteTensor* output_index_tensor;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutputIndexTensor,
                                           &output_index_tensor));
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 1);
  TfLiteIntArray* output_index_shape = TfLiteIntArrayCopy(input->dims);
  SetTensorToDynamic(output_unique_tensor);
  return context->ResizeTensor(context, output_index_tensor,
                               output_index_shape);
}
namespace {
template <typename T, typename I>
TfLiteStatus EvalImpl(TfLiteContext* context, const TfLiteTensor* input,
                      TfLiteNode* node) {
  std::map<T, int> unique_values;
  TfLiteTensor* output_indexes;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 1, &output_indexes));
  std::vector<T> output_values;
  I* indexes = GetTensorData<I>(output_indexes);
  const T* data = GetTensorData<T>(input);
  const int num_elements = NumElements(input);
  for (int i = 0; i < num_elements; ++i) {
    const auto element_it = unique_values.find(data[i]);
    if (element_it != unique_values.end()) {
      indexes[i] = element_it->second;
    } else {
      const int unique_index = unique_values.size();
      unique_values[data[i]] = unique_index;
      indexes[i] = unique_index;
      output_values.push_back(data[i]);
    }
  }
  TfLiteTensor* unique_output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &unique_output));
  std::unique_ptr<TfLiteIntArray, void (*)(TfLiteIntArray*)> shape(
      TfLiteIntArrayCreate(NumDimensions(input)), TfLiteIntArrayFree);
  shape->data[0] = unique_values.size();
  TF_LITE_ENSURE_STATUS(
      context->ResizeTensor(context, unique_output, shape.release()));
  T* output_unique_values = GetTensorData<T>(unique_output);
  for (int i = 0; i < output_values.size(); ++i) {
    output_unique_values[i] = output_values[i];
  }
  return kTfLiteOk;
}
template <typename T>
TfLiteStatus EvalImpl(TfLiteContext* context, const TfLiteTensor* input,
                      TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteUniqueParams*>(node->builtin_data);
  if (params == nullptr) {
    TF_LITE_KERNEL_LOG(context, "Null params passed");
    return kTfLiteError;
  }
  switch (params->index_out_type) {
    case kTfLiteInt32:
      return EvalImpl<T, int32_t>(context, input, node);
    case kTfLiteInt64:
      return EvalImpl<T, int64_t>(context, input, node);
    default:
      TF_LITE_KERNEL_LOG(
          context,
          "Unique index output array can only be Int32 or In64, requested: %s",
          TfLiteTypeGetName(params->index_out_type));
  }
  return kTfLiteError;
}
}  
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output_index_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, 1, &output_index_tensor));
  TF_LITE_ENSURE_EQ(context, NumElements(output_index_tensor),
                    NumElements(input));
  switch (input->type) {
    case kTfLiteInt8:
      TF_LITE_ENSURE_STATUS(EvalImpl<int8_t>(context, input, node));
      break;
    case kTfLiteInt16:
      TF_LITE_ENSURE_STATUS(EvalImpl<int16_t>(context, input, node));
      break;
    case kTfLiteInt32:
      TF_LITE_ENSURE_STATUS(EvalImpl<int32_t>(context, input, node));
      break;
    case kTfLiteInt64:
      TF_LITE_ENSURE_STATUS(EvalImpl<int64_t>(context, input, node));
      break;
    case kTfLiteFloat32:
      TF_LITE_ENSURE_STATUS(EvalImpl<float>(context, input, node));
      break;
    case kTfLiteUInt8:
      TF_LITE_ENSURE_STATUS(EvalImpl<uint8_t>(context, input, node));
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Currently Unique doesn't support type: %s",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}
}  
TfLiteRegistration* Register_UNIQUE() {
  static TfLiteRegistration r = {unique::Init, unique::Free, unique::Prepare,
                                 unique::Eval};
  return &r;
}
}  
}  
}  