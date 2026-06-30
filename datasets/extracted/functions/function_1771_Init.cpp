#include <stddef.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/binary_function.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace logical {
namespace {
constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;
struct OpData {
  bool requires_broadcast;
};
void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new OpData;
  data->requires_broadcast = false;
  return data;
}
void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input1->type, input2->type);
  const TfLiteType type = input1->type;
  if (type != kTfLiteBool) {
    TF_LITE_KERNEL_LOG(context, "Logical ops only support bool type.");
    return kTfLiteError;
  }
  output->type = type;
  data->requires_broadcast = !HaveSameShapes(input1, input2);
  TfLiteIntArray* output_size = nullptr;
  if (data->requires_broadcast) {
    TF_LITE_ENSURE_OK(context, CalculateShapeForBroadcast(
                                   context, input1, input2, &output_size));
  } else {
    output_size = TfLiteIntArrayCopy(input1->dims);
  }
  return context->ResizeTensor(context, output, output_size);
}
TfLiteStatus LogicalImpl(TfLiteContext* context, TfLiteNode* node,
                         bool (*func)(bool, bool)) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  if (data->requires_broadcast) {
    reference_ops::BroadcastBinaryFunction4DSlow<bool, bool, bool>(
        GetTensorShape(input1), GetTensorData<bool>(input1),
        GetTensorShape(input2), GetTensorData<bool>(input2),
        GetTensorShape(output), GetTensorData<bool>(output), func);
  } else {
    reference_ops::BinaryFunction<bool, bool, bool>(
        GetTensorShape(input1), GetTensorData<bool>(input1),
        GetTensorShape(input2), GetTensorData<bool>(input2),
        GetTensorShape(output), GetTensorData<bool>(output), func);
  }
  return kTfLiteOk;
}
bool LogicalOr(bool x, bool y) { return x || y; }
TfLiteStatus LogicalOrEval(TfLiteContext* context, TfLiteNode* node) {
  return LogicalImpl(context, node, LogicalOr);
}
bool LogicalAnd(bool x, bool y) { return x && y; }
TfLiteStatus LogicalAndEval(TfLiteContext* context, TfLiteNode* node) {
  return LogicalImpl(context, node, LogicalAnd);
}
}  
}  
TfLiteRegistration* Register_LOGICAL_OR() {
  static TfLiteRegistration r = {logical::Init, logical::Free, logical::Prepare,
                                 logical::LogicalOrEval};
  return &r;
}
TfLiteRegistration* Register_LOGICAL_AND() {
  static TfLiteRegistration r = {logical::Init, logical::Free, logical::Prepare,
                                 logical::LogicalAndEval};
  return &r;
}
}  
}  
}  