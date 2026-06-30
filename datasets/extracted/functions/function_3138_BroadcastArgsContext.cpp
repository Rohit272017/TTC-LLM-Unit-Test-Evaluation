#include "tensorflow/lite/kernels/internal/reference/broadcast_args.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace broadcast_args {
constexpr int kShape1Tensor = 0;
constexpr int kShape2Tensor = 1;
constexpr int kOutputTensor = 0;
struct BroadcastArgsContext {
  BroadcastArgsContext(TfLiteContext* context, TfLiteNode* node) {
    shape1 = GetInput(context, node, kShape1Tensor);
    shape2 = GetInput(context, node, kShape2Tensor);
    output = GetOutput(context, node, kOutputTensor);
  }
  const TfLiteTensor* shape1;
  const TfLiteTensor* shape2;
  TfLiteTensor* output;
};
TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, NumInputs(node) == 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  BroadcastArgsContext op_context(context, node);
  TF_LITE_ENSURE(context, op_context.shape1->type == kTfLiteInt32 ||
                              op_context.shape1->type == kTfLiteInt64);
  TF_LITE_ENSURE_EQ(context, op_context.shape1->type, op_context.shape2->type);
  TF_LITE_ENSURE_EQ(context, op_context.shape1->type, op_context.output->type);
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context.shape1), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context.shape2), 1);
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(1);
  output_shape->data[0] = std::max(SizeOfDimension(op_context.shape1, 0),
                                   SizeOfDimension(op_context.shape2, 0));
  if (IsConstantOrPersistentTensor(op_context.shape1) &&
      IsConstantOrPersistentTensor(op_context.shape2)) {
    SetTensorToPersistentRo(op_context.output);
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, op_context.output,
                                                     output_shape));
    return EvalImpl(context, node);
  }
  return context->ResizeTensor(context, op_context.output, output_shape);
}
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  BroadcastArgsContext op_context(context, node);
  if (IsConstantOrPersistentTensor(op_context.output)) {
    return kTfLiteOk;
  } else {
    return EvalImpl(context, node);
  }
}
TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node) {
  BroadcastArgsContext op_context(context, node);
#define TF_LITE_BROADCAST_ARG(data_type)                                    \
  reference_ops::BroadcastArgs(GetTensorShape(op_context.shape1),           \
                               GetTensorData<data_type>(op_context.shape1), \
                               GetTensorShape(op_context.shape2),           \
                               GetTensorData<data_type>(op_context.shape2), \
                               GetTensorShape(op_context.output),           \
                               GetTensorData<data_type>(op_context.output))
  if (op_context.output->type == kTfLiteInt32) {
    TF_LITE_BROADCAST_ARG(int32_t);
  } else {
    TF_LITE_BROADCAST_ARG(int64_t);
  }
#undef TF_LITE_BROADCAST_ARG
  return kTfLiteOk;
}
}  
TfLiteRegistration* Register_BROADCAST_ARGS() {
  static TfLiteRegistration r = {nullptr, nullptr, broadcast_args::Prepare,
                                 broadcast_args::Eval};
  return &r;
}
}  
}  
}  