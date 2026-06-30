#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/l2normalization.h"
#include "tensorflow/lite/kernels/internal/reference/l2normalization.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace l2norm {
enum KernelType {
  kReference,
  kGenericOptimized,
};
constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteL2NormParams*>(node->builtin_data);
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TF_LITE_ENSURE(context, NumDimensions(input) <= 4);
  TF_LITE_ENSURE(context, output->type == kTfLiteFloat32 ||
                              output->type == kTfLiteUInt8 ||
                              output->type == kTfLiteInt8);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, output->params.scale, (1. / 128.));
    if (output->type == kTfLiteUInt8) {
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 128);
    }
    if (output->type == kTfLiteInt8) {
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    }
  }
  TF_LITE_ENSURE_EQ(context, params->activation, kTfLiteActNone);
  TfLiteIntArray* output_size = TfLiteIntArrayCopy(input->dims);
  return context->ResizeTensor(context, output, output_size);
}
template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  const float epsilon = 1e-6f;
  if (output->type == kTfLiteFloat32) {
#define TF_LITE_L2NORM(type)                                                 \
  tflite::L2NormalizationParams op_params;                                   \
  op_params.input_zero_point = 0;                                            \
  type::L2Normalization(op_params, GetTensorShape(input),                    \
                        GetTensorData<float>(input), GetTensorShape(output), \
                        GetTensorData<float>(output), epsilon)
    if (kernel_type == kReference) {
      TF_LITE_L2NORM(reference_ops);
    }
    if (kernel_type == kGenericOptimized) {
      TF_LITE_L2NORM(optimized_ops);
    }
#undef TF_LITE_L2NORM
  } else if (output->type == kTfLiteUInt8) {
#define TF_LITE_L2NORM(type)                                                 \
  tflite::L2NormalizationParams op_params;                                   \
  op_params.input_zero_point = input->params.zero_point;                     \
  type::L2Normalization(op_params, GetTensorShape(input),                    \
                        GetTensorData<uint8>(input), GetTensorShape(output), \
                        GetTensorData<uint8>(output))
    if (kernel_type == kReference) {
      TF_LITE_L2NORM(reference_ops);
    }
    if (kernel_type == kGenericOptimized) {
      TF_LITE_L2NORM(optimized_ops);
    }
#undef TF_LITE_L2NORM
  } else if (output->type == kTfLiteInt8) {
    const auto input_shape = GetTensorShape(input);
    const auto output_shape = GetTensorShape(output);
    const int trailing_dim = input_shape.DimensionsCount() - 1;
    const int depth =
        MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
    const int outer_size =
        MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
    reference_integer_ops::L2Normalization(input->params.zero_point, outer_size,
                                           depth, GetTensorData<int8>(input),
                                           GetTensorData<int8>(output));
  } else {
    TF_LITE_KERNEL_LOG(context, "Output type is %s, requires float.",
                       TfLiteTypeGetName(output->type));
    return kTfLiteError;
  }
  return kTfLiteOk;
}
}  
TfLiteRegistration* Register_L2NORM_REF() {
  static TfLiteRegistration r = {nullptr, nullptr, l2norm::Prepare,
                                 l2norm::Eval<l2norm::kReference>};
  return &r;
}
TfLiteRegistration* Register_L2NORM_GENERIC_OPT() {
  static TfLiteRegistration r = {nullptr, nullptr, l2norm::Prepare,
                                 l2norm::Eval<l2norm::kGenericOptimized>};
  return &r;
}
TfLiteRegistration* Register_L2_NORMALIZATION() {
  return Register_L2NORM_GENERIC_OPT();
}
}  
}  
}  