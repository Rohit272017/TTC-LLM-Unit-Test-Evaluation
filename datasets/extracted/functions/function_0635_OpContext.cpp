#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace fake_quant {
enum KernelType {
  kReference,
};
struct OpContext {
  OpContext(TfLiteContext* context, TfLiteNode* node) {
    input = GetInput(context, node, 0);
    output = GetOutput(context, node, 0);
  }
  const TfLiteTensor* input;
  TfLiteTensor* output;
};
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const auto* params =
      reinterpret_cast<TfLiteFakeQuantParams*>(node->builtin_data);
  if (params->narrow_range) {
    TF_LITE_KERNEL_LOG(
        context,
        "narrow_range FakeQuant is not currently supported at runtime. "
        "narrow_range is only meant to be applied to weights, not activations");
    return kTfLiteError;
  }
  OpContext op_context(context, node);
  TfLiteIntArray* output_dims = TfLiteIntArrayCopy(op_context.input->dims);
  op_context.output->type = op_context.input->type;
  return context->ResizeTensor(context, op_context.output, output_dims);
}
template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);
  const auto* params =
      reinterpret_cast<TfLiteFakeQuantParams*>(node->builtin_data);
  tflite::FakeQuantParams op_params;
  op_params.num_bits = params->num_bits;
  op_params.minmax.min = params->min;
  op_params.minmax.max = params->max;
  reference_ops::FakeQuant(op_params, GetTensorShape(op_context.input),
                           GetTensorData<float>(op_context.input),
                           GetTensorShape(op_context.output),
                           GetTensorData<float>(op_context.output));
  return kTfLiteOk;
}
}  
TfLiteRegistration* Register_FAKE_QUANT_REF() {
  static TfLiteRegistration r = {nullptr, nullptr, fake_quant::Prepare,
                                 fake_quant::Eval<fake_quant::kReference>};
  return &r;
}
TfLiteRegistration* Register_FAKE_QUANT() { return Register_FAKE_QUANT_REF(); }
}  
}  
}  