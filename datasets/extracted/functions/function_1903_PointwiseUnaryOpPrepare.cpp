#include <cmath>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/custom_ops_register.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
namespace tflite {
namespace ops {
namespace custom {
namespace sign {
TfLiteStatus PointwiseUnaryOpPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), 1);
  const TfLiteTensor* input = tflite::GetInput(context, node, 0);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TfLiteIntArray* output_shape = TfLiteIntArrayCopy(input->dims);
  return context->ResizeTensor(context, output, output_shape);
}
template <typename Op, typename T>
TfLiteStatus PointwiseUnaryOpDoEval(
    TfLiteContext* context,
    const TfLiteTensor* input,
    TfLiteTensor* output) {
  const T* data = tflite::GetTensorData<T>(input);
  T* data_output = tflite::GetTensorData<T>(output);
  const int64_t num_elements = NumElements(input);
  for (int64_t i = 0; i < num_elements; ++i) {
    data_output[i] = Op::template Eval<T>(data[i]);
  }
  return TfLiteStatus::kTfLiteOk;
}
template <typename Op>
TfLiteStatus PointwiseUnaryOpEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = tflite::GetInput(context, node, 0);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  switch (output->type) {
    case kTfLiteFloat32:
      TF_LITE_ENSURE_OK(
          context,
          (PointwiseUnaryOpDoEval<Op, float>(context, input, output)));
      break;
    case kTfLiteFloat64:
      TF_LITE_ENSURE_OK(
          context,
          (PointwiseUnaryOpDoEval<Op, double>(context, input, output)));
      break;
    default: {
      TF_LITE_KERNEL_LOG(context, "Unsupported datatype for sign output: %s",
                         TfLiteTypeGetName(output->type));
      return TfLiteStatus::kTfLiteError;
    }
  }
  return TfLiteStatus::kTfLiteOk;
}
struct Sign {
  template <typename T>
  static T Eval(T x) {
    if (x > 0) {
      return 1;
    }
    if (x < 0) {
      return -1;
    }
    return 0;
  }
};
}  
TfLiteRegistration* Register_SIGN() {
  static TfLiteRegistration r = {nullptr, nullptr,
                                 sign::PointwiseUnaryOpPrepare,
                                 sign::PointwiseUnaryOpEval<sign::Sign>};
  return &r;
}
}  
}  
}  