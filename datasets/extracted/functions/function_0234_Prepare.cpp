#include <complex>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace complex {
static const int kInputTensor = 0;
static const int kOutputTensor = 0;
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, input->type == kTfLiteComplex64 ||
                              input->type == kTfLiteComplex128);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  if (input->type == kTfLiteComplex64) {
    TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat32);
  } else {
    TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat64);
  }
  TfLiteIntArray* output_shape = TfLiteIntArrayCopy(input->dims);
  return context->ResizeTensor(context, output, output_shape);
}
template <typename T, typename ExtractF>
void ExtractData(const TfLiteTensor* input, ExtractF extract_func,
                 TfLiteTensor* output) {
  const std::complex<T>* input_data = GetTensorData<std::complex<T>>(input);
  T* output_data = GetTensorData<T>(output);
  const int input_size = NumElements(input);
  for (int i = 0; i < input_size; ++i) {
    *output_data++ = extract_func(*input_data++);
  }
}
TfLiteStatus EvalReal(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  switch (input->type) {
    case kTfLiteComplex64: {
      ExtractData<float>(
          input,
          static_cast<float (*)(const std::complex<float>&)>(std::real<float>),
          output);
      break;
    }
    case kTfLiteComplex128: {
      ExtractData<double>(input,
                          static_cast<double (*)(const std::complex<double>&)>(
                              std::real<double>),
                          output);
      break;
    }
    default: {
      TF_LITE_KERNEL_LOG(context,
                         "Unsupported input type, Real op only supports "
                         "complex input, but got: %s",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}
TfLiteStatus EvalImag(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  switch (input->type) {
    case kTfLiteComplex64: {
      ExtractData<float>(
          input,
          static_cast<float (*)(const std::complex<float>&)>(std::imag<float>),
          output);
      break;
    }
    case kTfLiteComplex128: {
      ExtractData<double>(input,
                          static_cast<double (*)(const std::complex<double>&)>(
                              std::imag<double>),
                          output);
      break;
    }
    default: {
      TF_LITE_KERNEL_LOG(context,
                         "Unsupported input type, Imag op only supports "
                         "complex input, but got: %s",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}
TfLiteStatus EvalAbs(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  switch (input->type) {
    case kTfLiteComplex64: {
      ExtractData<float>(
          input,
          static_cast<float (*)(const std::complex<float>&)>(std::abs<float>),
          output);
      break;
    }
    case kTfLiteComplex128: {
      ExtractData<double>(input,
                          static_cast<double (*)(const std::complex<double>&)>(
                              std::abs<double>),
                          output);
      break;
    }
    default: {
      TF_LITE_KERNEL_LOG(context,
                         "Unsupported input type, ComplexAbs op only supports "
                         "complex input, but got: %s",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}
}  
TfLiteRegistration* Register_REAL() {
  static TfLiteRegistration r = {nullptr, nullptr,
                                 complex::Prepare, complex::EvalReal};
  return &r;
}
TfLiteRegistration* Register_IMAG() {
  static TfLiteRegistration r = {nullptr, nullptr,
                                 complex::Prepare, complex::EvalImag};
  return &r;
}
TfLiteRegistration* Register_COMPLEX_ABS() {
  static TfLiteRegistration r = {nullptr, nullptr,
                                 complex::Prepare, complex::EvalAbs};
  return &r;
}
}  
}  
}  