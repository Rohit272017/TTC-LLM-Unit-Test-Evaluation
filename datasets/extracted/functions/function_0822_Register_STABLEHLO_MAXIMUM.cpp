#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/stablehlo_elementwise.h"
namespace tflite::ops::builtin {
TfLiteRegistration* Register_STABLEHLO_MAXIMUM() {
  static TfLiteRegistration r = {nullptr, nullptr, ElementwisePrepare,
                                 ElementwiseEval<ComputationType::kMax>};
  return &r;
}
TfLiteRegistration* Register_STABLEHLO_MINIMUM() {
  static TfLiteRegistration r = {nullptr, nullptr, ElementwisePrepare,
                                 ElementwiseEval<ComputationType::kMin>};
  return &r;
}
}  