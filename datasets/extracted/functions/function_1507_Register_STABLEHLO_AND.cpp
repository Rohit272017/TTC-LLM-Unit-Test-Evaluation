#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/stablehlo_elementwise.h"
namespace tflite::ops::builtin {
TfLiteRegistration* Register_STABLEHLO_AND() {
  static TfLiteRegistration r = {nullptr, nullptr, ElementwisePrepare,
                                 ElementwiseEval<ComputationType::kAnd>};
  return &r;
}
}  