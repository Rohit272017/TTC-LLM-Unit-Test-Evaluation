#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_ELU_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_ELU_H_
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/types.h"
namespace tflite {
namespace reference_ops {
inline void Elu(const RuntimeShape& input_shape, const float* input_data,
                const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    output_data[i] = val < 0.0f ? TfLiteExpm1(val) : val;
  }
}
}  
}  
#endif  