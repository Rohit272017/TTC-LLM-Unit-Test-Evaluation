#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_HARD_SWISH_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_HARD_SWISH_H_
#include <algorithm>
#include "ruy/profiler/instrumentation.h"  
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
namespace tflite {
namespace reference_ops {
inline int16_t SaturatingLeftShift(int16_t value, int amount) {
  int64_t result = static_cast<int64_t>(value) * (1 << amount);
  result = std::min<int64_t>(result, std::numeric_limits<int16_t>::max());
  result = std::max<int64_t>(result, std::numeric_limits<int16_t>::min());
  return result;
}
inline std::int16_t SaturatingDoublingHighMul(std::int16_t a, std::int16_t b) {
  bool overflow = a == b && a == std::numeric_limits<std::int16_t>::min();
  std::int32_t a_32(a);
  std::int32_t b_32(b);
  std::int32_t ab_32 = a_32 * b_32;
  std::int16_t ab_x2_high16 = static_cast<std::int16_t>((ab_32) / (1 << 15));
  return overflow ? std::numeric_limits<std::int16_t>::max() : ab_x2_high16;
}
template <typename T>
inline void HardSwish(const RuntimeShape& input_shape, const T* input_data,
                      const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("ReferenceHardSwish/Float");
  auto matching_size = MatchingFlatSize(input_shape, output_shape);
  const T* in_end = input_data + matching_size;
  for (; input_data < in_end; input_data++, output_data++) {
    const float in = *input_data;
    *output_data =
        in * std::min(static_cast<T>(6), std::max(static_cast<T>(0), in + 3)) /
        6;
  }
}
template <typename T>
inline void HardSwish(const HardSwishParams& params,
                      const RuntimeShape& input_shape, const T* input_data,
                      const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("ReferenceHardSwish/Quantized");
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; i++) {
    const int16_t input_value = input_data[i] - params.input_zero_point;
    const int16_t input_value_on_hires_input_scale = input_value * (1 << 7);
    const int16_t input_value_on_preshift_output_scale =
        gemmlowp::SaturatingRoundingDoublingHighMul(
            input_value_on_hires_input_scale,
            params.output_multiplier_fixedpoint_int16);
    int16_t reluish_value = input_value_on_hires_input_scale;
    if (params.reluish_multiplier_exponent > 0) {
      reluish_value = SaturatingLeftShift(
          reluish_value, params.reluish_multiplier_exponent - 1);
    }
    reluish_value = gemmlowp::SaturatingRoundingDoublingHighMul(
        reluish_value, params.reluish_multiplier_fixedpoint_int16);
    if (params.reluish_multiplier_exponent > 0) {
      reluish_value = SaturatingLeftShift(reluish_value, 1);
    }
    if (params.reluish_multiplier_exponent < 0) {
      reluish_value = gemmlowp::RoundingDivideByPOT(
          reluish_value, -params.reluish_multiplier_exponent);
    }
    reluish_value = (reluish_value + (1 << 15)) >> 1;
    const int16_t preshift_output_value = SaturatingDoublingHighMul(
        reluish_value, input_value_on_preshift_output_scale);
    int16_t output_value = gemmlowp::RoundingDivideByPOT(
        preshift_output_value, -params.output_multiplier_exponent);
    output_value += params.output_zero_point;
    output_value =
        std::min<int16_t>(output_value, std::numeric_limits<T>::max());
    output_value =
        std::max<int16_t>(output_value, std::numeric_limits<T>::min());
    output_data[i] = output_value;
  }
}
}  
}  
#endif  