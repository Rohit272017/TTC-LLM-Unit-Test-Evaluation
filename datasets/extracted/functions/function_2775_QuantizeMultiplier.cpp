#include "tensorflow/compiler/mlir/lite/kernels/internal/quantization_util.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include "tensorflow/compiler/mlir/lite/kernels/internal/compatibility_macros.h"
#include "tensorflow/compiler/mlir/lite/kernels/internal/cppmath.h"
namespace tflite_migration {
namespace {
constexpr uint64_t kSignMask = 0x8000000000000000LL;
constexpr uint64_t kExponentMask = 0x7ff0000000000000LL;
constexpr int32_t kExponentShift = 52;
constexpr int32_t kExponentBias = 1023;
constexpr uint32_t kExponentIsBadNum = 0x7ff;
constexpr uint64_t kFractionMask = 0x000fffffffc00000LL;
constexpr uint32_t kFractionShift = 22;
constexpr uint32_t kFractionRoundingMask = 0x003fffff;
constexpr uint32_t kFractionRoundingThreshold = 0x00200000;
}  
void QuantizeMultiplier(double double_multiplier, int32_t* quantized_multiplier,
                        int* shift) {
#if TFLITE_SINGLE_ROUNDING
#endif
  if (double_multiplier == 0.) {
    *quantized_multiplier = 0;
    *shift = 0;
    return;
  }
#ifdef TFLITE_EMULATE_FLOAT
  int64_t q_fixed = IntegerFrExp(double_multiplier, shift);
#else   
  const double q = std::frexp(double_multiplier, shift);
  auto q_fixed = static_cast<int64_t>(TfLiteRound(q * (1LL << 31)));
#endif  
  TFLITE_DCHECK(q_fixed <= (1LL << 31));
  if (q_fixed == (1LL << 31)) {
    q_fixed /= 2;
    ++*shift;
  }
  TFLITE_DCHECK_LE(q_fixed, std::numeric_limits<int32_t>::max());
  if (*shift < -31) {
    *shift = 0;
    q_fixed = 0;
  }
#if TFLITE_SINGLE_ROUNDING
  if (*shift > 30) {
    *shift = 30;
    q_fixed = (1LL << 31) - 1;
  }
#endif
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
}
void QuantizeMultiplierGreaterThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int* left_shift) {
  TFLITE_DCHECK_GT(double_multiplier, 1.);
  QuantizeMultiplier(double_multiplier, quantized_multiplier, left_shift);
  TFLITE_DCHECK_GE(*left_shift, 0);
}
int64_t IntegerFrExp(double input, int* shift) {
  TFLITE_DCHECK_EQ(8, sizeof(double));
  union {
    double double_value;
    uint64_t double_as_uint;
  } cast_union;
  cast_union.double_value = input;
  const uint64_t u = cast_union.double_as_uint;
  if ((u & ~kSignMask) == 0) {
    *shift = 0;
    return 0;
  }
  const uint32_t exponent_part = ((u & kExponentMask) >> kExponentShift);
  if (exponent_part == kExponentIsBadNum) {
    *shift = std::numeric_limits<int>::max();
    if (u & kFractionMask) {
      return 0;
    } else {
      if (u & kSignMask) {
        return std::numeric_limits<int64_t>::min();
      } else {
        return std::numeric_limits<int64_t>::max();
      }
    }
  }
  *shift = (exponent_part - kExponentBias) + 1;
  int64_t fraction = 0x40000000 + ((u & kFractionMask) >> kFractionShift);
  if ((u & kFractionRoundingMask) > kFractionRoundingThreshold) {
    fraction += 1;
  }
  if (u & kSignMask) {
    fraction *= -1;
  }
  return fraction;
}
double DoubleFromFractionAndShift(int64_t fraction, int shift) {
  union {
    double double_value;
    uint64_t double_as_uint;
  } result;
  if (shift == std::numeric_limits<int>::max()) {
    if (fraction == 0) {
      return std::numeric_limits<double>::quiet_NaN();
    } else if (fraction > 0) {
      return std::numeric_limits<double>::infinity();
    } else {
      return -std::numeric_limits<double>::infinity();
    }
  }
  if (fraction == 0) {
    result.double_as_uint = 0;
    return result.double_value;
  }
  bool is_negative = (fraction < 0);
  int64_t encoded_fraction = is_negative ? -fraction : fraction;
  int64_t encoded_shift = (shift - 1);
  while (encoded_fraction < 0x40000000) {
    encoded_fraction *= 2;
    encoded_shift -= 1;
  }
  while (encoded_fraction > 0x80000000) {
    encoded_fraction /= 2;
    encoded_shift += 1;
  }
  encoded_fraction -= 0x40000000;
  if (encoded_shift < -1022) {
    encoded_shift = -1023;
  } else if (encoded_shift > 1022) {
    encoded_shift = 1023;
  }
  encoded_shift += kExponentBias;
  uint64_t encoded_sign = is_negative ? kSignMask : 0;
  result.double_as_uint = encoded_sign | (encoded_shift << kExponentShift) |
                          (encoded_fraction << kFractionShift);
  return result.double_value;
}
double IntegerDoubleMultiply(double a, double b) {
  int a_shift;
  const int64_t a_fraction = IntegerFrExp(a, &a_shift);
  int b_shift;
  const int64_t b_fraction = IntegerFrExp(b, &b_shift);
  if (a_shift == std::numeric_limits<int>::max() ||
      (b_shift == std::numeric_limits<int>::max())) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const int result_shift = a_shift + b_shift + 1;
  const int64_t result_fraction = (a_fraction * b_fraction) >> 32;
  return DoubleFromFractionAndShift(result_fraction, result_shift);
}
int IntegerDoubleCompare(double a, double b) {
  int a_shift;
  const int64_t a_fraction = IntegerFrExp(a, &a_shift);
  int b_shift;
  const int64_t b_fraction = IntegerFrExp(b, &b_shift);
  if (a_shift == std::numeric_limits<int>::max() ||
      (b_shift == std::numeric_limits<int>::max())) {
    return 1;
  }
  if ((a_fraction == 0) && (b_fraction < 0)) {
    return 1;
  } else if ((a_fraction < 0) && (b_fraction == 0)) {
    return -1;
  } else if (a_shift < b_shift) {
    return -1;
  } else if (a_shift > b_shift) {
    return 1;
  } else if (a_fraction < b_fraction) {
    return -1;
  } else if (a_fraction > b_fraction) {
    return 1;
  } else {
    return 0;
  }
}
void PreprocessSoftmaxScaling(double beta, double input_scale,
                              int input_integer_bits,
                              int32_t* quantized_multiplier, int* left_shift) {
#if TFLITE_SINGLE_ROUNDING
  const double max_real_multiplier = (1LL << 30) - 1.0;
#else
  const double max_real_multiplier = (1LL << 31) - 1.0;
#endif
#ifdef TFLITE_EMULATE_FLOAT
  const double input_beta = IntegerDoubleMultiply(beta, input_scale);
  int shift;
  int64_t fraction = IntegerFrExp(input_beta, &shift);
  shift += (31 - input_integer_bits);
  double input_beta_real_multiplier =
      DoubleFromFractionAndShift(fraction, shift);
  if (IntegerDoubleCompare(input_beta_real_multiplier, max_real_multiplier) >
      0) {
    input_beta_real_multiplier = max_real_multiplier;
  }
#else   
  const double input_beta_real_multiplier =
      std::min<double>(beta * input_scale * (1 << (31 - input_integer_bits)),
                       max_real_multiplier);
#endif  
  QuantizeMultiplierGreaterThanOne(input_beta_real_multiplier,
                                   quantized_multiplier, left_shift);
}
int CalculateInputRadius(int input_integer_bits, int input_left_shift,
                         int total_signed_bits) {
#ifdef TFLITE_EMULATE_FLOAT
  int64_t result = (1 << input_integer_bits) - 1;
  result <<= (total_signed_bits - input_integer_bits);
  result >>= input_left_shift;
  return result;
#else   
  const double max_input_rescaled =
      1.0 * ((1 << input_integer_bits) - 1) *
      (1LL << (total_signed_bits - input_integer_bits)) /
      (1LL << input_left_shift);
  return static_cast<int>(std::floor(max_input_rescaled));
#endif  
}
}  