#ifndef XLA_FP_UTIL_H_
#define XLA_FP_UTIL_H_
#include <algorithm>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <utility>
#include "xla/types.h"
#include "xla/util.h"
namespace xla {
template <typename T>
constexpr bool IsZero(T x) {
  return x == static_cast<T>(0.0f);
}
template <typename T>
constexpr bool IsSignMinus(T x) {
  return x < 0;
}
template <typename T>
constexpr T Abs(T x) {
  if (IsZero(x)) {
    return x + static_cast<T>(0.0f);
  }
  return IsSignMinus(x) ? -x : x;
}
template <typename T>
constexpr bool IsNaN(T x) {
  return x != x;
}
template <typename T>
constexpr bool IsInfinite(T x) {
  return x == std::numeric_limits<T>::infinity() ||
         x == -std::numeric_limits<T>::infinity();
}
template <typename T>
constexpr bool IsFinite(T x) {
  return !IsNaN(x) && !IsInfinite(x);
}
template <typename T>
constexpr bool IsNormal(T x) {
  T abs_x = Abs(x);
  return abs_x >= std::numeric_limits<T>::min() &&
         abs_x <= std::numeric_limits<T>::max();
}
template <typename T>
constexpr bool IsSubnormal(T x) {
  T abs_x = Abs(x);
  return abs_x > static_cast<T>(0) && abs_x < std::numeric_limits<T>::min();
}
template <typename T>
constexpr T ScaleBase(T x, int n) {
  static_assert(is_specialized_floating_point_v<T>);
  while (n > 0 && IsFinite(x) && !IsZero(x)) {
    int multiplier_exponent =
        std::min(n, std::numeric_limits<T>::max_exponent - 1);
    x *= IPow(static_cast<T>(std::numeric_limits<T>::radix),
              multiplier_exponent);
    n -= multiplier_exponent;
  }
  for (; n < 0 && IsFinite(x) && !IsZero(x); ++n) {
    T shifted_x = x / std::numeric_limits<T>::radix;
    if (IsSubnormal(shifted_x)) {
      int scale_exponent = -((std::numeric_limits<T>::min_exponent - 1) -
                             (std::numeric_limits<T>::digits - 1)) +
                           n;
      if (scale_exponent < 0) {
        return x * static_cast<T>(0);
      }
      return x *
             ScaleBase(std::numeric_limits<T>::denorm_min(), scale_exponent);
    }
    x = shifted_x;
  }
  return x;
}
template <typename T>
constexpr std::optional<int> LogBase(T x) {
  if (IsNaN(x)) {
    return std::nullopt;
  }
  if (IsInfinite(x)) {
    return std::numeric_limits<int>::max();
  }
  if (IsZero(x)) {
    return std::numeric_limits<int>::min();
  }
  T abs_x = Abs(x);
  int exponent = 0;
  while (abs_x < static_cast<T>(1)) {
    abs_x *= std::numeric_limits<T>::radix;
    exponent -= 1;
  }
  while (abs_x >= std::numeric_limits<T>::radix) {
    abs_x /= std::numeric_limits<T>::radix;
    exponent += 1;
  }
  return exponent;
}
enum class RoundingDirection {
  kRoundTiesToEven,
  kRoundTowardsZero,
};
template <typename DstT, typename SrcT>
constexpr std::pair<DstT, DstT> SplitToFpPair(
    SrcT to_split, int num_high_trailing_zeros,
    RoundingDirection rounding_direction =
        RoundingDirection::kRoundTiesToEven) {
  constexpr auto kError =
      std::make_pair(std::numeric_limits<DstT>::quiet_NaN(),
                     std::numeric_limits<DstT>::quiet_NaN());
  if (num_high_trailing_zeros < 0) {
    return kError;
  }
  if (!IsFinite(to_split)) {
    return kError;
  }
  if (IsZero(to_split)) {
    DstT zero = static_cast<DstT>(to_split);
    return std::make_pair(zero, zero);
  }
  if (IsSignMinus(to_split)) {
    auto [high, low] =
        SplitToFpPair<DstT, SrcT>(Abs(to_split), num_high_trailing_zeros);
    return std::make_pair(-high, -low);
  }
  auto maybe_exponent = LogBase(to_split);
  if (!maybe_exponent.has_value()) {
    return kError;
  }
  int exponent = *maybe_exponent;
  constexpr int kMinNormalExponent =
      std::numeric_limits<DstT>::min_exponent - 1;
  const int effective_precision = std::numeric_limits<DstT>::digits -
                                  std::max(kMinNormalExponent - exponent, 0);
  const int high_bits_to_keep = effective_precision - num_high_trailing_zeros;
  if (high_bits_to_keep < 1) {
    return kError;
  }
  static_assert(std::numeric_limits<SrcT>::max_exponent - 1 >=
                std::numeric_limits<DstT>::digits);
  SrcT scaled_significand =
      ScaleBase(to_split, high_bits_to_keep - (exponent + 1));
  uint64_t integer_part = static_cast<uint64_t>(scaled_significand);
  SrcT fractional_part = scaled_significand - static_cast<SrcT>(integer_part);
  switch (rounding_direction) {
    case RoundingDirection::kRoundTiesToEven: {
      if (fractional_part > static_cast<SrcT>(0.5f) ||
          (fractional_part == static_cast<SrcT>(0.5f) &&
           integer_part % 2 == 1)) {
        integer_part += 1;
      }
      break;
    }
    case RoundingDirection::kRoundTowardsZero: {
      break;
    }
  }
  SrcT rounded = ScaleBase(static_cast<SrcT>(integer_part),
                           (exponent + 1) - high_bits_to_keep);
  DstT high = static_cast<DstT>(rounded);
  if (static_cast<SrcT>(high) != rounded) {
    return kError;
  }
  DstT low = static_cast<DstT>(to_split - double{high});
  return std::make_pair(high, low);
}
template <typename DstT, typename SrcT>
constexpr DstT RoundToPrecision(
    SrcT to_round, int precision = std::numeric_limits<DstT>::digits,
    RoundingDirection rounding_direction =
        RoundingDirection::kRoundTiesToEven) {
  auto [high, low] = SplitToFpPair<DstT, SrcT>(
      to_round,
      std::numeric_limits<DstT>::digits - precision,
      rounding_direction);
  return high;
}
template <typename DstT>
constexpr std::pair<DstT, DstT> Log2FloatPair(int num_high_trailing_zeros) {
  return SplitToFpPair<DstT>(M_LN2, num_high_trailing_zeros);
}
template <typename T>
constexpr T GoldbergUlp(T x) {
  if (IsZero(x) || IsSubnormal(x)) {
    return GoldbergUlp(std::numeric_limits<T>::min());
  }
  std::optional<int> maybe_exponent = LogBase(x);
  if (maybe_exponent.has_value(); const int exponent = *maybe_exponent) {
    return ScaleBase(std::numeric_limits<T>::epsilon(), exponent);
  }
  if constexpr (std::numeric_limits<T>::has_quiet_NaN) {
    return std::numeric_limits<T>::quiet_NaN();
  } else if constexpr (std::numeric_limits<T>::has_infinity) {
    return std::numeric_limits<T>::infinity();
  } else {
    return GoldbergUlp(std::numeric_limits<T>::max());
  }
}
template <typename T>
int64_t CalculateDistanceInFloats(T a, T b) {
  auto a_sign_and_magnitude = SignAndMagnitude(a);
  auto b_sign_and_magnitude = SignAndMagnitude(b);
  uint64_t a_distance_from_zero = a_sign_and_magnitude.first
                                      ? -a_sign_and_magnitude.second
                                      : a_sign_and_magnitude.second;
  uint64_t b_distance_from_zero = b_sign_and_magnitude.first
                                      ? -b_sign_and_magnitude.second
                                      : b_sign_and_magnitude.second;
  int64_t signed_distance = a_distance_from_zero - b_distance_from_zero;
  return std::abs(signed_distance);
}
}  
#endif  