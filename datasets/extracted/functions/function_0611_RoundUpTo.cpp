#ifndef TENSORSTORE_UTIL_DIVISION_H_
#define TENSORSTORE_UTIL_DIVISION_H_
#include <cassert>
#include <limits>
#include <type_traits>
namespace tensorstore {
template <typename IntegralType>
constexpr IntegralType RoundUpTo(IntegralType input,
                                 IntegralType rounding_value) {
  static_assert(std::is_integral<IntegralType>::value,
                "IntegralType must be an integral type.");
  assert(input >= 0 && rounding_value > 0);
  return ((input + rounding_value - 1) / rounding_value) * rounding_value;
}
template <typename IntegralType, bool ceil>
constexpr IntegralType CeilOrFloorOfRatio(IntegralType numerator,
                                          IntegralType denominator);
template <typename IntegralType>
constexpr IntegralType CeilOfRatio(IntegralType numerator,
                                   IntegralType denominator) {
  return CeilOrFloorOfRatio<IntegralType, true>(numerator, denominator);
}
template <typename IntegralType>
constexpr IntegralType FloorOfRatio(IntegralType numerator,
                                    IntegralType denominator) {
  return CeilOrFloorOfRatio<IntegralType, false>(numerator, denominator);
}
template <typename IntegralType, bool ceil>
constexpr IntegralType CeilOrFloorOfRatio(IntegralType numerator,
                                          IntegralType denominator) {
  const IntegralType rounded_toward_zero = numerator / denominator;
  const IntegralType intermediate_product = rounded_toward_zero * denominator;
  if constexpr (ceil) {  
    const bool needs_adjustment =
        (rounded_toward_zero >= 0) &&
        ((denominator > 0 && numerator > intermediate_product) ||
         (denominator < 0 && numerator < intermediate_product));
    const IntegralType adjustment = static_cast<IntegralType>(needs_adjustment);
    const IntegralType ceil_of_ratio = rounded_toward_zero + adjustment;
    return ceil_of_ratio;
  } else {
    const bool needs_adjustment =
        (rounded_toward_zero <= 0) &&
        ((denominator > 0 && numerator < intermediate_product) ||
         (denominator < 0 && numerator > intermediate_product));
    const IntegralType adjustment = static_cast<IntegralType>(needs_adjustment);
    const IntegralType floor_of_ratio = rounded_toward_zero - adjustment;
    return floor_of_ratio;
  }
}
template <typename IntegralType>
constexpr IntegralType NonnegativeMod(IntegralType numerator,
                                      IntegralType denominator) {
  assert(denominator > 0);
  IntegralType modulus = numerator % denominator;
  return modulus + (modulus < 0) * denominator;
}
template <typename IntegralType>
constexpr IntegralType GreatestCommonDivisor(IntegralType x, IntegralType y) {
  assert(x != 0 || y != 0);
  if (std::is_signed_v<IntegralType> &&
      x == std::numeric_limits<IntegralType>::min()) {
    x = x % y;
  }
  if (std::is_signed_v<IntegralType> &&
      y == std::numeric_limits<IntegralType>::min()) {
    y = y % x;
  }
  if (std::is_signed_v<IntegralType> && x < 0) x = -x;
  if (std::is_signed_v<IntegralType> && y < 0) y = -y;
  while (y != 0) {
    IntegralType r = x % y;
    x = y;
    y = r;
  }
  return x;
}
}  
#endif  