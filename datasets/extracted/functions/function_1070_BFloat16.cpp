#ifndef TENSORSTORE_UTIL_BFLOAT16_H_
#define TENSORSTORE_UTIL_BFLOAT16_H_
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include "absl/base/casts.h"
#include <nlohmann/json_fwd.hpp>
namespace tensorstore {
class BFloat16;
}  
namespace std {
template <>
struct numeric_limits<::tensorstore::BFloat16>;
}  
namespace tensorstore {
namespace internal {
BFloat16 NumericFloat32ToBfloat16RoundNearestEven(float v);
BFloat16 Float32ToBfloat16RoundNearestEven(float v);
float Bfloat16ToFloat(BFloat16 v);
}  
class BFloat16 {
 public:
  constexpr BFloat16() : rep_(0) {}
  template <typename T,
            typename = std::enable_if_t<std::is_convertible_v<T, float>>>
  explicit BFloat16(T x) {
    if constexpr (std::is_same_v<T, bool>) {
      rep_ = static_cast<uint16_t>(x) * 0x3f80;
    } else if constexpr (std::numeric_limits<T>::is_integer) {
      *this = internal::NumericFloat32ToBfloat16RoundNearestEven(
          static_cast<float>(x));
    } else {
      *this =
          internal::Float32ToBfloat16RoundNearestEven(static_cast<float>(x));
    }
  }
  operator float() const { return internal::Bfloat16ToFloat(*this); }
  BFloat16& operator=(float v) { return *this = static_cast<BFloat16>(v); }
  BFloat16& operator=(bool v) { return *this = static_cast<BFloat16>(v); }
  template <typename T>
  std::enable_if_t<std::numeric_limits<T>::is_integer, BFloat16&> operator=(
      T v) {  
    return *this = static_cast<BFloat16>(v);
  }
#define TENSORSTORE_INTERNAL_BFLOAT16_ARITHMETIC_OP(OP)                 \
  friend BFloat16 operator OP(BFloat16 a, BFloat16 b) {                 \
    return BFloat16(static_cast<float>(a) OP static_cast<float>(b));    \
  }                                                                     \
  template <typename T>                                                 \
  friend std::enable_if_t<std::numeric_limits<T>::is_integer, BFloat16> \
  operator OP(BFloat16 a, T b) {                                        \
    return BFloat16(static_cast<float>(a) OP b);                        \
  }                                                                     \
  template <typename T>                                                 \
  friend std::enable_if_t<std::numeric_limits<T>::is_integer, BFloat16> \
  operator OP(T a, BFloat16 b) {                                        \
    return BFloat16(a OP static_cast<float>(b));                        \
  }                                                                     \
#define TENSORSTORE_INTERNAL_BFLOAT16_ARITHMETIC_ASSIGN_OP(OP)           \
  friend BFloat16& operator OP##=(BFloat16& a, BFloat16 b) {             \
    return a = BFloat16(static_cast<float>(a) OP static_cast<float>(b)); \
  }                                                                      \
  template <typename T>                                                  \
  friend std::enable_if_t<std::numeric_limits<T>::is_integer, BFloat16&> \
  operator OP##=(BFloat16& a, T b) {                                     \
    return a = BFloat16(static_cast<float>(a) OP b);                     \
  }                                                                      \
  TENSORSTORE_INTERNAL_BFLOAT16_ARITHMETIC_OP(+)
  TENSORSTORE_INTERNAL_BFLOAT16_ARITHMETIC_ASSIGN_OP(+)
  TENSORSTORE_INTERNAL_BFLOAT16_ARITHMETIC_OP(-)
  TENSORSTORE_INTERNAL_BFLOAT16_ARITHMETIC_ASSIGN_OP(-)
  TENSORSTORE_INTERNAL_BFLOAT16_ARITHMETIC_OP(*)
  TENSORSTORE_INTERNAL_BFLOAT16_ARITHMETIC_ASSIGN_OP(*)
  TENSORSTORE_INTERNAL_BFLOAT16_ARITHMETIC_OP(/)
  TENSORSTORE_INTERNAL_BFLOAT16_ARITHMETIC_ASSIGN_OP(/)
#undef TENSORSTORE_INTERNAL_BFLOAT16_ARITHMETIC_OP
#undef TENSORSTORE_INTERNAL_BFLOAT16_ARITHMETIC_ASSIGN_OP
  friend BFloat16 operator-(BFloat16 a) {
    BFloat16 result;
    result.rep_ = a.rep_ ^ 0x8000;
    return result;
  }
  friend BFloat16 operator+(BFloat16 a) { return a; }
  friend BFloat16 operator++(BFloat16& a) {
    a += BFloat16(1);
    return a;
  }
  friend BFloat16 operator--(BFloat16& a) {
    a -= BFloat16(1);
    return a;
  }
  friend BFloat16 operator++(BFloat16& a, int) {
    BFloat16 original_value = a;
    ++a;
    return original_value;
  }
  friend BFloat16 operator--(BFloat16& a, int) {
    BFloat16 original_value = a;
    --a;
    return original_value;
  }
  template <template <typename U, typename V, typename... Args>
            class ObjectType ,
            template <typename U, typename... Args>
            class ArrayType ,
            class StringType , class BooleanType ,
            class NumberIntegerType ,
            class NumberUnsignedType ,
            class NumberFloatType ,
            template <typename U> class AllocatorType ,
            template <typename T, typename SFINAE = void>
            class JSONSerializer ,
            class BinaryType >
  friend void to_json(
      ::nlohmann::basic_json<ObjectType, ArrayType, StringType, BooleanType,
                             NumberIntegerType, NumberUnsignedType,
                             NumberFloatType, AllocatorType, JSONSerializer,
                             BinaryType>& j,
      BFloat16 v) {
    j = static_cast<NumberFloatType>(v);
  }
  struct bitcast_construct_t {};
  explicit constexpr BFloat16(bitcast_construct_t, uint16_t rep) : rep_(rep) {}
  uint16_t rep_;
};
inline bool isinf(BFloat16 x) { return std::isinf(static_cast<float>(x)); }
inline bool signbit(BFloat16 x) { return std::signbit(static_cast<float>(x)); }
inline bool isnan(BFloat16 x) { return std::isnan(static_cast<float>(x)); }
inline bool isfinite(BFloat16 x) {
  return std::isfinite(static_cast<float>(x));
}
inline BFloat16 abs(BFloat16 x) {
  x.rep_ &= 0x7fff;
  return x;
}
inline BFloat16 exp(BFloat16 x) {
  return BFloat16(std::exp(static_cast<float>(x)));
}
inline BFloat16 exp2(BFloat16 x) {
  return BFloat16(std::exp2(static_cast<float>(x)));
}
inline BFloat16 expm1(BFloat16 x) {
  return BFloat16(std::expm1(static_cast<float>(x)));
}
inline BFloat16 log(BFloat16 x) {
  return BFloat16(std::log(static_cast<float>(x)));
}
inline BFloat16 log1p(BFloat16 x) {
  return BFloat16(std::log1p(static_cast<float>(x)));
}
inline BFloat16 log10(BFloat16 x) {
  return BFloat16(std::log10(static_cast<float>(x)));
}
inline BFloat16 log2(BFloat16 x) {
  return BFloat16(std::log2(static_cast<float>(x)));
}
inline BFloat16 sqrt(BFloat16 x) {
  return BFloat16(std::sqrt(static_cast<float>(x)));
}
inline BFloat16 pow(BFloat16 x, BFloat16 y) {
  return BFloat16(std::pow(static_cast<float>(x), static_cast<float>(y)));
}
inline BFloat16 sin(BFloat16 x) {
  return BFloat16(std::sin(static_cast<float>(x)));
}
inline BFloat16 cos(BFloat16 x) {
  return BFloat16(std::cos(static_cast<float>(x)));
}
inline BFloat16 tan(BFloat16 x) {
  return BFloat16(std::tan(static_cast<float>(x)));
}
inline BFloat16 asin(BFloat16 x) {
  return BFloat16(std::asin(static_cast<float>(x)));
}
inline BFloat16 acos(BFloat16 x) {
  return BFloat16(std::acos(static_cast<float>(x)));
}
inline BFloat16 atan(BFloat16 x) {
  return BFloat16(std::atan(static_cast<float>(x)));
}
inline BFloat16 sinh(BFloat16 x) {
  return BFloat16(std::sinh(static_cast<float>(x)));
}
inline BFloat16 cosh(BFloat16 x) {
  return BFloat16(std::cosh(static_cast<float>(x)));
}
inline BFloat16 tanh(BFloat16 x) {
  return BFloat16(std::tanh(static_cast<float>(x)));
}
inline BFloat16 asinh(BFloat16 x) {
  return BFloat16(std::asinh(static_cast<float>(x)));
}
inline BFloat16 acosh(BFloat16 x) {
  return BFloat16(std::acosh(static_cast<float>(x)));
}
inline BFloat16 atanh(BFloat16 x) {
  return BFloat16(std::atanh(static_cast<float>(x)));
}
inline BFloat16 floor(BFloat16 x) {
  return BFloat16(std::floor(static_cast<float>(x)));
}
inline BFloat16 trunc(BFloat16 x) {
  return BFloat16(std::trunc(static_cast<float>(x)));
}
inline BFloat16 rint(BFloat16 x) {
  return BFloat16(std::rint(static_cast<float>(x)));
}
inline BFloat16 ceil(BFloat16 x) {
  return BFloat16(std::ceil(static_cast<float>(x)));
}
inline BFloat16 fmod(BFloat16 x, BFloat16 y) {
  return BFloat16(std::fmod(static_cast<float>(x), static_cast<float>(y)));
}
inline BFloat16 fmin(BFloat16 a, BFloat16 b) {
  return BFloat16(std::fmin(static_cast<float>(a), static_cast<float>(b)));
}
inline BFloat16 fmax(BFloat16 a, BFloat16 b) {
  return BFloat16(std::fmax(static_cast<float>(a), static_cast<float>(b)));
}
inline BFloat16 nextafter(BFloat16 from, BFloat16 to) {
  const uint16_t from_as_int = absl::bit_cast<uint16_t>(from),
                 to_as_int = absl::bit_cast<uint16_t>(to);
  const uint16_t sign_mask = 1 << 15;
  float from_as_float(from), to_as_float(to);
  if (std::isnan(from_as_float) || std::isnan(to_as_float)) {
    return BFloat16(std::numeric_limits<float>::quiet_NaN());
  }
  if (from_as_int == to_as_int) {
    return to;
  }
  if (from_as_float == 0) {
    if (to_as_float == 0) {
      return to;
    } else {
      return absl::bit_cast<BFloat16, uint16_t>((to_as_int & sign_mask) | 1);
    }
  }
  uint16_t from_sign = from_as_int & sign_mask;
  uint16_t to_sign = to_as_int & sign_mask;
  uint16_t from_abs = from_as_int & ~sign_mask;
  uint16_t to_abs = to_as_int & ~sign_mask;
  uint16_t magnitude_adjustment =
      (from_abs > to_abs || from_sign != to_sign) ? 0xFFFF : 0x0001;
  return absl::bit_cast<BFloat16, uint16_t>(from_as_int + magnitude_adjustment);
}
namespace internal {
inline uint16_t GetFloat32High16(float v) {
  return static_cast<uint16_t>(absl::bit_cast<uint32_t>(v) >> 16);
}
inline BFloat16 Float32ToBfloat16Truncate(float v) {
  uint32_t bits = absl::bit_cast<uint32_t>(v);
  if (std::isnan(v)) {
    bits |= (static_cast<uint32_t>(1) << 21);
  }
  return absl::bit_cast<BFloat16, uint16_t>(bits >> 16);
}
inline BFloat16 NumericFloat32ToBfloat16RoundNearestEven(float v) {
  assert(!std::isnan(v));
  uint32_t input = absl::bit_cast<uint32_t>(v);
  const uint32_t lsb = (input >> 16) & 1;
  const uint32_t rounding_bias = 0x7fff + lsb;
  input += rounding_bias;
  return absl::bit_cast<BFloat16, uint16_t>(input >> 16);
}
inline BFloat16 Float32ToBfloat16RoundNearestEven(float v) {
  if (std::isnan(v)) {
    return tensorstore::BFloat16(
        tensorstore::BFloat16::bitcast_construct_t{},
        static_cast<uint16_t>((absl::bit_cast<uint32_t>(v) | 0x00200000u) >>
                              16));
  }
  return NumericFloat32ToBfloat16RoundNearestEven(v);
}
inline float Bfloat16ToFloat(BFloat16 v) {
  return absl::bit_cast<float>(
      static_cast<uint32_t>(absl::bit_cast<uint16_t>(v)) << 16);
}
}  
}  
namespace std {
template <>
struct numeric_limits<tensorstore::BFloat16> {
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr float_denorm_style has_denorm = std::denorm_present;
  static constexpr bool has_denorm_loss = false;
  static constexpr std::float_round_style round_style =
      numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 8;
  static constexpr int digits10 = 2;
  static constexpr int max_digits10 = 4;
  static constexpr int radix = 2;
  static constexpr int min_exponent = numeric_limits<float>::min_exponent;
  static constexpr int min_exponent10 = numeric_limits<float>::min_exponent10;
  static constexpr int max_exponent = numeric_limits<float>::max_exponent;
  static constexpr int max_exponent10 = numeric_limits<float>::max_exponent10;
  static constexpr bool traps = numeric_limits<float>::traps;
  static constexpr bool tinyness_before =
      numeric_limits<float>::tinyness_before;
  static constexpr tensorstore::BFloat16 min() {
    return tensorstore::BFloat16(tensorstore::BFloat16::bitcast_construct_t{},
                                 static_cast<uint16_t>(0x0080));
  }
  static constexpr tensorstore::BFloat16 lowest() {
    return tensorstore::BFloat16(tensorstore::BFloat16::bitcast_construct_t{},
                                 static_cast<uint16_t>(0xff7f));
  }
  static constexpr tensorstore::BFloat16 max() {
    return tensorstore::BFloat16(tensorstore::BFloat16::bitcast_construct_t{},
                                 static_cast<uint16_t>(0x7f7f));
  }
  static constexpr tensorstore::BFloat16 epsilon() {
    return tensorstore::BFloat16(tensorstore::BFloat16::bitcast_construct_t{},
                                 static_cast<uint16_t>(0x3c00));
  }
  static constexpr tensorstore::BFloat16 round_error() {
    return tensorstore::BFloat16(tensorstore::BFloat16::bitcast_construct_t{},
                                 static_cast<uint16_t>(0x3f00));
  }
  static constexpr tensorstore::BFloat16 infinity() {
    return tensorstore::BFloat16(tensorstore::BFloat16::bitcast_construct_t{},
                                 static_cast<uint16_t>(0x7f80));
  }
  static constexpr tensorstore::BFloat16 quiet_NaN() {
    return tensorstore::BFloat16(tensorstore::BFloat16::bitcast_construct_t{},
                                 static_cast<uint16_t>(0x7fc0));
  }
  static constexpr tensorstore::BFloat16 signaling_NaN() {
    return tensorstore::BFloat16(tensorstore::BFloat16::bitcast_construct_t{},
                                 static_cast<uint16_t>(0x7f81));
  }
  static constexpr tensorstore::BFloat16 denorm_min() {
    return tensorstore::BFloat16(tensorstore::BFloat16::bitcast_construct_t{},
                                 static_cast<uint16_t>(0x0001));
  }
};
}  
#endif  