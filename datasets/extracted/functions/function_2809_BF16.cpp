#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_BF16_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_BF16_H_
#include "tensorflow/lite/experimental/shlo/legacy/src/has_keyword.h"
#if defined(__STDCPP_BFLOAT16_T__)
#include <stdfloat>
namespace stablehlo {
using BF16 = bfloat16_t;
}  
#elif __has_keyword(__bf16) && __x86_64__
namespace stablehlo {
using BF16 = __bf16;
}  
#elif __has_keyword(__bf16) && __aarch64__
#include <cmath>
#include <cstdint>
namespace stablehlo {
class BF16 {
 public:
  BF16(float f = 0.0f) {
    if (std::isnan(f)) {
      value_ = std::signbit(f) ? 0xFFC0 : 0x7FC0;
    } else {
      uint32_t input = *reinterpret_cast<const uint32_t*>(&f);
      uint32_t lsb = (input >> 16) & 1;
      uint32_t rounding_bias = 0x7fff + lsb;
      input += rounding_bias;
      value_ = static_cast<uint16_t>(input >> 16u);
    }
  }
  BF16& operator=(BF16 other) {
    value_ = other.value_;
    return *this;
  }
  bool operator==(BF16 other) const { return value_ == other.value_; }
  bool operator!=(BF16 other) const { return !(*this == other); }
  operator float() const {
    uint32_t tmp = value_ << 16;
    return *reinterpret_cast<float*>(&tmp);
  }
  BF16 operator-() const { return BF16(-static_cast<float>(*this)); }
  BF16& operator+=(BF16 other) {
    value_ = BF16(static_cast<float>(*this) + static_cast<float>(other)).value_;
    return *this;
  }
  BF16& operator-=(BF16 other) {
    value_ = BF16(static_cast<float>(*this) - static_cast<float>(other)).value_;
    return *this;
  }
  BF16& operator*=(BF16 other) {
    value_ = BF16(static_cast<float>(*this) * static_cast<float>(other)).value_;
    return *this;
  }
  BF16& operator/=(BF16 other) {
    value_ = BF16(static_cast<float>(*this) / static_cast<float>(other)).value_;
    return *this;
  }
 private:
  uint16_t value_;
};
inline BF16 operator+(BF16 x, BF16 y) {
  x += y;
  return x;
}
inline BF16 operator-(BF16 x, BF16 y) {
  x -= y;
  return x;
}
inline BF16 operator*(BF16 x, BF16 y) {
  x *= y;
  return x;
}
inline BF16 operator/(BF16 x, BF16 y) {
  x /= y;
  return x;
}
}  
#else
#error Type BF16 is not available
#endif
#endif  