#ifndef XLA_BIT_CAST_H_
#define XLA_BIT_CAST_H_
#include <cstdint>
#include "absl/base/casts.h"
#include "Eigen/Core"
#include "xla/types.h"
#include "tsl/platform/bfloat16.h"
namespace xla {
template <typename T, typename U>
T BitCast(U src) {
  static_assert(sizeof(T) == sizeof(U), "sizes don't match");
  return absl::bit_cast<T>(src);
}
template <>
inline tsl::bfloat16 BitCast<tsl::bfloat16, uint16_t>(uint16_t src) {
  return Eigen::numext::bit_cast<tsl::bfloat16>(src);
}
template <>
inline uint16_t BitCast<uint16_t, tsl::bfloat16>(tsl::bfloat16 src) {
  return Eigen::numext::bit_cast<uint16_t>(src);
}
template <>
inline Eigen::half BitCast<Eigen::half, uint16_t>(uint16_t src) {
  return Eigen::numext::bit_cast<Eigen::half>(src);
}
template <>
inline uint16_t BitCast<uint16_t, Eigen::half>(Eigen::half src) {
  return Eigen::numext::bit_cast<uint16_t>(src);
}
}  
#endif  