#ifndef TENSORFLOW_CORE_UTIL_OVERFLOW_H_
#define TENSORFLOW_CORE_UTIL_OVERFLOW_H_
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
namespace tensorflow {
inline int64_t MultiplyWithoutOverflow(int64_t x, int64_t y) {
  if (TF_PREDICT_FALSE(x < 0)) return -1;
  if (TF_PREDICT_FALSE(y < 0)) return -1;
  if (TF_PREDICT_FALSE(x == 0)) return 0;
  const uint64 ux = x;
  const uint64 uy = y;
  const uint64 uxy = ux * uy;
  if (TF_PREDICT_FALSE((ux | uy) >> 32 != 0)) {
    if (uxy / ux != uy) return -1;
  }
  return static_cast<int64_t>(uxy);
}
inline int64_t AddWithoutOverflow(int64_t x, int64_t y) {
  if (TF_PREDICT_FALSE((x < 0)) || (y < 0)) return -1;
  const uint64 ux = x;
  const uint64 uy = y;
  const uint64 uxy = ux + uy;
  return static_cast<int64_t>(uxy);
}
}  
#endif  