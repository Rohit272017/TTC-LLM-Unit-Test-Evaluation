#include "tensorflow/core/framework/bfloat16.h"
#include "Eigen/Core"  
namespace tensorflow {
void RoundFloatToBFloat16(const float* src, bfloat16* dst, int64_t size) {
  Eigen::Map<const Eigen::ArrayXf> src_eigen(src, size);
  Eigen::Map<Eigen::Array<bfloat16, Eigen::Dynamic, 1>> dst_eigen(dst, size);
  dst_eigen = src_eigen.cast<bfloat16>();
}
void FloatToBFloat16(const float* src, bfloat16* dst, int64_t size) {
  for (; size != 0; src++, dst++, size--) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    memcpy(dst, src, sizeof(bfloat16));
#else
    memcpy(
        dst,
        reinterpret_cast<const char*>(src) + sizeof(float) - sizeof(bfloat16),
        sizeof(bfloat16));
#endif
  }
}
void BFloat16ToFloat(const bfloat16* src, float* dst, int64_t size) {
  Eigen::Map<const Eigen::Array<bfloat16, Eigen::Dynamic, 1>> src_eigen(src,
                                                                        size);
  Eigen::Map<Eigen::ArrayXf> dst_eigen(dst, size);
  dst_eigen = src_eigen.cast<float>();
}
}  