#ifndef TENSORSTORE_UTIL_CONSTANT_BIT_VECTOR_H_
#define TENSORSTORE_UTIL_CONSTANT_BIT_VECTOR_H_
#include <cstddef>
#include <type_traits>
#include "tensorstore/util/bit_span.h"
#include "tensorstore/util/constant_vector.h"
namespace tensorstore {
template <typename Block, bool value, std::ptrdiff_t Length>
constexpr BitSpan<const Block, Length> GetConstantBitVector(
    std::integral_constant<std::ptrdiff_t, Length> = {}) {
  return {GetConstantVector<
              Block, (value ? ~static_cast<Block>(0) : static_cast<Block>(0)),
              BitVectorSizeInBlocks<Block>(Length)>()
              .data(),
          0, Length};
}
template <typename Block, bool value>
BitSpan<const Block> GetConstantBitVector(std::ptrdiff_t length) {
  return {GetConstantVector<Block, (value ? ~static_cast<Block>(0)
                                          : static_cast<Block>(0))>(
              BitVectorSizeInBlocks<Block>(length))
              .data(),
          0, length};
}
}  
#endif  