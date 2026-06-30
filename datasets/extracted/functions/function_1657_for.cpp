#ifndef AROLLA_JAGGED_SHAPE_DENSE_ARRAY_UTIL_CONCAT_H_
#define AROLLA_JAGGED_SHAPE_DENSE_ARRAY_UTIL_CONCAT_H_
#include <cstdint>
#include "absl/types/span.h"
#include "arolla/dense_array/dense_array.h"
#include "arolla/jagged_shape/util/concat.h"
namespace arolla {
namespace jagged_shape_internal {
template <typename T>
struct ConcatResultArrayBuilderHelper<DenseArray<T>> {
  DenseArrayBuilder<T> operator()(
      absl::Span<const DenseArray<T>> arrays) const {
    int64_t result_size = 0;
    for (const auto& array : arrays) {
      result_size += array.size();
    }
    return DenseArrayBuilder<T>(result_size);
  }
};
}  
}  
#endif  