#include "tensorstore/contiguous_layout.h"
#include <stddef.h>
#include <cassert>
#include <ostream>
#include "tensorstore/index.h"
#include "tensorstore/util/span.h"
namespace tensorstore {
void ComputeStrides(ContiguousLayoutOrder order, ptrdiff_t element_stride,
                    tensorstore::span<const Index> shape,
                    tensorstore::span<Index> strides) {
  const DimensionIndex rank = shape.size();
  assert(strides.size() == rank);
  if (order == ContiguousLayoutOrder::right) {
    for (DimensionIndex i = rank - 1; i >= 0; --i) {
      strides[i] = element_stride;
      element_stride *= shape[i];
    }
  } else {
    for (DimensionIndex i = 0; i < rank; ++i) {
      strides[i] = element_stride;
      element_stride *= shape[i];
    }
  }
}
std::ostream& operator<<(std::ostream& os, ContiguousLayoutOrder order) {
  return os << (order == ContiguousLayoutOrder::c ? 'C' : 'F');
}
}  