#include "tensorflow/core/util/bcast.h"
#include "tensorflow/core/platform/logging.h"
namespace tensorflow {
BCast::Vec BCast::FromShape(const TensorShape& shape) {
  const int N = shape.dims();
  BCastList::Vec ret(N);
  for (int i = 0; i < N; ++i) {
    ret[i] = shape.dim_size(i);
  }
  return ret;
}
TensorShape BCast::ToShape(const BCastList::Vec& vec) {
  TensorShape shape(vec);
  return shape;
}
}  