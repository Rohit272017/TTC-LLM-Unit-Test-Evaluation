#include "tensorflow/compiler/mlir/lite/kernels/internal/runtime_shape.h"
#include <cstdint>
#include <cstring>
#include "tensorflow/compiler/mlir/lite/kernels/internal/compatibility_macros.h"
namespace mlir {
RuntimeShape::~RuntimeShape() {
  if (size_ > kMaxSmallSize) {
    delete[] dims_pointer_;
  }
}
int32_t RuntimeShape::Dims(int i) const {
  TFLITE_DCHECK_GE(i, 0);
  TFLITE_DCHECK_LT(i, size_);
  return size_ > kMaxSmallSize ? dims_pointer_[i] : dims_[i];
}
void RuntimeShape::ReplaceWith(int dimensions_count, const int32_t* dims_data) {
  Resize(dimensions_count);
  int32_t* dst_dims = DimsData();
  std::memcpy(dst_dims, dims_data, dimensions_count * sizeof(int32_t));
}
int RuntimeShape::FlatSize() const {
  int buffer_size = 1;
  const int* dims_data = reinterpret_cast<const int*>(DimsData());
  for (int i = 0; i < size_; i++) {
    buffer_size *= dims_data[i];
  }
  return buffer_size;
}
}  