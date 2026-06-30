#include "tensorflow/c/kernels/tensor_shape_utils.h"
#include <string>
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/strcat.h"
namespace tensorflow {
std::string ShapeDebugString(TF_Tensor* tensor) {
  CHECK_GE(TF_NumDims(tensor), 0);
  tensorflow::string s = "[";
  for (int i = 0; i < TF_NumDims(tensor); ++i) {
    if (i > 0) tensorflow::strings::StrAppend(&s, ",");
    int64_t dim = TF_Dim(tensor, i);
    CHECK_GE(dim, 0);
    tensorflow::strings::StrAppend(&s, dim);
  }
  tensorflow::strings::StrAppend(&s, "]");
  return s;
}
}  