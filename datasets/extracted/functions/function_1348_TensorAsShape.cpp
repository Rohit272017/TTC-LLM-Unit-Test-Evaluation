#include "tensorflow/lite/kernels/variants/list_ops_util.h"
#include "tensorflow/lite/array.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/variants/tensor_array.h"
namespace tflite {
namespace variants {
IntArrayUniquePtr TensorAsShape(const TfLiteTensor& shape) {
  if (shape.dims->size == 0) {
    return BuildTfLiteArray({});
  }
  const int rank = shape.dims->data[0];
  const int* begin = reinterpret_cast<const int*>(shape.data.data);
  const int* end = begin + rank;
  return BuildTfLiteArray(std::vector<int>(begin, end));
}
IntArrayUniquePtr MergeShapesOrNull(IntArrayUniquePtr l, IntArrayUniquePtr r) {
  if (l == nullptr) {
    return r;
  }
  if (r == nullptr) {
    return l;
  }
  if (l->size == 0) {
    return r;
  }
  if (r->size == 0) {
    return l;
  }
  if (l->size != r->size) {
    return nullptr;
  }
  for (int i = 0; i < r->size; ++i) {
    if (l->data[i] == -1) {
      l->data[i] = r->data[i];
    } else if (r->data[i] != -1 && l->data[i] != r->data[i]) {
      return nullptr;
    }
  }
  return l;
}
bool IsShapeFullyDefined(const TfLiteIntArray& shape) {
  for (int i = 0; i < shape.size; ++i) {
    if (shape.data[i] < 0) {
      return false;
    }
  }
  return true;
}
TfLiteStatus GetShapeIfAllEqual(const TensorArray& arr,
                                IntArrayUniquePtr& result) {
  const TfLiteIntArray* common_shape = nullptr;
  for (int i = 0; i < arr.NumElements(); ++i) {
    const TfLiteTensor* cur_element = arr.At(i);
    if (cur_element == nullptr) {
      continue;
    }
    if (common_shape == nullptr) {
      common_shape = cur_element->dims;
      continue;
    }
    if (!TfLiteIntArrayEqual(common_shape, cur_element->dims)) {
      return kTfLiteError;
    }
  }
  result = common_shape != nullptr ? BuildTfLiteArray(*common_shape) : nullptr;
  return kTfLiteOk;
}
}  
}  