#include "tensorflow/lite/kernels/variants/tensor_array.h"
#include <cstring>
#include "tensorflow/lite/array.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/util.h"
namespace tflite {
namespace variants {
TensorArray::TensorArray(const TensorArray& other) {
  TfLiteIntArray* copied_shape = TfLiteIntArrayCopy(other.element_shape_.get());
  element_shape_ = IntArrayUniquePtr(copied_shape);
  element_type_ = other.element_type_;
  num_elements_ = other.num_elements_;
  elements_ =
      (RefCountedTensor*)malloc(sizeof(RefCountedTensor) * other.num_elements_);
  other.AssignBuffer(elements_);
}
TensorArray& TensorArray::operator=(const TensorArray& other) {
  TfLiteIntArray* copied_shape = TfLiteIntArrayCopy(other.element_shape_.get());
  element_shape_ = IntArrayUniquePtr(copied_shape);
  Resize(other.num_elements_);
  Clear();
  other.AssignBuffer(elements_);
  return *this;
}
void TensorArray::Resize(int num_elements) {
  if (num_elements == NumElements() || num_elements < 0) return;
  if (num_elements > NumElements()) {
    elements_ = (RefCountedTensor*)realloc(
        elements_, num_elements * sizeof(RefCountedTensor));
    for (int i = NumElements(); i < num_elements; ++i) {
      elements_[i].count = nullptr;
      elements_[i].tensor = nullptr;
    }
  } else {
    for (int i = num_elements; i < NumElements(); ++i) {
      Drop(i);
    }
    elements_ = (RefCountedTensor*)realloc(
        elements_, num_elements * sizeof(RefCountedTensor));
  }
  num_elements_ = num_elements;
}
const TfLiteTensor* TensorArray::At(int index) const {
  if (index < 0 || index >= NumElements()) {
    return nullptr;
  }
  return elements_[index].tensor;
}
bool TensorArray::Set(int index, TensorUniquePtr tensor) {
  if (index < 0 || index >= NumElements()) {
    return false;
  }
  Drop(index);
  int* c = (int*)malloc(sizeof(int));
  *c = 1;
  elements_[index].tensor = tensor.release();
  elements_[index].count = c;
  return true;
}
TensorArray::~TensorArray() {
  Clear();
  free(elements_);
  elements_ = nullptr;
}
void TensorArray::Drop(int i) {
  RefCountedTensor* t = elements_ + i;
  int* count = t->count;
  if (count == nullptr) {
    return;
  }
  if (*count == 1) {
    TfLiteTensorFree(t->tensor);
    free(t->tensor);
    free(t->count);
    t->tensor = nullptr;
    t->count = nullptr;
    return;
  }
  (*count)--;
}
void TensorArray::Clear() {
  for (int i = 0; i < num_elements_; ++i) {
    Drop(i);
  }
}
void TensorArray::AssignBuffer(RefCountedTensor* dst) const {
  std::memcpy(dst, elements_, sizeof(RefCountedTensor) * num_elements_);
  for (int i = 0; i < num_elements_; ++i) {
    if (dst[i].count == nullptr) {
      continue;
    }
    (*dst[i].count)++;
  }
}
}  
}  