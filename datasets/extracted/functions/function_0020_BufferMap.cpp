#include "tensorflow/lite/delegates/flex/buffer_map.h"
#include <utility>
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/lite/delegates/flex/buffer_map_util.h"
#include "tensorflow/lite/delegates/flex/util.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/string_type.h"
namespace tflite {
namespace flex {
BufferMap::BufferMap() {}
BufferMap::~BufferMap() {}
bool BufferMap::HasTensor(int tensor_index) const {
  return id_to_tensor_.count(tensor_index) != 0;
}
tensorflow::Tensor BufferMap::GetTensor(int tensor_index) const {
  return id_to_tensor_.at(tensor_index);
}
const tensorflow::Tensor* BufferMap::GetTensorPtr(int tensor_index) const {
  auto& tensor = id_to_tensor_.at(tensor_index);
  return &tensor;
}
void BufferMap::SetFromTfLite(int tensor_index, const TfLiteTensor* tensor,
                              bool allow_reusing) {
  TFLITE_CHECK(
      SetTfTensorFromTfLite(tensor, &id_to_tensor_[tensor_index], allow_reusing)
          .ok());
}
void BufferMap::SetFromTensorFlow(int tensor_index, tensorflow::Tensor tensor) {
  id_to_tensor_[tensor_index] = std::move(tensor);
}
}  
}  