#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include <utility>
#include "tensorflow/core/common_runtime/dma_helper.h"
namespace tensorflow {
namespace tfrt_stub {
namespace {
class ImmutableTensorBuffer final : public tensorflow::TensorBuffer {
 public:
  static tensorflow::core::RefCountPtr<ImmutableTensorBuffer> Create(
      tensorflow::Tensor tensor);
  explicit ImmutableTensorBuffer(tensorflow::Tensor tensor)
      : tensorflow::TensorBuffer(tensor.data()), tensor_(std::move(tensor)) {
    if (auto* buf = tensorflow::DMAHelper::buffer(&tensor_)) {
      root_buffer_ = buf->root_buffer();
    } else {
      root_buffer_ = this;
    }
  }
  ~ImmutableTensorBuffer() override = default;
  size_t size() const override {
    return tensorflow::DMAHelper::buffer(&tensor_)->size();
  }
  bool OwnsMemory() const override { return false; }
  tensorflow::TensorBuffer* root_buffer() override { return root_buffer_; }
  void FillAllocationDescription(AllocationDescription* proto) const override {}
  bool GetAllocatedBytes(size_t*) const override { return false; }
 private:
  tensorflow::Tensor tensor_;
  tensorflow::TensorBuffer* root_buffer_ = nullptr;
};
tensorflow::core::RefCountPtr<ImmutableTensorBuffer>
ImmutableTensorBuffer::Create(tensorflow::Tensor tensor) {
  return tensorflow::core::RefCountPtr<ImmutableTensorBuffer>(
      new ImmutableTensorBuffer(std::move(tensor)));
}
}  
ImmutableTensor ImmutableTensor::Create(tensorflow::Tensor tensor) {
  auto dtype = tensor.dtype();
  auto shape = tensor.shape();
  auto immutable_buffer = ImmutableTensorBuffer::Create(std::move(tensor));
  return ImmutableTensor(
      tensorflow::Tensor(dtype, shape, std::move(immutable_buffer)));
}
}  
}  