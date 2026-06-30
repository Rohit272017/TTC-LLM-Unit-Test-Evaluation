#include "tensorflow/compiler/jit/pjrt_tensor_buffer_util.h"
#include <cstddef>
#include <memory>
#include <utility>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/compiler/jit/pjrt_tensor_buffer.h"
#include "xla/pjrt/pjrt_client.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tsl/platform/statusor.h"
namespace tensorflow {
static size_t GetTensorSize(const TensorShape& shape, const DataType dtype) {
  return shape.num_elements() * DataTypeSize(dtype);
}
absl::StatusOr<Tensor> MakeTensorFromPjRtBuffer(
    const DataType dtype, const TensorShape& shape,
    std::unique_ptr<xla::PjRtBuffer> pjrt_buffer) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtBuffer::ExternalReference> ref,
                      pjrt_buffer->AcquireExternalReference());
  auto* tensor_buffer =
      new PjRtTensorBuffer(ref->OpaqueDeviceMemoryDataPointer(),
                           GetTensorSize(shape, dtype), std::move(pjrt_buffer));
  Tensor result(dtype, shape, tensor_buffer);
  tensor_buffer->Unref();
  return result;
}
static bool ShouldReuseTensor(void* opaque_device_memory,
                              const size_t expected_size,
                              const Tensor* existing_tensor) {
  const PjRtTensorBuffer* input_pjrt_tensor_buffer =
      dynamic_cast<const PjRtTensorBuffer*>(DMAHelper::buffer(existing_tensor));
  if (input_pjrt_tensor_buffer != nullptr) {
    return false;
  }
  const size_t current_size =
      GetTensorSize(existing_tensor->shape(), existing_tensor->dtype());
  return existing_tensor->tensor_data().data() == opaque_device_memory &&
         current_size == expected_size;
}
absl::Status PjRtTensorBufferUtil::UpdateOrMakeTensorWithPjRtBuffer(
    const DataType dtype, const TensorShape& shape,
    std::unique_ptr<xla::PjRtBuffer> pjrt_buffer, Tensor* output_tensor) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtBuffer::ExternalReference> ref,
                      pjrt_buffer->AcquireExternalReference());
  const size_t expected_size = GetTensorSize(shape, dtype);
  void* opaque_device_memory = ref->OpaqueDeviceMemoryDataPointer();
  auto* tensor_buffer = new PjRtTensorBuffer(
      opaque_device_memory, expected_size, std::move(pjrt_buffer));
  if (ShouldReuseTensor(opaque_device_memory, expected_size, output_tensor)) {
    output_tensor->buf_ = tensor_buffer;
    return absl::OkStatus();
  }
  Tensor result(dtype, shape, tensor_buffer);
  tensor_buffer->Unref();
  *output_tensor = result;
  return absl::OkStatus();
}
}  