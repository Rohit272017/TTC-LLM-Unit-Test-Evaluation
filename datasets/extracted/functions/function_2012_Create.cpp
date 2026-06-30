#include "tensorflow/lite/experimental/shlo/ops/multiply.h"
#include <functional>
#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/binary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"
namespace shlo_ref {
template <DataType expressed_type>
struct Multiply : std::multiplies<void> {};
template <>
struct Multiply<DataType::kI1> {
  template <class T>
  T operator()(const T& lhs, const T& rhs) const {
    return static_cast<T>(lhs && rhs);
  }
};
MultiplyOp Create(MultiplyOp::Attributes) { return {}; }
absl::Status Prepare(MultiplyOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(lhs.shape(), rhs.shape(), output.shape()));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSupportedTypes(CheckCtx("multiply"), lhs, IsBoolTensor, IsIntTensor,
                          IsFloatTensor, IsQuantizedPerTensorTensor));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("multiply"), lhs, output));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("multiply"), rhs, output));
  return absl::OkStatus();
}
absl::Status Evaluate(MultiplyOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output) {
  if (IsBoolTensor(lhs)) {
    detail::EvaluateNoQuantization<DataType::kI1>(Multiply<DataType::kI1>(),
                                                  lhs, rhs, output);
    return absl::OkStatus();
  } else if (IsIntTensor(lhs) || IsFloatTensor(lhs)) {
    Multiply<DataType::kF32> multiply;
    DISPATCH_INT_FLOAT(detail::EvaluateNoQuantization,
                       lhs.tensor_element_type(), multiply, lhs, rhs, output);
  } else if (IsQuantizedPerTensorTensor(lhs)) {
    Multiply<DataType::kF32> multiply;
    DISPATCH_QUANTIZED(detail::DequantizeOpQuantizePerTensor,
                       lhs.quantized_per_tensor_element_type().StorageType(),
                       lhs.quantized_per_tensor_element_type().ExpressedType(),
                       multiply, lhs, rhs, output)
  }
  return absl::FailedPreconditionError(
      "stablehlo.multiply: Unsupported tensor type.");
}
}  