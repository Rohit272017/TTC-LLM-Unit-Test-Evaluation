#include "tensorflow/lite/experimental/shlo/ops/maximum.h"
#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/binary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"
namespace shlo_ref {
struct Maximum {
  template <class T>
  constexpr auto operator()(const T a, const T b) {
    return a > b ? a : b;
  }
};
MaximumOp Create(MaximumOp::Attributes) { return {}; }
absl::Status Prepare(MaximumOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(lhs.shape(), rhs.shape(), output.shape()));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSupportedTypes(CheckCtx("maximum"), lhs, IsBoolTensor, IsIntTensor,
                          IsFloatTensor, IsQuantizedPerTensorTensor));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("maximum"), lhs, output));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("maximum"), rhs, output));
  return absl::OkStatus();
}
absl::Status Evaluate(MaximumOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output) {
  Maximum maximum;
  if (IsBoolTensor(lhs) || IsIntTensor(lhs) || IsFloatTensor(lhs)) {
    DISPATCH_BOOL_INT_FLOAT(detail::EvaluateNoQuantization,
                            lhs.tensor_element_type(), maximum, lhs, rhs,
                            output);
  } else if (IsQuantizedPerTensorTensor(lhs)) {
    DISPATCH_QUANTIZED(detail::DequantizeOpQuantizePerTensor,
                       lhs.quantized_per_tensor_element_type().StorageType(),
                       lhs.quantized_per_tensor_element_type().ExpressedType(),
                       maximum, lhs, rhs, output)
  }
  return absl::FailedPreconditionError(
      "stablehlo.maximum: Unsupported tensor type.");
}
}  