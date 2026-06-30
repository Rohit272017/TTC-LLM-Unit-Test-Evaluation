#include "tensorflow/lite/experimental/shlo/ops/subtract.h"
#include <functional>
#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/binary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"
namespace shlo_ref {
struct Subtract : std::minus<void> {};
SubtractOp Create(SubtractOp::Attributes) { return {}; }
absl::Status Prepare(SubtractOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(lhs.shape(), rhs.shape(), output.shape()));
  SHLO_REF_RETURN_ON_ERROR(CheckSupportedTypes(CheckCtx("subtract"), lhs,
                                               IsIntTensor, IsFloatTensor,
                                               IsQuantizedPerTensorTensor));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("subtract"), lhs, output));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("subtract"), rhs, output));
  return absl::OkStatus();
}
absl::Status Evaluate(SubtractOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output) {
  Subtract subtract;
  if (IsIntTensor(lhs) || IsFloatTensor(lhs)) {
    DISPATCH_INT_FLOAT(detail::EvaluateNoQuantization,
                       lhs.tensor_element_type(), subtract, lhs, rhs, output);
  } else if (IsQuantizedPerTensorTensor(lhs)) {
    DISPATCH_QUANTIZED(detail::DequantizeOpQuantizePerTensor,
                       lhs.quantized_per_tensor_element_type().StorageType(),
                       lhs.quantized_per_tensor_element_type().ExpressedType(),
                       subtract, lhs, rhs, output)
  }
  return absl::FailedPreconditionError(
      "stablehlo.subtract: Unsupported tensor type.");
}
}  