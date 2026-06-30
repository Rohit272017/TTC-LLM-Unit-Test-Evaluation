#include "tensorflow/lite/experimental/shlo/ops/negate.h"
#include <functional>
#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/unary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"
namespace shlo_ref {
struct Negate : std::negate<void> {};
NegateOp Create(NegateOp::Attributes) { return {}; }
absl::Status Prepare(NegateOp& op, const Tensor& input, Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(input.shape(), output.shape()));
  SHLO_REF_RETURN_ON_ERROR(CheckSupportedTypes(CheckCtx("negate"), input,
                                               IsSignedIntTensor, IsFloatTensor,
                                               IsQuantizedPerTensorTensor));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("negate"), input, output));
  return absl::OkStatus();
}
absl::Status Evaluate(NegateOp& op, const Tensor& input, Tensor& output) {
  Negate negate;
  if (input.IsPerTensorQuantized()) {
    DISPATCH_QUANTIZED(
        detail::DequantizeOpQuantizePerTensor,
        input.quantized_per_tensor_element_type().StorageType(),
        input.quantized_per_tensor_element_type().ExpressedType(), negate,
        input, output)
  } else if (IsSignedIntTensor(input) || IsFloatTensor(input)) {
    DISPATCH_INT_FLOAT(detail::EvaluateNoQuantization,
                       input.tensor_element_type(), negate, input, output);
  }
  return absl::FailedPreconditionError(
      "stablehlo.negate: Unsupported tensor type.");
}
};  