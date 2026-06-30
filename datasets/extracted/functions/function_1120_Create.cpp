#include "tensorflow/lite/experimental/shlo/ops/exponential_minus_one.h"
#include <cmath>
#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/bf16.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/f16.h"
#include "tensorflow/lite/experimental/shlo/ops/unary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"
namespace shlo_ref {
struct ExponentialMinusOne {
  template <class T>
  T operator()(T v) const {
    return std::expm1(v);
  }
};
template <>
F16 ExponentialMinusOne::operator()(F16 v) const {
  return F16(operator()(static_cast<float>(v)));
}
template <>
BF16 ExponentialMinusOne::operator()(BF16 v) const {
  return BF16(operator()(static_cast<float>(v)));
}
ExponentialMinusOneOp Create(ExponentialMinusOneOp::Attributes) { return {}; }
absl::Status Prepare(ExponentialMinusOneOp& op, const Tensor& input,
                     Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(input.shape(), output.shape()));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSupportedTypes(CheckCtx("exponential_minus_one"), input,
                          IsFloatTensor, IsQuantizedPerTensorTensor));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("exponential_minus_one"), input, output));
  return absl::OkStatus();
}
absl::Status Evaluate(ExponentialMinusOneOp& op, const Tensor& input,
                      Tensor& output) {
  ExponentialMinusOne exponential_minus_one;
  if (input.IsPerTensorQuantized()) {
    DISPATCH_QUANTIZED(
        detail::DequantizeOpQuantizePerTensor,
        input.quantized_per_tensor_element_type().StorageType(),
        input.quantized_per_tensor_element_type().ExpressedType(),
        exponential_minus_one, input, output)
  } else if (IsFloatTensor(input)) {
    DISPATCH_FLOAT(detail::EvaluateNoQuantization, input.tensor_element_type(),
                   exponential_minus_one, input, output);
  }
  return absl::FailedPreconditionError(
      "stablehlo.exponential_minus_one: Unsupported tensor type.");
}
};  