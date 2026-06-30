#include "tensorflow/lite/experimental/shlo/ops/logistic.h"
#include <cmath>
#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/bf16.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/f16.h"
#include "tensorflow/lite/experimental/shlo/ops/unary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"
namespace shlo_ref {
struct Logistic {
  template <class T>
  T operator()(T v) const {
    constexpr T one = static_cast<T>(1);
    return one / (one + std::exp(-v));
  }
};
template <>
F16 Logistic::operator()(F16 v) const {
  return F16(operator()(static_cast<float>(v)));
}
template <>
BF16 Logistic::operator()(BF16 v) const {
  return BF16(operator()(static_cast<float>(v)));
}
LogisticOp Create(LogisticOp::Attributes) { return {}; }
absl::Status Prepare(LogisticOp& op, const Tensor& input, Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(input.shape(), output.shape()));
  SHLO_REF_RETURN_ON_ERROR(CheckSupportedTypes(
      CheckCtx("logistic"), input, IsFloatTensor, IsQuantizedPerTensorTensor));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("logistic"), input, output));
  return absl::OkStatus();
}
absl::Status Evaluate(LogisticOp& op, const Tensor& input, Tensor& output) {
  Logistic logistic;
  if (input.IsPerTensorQuantized()) {
    DISPATCH_QUANTIZED(
        detail::DequantizeOpQuantizePerTensor,
        input.quantized_per_tensor_element_type().StorageType(),
        input.quantized_per_tensor_element_type().ExpressedType(), logistic,
        input, output)
  } else if (IsFloatTensor(input)) {
    DISPATCH_FLOAT(detail::EvaluateNoQuantization, input.tensor_element_type(),
                   logistic, input, output);
  }
  return absl::FailedPreconditionError(
      "stablehlo.logistic: Unsupported tensor type.");
}
};  