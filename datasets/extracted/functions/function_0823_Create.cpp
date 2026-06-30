#include "tensorflow/lite/experimental/shlo/ops/tanh.h"
#include <cmath>
#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/bf16.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/f16.h"
#include "tensorflow/lite/experimental/shlo/ops/unary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"
namespace shlo_ref {
struct Tanh {
  template <class T>
  T operator()(T v) const {
    return std::tanh(v);
  }
};
template <>
F16 Tanh::operator()<F16>(F16 val) const {
  return F16(operator()(static_cast<float>(val)));
}
template <>
BF16 Tanh::operator()<BF16>(BF16 val) const {
  return BF16(operator()(static_cast<float>(val)));
}
TanhOp Create(TanhOp::Attributes) { return {}; }
absl::Status Prepare(TanhOp& op, const Tensor& input, Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(input.shape(), output.shape()));
  SHLO_REF_RETURN_ON_ERROR(CheckSupportedTypes(
      CheckCtx("tanh"), input, IsFloatTensor, IsQuantizedPerTensorTensor));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("tanh"), input, output));
  return absl::OkStatus();
}
absl::Status Evaluate(TanhOp& op, const Tensor& input, Tensor& output) {
  Tanh tanh;
  if (input.IsPerTensorQuantized()) {
    DISPATCH_QUANTIZED(
        detail::DequantizeOpQuantizePerTensor,
        input.quantized_per_tensor_element_type().StorageType(),
        input.quantized_per_tensor_element_type().ExpressedType(), tanh, input,
        output)
  } else if (!input.IsQuantized() && IsFloat(input.StorageType())) {
    DISPATCH_FLOAT(detail::EvaluateNoQuantization, input.tensor_element_type(),
                   tanh, input, output);
  }
  return absl::FailedPreconditionError(
      "stablehlo.tanh: Unsupported tensor type.");
}
};  