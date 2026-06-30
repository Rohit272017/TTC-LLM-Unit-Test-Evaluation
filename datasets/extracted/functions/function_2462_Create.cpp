#include "tensorflow/lite/experimental/shlo/ops/sine.h"
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
struct Sine {
  template <class T>
  T operator()(T v) const {
    return std::sin(v);
  }
};
template <>
F16 Sine::operator()<F16>(F16 val) const {
  return F16(operator()(static_cast<float>(val)));
}
template <>
BF16 Sine::operator()<BF16>(BF16 val) const {
  return BF16(operator()(static_cast<float>(val)));
}
SineOp Create(SineOp::Attributes) { return {}; }
absl::Status Prepare(SineOp& op, const Tensor& input, Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(input.shape(), output.shape()));
  SHLO_REF_RETURN_ON_ERROR(CheckSupportedTypes(
      CheckCtx("sine"), input, IsFloatTensor, IsQuantizedPerTensorTensor));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("sine"), input, output));
  return absl::OkStatus();
}
absl::Status Evaluate(SineOp& op, const Tensor& input, Tensor& output) {
  Sine sine;
  if (input.IsPerTensorQuantized()) {
    DISPATCH_QUANTIZED(
        detail::DequantizeOpQuantizePerTensor,
        input.quantized_per_tensor_element_type().StorageType(),
        input.quantized_per_tensor_element_type().ExpressedType(), sine, input,
        output)
  } else if (!input.IsQuantized() && IsFloat(input.StorageType())) {
    DISPATCH_FLOAT(detail::EvaluateNoQuantization, input.tensor_element_type(),
                   sine, input, output);
  }
  return absl::FailedPreconditionError("Unsupported tensor type.");
}
};  