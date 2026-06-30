#include "tensorflow/lite/experimental/shlo/ops/log.h"
#include <cmath>
#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/bf16.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/f16.h"
#include "tensorflow/lite/experimental/shlo/ops/unary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"
namespace shlo_ref {
struct Log {
  template <class T>
  T operator()(T v) const {
    return std::log(v);
  }
};
template <>
F16 Log::operator()<F16>(F16 val) const {
  return F16(operator()(static_cast<float>(val)));
}
template <>
BF16 Log::operator()<BF16>(BF16 val) const {
  return BF16(operator()(static_cast<float>(val)));
}
LogOp Create(LogOp::Attributes) { return {}; }
absl::Status Prepare(LogOp& op, const Tensor& input, Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(input.shape(), output.shape()));
  SHLO_REF_RETURN_ON_ERROR(CheckSupportedTypes(
      CheckCtx("log"), input, IsFloatTensor, IsQuantizedPerTensorTensor));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("log"), input, output));
  return absl::OkStatus();
}
absl::Status Evaluate(LogOp& op, const Tensor& input, Tensor& output) {
  Log log;
  if (input.IsPerTensorQuantized()) {
    DISPATCH_QUANTIZED(
        detail::DequantizeOpQuantizePerTensor,
        input.quantized_per_tensor_element_type().StorageType(),
        input.quantized_per_tensor_element_type().ExpressedType(), log, input,
        output)
  } else if (IsFloatTensor(input)) {
    DISPATCH_FLOAT(detail::EvaluateNoQuantization, input.tensor_element_type(),
                   log, input, output);
  }
  return absl::FailedPreconditionError(
      "stablehlo.log: Unsupported tensor type.");
}
};  