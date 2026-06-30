#include "tensorflow/lite/experimental/shlo/ops/abs.h"
#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/unary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"
namespace shlo_ref {
struct Abs {
  template <class T>
  T operator()(const T& val) {
    return val < static_cast<T>(0) ? static_cast<T>(-val) : val;
  }
};
AbsOp Create(typename AbsOp::Attributes) { return AbsOp{}; }
absl::Status Prepare(AbsOp& op, const Tensor& input, Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(input.shape(), output.shape()));
  SHLO_REF_RETURN_ON_ERROR(CheckSupportedTypes(CheckCtx("abs"), input,
                                               IsSignedIntTensor, IsFloatTensor,
                                               IsQuantizedPerTensorTensor));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("abs"), input, output));
  return absl::OkStatus();
}
absl::Status Evaluate(AbsOp& op, const Tensor& input, Tensor& output) {
  Abs abs;
  if (input.IsPerTensorQuantized()) {
    DISPATCH_QUANTIZED(
        detail::DequantizeOpQuantizePerTensor,
        input.quantized_per_tensor_element_type().StorageType(),
        input.quantized_per_tensor_element_type().ExpressedType(), abs, input,
        output)
  } else if (IsSignedIntTensor(input) || IsFloatTensor(input)) {
    DISPATCH_INT_FLOAT(detail::EvaluateNoQuantization,
                       input.tensor_element_type(), abs, input, output);
  }
  return absl::FailedPreconditionError("Unsupported tensor type.");
}
}  