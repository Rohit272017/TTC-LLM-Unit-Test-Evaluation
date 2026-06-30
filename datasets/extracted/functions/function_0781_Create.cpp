#include "tensorflow/lite/experimental/shlo/ops/not.h"
#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/unary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"
namespace shlo_ref {
struct Not {
  template <class T>
  T operator()(T v) const {
    return static_cast<T>(~v);
  }
};
template <>
bool Not::operator()(bool v) const {
  return !v;
}
NotOp Create(NotOp::Attributes) { return {}; }
absl::Status Prepare(NotOp& op, const Tensor& input, Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(input.shape(), output.shape()));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSupportedTypes(CheckCtx("not"), input, IsBoolTensor, IsIntTensor));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("not"), input, output));
  return absl::OkStatus();
}
absl::Status Evaluate(NotOp& op, const Tensor& input, Tensor& output) {
  Not not_func;
  if (IsIntTensor(input) || IsBoolTensor(input)) {
    DISPATCH_BOOL_INT(detail::EvaluateNoQuantization,
                      input.tensor_element_type(), not_func, input, output);
  }
  return absl::FailedPreconditionError(
      "stablehlo.not: Unsupported tensor type.");
}
};  