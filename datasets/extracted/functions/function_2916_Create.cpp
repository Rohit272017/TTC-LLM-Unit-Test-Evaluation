#include "tensorflow/lite/experimental/shlo/ops/or.h"
#include <functional>
#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/binary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"
namespace shlo_ref {
template <DataType>
struct Or : std::bit_or<void> {};
template <>
struct Or<DataType::kI1> : std::logical_or<void> {};
OrOp Create(OrOp::Attributes) { return {}; }
absl::Status Prepare(OrOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(lhs.shape(), rhs.shape(), output.shape()));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSupportedTypes(CheckCtx("or"), lhs, IsBoolTensor, IsIntTensor));
  SHLO_REF_RETURN_ON_ERROR(CheckSameBaselineType(CheckCtx("or"), lhs, output));
  SHLO_REF_RETURN_ON_ERROR(CheckSameBaselineType(CheckCtx("or"), rhs, output));
  return absl::OkStatus();
}
absl::Status Evaluate(OrOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output) {
  if (IsIntTensor(lhs)) {
    Or<DataType::kSI32> or_func;
    DISPATCH_INT(detail::EvaluateNoQuantization, lhs.tensor_element_type(),
                 or_func, lhs, rhs, output);
  } else if (IsBoolTensor(lhs)) {
    Or<DataType::kI1> or_func;
    detail::EvaluateNoQuantization<DataType::kI1>(or_func, lhs, rhs, output);
    return absl::OkStatus();
  }
  return absl::FailedPreconditionError(
      "stablehlo.or: Unsupported tensor type in Evaluate.");
}
}  