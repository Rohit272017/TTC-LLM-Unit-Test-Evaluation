#include "tensorflow/lite/experimental/shlo/ops/and.h"
#include <functional>
#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/binary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"
namespace shlo_ref {
template <DataType>
struct And : std::bit_and<void> {};
template <>
struct And<DataType::kI1> : std::logical_and<void> {};
AndOp Create(AndOp::Attributes) { return {}; }
absl::Status Prepare(AndOp& op, const Tensor& lhs, const Tensor& rhs,
                     Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(lhs.shape(), rhs.shape(), output.shape()));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSupportedTypes(CheckCtx("and"), lhs, IsBoolTensor, IsIntTensor));
  SHLO_REF_RETURN_ON_ERROR(CheckSameBaselineType(CheckCtx("and"), lhs, output));
  SHLO_REF_RETURN_ON_ERROR(CheckSameBaselineType(CheckCtx("and"), rhs, output));
  return absl::OkStatus();
}
absl::Status Evaluate(AndOp& op, const Tensor& lhs, const Tensor& rhs,
                      Tensor& output) {
  if (IsIntTensor(lhs)) {
    And<DataType::kSI32> and_func;
    DISPATCH_INT(detail::EvaluateNoQuantization, lhs.tensor_element_type(),
                 and_func, lhs, rhs, output);
  } else if (IsBoolTensor(lhs)) {
    And<DataType::kI1> and_func;
    detail::EvaluateNoQuantization<DataType::kI1>(and_func, lhs, rhs, output);
    return absl::OkStatus();
  }
  return absl::FailedPreconditionError(
      "stablehlo.and: Unsupported tensor type in Evaluate.");
}
}  