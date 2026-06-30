#include "tensorflow/lite/experimental/shlo/ops/popcnt.h"
#include <cstdint>
#include <type_traits>
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/i4.h"
#include "tensorflow/lite/experimental/shlo/ops/unary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"
namespace shlo_ref {
struct Popcnt {
  template <class T>
  T operator()(T v) const {
    if constexpr (std::is_same_v<I4, T>) {
      return I4(absl::popcount(static_cast<uint8_t>(v & 0xf)));
    } else {
      return absl::popcount(static_cast<std::make_unsigned_t<T>>(v));
    }
  }
};
PopcntOp Create(PopcntOp::Attributes) { return {}; }
absl::Status Prepare(PopcntOp& op, const Tensor& input, Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(input.shape(), output.shape()));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSupportedTypes(CheckCtx("popcnt"), input, IsIntTensor));
  SHLO_REF_RETURN_ON_ERROR(
      CheckSameBaselineType(CheckCtx("popcnt"), input, output));
  return absl::OkStatus();
}
absl::Status Evaluate(PopcntOp& op, const Tensor& input, Tensor& output) {
  Popcnt popcnt;
  if (IsIntTensor(input)) {
    DISPATCH_INT(detail::EvaluateNoQuantization, input.tensor_element_type(),
                 popcnt, input, output);
  }
  return absl::FailedPreconditionError(
      "stablehlo.popcnt: Unsupported tensor type.");
}
};  