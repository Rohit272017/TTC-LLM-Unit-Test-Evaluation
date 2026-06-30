#include "xla/hlo/builder/lib/logdet.h"
#include <limits>
#include <memory>
#include <vector>
#include "absl/status/statusor.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/matrix.h"
#include "xla/hlo/builder/lib/qr.h"
#include "xla/hlo/builder/lib/slicing.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"
namespace xla {
SignAndLogDet SLogDet(XlaOp a) {
  absl::StatusOr<SignAndLogDet> result =
      [&]() -> absl::StatusOr<SignAndLogDet> {
    TF_ASSIGN_OR_RETURN(Shape a_shape, a.builder()->GetShape(a));
    auto qr = Qr(a);
    int64_t m = ShapeUtil::GetDimension(a_shape, -2);
    int64_t n = ShapeUtil::GetDimension(a_shape, -1);
    if (m != n) {
      return InvalidArgument(
          "Arguments to logdet must be (batched) square matrices, got: %s",
          a_shape.ToString());
    }
    auto log_abs_det = Einsum(Log(Abs(qr.q_and_r)), "...aa->...");
    auto sign_diag = Reduce(
        Sign(Einsum(qr.q_and_r, "...aa->...a")),
        One(a.builder(), a_shape.element_type()),
        CreateScalarMultiplyComputation(a_shape.element_type(), a.builder()),
        {a_shape.rank() - 2});
    auto sliced_taus = SliceInMinorDims(qr.taus, {0}, {n - 1});
    auto sign_taus = Reduce(
        Select(Ne(sliced_taus, ZerosLike(sliced_taus)),
               FullLike(sliced_taus, -1), FullLike(sliced_taus, 1)),
        One(a.builder(), a_shape.element_type()),
        CreateScalarMultiplyComputation(a_shape.element_type(), a.builder()),
        {a_shape.rank() - 2});
    return SignAndLogDet{sign_diag * sign_taus, log_abs_det};
  }();
  if (!result.ok()) {
    XlaOp error = a.builder()->ReportError(result.status());
    return SignAndLogDet{error, error};
  }
  return result.value();
}
XlaOp LogDet(XlaOp a) {
  SignAndLogDet slogdet = SLogDet(a);
  return Select(
      Ge(slogdet.sign, ZerosLike(slogdet.sign)), slogdet.logdet,
      FullLike(slogdet.logdet, std::numeric_limits<float>::quiet_NaN()));
}
}  