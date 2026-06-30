#include "xla/hlo/builder/lib/self_adjoint_eig.h"
#include <memory>
#include <string>
#include <vector>
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/hlo/builder/lib/slicing.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"
namespace xla {
SelfAdjointEigResult SelfAdjointEig(XlaOp a, bool lower, int64_t max_iter,
                                    float tol, bool sort_eigenvalues) {
  XlaBuilder* builder = a.builder();
  XlaOp result = builder->ReportErrorOrReturn([&]() -> absl::StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape a_shape, builder->GetShape(a));
    const int64_t num_dims = a_shape.rank();
    if (num_dims < 2) {
      return InvalidArgument(
          "Arguments to Eigen decomposition must have rank >= 2: got shape %s.",
          a_shape.ToString());
    }
    PrimitiveType type = a_shape.element_type();
    if (!primitive_util::IsFloatingPointType(type) &&
        !primitive_util::IsComplexType(type)) {
      return InvalidArgument(
          "Type of the input matrix must be floating point "
          "or complex: got %s.",
          a_shape.ToString());
    }
    const int64_t m = ShapeUtil::GetDimension(a_shape, -2);
    const int64_t n = ShapeUtil::GetDimension(a_shape, -1);
    if (m != n) {
      return InvalidArgument(
          "Arguments to symmetric eigendecomposition must be square matrices: "
          "got shape (%d, %d).",
          m, n);
    }
    const int num_batch_dims = a_shape.dimensions().size() - 2;
    const std::vector<int64_t> batch_dims(
        a_shape.dimensions().begin(),
        a_shape.dimensions().begin() + num_batch_dims);
    PrimitiveType eigvals_type =
        primitive_util::IsComplexType(type)
            ? primitive_util::ComplexComponentType(type)
            : type;
    std::vector<int64_t> eigvals_dims = batch_dims;
    eigvals_dims.push_back(m);
    Shape eigh_shape = ShapeUtil::MakeTupleShape(
        {a_shape, ShapeUtil::MakeShape(eigvals_type, eigvals_dims)});
    std::string opaque =
        absl::StrFormat("%d,%d,%d,%f", lower, sort_eigenvalues, max_iter, tol);
    return CustomCall(a.builder(), "Eigh", {a}, eigh_shape, opaque);
  });
  return SelfAdjointEigResult{GetTupleElement(result, 0),
                              GetTupleElement(result, 1)};
}
}  