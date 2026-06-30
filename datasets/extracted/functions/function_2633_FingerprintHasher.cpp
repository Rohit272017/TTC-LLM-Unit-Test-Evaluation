#include "arolla/expr/operators/weak_qtype_operators.h"
#include <cstdint>
#include <memory>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "arolla/expr/basic_expr_operator.h"
#include "arolla/expr/derived_qtype_cast_operator.h"
#include "arolla/expr/expr.h"
#include "arolla/expr/expr_node.h"
#include "arolla/expr/expr_operator.h"
#include "arolla/expr/expr_operator_signature.h"
#include "arolla/expr/operators/type_meta_eval_strategies.h"
#include "arolla/qtype/array_like/array_like_qtype.h"
#include "arolla/qtype/base_types.h"
#include "arolla/qtype/optional_qtype.h"
#include "arolla/qtype/qtype.h"
#include "arolla/qtype/qtype_traits.h"
#include "arolla/qtype/standard_type_properties/properties.h"
#include "arolla/qtype/weak_qtype.h"
#include "arolla/util/fingerprint.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla::expr_operators {
namespace {
using ::arolla::expr::CallOp;
using ::arolla::expr::ExprNodePtr;
using ::arolla::expr::ExprOperatorPtr;
using ::arolla::expr::ExprOperatorSignature;
class CoreToWeakFloatOp final : public expr::BasicExprOperator {
 public:
  CoreToWeakFloatOp()
      : expr::BasicExprOperator(
            "core.to_weak_float", ExprOperatorSignature{{"x"}},
            "Casts a floating point value to the corresponding weak float "
            "type.",
            FingerprintHasher("::arolla::expr_operators::CoreToWeakFloatOp")
                .Finish()) {}
  absl::StatusOr<QTypePtr> GetOutputQType(
      absl::Span<const QTypePtr> inputs) const override {
    ASSIGN_OR_RETURN(auto scalar_type, GetScalarQType(inputs[0]));
    if (!(IsNumeric(scalar_type) || IsBoolean(scalar_type) ||
          scalar_type == GetQType<uint64_t>())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "expected a numeric or boolean number, got: %s", inputs[0]->name()));
    }
    if (IsOptionalQType(inputs[0])) {
      return GetOptionalWeakFloatQType();
    }
    if (IsArrayLikeQType(inputs[0])) {
      ASSIGN_OR_RETURN(auto shape_qtype, GetShapeQType(inputs[0]));
      return shape_qtype->WithValueQType(GetWeakFloatQType());
    }
    return GetWeakFloatQType();
  }
  absl::StatusOr<ExprNodePtr> ToLowerLevel(
      const ExprNodePtr& node) const final {
    RETURN_IF_ERROR(ValidateNodeDepsCount(*node));
    auto op =
        std::make_shared<expr::DerivedQTypeDowncastOperator>(node->qtype());
    return CallOp(op, {CallOp("core.to_float64", {node->node_deps()[0]})});
  }
};
}  
absl::StatusOr<ExprOperatorPtr> MakeCoreToWeakFloatOperator() {
  return std::make_shared<CoreToWeakFloatOp>();
}
}  