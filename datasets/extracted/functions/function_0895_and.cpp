#include "arolla/expr/operators/factory_operators.h"
#include <cstdint>
#include <memory>
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "arolla/expr/basic_expr_operator.h"
#include "arolla/expr/expr.h"
#include "arolla/expr/expr_node.h"
#include "arolla/expr/expr_operator.h"
#include "arolla/expr/expr_operator_signature.h"
#include "arolla/qtype/optional_qtype.h"
#include "arolla/qtype/qtype.h"
#include "arolla/qtype/standard_type_properties/properties.h"
#include "arolla/util/fingerprint.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla::expr_operators {
namespace {
using ::arolla::expr::CallOp;
using ::arolla::expr::ExprNodePtr;
using ::arolla::expr::ExprOperatorPtr;
using ::arolla::expr::ExprOperatorSignature;
using ::arolla::expr::Literal;
using NarrowestNumericType = int32_t;
class EmptyLikeOp final : public expr::BasicExprOperator {
 public:
  EmptyLikeOp()
      : BasicExprOperator(
            "core.empty_like", ExprOperatorSignature{{"target"}},
            "Creates an empty value with shape and (optional) type like "
            "target.",
            FingerprintHasher("arolla::expr_operators::EmptyLikeOp").Finish()) {
  }
  absl::StatusOr<expr::ExprNodePtr> ToLowerLevel(
      const expr::ExprNodePtr& node) const final {
    RETURN_IF_ERROR(ValidateNodeDepsCount(*node));
    auto target_qtype = node->node_deps()[0]->qtype();
    ASSIGN_OR_RETURN(auto scalar_qtype, GetScalarQType(target_qtype));
    ASSIGN_OR_RETURN(auto optional_scalar_qtype, ToOptionalQType(scalar_qtype));
    ASSIGN_OR_RETURN(auto missing, CreateMissingValue(optional_scalar_qtype));
    return CallOp("core.const_like", {node->node_deps()[0], Literal(missing)});
  }
  absl::StatusOr<QTypePtr> GetOutputQType(
      absl::Span<const QTypePtr> input_qtypes) const final {
    return ToOptionalLikeQType(input_qtypes[0]);
  }
};
}  
absl::StatusOr<ExprOperatorPtr> MakeEmptyLikeOp() {
  return std::make_shared<EmptyLikeOp>();
}
}  