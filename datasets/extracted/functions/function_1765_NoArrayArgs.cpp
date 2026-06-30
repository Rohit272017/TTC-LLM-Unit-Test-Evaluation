#include "arolla/expr/operators/dynamic_lifting.h"
#include <utility>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "arolla/expr/expr.h"
#include "arolla/expr/expr_operator.h"
#include "arolla/expr/expr_operator_signature.h"
#include "arolla/expr/lambda_expr_operator.h"
#include "arolla/expr/operators/restricted_operator.h"
#include "arolla/expr/operators/type_meta_eval_strategies.h"
#include "arolla/expr/overloaded_expr_operator.h"
#include "arolla/expr/registered_expr_operator.h"
#include "arolla/qtype/array_like/array_like_qtype.h"
#include "arolla/qtype/qtype.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla::expr_operators {
namespace {
using ::arolla::expr::ExprOperatorPtr;
using ::arolla::expr::ExprOperatorSignature;
using ::arolla::expr::Literal;
using ::arolla::expr::MakeOverloadedOperator;
using ::arolla::expr::Placeholder;
using ::arolla::expr_operators::type_meta::QTypes;
absl::StatusOr<QTypes> NoArrayArgs(absl::Span<const QTypePtr> types) {
  for (QTypePtr t : types) {
    if (IsArrayLikeQType(t)) {
      return absl::InvalidArgumentError("array argument found");
    }
  }
  return QTypes{types.begin(), types.end()};
}
}  
absl::StatusOr<ExprOperatorPtr> LiftDynamically(
    const absl::StatusOr<ExprOperatorPtr>& op_or) {
  ASSIGN_OR_RETURN(const ExprOperatorPtr& op, op_or);
  ASSIGN_OR_RETURN(ExprOperatorPtr map_op, expr::LookupOperator("core.map"));
  return MakeOverloadedOperator(
      op->display_name(), RestrictOperator(op, NoArrayArgs),
      MakeLambdaOperator(
          ExprOperatorSignature::Make("*args"),
          ::arolla::expr::CallOp(
              "core.apply_varargs",
              {Literal(std::move(map_op)), Literal(op), Placeholder("args")})));
}
}  