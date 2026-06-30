#include "arolla/expr/operators/restricted_operator.h"
#include <memory>
#include <utility>
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "arolla/expr/expr.h"
#include "arolla/expr/expr_attributes.h"
#include "arolla/expr/expr_node.h"
#include "arolla/expr/expr_operator.h"
#include "arolla/expr/expr_operator_signature.h"
#include "arolla/expr/operators/type_meta_eval_strategies.h"
#include "arolla/expr/qtype_utils.h"
#include "arolla/util/fingerprint.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla::expr_operators {
namespace {
using ::arolla::expr::ExprAttributes;
using ::arolla::expr::ExprNodePtr;
using ::arolla::expr::ExprOperator;
using ::arolla::expr::ExprOperatorPtr;
using ::arolla::expr::ExprOperatorSignature;
using ::arolla::expr::GetAttrQTypes;
using ::arolla::expr::HasAllAttrQTypes;
using ::arolla::expr::WithNewOperator;
class RestrictedOp final : public ExprOperator {
 public:
  RestrictedOp(ExprOperatorPtr wrapped_op, type_meta::Strategy restriction)
      : ExprOperator(wrapped_op->display_name(),
                     FingerprintHasher("::arolla::expr_operators::RestrictedOp")
                         .Combine(wrapped_op)
                         .Finish()),
        wrapped_op_(std::move(wrapped_op)),
        restriction_(std::move(restriction)) {}
  absl::StatusOr<ExprOperatorSignature> GetSignature() const final {
    return wrapped_op_->GetSignature();
  }
  absl::StatusOr<ExprNodePtr> ToLowerLevel(
      const ExprNodePtr& node) const final {
    if (!node->qtype()) {
      return node;
    }
    ASSIGN_OR_RETURN(auto unwrapped_node, WithNewOperator(node, wrapped_op_));
    return wrapped_op_->ToLowerLevel(unwrapped_node);
  }
  absl::StatusOr<ExprAttributes> InferAttributes(
      absl::Span<const ExprAttributes> inputs) const final {
    if (!HasAllAttrQTypes(inputs)) {
      return ExprAttributes{};
    }
    RETURN_IF_ERROR(restriction_(GetAttrQTypes(inputs)).status())
        << "in restriction for " << display_name() << " operator";
    return wrapped_op_->InferAttributes(inputs);
  }
 private:
  ExprOperatorPtr wrapped_op_;
  type_meta::Strategy restriction_;
};
}  
ExprOperatorPtr RestrictOperator(ExprOperatorPtr wrapped_op,
                                 type_meta::Strategy restriction) {
  return std::make_shared<RestrictedOp>(std::move(wrapped_op),
                                        std::move(restriction));
}
absl::StatusOr<ExprOperatorPtr> RestrictOperator(
    absl::StatusOr<ExprOperatorPtr> wrapped_op,
    absl::StatusOr<type_meta::Strategy> restriction) {
  RETURN_IF_ERROR(wrapped_op.status());
  RETURN_IF_ERROR(restriction.status());
  return RestrictOperator(*wrapped_op, *restriction);
}
}  