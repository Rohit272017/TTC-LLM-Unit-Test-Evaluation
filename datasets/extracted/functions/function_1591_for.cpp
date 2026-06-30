#include "arolla/expr/overloaded_expr_operator.h"
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "arolla/expr/expr_attributes.h"
#include "arolla/expr/expr_node.h"
#include "arolla/expr/expr_operator.h"
#include "arolla/expr/expr_operator_signature.h"
#include "arolla/expr/qtype_utils.h"
#include "arolla/qtype/qtype.h"
#include "arolla/util/fingerprint.h"
namespace arolla::expr {
OverloadedOperator::OverloadedOperator(absl::string_view name,
                                       std::vector<ExprOperatorPtr> base_ops)
    : ExprOperator(
          name,
          [name, &base_ops] {
            FingerprintHasher hasher("arolla::expr::OverloadedOperator");
            hasher.Combine(name, base_ops.size());
            for (const auto& base_op : base_ops) {
              hasher.Combine(base_op->fingerprint());
            }
            return std::move(hasher).Finish();
          }()),
      base_ops_(std::move(base_ops)) {}
absl::StatusOr<ExprOperatorSignature> OverloadedOperator::GetSignature() const {
  if (base_ops_.empty()) {
    return absl::InvalidArgumentError("no base operators");
  }
  return base_ops_.front()->GetSignature();
}
absl::StatusOr<std::string> OverloadedOperator::GetDoc() const {
  if (base_ops_.empty()) {
    return absl::InvalidArgumentError("no base operators");
  }
  return base_ops_.front()->GetDoc();
}
absl::Span<const ExprOperatorPtr> OverloadedOperator::base_ops() const {
  return base_ops_;
}
absl::StatusOr<ExprOperatorPtr> OverloadedOperator::LookupOp(
    absl::Span<const ExprAttributes> inputs) const {
  auto lookup_result = LookupImpl(inputs);
  if (!lookup_result.ok()) {
    return std::move(lookup_result).status();
  }
  return std::get<ExprOperatorPtr>(*lookup_result);
}
absl::StatusOr<ExprAttributes> OverloadedOperator::InferAttributes(
    absl::Span<const ExprAttributes> inputs) const {
  auto lookup_result = LookupImpl(inputs);
  if (!lookup_result.ok()) {
    return std::move(lookup_result).status();
  }
  return std::get<ExprAttributes>(*lookup_result);
}
absl::StatusOr<ExprNodePtr> OverloadedOperator::ToLowerLevel(
    const ExprNodePtr& node) const {
  auto lookup_result = LookupImpl(GetExprAttrs(node->node_deps()));
  if (!lookup_result.ok()) {
    return std::move(lookup_result).status();
  }
  auto& op = std::get<ExprOperatorPtr>(*lookup_result);
  auto& attr = std::get<ExprAttributes>(*lookup_result);
  if (op == nullptr) {
    return node;  
  }
  return ExprNode::UnsafeMakeOperatorNode(
      std::move(op), std::vector(node->node_deps()), std::move(attr));
}
absl::StatusOr<std::tuple<ExprOperatorPtr, ExprAttributes>>
OverloadedOperator::LookupImpl(absl::Span<const ExprAttributes> inputs) const {
  for (const auto& base_op : base_ops_) {
    auto status_or = base_op->InferAttributes(inputs);
    if (absl::IsInvalidArgument(status_or.status())) {
      continue;
    }
    if (!status_or.ok()) {
      return status_or.status();
    }
    if (!status_or->qtype()) {
      return std::make_tuple(ExprOperatorPtr{}, ExprAttributes{});
    }
    return std::make_tuple(base_op, *std::move(status_or));
  }
  if (inputs.size() == 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("unsupported argument type %s",
                        inputs[0].qtype() ? inputs[0].qtype()->name() : "*"));
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("unsupported argument types (%s)",
                      absl::StrReplaceAll(JoinTypeNames(GetAttrQTypes(inputs)),
                                          {{"NULL", "*"}})));
}
absl::string_view OverloadedOperator::py_qvalue_specialization_key() const {
  return "::arolla::expr::OverloadedOperator";
}
}  