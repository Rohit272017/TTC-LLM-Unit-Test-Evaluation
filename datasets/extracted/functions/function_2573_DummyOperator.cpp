#include "arolla/expr/operator_loader/dummy_operator.h"
#include <utility>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "arolla/expr/basic_expr_operator.h"
#include "arolla/expr/expr_attributes.h"
#include "arolla/expr/expr_operator.h"
#include "arolla/expr/expr_operator_signature.h"
#include "arolla/qtype/qtype.h"
#include "arolla/util/fingerprint.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla::operator_loader {
using ::arolla::expr::ExprAttributes;
using ::arolla::expr::ExprOperatorPtr;
using ::arolla::expr::ExprOperatorSignature;
DummyOperator::DummyOperator(absl::string_view name,
                             ExprOperatorSignature signature,
                             absl::string_view doc, QTypePtr result_qtype)
    : ExprOperatorWithFixedSignature(
          name, signature, doc,
          FingerprintHasher("::arolla::operator_loader::DummyOperator")
              .Combine(name, signature, doc, result_qtype)
              .Finish()),
      result_qtype_(std::move(result_qtype)) {}
absl::string_view DummyOperator::py_qvalue_specialization_key() const {
  return "::arolla::operator_loader::DummyOperator";
}
absl::StatusOr<ExprAttributes> DummyOperator::InferAttributes(
    absl::Span<const ExprAttributes> inputs) const {
  RETURN_IF_ERROR(ValidateOpInputsCount(inputs));
  return ExprAttributes(result_qtype_);
}
}  