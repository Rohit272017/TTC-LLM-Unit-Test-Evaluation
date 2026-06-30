#include "arolla/expr/expr_operator.h"
#include <memory>
#include <string>
#include "absl/base/no_destructor.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "arolla/expr/expr_node.h"
#include "arolla/qtype/qtype.h"
#include "arolla/qtype/simple_qtype.h"
#include "arolla/util/demangle.h"
#include "arolla/util/fingerprint.h"
#include "arolla/util/meta.h"
#include "arolla/util/repr.h"
namespace arolla::expr {
absl::StatusOr<std::string> ExprOperator::GetDoc() const { return ""; }
absl::StatusOr<ExprNodePtr> ExprOperator::ToLowerLevel(
    const ExprNodePtr& node) const {
  return node;
}
ReprToken ExprOperator::GenReprToken() const {
  const auto name = absl::CEscape(display_name_);
  const auto hash = fingerprint_.PythonHash();
  const auto cxx_type = TypeName(typeid(*this));
  const auto short_cxx_type = cxx_type.substr(cxx_type.rfind(':') + 1);
  const auto key = absl::CEscape(py_qvalue_specialization_key());
  struct ReprToken result;
  if (key.empty()) {
    result.str =
        absl::StrFormat("<Operator with name='%s', hash=0x%x, cxx_type='%s'>",
                        name, hash, short_cxx_type);
  } else {
    result.str = absl::StrFormat(
        "<Operator with name='%s', hash=0x%x, cxx_type='%s', key='%s'>", name,
        hash, short_cxx_type, key);
  }
  return result;
}
absl::string_view ExprOperator::py_qvalue_specialization_key() const {
  return "";
}
bool IsBackendOperator(const ExprOperatorPtr&  op,
                       absl::string_view name) {
  return HasBackendExprOperatorTag(op) && op->display_name() == name;
}
}  
namespace arolla {
using ::arolla::expr::ExprOperatorPtr;
void FingerprintHasherTraits<ExprOperatorPtr>::operator()(
    FingerprintHasher* hasher, const ExprOperatorPtr& value) const {
  hasher->Combine(value->fingerprint());
}
ReprToken ReprTraits<ExprOperatorPtr>::operator()(
    const ExprOperatorPtr& value) const {
  DCHECK(value != nullptr);
  if (value == nullptr) {
    return ReprToken{"<Operator nullptr>"};
  }
  return value->GenReprToken();
}
QTypePtr QTypeTraits<ExprOperatorPtr>::type() {
  struct ExprOperatorQType final : SimpleQType {
    ExprOperatorQType()
        : SimpleQType(meta::type<ExprOperatorPtr>(), "EXPR_OPERATOR") {}
    absl::string_view UnsafePyQValueSpecializationKey(
        const void* source) const final {
      if (const auto& op = *static_cast<const ExprOperatorPtr*>(source)) {
        return op->py_qvalue_specialization_key();
      }
      return "";
    }
  };
  static const absl::NoDestructor<ExprOperatorQType> result;
  return result.get();
}
}  