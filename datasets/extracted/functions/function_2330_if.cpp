#include "arolla/qtype/any_qtype.h"
#include <typeinfo>
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "arolla/qtype/simple_qtype.h"
#include "arolla/util/demangle.h"
namespace arolla {
absl::Status Any::InvalidCast(const std::type_info& t) const {
  if (value_.has_value()) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "can not cast Any(%s) to %s", TypeName(value_.type()), TypeName(t)));
  } else {
    return absl::FailedPreconditionError("can not cast an empty ::arolla::Any");
  }
}
AROLLA_DEFINE_SIMPLE_QTYPE(ANY, Any);
}  