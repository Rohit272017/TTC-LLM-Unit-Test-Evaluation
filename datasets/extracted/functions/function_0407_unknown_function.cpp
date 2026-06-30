#include <cstddef>
#include <string>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "common/any.h"
#include "common/json.h"
#include "common/value.h"
namespace cel {
absl::Status UnknownValue::SerializeTo(AnyToJsonConverter&, absl::Cord&) const {
  return absl::FailedPreconditionError(
      absl::StrCat(GetTypeName(), " is unserializable"));
}
absl::StatusOr<Json> UnknownValue::ConvertToJson(AnyToJsonConverter&) const {
  return absl::FailedPreconditionError(
      absl::StrCat(GetTypeName(), " is not convertable to JSON"));
}
absl::Status UnknownValue::Equal(ValueManager&, const Value&,
                                 Value& result) const {
  result = BoolValue{false};
  return absl::OkStatus();
}
}  