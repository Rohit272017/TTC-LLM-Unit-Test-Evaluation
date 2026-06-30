#include <string>
#include "absl/base/no_destructor.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "common/json.h"
#include "common/type.h"
#include "common/value.h"
namespace cel {
namespace {
std::string ErrorDebugString(const absl::Status& value) {
  ABSL_DCHECK(!value.ok()) << "use of moved-from ErrorValue";
  return value.ToString(absl::StatusToStringMode::kWithEverything);
}
const absl::Status& DefaultErrorValue() {
  static const absl::NoDestructor<absl::Status> value(
      absl::UnknownError("unknown error"));
  return *value;
}
}  
ErrorValue::ErrorValue() : ErrorValue(DefaultErrorValue()) {}
ErrorValue NoSuchFieldError(absl::string_view field) {
  return ErrorValue(absl::NotFoundError(
      absl::StrCat("no_such_field", field.empty() ? "" : " : ", field)));
}
ErrorValue NoSuchKeyError(absl::string_view key) {
  return ErrorValue(
      absl::NotFoundError(absl::StrCat("Key not found in map : ", key)));
}
ErrorValue NoSuchTypeError(absl::string_view type) {
  return ErrorValue(
      absl::NotFoundError(absl::StrCat("type not found: ", type)));
}
ErrorValue DuplicateKeyError() {
  return ErrorValue(absl::AlreadyExistsError("duplicate key in map"));
}
ErrorValue TypeConversionError(absl::string_view from, absl::string_view to) {
  return ErrorValue(absl::InvalidArgumentError(
      absl::StrCat("type conversion error from '", from, "' to '", to, "'")));
}
ErrorValue TypeConversionError(const Type& from, const Type& to) {
  return TypeConversionError(from.DebugString(), to.DebugString());
}
bool IsNoSuchField(const ErrorValue& value) {
  return absl::IsNotFound(value.NativeValue()) &&
         absl::StartsWith(value.NativeValue().message(), "no_such_field");
}
bool IsNoSuchKey(const ErrorValue& value) {
  return absl::IsNotFound(value.NativeValue()) &&
         absl::StartsWith(value.NativeValue().message(),
                          "Key not found in map");
}
std::string ErrorValue::DebugString() const { return ErrorDebugString(value_); }
absl::Status ErrorValue::SerializeTo(AnyToJsonConverter&, absl::Cord&) const {
  return absl::FailedPreconditionError(
      absl::StrCat(GetTypeName(), " is unserializable"));
}
absl::StatusOr<Json> ErrorValue::ConvertToJson(AnyToJsonConverter&) const {
  return absl::FailedPreconditionError(
      absl::StrCat(GetTypeName(), " is not convertable to JSON"));
}
absl::Status ErrorValue::Equal(ValueManager&, const Value&,
                               Value& result) const {
  result = BoolValue{false};
  return absl::OkStatus();
}
}  