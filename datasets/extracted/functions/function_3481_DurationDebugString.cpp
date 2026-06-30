#include <cstddef>
#include <string>
#include <utility>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "common/any.h"
#include "common/casting.h"
#include "common/json.h"
#include "common/value.h"
#include "internal/serialize.h"
#include "internal/status_macros.h"
#include "internal/time.h"
namespace cel {
namespace {
std::string DurationDebugString(absl::Duration value) {
  return internal::DebugStringDuration(value);
}
}  
std::string DurationValue::DebugString() const {
  return DurationDebugString(NativeValue());
}
absl::Status DurationValue::SerializeTo(AnyToJsonConverter&,
                                        absl::Cord& value) const {
  return internal::SerializeDuration(NativeValue(), value);
}
absl::StatusOr<Json> DurationValue::ConvertToJson(AnyToJsonConverter&) const {
  CEL_ASSIGN_OR_RETURN(auto json,
                       internal::EncodeDurationToJson(NativeValue()));
  return JsonString(std::move(json));
}
absl::Status DurationValue::Equal(ValueManager&, const Value& other,
                                  Value& result) const {
  if (auto other_value = As<DurationValue>(other); other_value.has_value()) {
    result = BoolValue{NativeValue() == other_value->NativeValue()};
    return absl::OkStatus();
  }
  result = BoolValue{false};
  return absl::OkStatus();
}
absl::StatusOr<Value> DurationValue::Equal(ValueManager& value_manager,
                                           const Value& other) const {
  Value result;
  CEL_RETURN_IF_ERROR(Equal(value_manager, other, result));
  return result;
}
}  