#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "common/any.h"
#include "common/casting.h"
#include "common/json.h"
#include "common/value.h"
#include "internal/number.h"
#include "internal/serialize.h"
#include "internal/status_macros.h"
namespace cel {
namespace {
std::string IntDebugString(int64_t value) { return absl::StrCat(value); }
}  
std::string IntValue::DebugString() const {
  return IntDebugString(NativeValue());
}
absl::Status IntValue::SerializeTo(AnyToJsonConverter&,
                                   absl::Cord& value) const {
  return internal::SerializeInt64Value(NativeValue(), value);
}
absl::StatusOr<Json> IntValue::ConvertToJson(AnyToJsonConverter&) const {
  return JsonInt(NativeValue());
}
absl::Status IntValue::Equal(ValueManager&, const Value& other,
                             Value& result) const {
  if (auto other_value = As<IntValue>(other); other_value.has_value()) {
    result = BoolValue{NativeValue() == other_value->NativeValue()};
    return absl::OkStatus();
  }
  if (auto other_value = As<DoubleValue>(other); other_value.has_value()) {
    result =
        BoolValue{internal::Number::FromInt64(NativeValue()) ==
                  internal::Number::FromDouble(other_value->NativeValue())};
    return absl::OkStatus();
  }
  if (auto other_value = As<UintValue>(other); other_value.has_value()) {
    result =
        BoolValue{internal::Number::FromInt64(NativeValue()) ==
                  internal::Number::FromUint64(other_value->NativeValue())};
    return absl::OkStatus();
  }
  result = BoolValue{false};
  return absl::OkStatus();
}
absl::StatusOr<Value> IntValue::Equal(ValueManager& value_manager,
                                      const Value& other) const {
  Value result;
  CEL_RETURN_IF_ERROR(Equal(value_manager, other, result));
  return result;
}
}  