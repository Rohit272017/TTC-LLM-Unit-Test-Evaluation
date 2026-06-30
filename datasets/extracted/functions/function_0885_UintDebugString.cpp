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
std::string UintDebugString(int64_t value) { return absl::StrCat(value, "u"); }
}  
std::string UintValue::DebugString() const {
  return UintDebugString(NativeValue());
}
absl::Status UintValue::SerializeTo(AnyToJsonConverter&,
                                    absl::Cord& value) const {
  return internal::SerializeUInt64Value(NativeValue(), value);
}
absl::StatusOr<Json> UintValue::ConvertToJson(AnyToJsonConverter&) const {
  return JsonUint(NativeValue());
}
absl::Status UintValue::Equal(ValueManager&, const Value& other,
                              Value& result) const {
  if (auto other_value = As<UintValue>(other); other_value.has_value()) {
    result = BoolValue{NativeValue() == other_value->NativeValue()};
    return absl::OkStatus();
  }
  if (auto other_value = As<DoubleValue>(other); other_value.has_value()) {
    result =
        BoolValue{internal::Number::FromUint64(NativeValue()) ==
                  internal::Number::FromDouble(other_value->NativeValue())};
    return absl::OkStatus();
  }
  if (auto other_value = As<IntValue>(other); other_value.has_value()) {
    result = BoolValue{internal::Number::FromUint64(NativeValue()) ==
                       internal::Number::FromInt64(other_value->NativeValue())};
    return absl::OkStatus();
  }
  result = BoolValue{false};
  return absl::OkStatus();
}
absl::StatusOr<Value> UintValue::Equal(ValueManager& value_manager,
                                       const Value& other) const {
  Value result;
  CEL_RETURN_IF_ERROR(Equal(value_manager, other, result));
  return result;
}
}  