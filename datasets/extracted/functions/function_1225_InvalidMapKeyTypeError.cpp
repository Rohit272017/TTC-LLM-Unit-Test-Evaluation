#include <cstddef>
#include <utility>
#include "absl/base/attributes.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "common/casting.h"
#include "common/value.h"
#include "common/value_kind.h"
#include "internal/status_macros.h"
namespace cel {
namespace {
absl::Status InvalidMapKeyTypeError(ValueKind kind) {
  return absl::InvalidArgumentError(
      absl::StrCat("Invalid map key type: '", ValueKindToString(kind), "'"));
}
}  
absl::string_view MapValue::GetTypeName() const {
  return absl::visit(
      [](const auto& alternative) -> absl::string_view {
        return alternative.GetTypeName();
      },
      variant_);
}
std::string MapValue::DebugString() const {
  return absl::visit(
      [](const auto& alternative) -> std::string {
        return alternative.DebugString();
      },
      variant_);
}
absl::Status MapValue::SerializeTo(AnyToJsonConverter& converter,
                                   absl::Cord& value) const {
  return absl::visit(
      [&converter, &value](const auto& alternative) -> absl::Status {
        return alternative.SerializeTo(converter, value);
      },
      variant_);
}
absl::StatusOr<Json> MapValue::ConvertToJson(
    AnyToJsonConverter& converter) const {
  return absl::visit(
      [&converter](const auto& alternative) -> absl::StatusOr<Json> {
        return alternative.ConvertToJson(converter);
      },
      variant_);
}
absl::StatusOr<JsonObject> MapValue::ConvertToJsonObject(
    AnyToJsonConverter& converter) const {
  return absl::visit(
      [&converter](const auto& alternative) -> absl::StatusOr<JsonObject> {
        return alternative.ConvertToJsonObject(converter);
      },
      variant_);
}
bool MapValue::IsZeroValue() const {
  return absl::visit(
      [](const auto& alternative) -> bool { return alternative.IsZeroValue(); },
      variant_);
}
absl::StatusOr<bool> MapValue::IsEmpty() const {
  return absl::visit(
      [](const auto& alternative) -> bool { return alternative.IsEmpty(); },
      variant_);
}
absl::StatusOr<size_t> MapValue::Size() const {
  return absl::visit(
      [](const auto& alternative) -> size_t { return alternative.Size(); },
      variant_);
}
namespace common_internal {
absl::Status MapValueEqual(ValueManager& value_manager, const MapValue& lhs,
                           const MapValue& rhs, Value& result) {
  if (Is(lhs, rhs)) {
    result = BoolValue{true};
    return absl::OkStatus();
  }
  CEL_ASSIGN_OR_RETURN(auto lhs_size, lhs.Size());
  CEL_ASSIGN_OR_RETURN(auto rhs_size, rhs.Size());
  if (lhs_size != rhs_size) {
    result = BoolValue{false};
    return absl::OkStatus();
  }
  CEL_ASSIGN_OR_RETURN(auto lhs_iterator, lhs.NewIterator(value_manager));
  Value lhs_key;
  Value lhs_value;
  Value rhs_value;
  for (size_t index = 0; index < lhs_size; ++index) {
    ABSL_CHECK(lhs_iterator->HasNext());  
    CEL_RETURN_IF_ERROR(lhs_iterator->Next(value_manager, lhs_key));
    bool rhs_value_found;
    CEL_ASSIGN_OR_RETURN(rhs_value_found,
                         rhs.Find(value_manager, lhs_key, rhs_value));
    if (!rhs_value_found) {
      result = BoolValue{false};
      return absl::OkStatus();
    }
    CEL_RETURN_IF_ERROR(lhs.Get(value_manager, lhs_key, lhs_value));
    CEL_RETURN_IF_ERROR(lhs_value.Equal(value_manager, rhs_value, result));
    if (auto bool_value = As<BoolValue>(result);
        bool_value.has_value() && !bool_value->NativeValue()) {
      return absl::OkStatus();
    }
  }
  ABSL_DCHECK(!lhs_iterator->HasNext());
  result = BoolValue{true};
  return absl::OkStatus();
}
absl::Status MapValueEqual(ValueManager& value_manager,
                           const ParsedMapValueInterface& lhs,
                           const MapValue& rhs, Value& result) {
  auto lhs_size = lhs.Size();
  CEL_ASSIGN_OR_RETURN(auto rhs_size, rhs.Size());
  if (lhs_size != rhs_size) {
    result = BoolValue{false};
    return absl::OkStatus();
  }
  CEL_ASSIGN_OR_RETURN(auto lhs_iterator, lhs.NewIterator(value_manager));
  Value lhs_key;
  Value lhs_value;
  Value rhs_value;
  for (size_t index = 0; index < lhs_size; ++index) {
    ABSL_CHECK(lhs_iterator->HasNext());  
    CEL_RETURN_IF_ERROR(lhs_iterator->Next(value_manager, lhs_key));
    bool rhs_value_found;
    CEL_ASSIGN_OR_RETURN(rhs_value_found,
                         rhs.Find(value_manager, lhs_key, rhs_value));
    if (!rhs_value_found) {
      result = BoolValue{false};
      return absl::OkStatus();
    }
    CEL_RETURN_IF_ERROR(lhs.Get(value_manager, lhs_key, lhs_value));
    CEL_RETURN_IF_ERROR(lhs_value.Equal(value_manager, rhs_value, result));
    if (auto bool_value = As<BoolValue>(result);
        bool_value.has_value() && !bool_value->NativeValue()) {
      return absl::OkStatus();
    }
  }
  ABSL_DCHECK(!lhs_iterator->HasNext());
  result = BoolValue{true};
  return absl::OkStatus();
}
}  
absl::Status CheckMapKey(const Value& key) {
  switch (key.kind()) {
    case ValueKind::kBool:
      ABSL_FALLTHROUGH_INTENDED;
    case ValueKind::kInt:
      ABSL_FALLTHROUGH_INTENDED;
    case ValueKind::kUint:
      ABSL_FALLTHROUGH_INTENDED;
    case ValueKind::kString:
      return absl::OkStatus();
    default:
      return InvalidMapKeyTypeError(key.kind());
  }
}
common_internal::ValueVariant MapValue::ToValueVariant() const& {
  return absl::visit(
      [](const auto& alternative) -> common_internal::ValueVariant {
        return alternative;
      },
      variant_);
}
common_internal::ValueVariant MapValue::ToValueVariant() && {
  return absl::visit(
      [](auto&& alternative) -> common_internal::ValueVariant {
        return std::move(alternative);
      },
      std::move(variant_));
}
}  