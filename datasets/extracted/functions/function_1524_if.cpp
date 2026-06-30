#include "common/values/parsed_map_field_value.h"
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include "absl/base/nullability.h"
#include "absl/base/optimization.h"
#include "absl/log/absl_check.h"
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "common/json.h"
#include "common/value.h"
#include "internal/status_macros.h"
#include "google/protobuf/message.h"
namespace cel {
std::string ParsedMapFieldValue::DebugString() const {
  if (ABSL_PREDICT_FALSE(field_ == nullptr)) {
    return "INVALID";
  }
  return "VALID";
}
absl::Status ParsedMapFieldValue::SerializeTo(AnyToJsonConverter& converter,
                                              absl::Cord& value) const {
  return absl::UnimplementedError("SerializeTo is not yet implemented");
}
absl::StatusOr<Json> ParsedMapFieldValue::ConvertToJson(
    AnyToJsonConverter& converter) const {
  return absl::UnimplementedError("ConvertToJson is not yet implemented");
}
absl::StatusOr<JsonObject> ParsedMapFieldValue::ConvertToJsonObject(
    AnyToJsonConverter& converter) const {
  return absl::UnimplementedError("ConvertToJsonObject is not yet implemented");
}
absl::Status ParsedMapFieldValue::Equal(ValueManager& value_manager,
                                        const Value& other,
                                        Value& result) const {
  return absl::UnimplementedError("Equal is not yet implemented");
}
absl::StatusOr<Value> ParsedMapFieldValue::Equal(ValueManager& value_manager,
                                                 const Value& other) const {
  Value result;
  CEL_RETURN_IF_ERROR(Equal(value_manager, other, result));
  return result;
}
bool ParsedMapFieldValue::IsZeroValue() const { return IsEmpty(); }
bool ParsedMapFieldValue::IsEmpty() const { return Size() == 0; }
size_t ParsedMapFieldValue::Size() const {
  ABSL_DCHECK(*this);
  if (ABSL_PREDICT_FALSE(field_ == nullptr)) {
    return 0;
  }
  return static_cast<size_t>(
      GetReflectionOrDie()->FieldSize(*message_, field_));
}
absl::Status ParsedMapFieldValue::Get(ValueManager& value_manager,
                                      const Value& key, Value& result) const {
  return absl::UnimplementedError("Get is not yet implemented");
}
absl::StatusOr<Value> ParsedMapFieldValue::Get(ValueManager& value_manager,
                                               const Value& key) const {
  Value result;
  CEL_RETURN_IF_ERROR(Get(value_manager, key, result));
  return result;
}
absl::StatusOr<bool> ParsedMapFieldValue::Find(ValueManager& value_manager,
                                               const Value& key,
                                               Value& result) const {
  return absl::UnimplementedError("Find is not yet implemented");
}
absl::StatusOr<std::pair<Value, bool>> ParsedMapFieldValue::Find(
    ValueManager& value_manager, const Value& key) const {
  Value result;
  CEL_ASSIGN_OR_RETURN(auto found, Find(value_manager, key, result));
  if (found) {
    return std::pair{std::move(result), found};
  }
  return std::pair{NullValue(), found};
}
absl::Status ParsedMapFieldValue::Has(ValueManager& value_manager,
                                      const Value& key, Value& result) const {
  return absl::UnimplementedError("Has is not yet implemented");
}
absl::StatusOr<Value> ParsedMapFieldValue::Has(ValueManager& value_manager,
                                               const Value& key) const {
  Value result;
  CEL_RETURN_IF_ERROR(Has(value_manager, key, result));
  return result;
}
absl::Status ParsedMapFieldValue::ListKeys(ValueManager& value_manager,
                                           ListValue& result) const {
  return absl::UnimplementedError("ListKeys is not yet implemented");
}
absl::StatusOr<ListValue> ParsedMapFieldValue::ListKeys(
    ValueManager& value_manager) const {
  ListValue result;
  CEL_RETURN_IF_ERROR(ListKeys(value_manager, result));
  return result;
}
absl::Status ParsedMapFieldValue::ForEach(ValueManager& value_manager,
                                          ForEachCallback callback) const {
  return absl::UnimplementedError("ForEach is not yet implemented");
}
absl::StatusOr<absl::Nonnull<std::unique_ptr<ValueIterator>>>
ParsedMapFieldValue::NewIterator(ValueManager& value_manager) const {
  return absl::UnimplementedError("NewIterator is not yet implemented");
}
absl::Nonnull<const google::protobuf::Reflection*>
ParsedMapFieldValue::GetReflectionOrDie() const {
  return ABSL_DIE_IF_NULL(message_->GetReflection());  
}
}  