#include <cstddef>
#include <string>
#include <utility>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "common/any.h"
#include "common/casting.h"
#include "common/json.h"
#include "common/value.h"
#include "internal/serialize.h"
#include "internal/status_macros.h"
namespace cel {
absl::Status NullValue::SerializeTo(AnyToJsonConverter&,
                                    absl::Cord& value) const {
  return internal::SerializeValue(kJsonNull, value);
}
absl::Status NullValue::Equal(ValueManager&, const Value& other,
                              Value& result) const {
  result = BoolValue{InstanceOf<NullValue>(other)};
  return absl::OkStatus();
}
absl::StatusOr<Value> NullValue::Equal(ValueManager& value_manager,
                                       const Value& other) const {
  Value result;
  CEL_RETURN_IF_ERROR(Equal(value_manager, other, result));
  return result;
}
}  