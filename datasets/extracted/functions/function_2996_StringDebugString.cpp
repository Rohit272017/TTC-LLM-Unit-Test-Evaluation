#include <cstddef>
#include <string>
#include <utility>
#include "absl/functional/overload.h"
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
#include "internal/strings.h"
#include "internal/utf8.h"
namespace cel {
namespace {
template <typename Bytes>
std::string StringDebugString(const Bytes& value) {
  return value.NativeValue(absl::Overload(
      [](absl::string_view string) -> std::string {
        return internal::FormatStringLiteral(string);
      },
      [](const absl::Cord& cord) -> std::string {
        if (auto flat = cord.TryFlat(); flat.has_value()) {
          return internal::FormatStringLiteral(*flat);
        }
        return internal::FormatStringLiteral(static_cast<std::string>(cord));
      }));
}
}  
std::string StringValue::DebugString() const {
  return StringDebugString(*this);
}
absl::Status StringValue::SerializeTo(AnyToJsonConverter&,
                                      absl::Cord& value) const {
  return NativeValue([&value](const auto& bytes) -> absl::Status {
    return internal::SerializeStringValue(bytes, value);
  });
}
absl::StatusOr<Json> StringValue::ConvertToJson(AnyToJsonConverter&) const {
  return NativeCord();
}
absl::Status StringValue::Equal(ValueManager&, const Value& other,
                                Value& result) const {
  if (auto other_value = As<StringValue>(other); other_value.has_value()) {
    result = NativeValue([other_value](const auto& value) -> BoolValue {
      return other_value->NativeValue(
          [&value](const auto& other_value) -> BoolValue {
            return BoolValue{value == other_value};
          });
    });
    return absl::OkStatus();
  }
  result = BoolValue{false};
  return absl::OkStatus();
}
size_t StringValue::Size() const {
  return NativeValue([](const auto& alternative) -> size_t {
    return internal::Utf8CodePointCount(alternative);
  });
}
bool StringValue::IsEmpty() const {
  return NativeValue(
      [](const auto& alternative) -> bool { return alternative.empty(); });
}
bool StringValue::Equals(absl::string_view string) const {
  return NativeValue([string](const auto& alternative) -> bool {
    return alternative == string;
  });
}
bool StringValue::Equals(const absl::Cord& string) const {
  return NativeValue([&string](const auto& alternative) -> bool {
    return alternative == string;
  });
}
bool StringValue::Equals(const StringValue& string) const {
  return string.NativeValue(
      [this](const auto& alternative) -> bool { return Equals(alternative); });
}
namespace {
int CompareImpl(absl::string_view lhs, absl::string_view rhs) {
  return lhs.compare(rhs);
}
int CompareImpl(absl::string_view lhs, const absl::Cord& rhs) {
  return -rhs.Compare(lhs);
}
int CompareImpl(const absl::Cord& lhs, absl::string_view rhs) {
  return lhs.Compare(rhs);
}
int CompareImpl(const absl::Cord& lhs, const absl::Cord& rhs) {
  return lhs.Compare(rhs);
}
}  
int StringValue::Compare(absl::string_view string) const {
  return NativeValue([string](const auto& alternative) -> int {
    return CompareImpl(alternative, string);
  });
}
int StringValue::Compare(const absl::Cord& string) const {
  return NativeValue([&string](const auto& alternative) -> int {
    return CompareImpl(alternative, string);
  });
}
int StringValue::Compare(const StringValue& string) const {
  return string.NativeValue(
      [this](const auto& alternative) -> int { return Compare(alternative); });
}
}  