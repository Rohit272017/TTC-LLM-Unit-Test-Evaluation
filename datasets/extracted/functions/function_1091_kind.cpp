#ifndef THIRD_PARTY_CEL_CPP_COMMON_TYPES_TIMESTAMP_TYPE_H_
#define THIRD_PARTY_CEL_CPP_COMMON_TYPES_TIMESTAMP_TYPE_H_
#include <ostream>
#include <string>
#include <utility>
#include "absl/strings/string_view.h"
#include "common/type_kind.h"
namespace cel {
class Type;
class TypeParameters;
class TimestampType final {
 public:
  static constexpr TypeKind kKind = TypeKind::kTimestamp;
  static constexpr absl::string_view kName = "google.protobuf.Timestamp";
  TimestampType() = default;
  TimestampType(const TimestampType&) = default;
  TimestampType(TimestampType&&) = default;
  TimestampType& operator=(const TimestampType&) = default;
  TimestampType& operator=(TimestampType&&) = default;
  static TypeKind kind() { return kKind; }
  static absl::string_view name() { return kName; }
  static TypeParameters GetParameters();
  static std::string DebugString() { return std::string(name()); }
  constexpr void swap(TimestampType&) noexcept {}
};
inline constexpr void swap(TimestampType& lhs, TimestampType& rhs) noexcept {
  lhs.swap(rhs);
}
inline constexpr bool operator==(TimestampType, TimestampType) { return true; }
inline constexpr bool operator!=(TimestampType lhs, TimestampType rhs) {
  return !operator==(lhs, rhs);
}
template <typename H>
H AbslHashValue(H state, TimestampType) {
  return std::move(state);
}
inline std::ostream& operator<<(std::ostream& out, const TimestampType& type) {
  return out << type.DebugString();
}
}  
#endif  