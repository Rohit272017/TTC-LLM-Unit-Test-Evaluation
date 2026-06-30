#ifndef THIRD_PARTY_CEL_CPP_COMMON_TYPES_DURATION_TYPE_H_
#define THIRD_PARTY_CEL_CPP_COMMON_TYPES_DURATION_TYPE_H_
#include <ostream>
#include <string>
#include <utility>
#include "absl/strings/string_view.h"
#include "common/type_kind.h"
namespace cel {
class Type;
class TypeParameters;
class DurationType final {
 public:
  static constexpr TypeKind kKind = TypeKind::kDuration;
  static constexpr absl::string_view kName = "google.protobuf.Duration";
  DurationType() = default;
  DurationType(const DurationType&) = default;
  DurationType(DurationType&&) = default;
  DurationType& operator=(const DurationType&) = default;
  DurationType& operator=(DurationType&&) = default;
  static TypeKind kind() { return kKind; }
  static absl::string_view name() { return kName; }
  static TypeParameters GetParameters();
  static std::string DebugString() { return std::string(name()); }
  constexpr void swap(DurationType&) noexcept {}
};
inline constexpr void swap(DurationType& lhs, DurationType& rhs) noexcept {
  lhs.swap(rhs);
}
inline constexpr bool operator==(DurationType, DurationType) { return true; }
inline constexpr bool operator!=(DurationType lhs, DurationType rhs) {
  return !operator==(lhs, rhs);
}
template <typename H>
H AbslHashValue(H state, DurationType) {
  return std::move(state);
}
inline std::ostream& operator<<(std::ostream& out, const DurationType& type) {
  return out << type.DebugString();
}
}  
#endif  