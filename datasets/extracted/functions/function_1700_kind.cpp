#ifndef THIRD_PARTY_CEL_CPP_COMMON_TYPES_UNKNOWN_TYPE_H_
#define THIRD_PARTY_CEL_CPP_COMMON_TYPES_UNKNOWN_TYPE_H_
#include <ostream>
#include <string>
#include <utility>
#include "absl/strings/string_view.h"
#include "common/type_kind.h"
namespace cel {
class Type;
class TypeParameters;
class UnknownType final {
 public:
  static constexpr TypeKind kKind = TypeKind::kUnknown;
  static constexpr absl::string_view kName = "*unknown*";
  UnknownType() = default;
  UnknownType(const UnknownType&) = default;
  UnknownType(UnknownType&&) = default;
  UnknownType& operator=(const UnknownType&) = default;
  UnknownType& operator=(UnknownType&&) = default;
  static TypeKind kind() { return kKind; }
  static absl::string_view name() { return kName; }
  static TypeParameters GetParameters();
  static std::string DebugString() { return std::string(name()); }
  constexpr void swap(UnknownType&) noexcept {}
};
inline constexpr void swap(UnknownType& lhs, UnknownType& rhs) noexcept {
  lhs.swap(rhs);
}
inline constexpr bool operator==(UnknownType, UnknownType) { return true; }
inline constexpr bool operator!=(UnknownType lhs, UnknownType rhs) {
  return !operator==(lhs, rhs);
}
template <typename H>
H AbslHashValue(H state, UnknownType) {
  return std::move(state);
}
inline std::ostream& operator<<(std::ostream& out, const UnknownType& type) {
  return out << type.DebugString();
}
}  
#endif  