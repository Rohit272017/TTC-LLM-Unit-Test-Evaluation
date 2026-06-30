#ifndef THIRD_PARTY_CEL_CPP_COMMON_TYPES_NULL_TYPE_H_
#define THIRD_PARTY_CEL_CPP_COMMON_TYPES_NULL_TYPE_H_
#include <ostream>
#include <string>
#include <utility>
#include "absl/strings/string_view.h"
#include "common/type_kind.h"
namespace cel {
class Type;
class TypeParameters;
class NullType final {
 public:
  static constexpr TypeKind kKind = TypeKind::kNull;
  static constexpr absl::string_view kName = "null_type";
  NullType() = default;
  NullType(const NullType&) = default;
  NullType(NullType&&) = default;
  NullType& operator=(const NullType&) = default;
  NullType& operator=(NullType&&) = default;
  static TypeKind kind() { return kKind; }
  static absl::string_view name() { return kName; }
  static TypeParameters GetParameters();
  static std::string DebugString() { return std::string(name()); }
  constexpr void swap(NullType&) noexcept {}
};
inline constexpr void swap(NullType& lhs, NullType& rhs) noexcept {
  lhs.swap(rhs);
}
inline constexpr bool operator==(NullType, NullType) { return true; }
inline constexpr bool operator!=(NullType lhs, NullType rhs) {
  return !operator==(lhs, rhs);
}
template <typename H>
H AbslHashValue(H state, NullType) {
  return std::move(state);
}
inline std::ostream& operator<<(std::ostream& out, const NullType& type) {
  return out << type.DebugString();
}
}  
#endif  