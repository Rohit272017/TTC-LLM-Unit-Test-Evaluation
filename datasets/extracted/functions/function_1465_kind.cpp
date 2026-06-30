#ifndef THIRD_PARTY_CEL_CPP_COMMON_TYPES_BOOL_TYPE_H_
#define THIRD_PARTY_CEL_CPP_COMMON_TYPES_BOOL_TYPE_H_
#include <ostream>
#include <string>
#include <utility>
#include "absl/strings/string_view.h"
#include "common/type_kind.h"
namespace cel {
class Type;
class TypeParameters;
class BoolType final {
 public:
  static constexpr TypeKind kKind = TypeKind::kBool;
  static constexpr absl::string_view kName = "bool";
  BoolType() = default;
  BoolType(const BoolType&) = default;
  BoolType(BoolType&&) = default;
  BoolType& operator=(const BoolType&) = default;
  BoolType& operator=(BoolType&&) = default;
  static TypeKind kind() { return kKind; }
  static absl::string_view name() { return kName; }
  static TypeParameters GetParameters();
  static std::string DebugString() { return std::string(name()); }
  constexpr void swap(BoolType&) noexcept {}
};
inline constexpr void swap(BoolType& lhs, BoolType& rhs) noexcept {
  lhs.swap(rhs);
}
inline constexpr bool operator==(BoolType, BoolType) { return true; }
inline constexpr bool operator!=(BoolType lhs, BoolType rhs) {
  return !operator==(lhs, rhs);
}
template <typename H>
H AbslHashValue(H state, BoolType) {
  return std::move(state);
}
inline std::ostream& operator<<(std::ostream& out, const BoolType& type) {
  return out << type.DebugString();
}
}  
#endif  