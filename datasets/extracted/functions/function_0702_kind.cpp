#ifndef THIRD_PARTY_CEL_CPP_COMMON_TYPES_INT_TYPE_H_
#define THIRD_PARTY_CEL_CPP_COMMON_TYPES_INT_TYPE_H_
#include <ostream>
#include <string>
#include <utility>
#include "absl/strings/string_view.h"
#include "common/type_kind.h"
namespace cel {
class Type;
class TypeParameters;
class IntType final {
 public:
  static constexpr TypeKind kKind = TypeKind::kInt;
  static constexpr absl::string_view kName = "int";
  IntType() = default;
  IntType(const IntType&) = default;
  IntType(IntType&&) = default;
  IntType& operator=(const IntType&) = default;
  IntType& operator=(IntType&&) = default;
  static TypeKind kind() { return kKind; }
  static absl::string_view name() { return kName; }
  static TypeParameters GetParameters();
  static std::string DebugString() { return std::string(name()); }
  constexpr void swap(IntType&) noexcept {}
};
inline constexpr void swap(IntType& lhs, IntType& rhs) noexcept {
  lhs.swap(rhs);
}
inline constexpr bool operator==(IntType, IntType) { return true; }
inline constexpr bool operator!=(IntType lhs, IntType rhs) {
  return !operator==(lhs, rhs);
}
template <typename H>
H AbslHashValue(H state, IntType) {
  return std::move(state);
}
inline std::ostream& operator<<(std::ostream& out, const IntType& type) {
  return out << type.DebugString();
}
}  
#endif  