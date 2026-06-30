#ifndef THIRD_PARTY_CEL_CPP_COMMON_TYPES_UINT_TYPE_H_
#define THIRD_PARTY_CEL_CPP_COMMON_TYPES_UINT_TYPE_H_
#include <ostream>
#include <string>
#include <utility>
#include "absl/strings/string_view.h"
#include "common/type_kind.h"
namespace cel {
class Type;
class TypeParameters;
class UintType final {
 public:
  static constexpr TypeKind kKind = TypeKind::kUint;
  static constexpr absl::string_view kName = "uint";
  UintType() = default;
  UintType(const UintType&) = default;
  UintType(UintType&&) = default;
  UintType& operator=(const UintType&) = default;
  UintType& operator=(UintType&&) = default;
  static TypeKind kind() { return kKind; }
  static absl::string_view name() { return kName; }
  static TypeParameters GetParameters();
  static std::string DebugString() { return std::string(name()); }
  constexpr void swap(UintType&) noexcept {}
};
inline constexpr void swap(UintType& lhs, UintType& rhs) noexcept {
  lhs.swap(rhs);
}
inline constexpr bool operator==(UintType, UintType) { return true; }
inline constexpr bool operator!=(UintType lhs, UintType rhs) {
  return !operator==(lhs, rhs);
}
template <typename H>
H AbslHashValue(H state, UintType) {
  return std::move(state);
}
inline std::ostream& operator<<(std::ostream& out, const UintType& type) {
  return out << type.DebugString();
}
}  
#endif  