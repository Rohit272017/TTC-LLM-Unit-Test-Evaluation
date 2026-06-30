#ifndef THIRD_PARTY_CEL_CPP_COMMON_TYPES_ANY_TYPE_H_
#define THIRD_PARTY_CEL_CPP_COMMON_TYPES_ANY_TYPE_H_
#include <ostream>
#include <string>
#include <utility>
#include "absl/strings/string_view.h"
#include "common/type_kind.h"
namespace cel {
class Type;
class TypeParameters;
class AnyType final {
 public:
  static constexpr TypeKind kKind = TypeKind::kAny;
  static constexpr absl::string_view kName = "google.protobuf.Any";
  AnyType() = default;
  AnyType(const AnyType&) = default;
  AnyType(AnyType&&) = default;
  AnyType& operator=(const AnyType&) = default;
  AnyType& operator=(AnyType&&) = default;
  static TypeKind kind() { return kKind; }
  static absl::string_view name() { return kName; }
  static TypeParameters GetParameters();
  static std::string DebugString() { return std::string(name()); }
  constexpr void swap(AnyType&) noexcept {}
};
inline constexpr void swap(AnyType& lhs, AnyType& rhs) noexcept {
  lhs.swap(rhs);
}
inline constexpr bool operator==(AnyType, AnyType) { return true; }
inline constexpr bool operator!=(AnyType lhs, AnyType rhs) {
  return !operator==(lhs, rhs);
}
template <typename H>
H AbslHashValue(H state, AnyType) {
  return std::move(state);
}
inline std::ostream& operator<<(std::ostream& out, const AnyType& type) {
  return out << type.DebugString();
}
}  
#endif  