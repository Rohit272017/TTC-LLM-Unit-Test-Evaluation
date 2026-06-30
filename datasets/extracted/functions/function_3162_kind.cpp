#ifndef THIRD_PARTY_CEL_CPP_COMMON_TYPES_DOUBLE_TYPE_H_
#define THIRD_PARTY_CEL_CPP_COMMON_TYPES_DOUBLE_TYPE_H_
#include <ostream>
#include <string>
#include <utility>
#include "absl/strings/string_view.h"
#include "common/type_kind.h"
namespace cel {
class Type;
class TypeParameters;
class DoubleType final {
 public:
  static constexpr TypeKind kKind = TypeKind::kDouble;
  static constexpr absl::string_view kName = "double";
  DoubleType() = default;
  DoubleType(const DoubleType&) = default;
  DoubleType(DoubleType&&) = default;
  DoubleType& operator=(const DoubleType&) = default;
  DoubleType& operator=(DoubleType&&) = default;
  static TypeKind kind() { return kKind; }
  static absl::string_view name() { return kName; }
  static TypeParameters GetParameters();
  static std::string DebugString() { return std::string(name()); }
  constexpr void swap(DoubleType&) noexcept {}
};
inline constexpr void swap(DoubleType& lhs, DoubleType& rhs) noexcept {
  lhs.swap(rhs);
}
inline constexpr bool operator==(DoubleType, DoubleType) { return true; }
inline constexpr bool operator!=(DoubleType lhs, DoubleType rhs) {
  return !operator==(lhs, rhs);
}
template <typename H>
H AbslHashValue(H state, DoubleType) {
  return std::move(state);
}
inline std::ostream& operator<<(std::ostream& out, const DoubleType& type) {
  return out << type.DebugString();
}
}  
#endif  