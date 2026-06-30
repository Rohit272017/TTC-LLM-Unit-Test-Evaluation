#ifndef THIRD_PARTY_CEL_CPP_COMMON_TYPES_DYN_TYPE_H_
#define THIRD_PARTY_CEL_CPP_COMMON_TYPES_DYN_TYPE_H_
#include <ostream>
#include <string>
#include <utility>
#include "absl/strings/string_view.h"
#include "common/type_kind.h"
namespace cel {
class Type;
class TypeParameters;
class DynType final {
 public:
  static constexpr TypeKind kKind = TypeKind::kDyn;
  static constexpr absl::string_view kName = "dyn";
  DynType() = default;
  DynType(const DynType&) = default;
  DynType(DynType&&) = default;
  DynType& operator=(const DynType&) = default;
  DynType& operator=(DynType&&) = default;
  static TypeKind kind() { return kKind; }
  static absl::string_view name() { return kName; }
  static TypeParameters GetParameters();
  static std::string DebugString() { return std::string(name()); }
  constexpr void swap(DynType&) noexcept {}
};
inline constexpr void swap(DynType& lhs, DynType& rhs) noexcept {
  lhs.swap(rhs);
}
inline constexpr bool operator==(DynType, DynType) { return true; }
inline constexpr bool operator!=(DynType lhs, DynType rhs) {
  return !operator==(lhs, rhs);
}
template <typename H>
H AbslHashValue(H state, DynType) {
  return std::move(state);
}
inline std::ostream& operator<<(std::ostream& out, const DynType& type) {
  return out << type.DebugString();
}
}  
#endif  