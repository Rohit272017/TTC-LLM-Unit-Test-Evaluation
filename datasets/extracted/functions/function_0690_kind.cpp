#ifndef THIRD_PARTY_CEL_CPP_COMMON_TYPES_UINT_WRAPPER_TYPE_H_
#define THIRD_PARTY_CEL_CPP_COMMON_TYPES_UINT_WRAPPER_TYPE_H_
#include <ostream>
#include <string>
#include <utility>
#include "absl/strings/string_view.h"
#include "common/type_kind.h"
namespace cel {
class Type;
class TypeParameters;
class UintWrapperType final {
 public:
  static constexpr TypeKind kKind = TypeKind::kUintWrapper;
  static constexpr absl::string_view kName = "google.protobuf.UInt64Value";
  UintWrapperType() = default;
  UintWrapperType(const UintWrapperType&) = default;
  UintWrapperType(UintWrapperType&&) = default;
  UintWrapperType& operator=(const UintWrapperType&) = default;
  UintWrapperType& operator=(UintWrapperType&&) = default;
  static TypeKind kind() { return kKind; }
  static absl::string_view name() { return kName; }
  static TypeParameters GetParameters();
  static std::string DebugString() { return std::string(name()); }
  constexpr void swap(UintWrapperType&) noexcept {}
};
inline constexpr void swap(UintWrapperType& lhs,
                           UintWrapperType& rhs) noexcept {
  lhs.swap(rhs);
}
inline constexpr bool operator==(UintWrapperType, UintWrapperType) {
  return true;
}
inline constexpr bool operator!=(UintWrapperType lhs, UintWrapperType rhs) {
  return !operator==(lhs, rhs);
}
template <typename H>
H AbslHashValue(H state, UintWrapperType) {
  return std::move(state);
}
inline std::ostream& operator<<(std::ostream& out,
                                const UintWrapperType& type) {
  return out << type.DebugString();
}
}  
#endif  