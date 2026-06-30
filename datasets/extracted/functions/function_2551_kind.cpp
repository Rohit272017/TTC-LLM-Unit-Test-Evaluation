#ifndef THIRD_PARTY_CEL_CPP_COMMON_TYPES_INT_WRAPPER_TYPE_H_
#define THIRD_PARTY_CEL_CPP_COMMON_TYPES_INT_WRAPPER_TYPE_H_
#include <ostream>
#include <string>
#include <utility>
#include "absl/strings/string_view.h"
#include "common/type_kind.h"
namespace cel {
class Type;
class TypeParameters;
class IntWrapperType final {
 public:
  static constexpr TypeKind kKind = TypeKind::kIntWrapper;
  static constexpr absl::string_view kName = "google.protobuf.Int64Value";
  IntWrapperType() = default;
  IntWrapperType(const IntWrapperType&) = default;
  IntWrapperType(IntWrapperType&&) = default;
  IntWrapperType& operator=(const IntWrapperType&) = default;
  IntWrapperType& operator=(IntWrapperType&&) = default;
  static TypeKind kind() { return kKind; }
  static absl::string_view name() { return kName; }
  static TypeParameters GetParameters();
  static std::string DebugString() { return std::string(name()); }
  constexpr void swap(IntWrapperType&) noexcept {}
};
inline constexpr void swap(IntWrapperType& lhs, IntWrapperType& rhs) noexcept {
  lhs.swap(rhs);
}
inline constexpr bool operator==(IntWrapperType, IntWrapperType) {
  return true;
}
inline constexpr bool operator!=(IntWrapperType lhs, IntWrapperType rhs) {
  return !operator==(lhs, rhs);
}
template <typename H>
H AbslHashValue(H state, IntWrapperType) {
  return std::move(state);
}
inline std::ostream& operator<<(std::ostream& out, const IntWrapperType& type) {
  return out << type.DebugString();
}
}  
#endif  