#ifndef THIRD_PARTY_CEL_CPP_COMMON_TYPES_DOUBLE_WRAPPER_TYPE_H_
#define THIRD_PARTY_CEL_CPP_COMMON_TYPES_DOUBLE_WRAPPER_TYPE_H_
#include <ostream>
#include <string>
#include <utility>
#include "absl/strings/string_view.h"
#include "common/type_kind.h"
namespace cel {
class Type;
class TypeParameters;
class DoubleWrapperType final {
 public:
  static constexpr TypeKind kKind = TypeKind::kDoubleWrapper;
  static constexpr absl::string_view kName = "google.protobuf.DoubleValue";
  DoubleWrapperType() = default;
  DoubleWrapperType(const DoubleWrapperType&) = default;
  DoubleWrapperType(DoubleWrapperType&&) = default;
  DoubleWrapperType& operator=(const DoubleWrapperType&) = default;
  DoubleWrapperType& operator=(DoubleWrapperType&&) = default;
  static TypeKind kind() { return kKind; }
  static absl::string_view name() { return kName; }
  static TypeParameters GetParameters();
  static std::string DebugString() { return std::string(name()); }
  constexpr void swap(DoubleWrapperType&) noexcept {}
};
inline constexpr void swap(DoubleWrapperType& lhs,
                           DoubleWrapperType& rhs) noexcept {
  lhs.swap(rhs);
}
inline constexpr bool operator==(DoubleWrapperType, DoubleWrapperType) {
  return true;
}
inline constexpr bool operator!=(DoubleWrapperType lhs, DoubleWrapperType rhs) {
  return !operator==(lhs, rhs);
}
template <typename H>
H AbslHashValue(H state, DoubleWrapperType) {
  return std::move(state);
}
inline std::ostream& operator<<(std::ostream& out,
                                const DoubleWrapperType& type) {
  return out << type.DebugString();
}
}  
#endif  