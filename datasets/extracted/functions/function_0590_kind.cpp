#ifndef THIRD_PARTY_CEL_CPP_COMMON_TYPES_BOOL_WRAPPER_TYPE_H_
#define THIRD_PARTY_CEL_CPP_COMMON_TYPES_BOOL_WRAPPER_TYPE_H_
#include <ostream>
#include <string>
#include <utility>
#include "absl/strings/string_view.h"
#include "common/type_kind.h"
namespace cel {
class Type;
class TypeParameters;
class BoolWrapperType final {
 public:
  static constexpr TypeKind kKind = TypeKind::kBoolWrapper;
  static constexpr absl::string_view kName = "google.protobuf.BoolValue";
  BoolWrapperType() = default;
  BoolWrapperType(const BoolWrapperType&) = default;
  BoolWrapperType(BoolWrapperType&&) = default;
  BoolWrapperType& operator=(const BoolWrapperType&) = default;
  BoolWrapperType& operator=(BoolWrapperType&&) = default;
  static TypeKind kind() { return kKind; }
  static absl::string_view name() { return kName; }
  static TypeParameters GetParameters();
  static std::string DebugString() { return std::string(name()); }
  constexpr void swap(BoolWrapperType&) noexcept {}
};
inline constexpr void swap(BoolWrapperType& lhs,
                           BoolWrapperType& rhs) noexcept {
  lhs.swap(rhs);
}
inline constexpr bool operator==(BoolWrapperType, BoolWrapperType) {
  return true;
}
inline constexpr bool operator!=(BoolWrapperType lhs, BoolWrapperType rhs) {
  return !operator==(lhs, rhs);
}
template <typename H>
H AbslHashValue(H state, BoolWrapperType) {
  return std::move(state);
}
inline std::ostream& operator<<(std::ostream& out,
                                const BoolWrapperType& type) {
  return out << type.DebugString();
}
}  
#endif  