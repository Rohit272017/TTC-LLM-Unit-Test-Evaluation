#ifndef THIRD_PARTY_CEL_CPP_COMMON_TYPES_STRING_WRAPPER_TYPE_H_
#define THIRD_PARTY_CEL_CPP_COMMON_TYPES_STRING_WRAPPER_TYPE_H_
#include <ostream>
#include <string>
#include <utility>
#include "absl/strings/string_view.h"
#include "common/type_kind.h"
namespace cel {
class Type;
class TypeParameters;
class StringWrapperType final {
 public:
  static constexpr TypeKind kKind = TypeKind::kStringWrapper;
  static constexpr absl::string_view kName = "google.protobuf.StringValue";
  StringWrapperType() = default;
  StringWrapperType(const StringWrapperType&) = default;
  StringWrapperType(StringWrapperType&&) = default;
  StringWrapperType& operator=(const StringWrapperType&) = default;
  StringWrapperType& operator=(StringWrapperType&&) = default;
  static TypeKind kind() { return kKind; }
  static absl::string_view name() { return kName; }
  static TypeParameters GetParameters();
  static std::string DebugString() { return std::string(name()); }
  constexpr void swap(StringWrapperType&) noexcept {}
};
inline constexpr void swap(StringWrapperType& lhs,
                           StringWrapperType& rhs) noexcept {
  lhs.swap(rhs);
}
inline constexpr bool operator==(StringWrapperType, StringWrapperType) {
  return true;
}
inline constexpr bool operator!=(StringWrapperType lhs, StringWrapperType rhs) {
  return !operator==(lhs, rhs);
}
template <typename H>
H AbslHashValue(H state, StringWrapperType) {
  return std::move(state);
}
inline std::ostream& operator<<(std::ostream& out,
                                const StringWrapperType& type) {
  return out << type.DebugString();
}
}  
#endif  