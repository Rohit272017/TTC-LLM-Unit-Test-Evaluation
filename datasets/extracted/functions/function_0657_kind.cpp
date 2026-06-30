#ifndef THIRD_PARTY_CEL_CPP_COMMON_TYPES_STRING_TYPE_H_
#define THIRD_PARTY_CEL_CPP_COMMON_TYPES_STRING_TYPE_H_
#include <ostream>
#include <string>
#include <utility>
#include "absl/strings/string_view.h"
#include "common/type_kind.h"
namespace cel {
class Type;
class TypeParameters;
class StringType final {
 public:
  static constexpr TypeKind kKind = TypeKind::kString;
  static constexpr absl::string_view kName = "string";
  StringType() = default;
  StringType(const StringType&) = default;
  StringType(StringType&&) = default;
  StringType& operator=(const StringType&) = default;
  StringType& operator=(StringType&&) = default;
  static TypeKind kind() { return kKind; }
  static absl::string_view name() { return kName; }
  static TypeParameters GetParameters();
  std::string DebugString() const { return std::string(name()); }
  constexpr void swap(StringType&) noexcept {}
};
inline constexpr void swap(StringType& lhs, StringType& rhs) noexcept {
  lhs.swap(rhs);
}
inline constexpr bool operator==(StringType, StringType) { return true; }
inline constexpr bool operator!=(StringType lhs, StringType rhs) {
  return !operator==(lhs, rhs);
}
template <typename H>
H AbslHashValue(H state, StringType) {
  return std::move(state);
}
inline std::ostream& operator<<(std::ostream& out, const StringType& type) {
  return out << type.DebugString();
}
}  
#endif  