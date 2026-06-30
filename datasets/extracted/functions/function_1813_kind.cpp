#ifndef THIRD_PARTY_CEL_CPP_COMMON_TYPES_ERROR_TYPE_H_
#define THIRD_PARTY_CEL_CPP_COMMON_TYPES_ERROR_TYPE_H_
#include <ostream>
#include <string>
#include <utility>
#include "absl/strings/string_view.h"
#include "common/type_kind.h"
namespace cel {
class Type;
class TypeParameters;
class ErrorType final {
 public:
  static constexpr TypeKind kKind = TypeKind::kError;
  static constexpr absl::string_view kName = "*error*";
  ErrorType() = default;
  ErrorType(const ErrorType&) = default;
  ErrorType(ErrorType&&) = default;
  ErrorType& operator=(const ErrorType&) = default;
  ErrorType& operator=(ErrorType&&) = default;
  static TypeKind kind() { return kKind; }
  static absl::string_view name() { return kName; }
  static TypeParameters GetParameters();
  static std::string DebugString() { return std::string(name()); }
  constexpr void swap(ErrorType&) noexcept {}
};
inline constexpr void swap(ErrorType& lhs, ErrorType& rhs) noexcept {
  lhs.swap(rhs);
}
inline constexpr bool operator==(ErrorType, ErrorType) { return true; }
inline constexpr bool operator!=(ErrorType lhs, ErrorType rhs) {
  return !operator==(lhs, rhs);
}
template <typename H>
H AbslHashValue(H state, ErrorType) {
  return std::move(state);
}
inline std::ostream& operator<<(std::ostream& out, const ErrorType& type) {
  return out << type.DebugString();
}
}  
#endif  