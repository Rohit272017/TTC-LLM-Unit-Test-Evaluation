#ifndef THIRD_PARTY_CEL_CPP_COMMON_TYPES_BYTES_WRAPPER_TYPE_H_
#define THIRD_PARTY_CEL_CPP_COMMON_TYPES_BYTES_WRAPPER_TYPE_H_
#include <ostream>
#include <string>
#include <utility>
#include "absl/strings/string_view.h"
#include "common/type_kind.h"
namespace cel {
class Type;
class TypeParameters;
class BytesWrapperType final {
 public:
  static constexpr TypeKind kKind = TypeKind::kBytesWrapper;
  static constexpr absl::string_view kName = "google.protobuf.BytesValue";
  BytesWrapperType() = default;
  BytesWrapperType(const BytesWrapperType&) = default;
  BytesWrapperType(BytesWrapperType&&) = default;
  BytesWrapperType& operator=(const BytesWrapperType&) = default;
  BytesWrapperType& operator=(BytesWrapperType&&) = default;
  static TypeKind kind() { return kKind; }
  static absl::string_view name() { return kName; }
  static TypeParameters GetParameters();
  static std::string DebugString() { return std::string(name()); }
  constexpr void swap(BytesWrapperType&) noexcept {}
};
inline constexpr void swap(BytesWrapperType& lhs,
                           BytesWrapperType& rhs) noexcept {
  lhs.swap(rhs);
}
inline constexpr bool operator==(BytesWrapperType, BytesWrapperType) {
  return true;
}
inline constexpr bool operator!=(BytesWrapperType lhs, BytesWrapperType rhs) {
  return !operator==(lhs, rhs);
}
template <typename H>
H AbslHashValue(H state, BytesWrapperType) {
  return std::move(state);
}
inline std::ostream& operator<<(std::ostream& out,
                                const BytesWrapperType& type) {
  return out << type.DebugString();
}
}  
#endif  