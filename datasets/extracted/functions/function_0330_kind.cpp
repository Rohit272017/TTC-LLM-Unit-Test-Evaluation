#ifndef THIRD_PARTY_CEL_CPP_COMMON_TYPES_BYTES_TYPE_H_
#define THIRD_PARTY_CEL_CPP_COMMON_TYPES_BYTES_TYPE_H_
#include <ostream>
#include <string>
#include <utility>
#include "absl/strings/string_view.h"
#include "common/type_kind.h"
namespace cel {
class Type;
class TypeParameters;
class BytesType final {
 public:
  static constexpr TypeKind kKind = TypeKind::kBytes;
  static constexpr absl::string_view kName = "bytes";
  BytesType() = default;
  BytesType(const BytesType&) = default;
  BytesType(BytesType&&) = default;
  BytesType& operator=(const BytesType&) = default;
  BytesType& operator=(BytesType&&) = default;
  static TypeKind kind() { return kKind; }
  static absl::string_view name() { return kName; }
  static TypeParameters GetParameters();
  static std::string DebugString() { return std::string(name()); }
  constexpr void swap(BytesType&) noexcept {}
};
inline constexpr void swap(BytesType& lhs, BytesType& rhs) noexcept {
  lhs.swap(rhs);
}
inline constexpr bool operator==(BytesType, BytesType) { return true; }
inline constexpr bool operator!=(BytesType lhs, BytesType rhs) {
  return !operator==(lhs, rhs);
}
template <typename H>
H AbslHashValue(H state, BytesType) {
  return std::move(state);
}
inline std::ostream& operator<<(std::ostream& out, const BytesType& type) {
  return out << type.DebugString();
}
}  
#endif  