#ifndef THIRD_PARTY_CEL_CPP_COMMON_TYPES_TYPE_PARAM_TYPE_H_
#define THIRD_PARTY_CEL_CPP_COMMON_TYPES_TYPE_PARAM_TYPE_H_
#include <ostream>
#include <string>
#include <utility>
#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"
#include "common/type_kind.h"
namespace cel {
class Type;
class TypeParameters;
class TypeParamType final {
 public:
  static constexpr TypeKind kKind = TypeKind::kTypeParam;
  explicit TypeParamType(absl::string_view name ABSL_ATTRIBUTE_LIFETIME_BOUND)
      : name_(name) {}
  TypeParamType() = default;
  TypeParamType(const TypeParamType&) = default;
  TypeParamType(TypeParamType&&) = default;
  TypeParamType& operator=(const TypeParamType&) = default;
  TypeParamType& operator=(TypeParamType&&) = default;
  static TypeKind kind() { return kKind; }
  absl::string_view name() const ABSL_ATTRIBUTE_LIFETIME_BOUND { return name_; }
  static TypeParameters GetParameters();
  std::string DebugString() const { return std::string(name()); }
  friend void swap(TypeParamType& lhs, TypeParamType& rhs) noexcept {
    using std::swap;
    swap(lhs.name_, rhs.name_);
  }
 private:
  absl::string_view name_;
};
inline bool operator==(const TypeParamType& lhs, const TypeParamType& rhs) {
  return lhs.name() == rhs.name();
}
inline bool operator!=(const TypeParamType& lhs, const TypeParamType& rhs) {
  return !operator==(lhs, rhs);
}
template <typename H>
H AbslHashValue(H state, const TypeParamType& type) {
  return H::combine(std::move(state), type.name());
}
inline std::ostream& operator<<(std::ostream& out, const TypeParamType& type) {
  return out << type.DebugString();
}
}  
#endif  