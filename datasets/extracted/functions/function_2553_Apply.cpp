#ifndef TENSORSTORE_UTIL_APPLY_MEMBERS_APPLY_MEMBERS_H_
#define TENSORSTORE_UTIL_APPLY_MEMBERS_APPLY_MEMBERS_H_
#include <type_traits>
#include <utility>
#include "absl/base/attributes.h"
namespace half_float {
class half;
}
namespace tensorstore {
class BFloat16;
template <typename T, typename SFINAE = void>
struct ApplyMembers {
  using NotSpecialized = void;
};
namespace internal_apply_members {
struct IgnoreMembers {
  template <typename... T>
  constexpr void operator()(const T&...) const {}
};
template <typename T, typename SFINAE = void>
struct SupportsApplyMembersImpl : public std::true_type {};
template <typename T>
struct SupportsApplyMembersImpl<T, typename ApplyMembers<T>::NotSpecialized>
    : public std::false_type {};
template <typename T>
using MemberApplyMembersCallExpr = decltype(T::ApplyMembers(
    std::declval<const T&>(), internal_apply_members::IgnoreMembers{}));
}  
template <typename T>
struct ApplyMembers<
    T,
    std::enable_if_t<
        !std::is_empty_v<T>,
        std::void_t<internal_apply_members::MemberApplyMembersCallExpr<T>>>> {
  template <typename X, typename F>
  ABSL_ATTRIBUTE_ALWAYS_INLINE static constexpr auto Apply(X&& x, F f) {
    return T::ApplyMembers(x, std::move(f));
  }
};
template <typename T>
struct ApplyMembers<T, std::enable_if_t<std::is_empty_v<T>>> {
  template <typename X, typename F>
  ABSL_ATTRIBUTE_ALWAYS_INLINE static constexpr auto Apply(X&& x, F f) {
    return f();
  }
};
template <typename T>
constexpr inline bool SupportsApplyMembers =
    internal_apply_members::SupportsApplyMembersImpl<T>::value;
template <typename T, typename SFINAE = void>
constexpr inline bool SerializeUsingMemcpy =
    std::is_integral_v<T> || std::is_floating_point_v<T> || std::is_enum_v<T>;
template <>
constexpr inline bool SerializeUsingMemcpy<BFloat16> = true;
template <>
constexpr inline bool SerializeUsingMemcpy<half_float::half> = true;
}  
#endif  