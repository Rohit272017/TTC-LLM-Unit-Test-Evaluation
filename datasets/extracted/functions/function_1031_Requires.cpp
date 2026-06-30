#ifndef TENSORSTORE_UTIL_STR_CAT_H_
#define TENSORSTORE_UTIL_STR_CAT_H_
#include <cstddef>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include "absl/strings/str_cat.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/span.h"
namespace tensorstore {
namespace internal_strcat {
template <typename... T, typename F>
constexpr bool Requires(F) {
  return std::is_invocable_v<F, T...>;
}
template <typename T>
auto ToAlphaNumOrString(const T& x);
template <typename T>
std::string StringifyUsingOstream(const T& x) {
  std::ostringstream ostr;
  ostr << x;
  return ostr.str();
}
template <typename... T>
std::string StringifyTuple(const std::tuple<T...>& x) {
  return std::apply(
      [](const auto&... item) {
        std::string result = "{";
        size_t i = 0;
        (absl::StrAppend(&result, ToAlphaNumOrString(item),
                         (++i == sizeof...(item) ? "}" : ", ")),
         ...);
        return result;
      },
      x);
}
template <typename A, typename B>
std::string StringifyPair(const std::pair<A, B>& x) {
  return absl::StrCat("{", ToAlphaNumOrString(x.first), ", ",
                      ToAlphaNumOrString(x.second), "}");
}
template <typename Iterator>
std::string StringifyContainer(Iterator begin, Iterator end) {
  std::string result = "{";
  if (begin != end) {
    absl::StrAppend(&result, ToAlphaNumOrString(*begin++));
  }
  for (; begin != end; ++begin) {
    absl::StrAppend(&result, ", ", ToAlphaNumOrString(*begin));
  }
  absl::StrAppend(&result, "}");
  return result;
}
template <typename T>
auto ToAlphaNumOrString(const T& x) {
  if constexpr (std::is_same_v<T, std::nullptr_t>) {
    return "null";
  } else if constexpr (std::is_convertible_v<T, absl::AlphaNum> &&
                       !std::is_enum_v<T>) {
    return x;
  } else if constexpr (internal::IsOstreamable<T>) {
    return StringifyUsingOstream(x);
  } else if constexpr (Requires<const T>(
                           [](auto&& v) -> decltype(StringifyPair(v)) {})) {
    return StringifyPair(x);
  } else if constexpr (Requires<const T>(
                           [](auto&& v) -> decltype(StringifyTuple(v)) {})) {
    return StringifyTuple(x);
  } else if constexpr (Requires<const T>(
                           [](auto&& v) -> decltype(v.begin(), v.end()) {})) {
    return StringifyContainer(x.begin(), x.end());
  } else if constexpr (std::is_enum_v<T>) {
    using I = typename std::underlying_type<T>::type;
    return static_cast<I>(x);
  } else {
    return StringifyUsingOstream(x);
  }
}
}  
template <typename Element, std::ptrdiff_t N>
std::enable_if_t<internal::IsOstreamable<Element>, std::ostream&> operator<<(
    std::ostream& os, ::tensorstore::span<Element, N> s) {
  os << "{";
  std::ptrdiff_t size = s.size();
  for (std::ptrdiff_t i = 0; i < size; ++i) {
    if (i != 0) os << ", ";
    os << s[i];
  }
  return os << "}";
}
template <typename... Arg>
std::string StrCat(const Arg&... arg) {
  return absl::StrCat(internal_strcat::ToAlphaNumOrString(arg)...);
}
template <typename... Arg>
void StrAppend(std::string* result, const Arg&... arg) {
  return absl::StrAppend(result, internal_strcat::ToAlphaNumOrString(arg)...);
}
}  
#endif  