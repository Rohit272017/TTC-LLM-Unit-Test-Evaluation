#ifndef AROLLA_UTIL_META_H_
#define AROLLA_UTIL_META_H_
#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
#include "absl/base/attributes.h"
namespace arolla::meta {
template <class T>
struct type_ {
  using type = T;
};
template <typename T>
using type = type_<T>;
template <class... TypeTraits>
struct type_list {
  using tuple = std::tuple<TypeTraits...>;
};
template <class L>
struct head;
template <class T, class... Ts>
struct head<type_list<T, Ts...>> : type<T> {};
template <class L>
using head_t = typename head<L>::type;
template <class L>
struct tail;
template <class T, class... Ts>
struct tail<type_list<T, Ts...>> {
  using type = type_list<Ts...>;
};
template <class L>
using tail_t = typename tail<L>::type;
template <class L1, class L2>
struct concat;
template <class... Ts1, class... Ts2>
struct concat<type_list<Ts1...>, type_list<Ts2...>> {
  using type = type_list<Ts1..., Ts2...>;
};
template <class L1, class L2>
using concat_t = typename concat<L1, L2>::type;
template <typename Fn, typename... Ts>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline constexpr void foreach_type(
    type_list<Ts...>, Fn fn) {
  (fn(type<Ts>()), ...);
}
template <typename TypeList, typename Fn>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline constexpr void foreach_type(Fn fn) {
  foreach_type(TypeList(), fn);
}
template <typename TypeList, typename T>
struct contains : std::false_type {};
template <typename T, typename... Ts>
struct contains<type_list<Ts...>, T>
    : std::disjunction<std::is_same<T, Ts>...> {};
template <typename TypeList, typename T>
constexpr bool contains_v = contains<TypeList, T>::value;
template <typename T>
struct function_traits : public function_traits<decltype(&T::operator())> {};
template <typename CLS, typename RES, typename... ARGs>
struct function_traits<RES (CLS::*)(ARGs...) const> {
  static constexpr int arity = sizeof...(ARGs);
  using arg_types = type_list<ARGs...>;
  using return_type = RES;
};
template <typename CLS, typename RES, typename... ARGs>
struct function_traits<RES (CLS::*)(ARGs...)> {
  static constexpr int arity = sizeof...(ARGs);
  using arg_types = type_list<ARGs...>;
  using return_type = RES;
};
template <typename RES, typename... ARGs>
struct function_traits<RES (*)(ARGs...)> {
  static constexpr int arity = sizeof...(ARGs);
  using arg_types = type_list<ARGs...>;
  using return_type = RES;
};
template <typename F>
struct function_traits<std::reference_wrapper<F>> : function_traits<F> {};
template <template <typename> class Wrapper, class T>
struct is_wrapped_with : public std::false_type {};
template <template <typename> class Wrapper, class T>
struct is_wrapped_with<Wrapper, Wrapper<T>> : public std::true_type {};
template <template <typename> class Wrapper, class T>
constexpr bool is_wrapped_with_v = is_wrapped_with<Wrapper, T>::value;
template <template <typename> class Wrapper, class T>
struct strip_template {
  using type = T;
};
template <template <typename> class Wrapper, class T>
struct strip_template<Wrapper, Wrapper<T>> {
  using type = T;
};
template <template <typename> class Wrapper, class T>
using strip_template_t = typename strip_template<Wrapper, T>::type;
template <typename T, typename ArgsTypeList, class = void>
struct has_create_method_impl : std::false_type {};
template <class T, class... Args>
struct has_create_method_impl<
    T, type_list<Args...>,
    std::void_t<decltype(T::Create(std::declval<Args>()...))>>
    : std::true_type {};
template <class T, class... Args>
struct has_create_method
    : public has_create_method_impl<T, type_list<Args...>> {};
template <class T, class... Args>
constexpr bool has_create_method_v = has_create_method<T, Args...>::value;
template <typename Tuple, typename Fn, size_t... Is>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline void foreach_tuple_element_impl(
    ABSL_ATTRIBUTE_UNUSED Tuple&& tuple, ABSL_ATTRIBUTE_UNUSED Fn&& fn,
    std::index_sequence<Is...>) {
  (fn(std::get<Is>(std::forward<Tuple>(tuple))), ...);
}
template <typename Tuple, typename Fn>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline void foreach_tuple_element(Tuple&& tuple,
                                                               Fn&& fn) {
  static_assert(std::tuple_size_v<std::decay_t<Tuple>> >= 0,
                "expected a tuple");
  foreach_tuple_element_impl(
      std::forward<Tuple>(tuple), std::forward<Fn>(fn),
      std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>>>());
}
template <typename Tuple, typename Fn, size_t... Is>
void foreach_tuple_element_type_impl(ABSL_ATTRIBUTE_UNUSED Fn&& fn,
                                     std::index_sequence<Is...>) {
  (fn(::arolla::meta::type<std::tuple_element_t<Is, Tuple>>()), ...);
}
template <typename Tuple, typename Fn>
void foreach_tuple_element_type(Fn fn) {
  static_assert(std::tuple_size_v<Tuple> >= 0, "expected a tuple");
  foreach_tuple_element_type_impl<Tuple>(
      std::forward<Fn>(fn),
      std::make_index_sequence<std::tuple_size_v<Tuple>>());
}
template <class, class = void>
struct is_transparent : std::false_type {};
template <class T>
struct is_transparent<T, std::void_t<typename T::is_transparent>>
    : std::true_type {};
template <typename T>
constexpr bool is_transparent_v = is_transparent<T>::value;
}  
#endif  