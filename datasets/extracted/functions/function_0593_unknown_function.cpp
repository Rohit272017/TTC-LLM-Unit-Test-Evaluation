#ifndef LANGSVR_TRAITS_H_
#define LANGSVR_TRAITS_H_
#include <string>
#include <tuple>
namespace langsvr {
template <typename SUCCESS_TYPE, typename FAILURE_TYPE>
struct Result;
}
namespace langsvr::detail {
template <int N, typename... Types>
using NthTypeOf = typename std::tuple_element<N, std::tuple<Types...>>::type;
template <typename RETURN, typename... PARAMETERS>
struct Signature {
    using ret = RETURN;
    using parameters = std::tuple<PARAMETERS...>;
    template <std::size_t N>
    using parameter = NthTypeOf<N, PARAMETERS...>;
    static constexpr std::size_t parameter_count = sizeof...(PARAMETERS);
};
template <typename F>
struct SignatureOf {
    using type = typename SignatureOf<decltype(&F::operator())>::type;
};
template <typename R, typename... ARGS>
struct SignatureOf<R (*)(ARGS...)> {
    using type = Signature<typename std::decay<R>::type, typename std::decay<ARGS>::type...>;
};
template <typename R, typename C, typename... ARGS>
struct SignatureOf<R (C::*)(ARGS...)> {
    using type = Signature<typename std::decay<R>::type, typename std::decay<ARGS>::type...>;
};
template <typename R, typename C, typename... ARGS>
struct SignatureOf<R (C::*)(ARGS...) const> {
    using type = Signature<typename std::decay<R>::type, typename std::decay<ARGS>::type...>;
};
template <typename T, typename... TYPES>
struct TypeIndexHelper;
template <typename T>
struct TypeIndexHelper<T> {
    static const size_t value = 0;
};
template <typename T, typename FIRST, typename... OTHERS>
struct TypeIndexHelper<T, FIRST, OTHERS...> {
    static const size_t value =
        std::is_same_v<T, FIRST> ? 0 : 1 + TypeIndexHelper<T, OTHERS...>::value;
};
template <typename T, typename... TYPES>
struct TypeIndex {
    static const size_t value = TypeIndexHelper<T, TYPES...>::value;
    static_assert(value < sizeof...(TYPES), "type not found");
};
template <typename LHS, typename RHS, typename = void>
struct HasOperatorShiftLeft : std::false_type {};
template <typename LHS, typename RHS>
struct HasOperatorShiftLeft<LHS,
                            RHS,
                            std::void_t<decltype((std::declval<LHS>() << std::declval<RHS>()))>>
    : std::true_type {};
template <typename T, typename = void>
struct IsResult : std::false_type {};
template <typename SUCCESS, typename FAILURE>
struct IsResult<Result<SUCCESS, FAILURE>> : std::true_type {};
}  
namespace langsvr {
template <typename F>
using SignatureOf = typename detail::SignatureOf<std::decay_t<F>>::type;
template <typename F, std::size_t N>
using ParameterType = typename SignatureOf<std::decay_t<F>>::template parameter<N>;
template <typename F>
using ReturnType = typename SignatureOf<std::decay_t<F>>::ret;
template <typename T, typename... TYPES>
static constexpr size_t TypeIndex = detail::TypeIndex<T, TYPES...>::value;
template <typename T, typename... TYPES>
static constexpr bool TypeIsIn = (std::is_same_v<T, TYPES> || ...);
template <typename LHS, typename RHS>
static constexpr bool HasOperatorShiftLeft = detail::HasOperatorShiftLeft<LHS, RHS>::value;
template <typename T>
static constexpr bool IsStringLike =
    std::is_same_v<std::decay_t<T>, std::string> ||
    std::is_same_v<std::decay_t<T>, std::string_view> ||
    std::is_same_v<std::decay_t<T>, const char*> || std::is_same_v<std::decay_t<T>, char*>;
template <typename T>
static constexpr bool IsResult = detail::IsResult<T>::value;
}  
#endif  