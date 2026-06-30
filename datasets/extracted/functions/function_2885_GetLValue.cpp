#ifndef TENSORSTORE_INTERNAL_TYPE_TRAITS_H_
#define TENSORSTORE_INTERNAL_TYPE_TRAITS_H_
#include <cstddef>
#include <initializer_list>
#include <iosfwd>
#include <type_traits>
#include <utility>
#if defined(__has_builtin)
#if __has_builtin(__type_pack_element)
#define TENSORSTORE_HAS_TYPE_PACK_ELEMENT
#endif
#endif
#ifndef TENSORSTORE_HAS_TYPE_PACK_ELEMENT
#include <tuple>
#endif
#include "absl/meta/type_traits.h"
#include "tensorstore/index.h"
namespace tensorstore {
namespace internal {
struct not_detected {
  ~not_detected() = delete;
  not_detected(not_detected const&) = delete;
  void operator=(not_detected const&) = delete;
};
template <class AlwaysVoid, template <class...> class Op, class... Args>
struct detector_impl {
  using value_t = std::false_type;
  using type = not_detected;
};
template <template <class...> class Op, class... Args>
struct detector_impl<std::void_t<Op<Args...>>, Op, Args...> {
  using value_t = std::true_type;
  using type = Op<Args...>;
};
template <template <class...> class Op, class... Args>
using is_detected = typename detector_impl<void, Op, Args...>::value_t;
template <template <class...> class Op, class... Args>
using detected_t = typename detector_impl<void, Op, Args...>::type;
template <typename T, typename U = T, typename = void>
constexpr inline bool IsEqualityComparable = false;
template <typename T, typename U>
constexpr inline bool IsEqualityComparable<
    T, U,
    std::enable_if_t<std::is_convertible_v<
        decltype(std::declval<T>() == std::declval<U>()), bool>>> = true;
template <typename To, typename, typename... From>
constexpr inline bool IsPackConvertibleWithoutNarrowingHelper = false;
template <typename To, typename... From>
constexpr inline bool IsPackConvertibleWithoutNarrowingHelper<
    To,
    std::void_t<decltype(std::initializer_list<To>{std::declval<From>()...})>,
    From...> = true;
template <typename Source, typename Target>
constexpr inline bool IsOnlyExplicitlyConvertible =
    (std::is_constructible_v<Target, Source> &&
     !std::is_convertible_v<Source, Target>);
template <typename To, typename... From>
constexpr inline bool IsPackConvertibleWithoutNarrowing =
    IsPackConvertibleWithoutNarrowingHelper<To, void, From...>;
template <typename... IndexType>
constexpr inline bool IsIndexPack =
    IsPackConvertibleWithoutNarrowing<Index, IndexType...>;
template <typename ASource, typename BSource, typename ADest, typename BDest>
constexpr inline bool IsPairImplicitlyConvertible =
    std::is_convertible_v<ASource, ADest> &&
    std::is_convertible_v<BSource, BDest>;
template <typename ASource, typename BSource, typename ADest, typename BDest>
constexpr inline bool IsPairExplicitlyConvertible =
    std::is_constructible_v<ADest, ASource> &&
    std::is_constructible_v<BDest, BSource>;
template <typename ASource, typename BSource, typename ADest, typename BDest>
constexpr inline bool IsPairOnlyExplicitlyConvertible =
    IsPairExplicitlyConvertible<ASource, BSource, ADest, BDest> &&
    !IsPairImplicitlyConvertible<ASource, BSource, ADest, BDest>;
template <typename ASource, typename BSource, typename ADest, typename BDest>
constexpr inline bool IsPairAssignable =
    std::is_assignable_v<ADest&, ASource> &&
    std::is_assignable_v<BDest&, BSource>;
template <typename From, typename To>
constexpr inline bool IsConvertibleOrVoid = std::is_convertible_v<From, To>;
template <typename From>
constexpr inline bool IsConvertibleOrVoid<From, void> = true;
template <typename T, typename = void>
constexpr inline bool IsOstreamable = false;
template <typename T>
constexpr inline bool
    IsOstreamable<T, std::void_t<decltype(std::declval<std::ostream&>()
                                          << std ::declval<const T&>())>> =
        true;
template <typename Qualified, typename T>
struct CopyQualifiersHelper {
  using type = T;
};
template <typename Qualified, typename T>
struct CopyQualifiersHelper<const Qualified, T> {
  using type = const typename CopyQualifiersHelper<Qualified, T>::type;
};
template <typename Qualified, typename T>
struct CopyQualifiersHelper<volatile Qualified, T> {
  using type = volatile typename CopyQualifiersHelper<Qualified, T>::type;
};
template <typename Qualified, typename T>
struct CopyQualifiersHelper<const volatile Qualified, T> {
  using type = const volatile typename CopyQualifiersHelper<Qualified, T>::type;
};
template <typename Qualified, typename T>
struct CopyQualifiersHelper<Qualified&, T> {
  using type = typename CopyQualifiersHelper<Qualified, T>::type&;
};
template <typename T, typename Qualified>
struct CopyQualifiersHelper<Qualified&&, T> {
  using type = typename CopyQualifiersHelper<Qualified, T>::type&&;
};
template <typename Qualified, typename T>
using CopyQualifiers =
    typename CopyQualifiersHelper<Qualified, absl::remove_cvref_t<T>>::type;
template <typename T>
inline T& GetLValue(T&& x) {
  return x;
}
template <typename T, typename... U>
using FirstType = T;
template <typename Source, typename Dest>
constexpr inline bool IsConstConvertible =
    (std::is_same_v<Source, Dest> || std::is_same_v<const Source, Dest>);
template <typename Source, typename Dest>
constexpr inline bool IsConstConvertibleOrVoid =
    (std::is_same_v<Source, Dest> || std::is_same_v<const Source, Dest> ||
     std::is_void_v<Dest>);
#ifdef TENSORSTORE_HAS_TYPE_PACK_ELEMENT
#if __clang__
template <size_t I, typename... Ts>
using TypePackElement = __type_pack_element<I, Ts...>;
#else
template <std::size_t I, typename... Ts>
struct TypePackElementImpl {
  using type = __type_pack_element<I, Ts...>;
};
template <size_t I, typename... Ts>
using TypePackElement = typename TypePackElementImpl<I, Ts...>::type;
#endif
#else
template <size_t I, typename... Ts>
using TypePackElement = typename std::tuple_element<I, std::tuple<Ts...>>::type;
#endif
template <typename T>
class EmptyObject {
  static_assert(std::is_empty_v<T>, "T must be an empty type.");
  static_assert(std::is_standard_layout_v<T>, "T must be standard layout.");
  struct T1 {
    char c;
  };
  struct T2 : T {
    char c;
  };
  union Storage {
    constexpr Storage() : t1{} {}
    T1 t1;
    T2 t2;
  };
  Storage storage{};
 public:
  T& get(T* = nullptr) {
    char* c = &storage.t2.c;
    T2* t2 = reinterpret_cast<T2*>(c);
    return *static_cast<T*>(t2);
  }
};
class NonEmptyObjectGetter {
 public:
  template <typename T>
  static T& get(T* pointer) {
    return *pointer;
  }
};
template <typename T>
using PossiblyEmptyObjectGetter =
    std::conditional_t<std::is_empty_v<T>, EmptyObject<T>,
                       NonEmptyObjectGetter>;
template <typename T>
struct DefaultConstructibleFunction {
  constexpr DefaultConstructibleFunction() = default;
  constexpr DefaultConstructibleFunction(const T&) {}
  template <typename... Arg>
  constexpr std::invoke_result_t<T&, Arg...> operator()(Arg&&... arg) const {
    EmptyObject<T> obj;
    return obj.get()(static_cast<Arg&&>(arg)...);
  }
};
template <typename T>
using DefaultConstructibleFunctionIfEmpty =
    std::conditional_t<(std::is_empty_v<T> &&
                        !std::is_default_constructible_v<T>),
                       DefaultConstructibleFunction<T>, T>;
template <typename T>
struct type_identity {
  using type = T;
};
template <typename T>
using type_identity_t = typename type_identity<T>::type;
struct identity {
  template <typename T>
  constexpr T&& operator()(T&& t) const noexcept {
    return static_cast<T&&>(t);
  }
};
template <typename Base, typename Derived>
Base* BaseCast(Derived* derived) {
  return derived;
}
template <typename Base, typename Derived>
const Base* BaseCast(const Derived* derived) {
  return derived;
}
template <typename Base, typename Derived>
Base& BaseCast(Derived& derived) {
  return derived;
}
template <typename Base, typename Derived>
const Base& BaseCast(const Derived& derived) {
  return derived;
}
template <typename T>
using Undocumented = T;
}  
}  
#endif  