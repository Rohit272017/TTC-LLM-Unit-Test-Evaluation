#ifndef TENSORSTORE_UTIL_ELEMENT_TRAITS_H_
#define TENSORSTORE_UTIL_ELEMENT_TRAITS_H_
#include <type_traits>
namespace tensorstore {
template <typename Source, typename Dest>
constexpr inline bool IsElementTypeImplicitlyConvertible =
    (std::is_const_v<Source> <= std::is_const_v<Dest>)&&  
    (std::is_same_v<const Source, const Dest> ||
     std::is_void_v<Source> < std::is_void_v<Dest>);
template <class Source, class Dest>
constexpr inline bool IsElementTypeOnlyExplicitlyConvertible =
    (std::is_void_v<Source> > std::is_void_v<Dest>)&&  
    (std::is_const_v<Source> <= std::is_const_v<Dest>);
template <typename Source, typename Dest>
constexpr inline bool IsElementTypeExplicitlyConvertible =
    (std::is_const_v<Source> <= std::is_const_v<Dest>)&&  
    (std::is_void_v<Source> || std::is_void_v<Dest> ||
     std::is_same_v<const Source, const Dest>);
template <typename A, typename B>
constexpr inline bool AreElementTypesCompatible =
    (std::is_void_v<A> || std::is_void_v<B> ||
     std::is_same_v<const A, const B>);
}  
#endif  