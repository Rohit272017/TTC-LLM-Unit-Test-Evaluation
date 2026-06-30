#ifndef AROLLA_QTYPE_QTYPE_TRAITS_H_
#define AROLLA_QTYPE_QTYPE_TRAITS_H_
#include <type_traits>
#include "absl/base/attributes.h"
#include "absl/log/check.h"
#include "arolla/qtype/qtype.h"
#include "arolla/qtype/qtype_traits.h"
#include "arolla/util/demangle.h"
namespace arolla {
template <typename T>
struct QTypeTraits;
template <typename T, typename = void>
struct has_qtype_traits : std::false_type {};
template <typename T>
struct has_qtype_traits<T, std::void_t<decltype(QTypeTraits<T>::type())>>
    : std::true_type {};
template <typename T>
constexpr bool has_qtype_traits_v = has_qtype_traits<T>::value;
template <typename T>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline QTypePtr GetQType() {
  static_assert(
      has_qtype_traits_v<T>,
      "QTypeTraits<T> specialization is not included. #include file with "
      "QTypeTraits<T> expliclty to fix this problem. "
      "E.g., #include \"arolla/qtype/base_types.h\" for standard "
      "Arolla scalar and OptionalValue types.");
  DCHECK(typeid(T) == QTypeTraits<T>::type()->type_info())
      << "There is an error in the QType implementation for "
      << QTypeTraits<T>::type()->name();
  DCHECK(sizeof(T) <= QTypeTraits<T>::type()->type_layout().AllocSize())
      << "QType " << QTypeTraits<T>::type()->name()
      << " has too small frame layout to carry a value of C++ type "
      << TypeName<T>();
  return QTypeTraits<T>::type();
}
#define AROLLA_DECLARE_QTYPE(...) \
  template <>                                 \
  struct QTypeTraits<__VA_ARGS__> {           \
    static QTypePtr type();                   \
  }
template <>
struct QTypeTraits<QTypePtr> {
  static QTypePtr type() { return GetQTypeQType(); }
};
}  
#endif  