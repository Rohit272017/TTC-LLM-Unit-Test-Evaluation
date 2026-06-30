#ifndef TENSORSTORE_INTERNAL_INTEGER_OVERFLOW_H_
#define TENSORSTORE_INTERNAL_INTEGER_OVERFLOW_H_
#include <limits>
#include <type_traits>
#include "absl/base/attributes.h"
namespace tensorstore {
namespace internal {
namespace wrap_on_overflow {
#if ABSL_HAVE_ATTRIBUTE(no_sanitize) && defined(__clang__)
#define TENSORSTORE_ATTRIBUTE_NO_SANITIZE_UNSIGNED_INTEGER_OVERFLOW \
  __attribute__((no_sanitize("unsigned-integer-overflow")))
#else
#define TENSORSTORE_ATTRIBUTE_NO_SANITIZE_UNSIGNED_INTEGER_OVERFLOW
#endif
#define TENSORSTORE_INTERNAL_DEFINE_WRAP_ON_OVERFLOW_OP(OP, NAME) \
  template <typename T>                                           \
  TENSORSTORE_ATTRIBUTE_NO_SANITIZE_UNSIGNED_INTEGER_OVERFLOW     \
      std::enable_if_t<std::is_integral<T>::value, T>             \
      NAME(T a, T b) {                                            \
    using UnsignedT = std::make_unsigned_t<T>;                    \
    return static_cast<T>(static_cast<UnsignedT>(                 \
        static_cast<UnsignedT>(a) OP static_cast<UnsignedT>(b))); \
  }                                                               \
TENSORSTORE_INTERNAL_DEFINE_WRAP_ON_OVERFLOW_OP(+, Add)
TENSORSTORE_INTERNAL_DEFINE_WRAP_ON_OVERFLOW_OP(-, Subtract)
TENSORSTORE_INTERNAL_DEFINE_WRAP_ON_OVERFLOW_OP(*, Multiply)
#undef TENSORSTORE_INTERNAL_DEFINE_WRAP_ON_OVERFLOW_OP
template <typename AccumType, typename T0, typename T1>
inline AccumType InnerProduct(std::ptrdiff_t n, const T0* a, const T1* b) {
  AccumType sum = 0;
  for (std::ptrdiff_t i = 0; i < n; ++i) {
    sum = Add(sum, Multiply(static_cast<AccumType>(a[i]),
                            static_cast<AccumType>(b[i])));
  }
  return sum;
}
template <ptrdiff_t N, typename AccumType, typename T0, typename T1>
inline AccumType InnerProduct(const T0* a, const T1* b) {
  AccumType sum = 0;
  for (std::ptrdiff_t i = 0; i < N; ++i) {
    sum = Add(sum, Multiply(static_cast<AccumType>(a[i]),
                            static_cast<AccumType>(b[i])));
  }
  return sum;
}
}  
template <typename T>
constexpr bool AddOverflow(T a, T b, T* result) {
#if defined(__clang__) || !defined(_MSC_VER)
  return __builtin_add_overflow(a, b, result);
#else
  *result = wrap_on_overflow::Add(a, b);
  return (a > 0 && (b > std::numeric_limits<T>::max() - a)) ||
         (a < 0 && (b < std::numeric_limits<T>::min() - a));
#endif
}
template <typename T>
constexpr T AddSaturate(T a, T b) {
  T result;
  if (AddOverflow(a, b, &result)) {
    result = (b >= 0 ? std::numeric_limits<T>::max()
                     : std::numeric_limits<T>::min());
  }
  return result;
}
template <typename T>
constexpr bool SubOverflow(T a, T b, T* result) {
#if defined(__clang__) || !defined(_MSC_VER)
  return __builtin_sub_overflow(a, b, result);
#else
  *result = wrap_on_overflow::Subtract(a, b);
  return (b < 0 && (a > std::numeric_limits<T>::max() + b)) ||
         (b > 0 && (a < std::numeric_limits<T>::min() + b));
#endif
}
template <typename T>
constexpr bool MulOverflow(T a, T b, T* result) {
#if defined(__clang__) || !defined(_MSC_VER)
  return __builtin_mul_overflow(a, b, result);
#else
  const T r = *result = wrap_on_overflow::Multiply(a, b);
  return b && (r / b) != a;
#endif
}
}  
}  
#endif  