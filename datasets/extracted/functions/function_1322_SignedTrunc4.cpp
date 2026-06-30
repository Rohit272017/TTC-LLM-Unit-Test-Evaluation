#ifndef TENSORSTORE_UTIL_INT4_H_
#define TENSORSTORE_UTIL_INT4_H_
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <nlohmann/json_fwd.hpp>
namespace tensorstore {
class Int4Padded;
}  
namespace std {
template <>
struct numeric_limits<::tensorstore::Int4Padded>;
}  
namespace tensorstore {
namespace internal {
constexpr int8_t SignedTrunc4(int8_t x) {
  return static_cast<int8_t>(static_cast<uint8_t>(x) << 4) >> 4;
}
}  
class Int4Padded {
 public:
  constexpr Int4Padded() : rep_(0) {}
  template <typename T,
            typename = std::enable_if_t<std::is_convertible_v<T, int8_t>>>
  constexpr explicit Int4Padded(T x)
      : rep_(internal::SignedTrunc4(static_cast<int8_t>(x))) {}
  constexpr operator int8_t() const {
    return internal::SignedTrunc4(rep_);
  }
  Int4Padded& operator=(bool v) { return *this = static_cast<Int4Padded>(v); }
  template <typename T>
  std::enable_if_t<std::numeric_limits<T>::is_integer, Int4Padded&> operator=(
      T v) {  
    return *this = static_cast<Int4Padded>(v);
  }
#define TENSORSTORE_INTERNAL_INT4_PADDED_ARITHMETIC_OP(OP)                \
  friend Int4Padded operator OP(Int4Padded a, Int4Padded b) {             \
    return Int4Padded(a.rep_ OP b.rep_);                                  \
  }                                                                       \
  template <typename T>                                                   \
  friend std::enable_if_t<std::numeric_limits<T>::is_integer, Int4Padded> \
  operator OP(Int4Padded a, T b) {                                        \
    return Int4Padded(a.rep_ OP b);                                       \
  }                                                                       \
  template <typename T>                                                   \
  friend std::enable_if_t<std::numeric_limits<T>::is_integer, Int4Padded> \
  operator OP(T a, Int4Padded b) {                                        \
    return Int4Padded(a OP b.rep_);                                       \
  }                                                                       \
#define TENSORSTORE_INTERNAL_INT4_PADDED_ARITHMETIC_ASSIGN_OP(OP)          \
  friend Int4Padded& operator OP##=(Int4Padded& a, Int4Padded b) {         \
    return a = Int4Padded(a.rep_ OP b.rep_);                               \
  }                                                                        \
  template <typename T>                                                    \
  friend std::enable_if_t<std::numeric_limits<T>::is_integer, Int4Padded&> \
  operator OP##=(Int4Padded& a, T b) {                                     \
    return a = Int4Padded(a.rep_ OP b);                                    \
  }                                                                        \
  TENSORSTORE_INTERNAL_INT4_PADDED_ARITHMETIC_OP(+)
  TENSORSTORE_INTERNAL_INT4_PADDED_ARITHMETIC_ASSIGN_OP(+)
  TENSORSTORE_INTERNAL_INT4_PADDED_ARITHMETIC_OP(-)
  TENSORSTORE_INTERNAL_INT4_PADDED_ARITHMETIC_ASSIGN_OP(-)
  TENSORSTORE_INTERNAL_INT4_PADDED_ARITHMETIC_OP(*)
  TENSORSTORE_INTERNAL_INT4_PADDED_ARITHMETIC_ASSIGN_OP(*)
  TENSORSTORE_INTERNAL_INT4_PADDED_ARITHMETIC_OP(/)
  TENSORSTORE_INTERNAL_INT4_PADDED_ARITHMETIC_ASSIGN_OP(/)
  TENSORSTORE_INTERNAL_INT4_PADDED_ARITHMETIC_OP(%)
  TENSORSTORE_INTERNAL_INT4_PADDED_ARITHMETIC_ASSIGN_OP(%)
  TENSORSTORE_INTERNAL_INT4_PADDED_ARITHMETIC_OP(&)
  TENSORSTORE_INTERNAL_INT4_PADDED_ARITHMETIC_ASSIGN_OP(&)
  TENSORSTORE_INTERNAL_INT4_PADDED_ARITHMETIC_OP(|)
  TENSORSTORE_INTERNAL_INT4_PADDED_ARITHMETIC_ASSIGN_OP(|)
  TENSORSTORE_INTERNAL_INT4_PADDED_ARITHMETIC_OP(^)
  TENSORSTORE_INTERNAL_INT4_PADDED_ARITHMETIC_ASSIGN_OP(^)
  TENSORSTORE_INTERNAL_INT4_PADDED_ARITHMETIC_OP(<<)
  TENSORSTORE_INTERNAL_INT4_PADDED_ARITHMETIC_ASSIGN_OP(<<)
  TENSORSTORE_INTERNAL_INT4_PADDED_ARITHMETIC_OP(>>)
  TENSORSTORE_INTERNAL_INT4_PADDED_ARITHMETIC_ASSIGN_OP(>>)
#undef TENSORSTORE_INTERNAL_INT4_PADDED_ARITHMETIC_OP
#undef TENSORSTORE_INTERNAL_INT4_PADDED_ARITHMETIC_ASSIGN_OP
  friend Int4Padded operator~(Int4Padded a) {
    Int4Padded result;
    result.rep_ = internal::SignedTrunc4(~a.rep_);
    return result;
  }
  friend Int4Padded operator-(Int4Padded a) {
    Int4Padded result;
    result.rep_ = internal::SignedTrunc4(-a.rep_);
    return result;
  }
  friend Int4Padded operator+(Int4Padded a) { return a; }
  friend Int4Padded operator++(Int4Padded& a) {
    a += Int4Padded(1);
    return a;
  }
  friend Int4Padded operator--(Int4Padded& a) {
    a -= Int4Padded(1);
    return a;
  }
  friend Int4Padded operator++(Int4Padded& a, int) {
    Int4Padded original_value = a;
    ++a;
    return original_value;
  }
  friend Int4Padded operator--(Int4Padded& a, int) {
    Int4Padded original_value = a;
    --a;
    return original_value;
  }
  template <template <typename U, typename V, typename... Args>
            class ObjectType ,
            template <typename U, typename... Args>
            class ArrayType ,
            class StringType , class BooleanType ,
            class NumberIntegerType ,
            class NumberUnsignedType ,
            class NumberFloatType ,
            template <typename U> class AllocatorType ,
            template <typename T, typename SFINAE = void>
            class JSONSerializer ,
            class BinaryType >
  friend void to_json(
      ::nlohmann::basic_json<ObjectType, ArrayType, StringType, BooleanType,
                             NumberIntegerType, NumberUnsignedType,
                             NumberFloatType, AllocatorType, JSONSerializer,
                             BinaryType>& j,
      Int4Padded v) {
    j = static_cast<NumberIntegerType>(v);
  }
  constexpr friend bool operator==(const Int4Padded& a, const Int4Padded& b) {
    return internal::SignedTrunc4(a.rep_) == internal::SignedTrunc4(b.rep_);
  }
  constexpr friend bool operator!=(const Int4Padded& a, const Int4Padded& b) {
    return !(a == b);
  }
  struct bitcast_construct_t {};
  explicit constexpr Int4Padded(bitcast_construct_t, int8_t rep) : rep_(rep) {}
  int8_t rep_;
};
inline Int4Padded abs(Int4Padded x) {
  x.rep_ = internal::SignedTrunc4(::std::abs(x.rep_));
  return x;
}
inline Int4Padded pow(Int4Padded x, Int4Padded y) {
  return Int4Padded(std::pow(static_cast<int8_t>(x), static_cast<int8_t>(y)));
}
}  
namespace std {
template <>
struct numeric_limits<tensorstore::Int4Padded> {
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = true;
  static constexpr bool is_exact = true;
  static constexpr bool has_infinity = false;
  static constexpr bool has_quiet_NaN = false;
  static constexpr bool has_signaling_NaN = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = true;
  static constexpr int digits = 3;
  static constexpr int digits10 = 0;
  static constexpr int max_digits10 = 0;
  static constexpr int radix = 2;
  static constexpr tensorstore::Int4Padded min() {
    return tensorstore::Int4Padded(
        tensorstore::Int4Padded::bitcast_construct_t{}, int8_t{-8});
  }
  static constexpr tensorstore::Int4Padded lowest() {
    return tensorstore::Int4Padded(
        tensorstore::Int4Padded::bitcast_construct_t{}, int8_t{-8});
  }
  static constexpr tensorstore::Int4Padded max() {
    return tensorstore::Int4Padded(
        tensorstore::Int4Padded::bitcast_construct_t{}, int8_t{7});
  }
};
}  
#endif  