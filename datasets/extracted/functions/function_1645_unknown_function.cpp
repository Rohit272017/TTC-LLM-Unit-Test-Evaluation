#ifndef AROLLA_QEXPR_OPERATORS_BITWISE_BITWISE_H_
#define AROLLA_QEXPR_OPERATORS_BITWISE_BITWISE_H_
#include <type_traits>
namespace arolla {
struct BitwiseAndOp {
  using run_on_missing = std::true_type;
  template <typename T>
  T operator()(T lhs, T rhs) const {
    return lhs & rhs;
  }
};
struct BitwiseOrOp {
  using run_on_missing = std::true_type;
  template <typename T>
  T operator()(T lhs, T rhs) const {
    return lhs | rhs;
  }
};
struct BitwiseXorOp {
  using run_on_missing = std::true_type;
  template <typename T>
  T operator()(T lhs, T rhs) const {
    return lhs ^ rhs;
  }
};
struct InvertOp {
  using run_on_missing = std::true_type;
  template <typename T>
  T operator()(T x) const {
    return ~x;
  }
};
}  
#endif  