#ifndef AROLLA_QEXPR_OPERATORS_BOOL_COMPARISON_H_
#define AROLLA_QEXPR_OPERATORS_BOOL_COMPARISON_H_
#include <type_traits>
namespace arolla {
struct EqualOp {
  using run_on_missing = std::true_type;
  template <typename T>
  bool operator()(const T& lhs, const T& rhs) const {
    return lhs == rhs;
  }
};
struct NotEqualOp {
  using run_on_missing = std::true_type;
  template <typename T>
  bool operator()(const T& lhs, const T& rhs) const {
    return lhs != rhs;
  }
};
struct LessOp {
  using run_on_missing = std::true_type;
  template <typename T>
  bool operator()(const T& lhs, const T& rhs) const {
    return lhs < rhs;
  }
};
struct LessEqualOp {
  using run_on_missing = std::true_type;
  template <typename T>
  bool operator()(const T& lhs, const T& rhs) const {
    return lhs <= rhs;
  }
};
}  
#endif  