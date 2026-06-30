#ifndef AROLLA_OPERATORS_MATH_MATH_H_
#define AROLLA_OPERATORS_MATH_MATH_H_
#include <cmath>
namespace arolla {
struct LogOp {
  float operator()(float x) const { return std::log(x); }
  double operator()(double x) const {
    return std::log(x);
  }
};
struct Log2Op {
  template <typename T>
  T operator()(T x) const {
    return std::log2(x);
  }
};
struct Log10Op {
  template <typename T>
  T operator()(T x) const {
    return std::log10(x);
  }
};
struct Log1pOp {
  template <typename T>
  T operator()(T x) const {
    return std::log1p(x);
  }
};
struct Symlog1pOp {
  template <typename T>
  T operator()(T x) const {
    return x >= 0 ? std::log1p(x) : -std::log1p(-x);
  }
};
struct ExpOp {
  float operator()(float x) const {
    return std::exp(x);
  }
  double operator()(double x) const {
    return std::exp(x);
  }
};
struct Expm1Op {
  template <typename T>
  T operator()(T x) const {
    return std::expm1(x);
  }
};
struct PowOp {
  template <typename T>
  T operator()(T a, T b) const {
    return std::pow(a, b);
  }
  double operator()(double a, double b) const {
    return std::pow(a, b);
  }
};
struct SigmoidOp {
  template <typename T>
  T operator()(T value, T half, T slope) const {
    return 1.0f / (1.0f + ExpOp()(slope * (half - value)));
  }
};
struct LogitOp {
  template <typename T>
  T operator()(T p) const {
    return LogOp()(p) - std::log1p(-p);
  }
};
struct LogSigmoidOp {
  template <typename T>
  T operator()(T x) const {
    if (x >= 0) {
      return -std::log1p(std::exp(-x));
    }
    return x - std::log1p(std::exp(x));
  }
};
struct SinOp {
  template <typename T>
  T operator()(T x) const {
    return std::sin(x);
  }
};
struct CosOp {
  template <typename T>
  T operator()(T x) const {
    return std::cos(x);
  }
};
struct SinhOp {
  template <typename T>
  T operator()(T x) const {
    return std::sinh(x);
  }
};
struct AtanOp {
  template <typename T>
  T operator()(T x) const {
    return std::atan(x);
  }
};
}  
#endif  