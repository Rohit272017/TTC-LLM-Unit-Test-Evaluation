#ifndef XLA_ARRAY3D_H_
#define XLA_ARRAY3D_H_
#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include "xla/array.h"
#include "xla/types.h"
#include "tsl/platform/logging.h"
namespace xla {
template <typename T>
class Array3D : public Array<T> {
 public:
  Array3D() : Array<T>(std::vector<int64_t>{0, 0, 0}) {}
  Array3D(const int64_t n1, const int64_t n2, const int64_t n3)
      : Array<T>(std::vector<int64_t>{n1, n2, n3}) {}
  Array3D(const int64_t n1, const int64_t n2, const int64_t n3, const T value)
      : Array<T>(std::vector<int64_t>{n1, n2, n3}, value) {}
  Array3D(std::initializer_list<std::initializer_list<std::initializer_list<T>>>
              values)
      : Array<T>(values) {}
  template <typename T2, array_impl::overload_for_float<T, T2> = true>
  Array3D(
      std::initializer_list<std::initializer_list<std::initializer_list<T2>>>
          values)
      : Array<T>(values) {}
  int64_t n1() const { return this->dim(0); }
  int64_t n2() const { return this->dim(1); }
  int64_t n3() const { return this->dim(2); }
};
}  
#endif  