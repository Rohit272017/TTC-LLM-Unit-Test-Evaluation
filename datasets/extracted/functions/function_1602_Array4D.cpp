#ifndef XLA_ARRAY4D_H_
#define XLA_ARRAY4D_H_
#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/array2d.h"
#include "xla/types.h"
#include "tsl/platform/logging.h"
namespace xla {
template <typename T>
class Array4D : public Array<T> {
 public:
  Array4D() : Array<T>(std::vector<int64_t>{0, 0, 0, 0}) {}
  Array4D(int64_t planes, int64_t depth, int64_t height, int64_t width)
      : Array<T>(std::vector<int64_t>{planes, depth, height, width}) {}
  Array4D(int64_t planes, int64_t depth, int64_t height, int64_t width, T value)
      : Array<T>(std::vector<int64_t>{planes, depth, height, width}, value) {}
  template <typename Container = std::initializer_list<T>>
  Array4D(int64_t planes, int64_t depth, int64_t height, int64_t width,
          const Container& values)
      : Array4D(planes, depth, height, width) {
    this->SetValues(values);
  }
  Array4D(std::initializer_list<std::initializer_list<
              std::initializer_list<std::initializer_list<T>>>>
              values)
      : Array<T>(values) {}
  template <typename T2, array_impl::overload_for_float<T, T2> = true>
  Array4D(std::initializer_list<std::initializer_list<
              std::initializer_list<std::initializer_list<T2>>>>
              values)
      : Array<T>(values) {}
  int64_t n4() const { return this->dim(3); }
  int64_t n3() const { return this->dim(2); }
  int64_t n2() const { return this->dim(1); }
  int64_t n1() const { return this->dim(0); }
  int64_t width() const { return this->dim(3); }
  int64_t height() const { return this->dim(2); }
  int64_t depth() const { return this->dim(1); }
  int64_t planes() const { return this->dim(0); }
  void FillWithYX(const Array2D<T>& value) {
    CHECK_EQ(value.height(), height());
    CHECK_EQ(value.width(), width());
    for (int64_t plane = 0; plane < planes(); ++plane) {
      for (int64_t depth = 0; depth < this->depth(); ++depth) {
        for (int64_t height = 0; height < this->height(); ++height) {
          for (int64_t width = 0; width < this->width(); ++width) {
            (*this)(plane, depth, height, width) = value(height, width);
          }
        }
      }
    }
  }
  void FillWithZY(const Array2D<T>& value) {
    CHECK_EQ(value.height(), depth());
    CHECK_EQ(value.width(), height());
    for (int64_t plane = 0; plane < planes(); ++plane) {
      for (int64_t depth = 0; depth < this->depth(); ++depth) {
        for (int64_t height = 0; height < this->height(); ++height) {
          for (int64_t width = 0; width < this->width(); ++width) {
            (*this)(plane, depth, height, width) = value(depth, height);
          }
        }
      }
    }
  }
  void FillWithPZ(const Array2D<T>& value) {
    CHECK_EQ(value.height(), planes());
    CHECK_EQ(value.width(), depth());
    for (int64_t height = 0; height < this->height(); ++height) {
      for (int64_t width = 0; width < this->width(); ++width) {
        for (int64_t plane = 0; plane < planes(); ++plane) {
          for (int64_t depth = 0; depth < this->depth(); ++depth) {
            (*this)(plane, depth, height, width) = value(plane, depth);
          }
        }
      }
    }
  }
  void FillWithMinorDimNum() {
    LOG(INFO) << "width: " << this->width();
    LOG(INFO) << "height: " << this->height();
    LOG(INFO) << "depth: " << this->depth();
    LOG(INFO) << "planes: " << this->planes();
    for (int64_t height = 0; height < this->height(); ++height) {
      for (int64_t width = 0; width < this->width(); ++width) {
        for (int64_t plane = 0; plane < planes(); ++plane) {
          for (int64_t depth = 0; depth < this->depth(); ++depth) {
            float this_val = plane * this->depth() + depth;
            (*this)(plane, depth, height, width) = this_val;
          }
        }
      }
    }
  }
};
}  
#endif  