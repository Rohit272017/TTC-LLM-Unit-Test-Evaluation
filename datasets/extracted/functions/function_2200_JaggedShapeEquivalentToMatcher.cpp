#ifndef AROLLA_JAGGED_SHAPE_TESTING_MATCHERS_H_
#define AROLLA_JAGGED_SHAPE_TESTING_MATCHERS_H_
#include <ostream>
#include "gtest/gtest.h"
#include "arolla/jagged_shape/jagged_shape.h"
#include "arolla/util/repr.h"
namespace arolla::testing {
namespace matchers_impl {
template <typename Edge>
class JaggedShapeEquivalentToMatcher {
 public:
  using is_gtest_matcher = void;
  explicit JaggedShapeEquivalentToMatcher(JaggedShape<Edge> expected_shape)
      : expected_shape_(std::move(expected_shape)) {}
  bool MatchAndExplain(const JaggedShape<Edge>& shape,
                       ::testing::MatchResultListener* listener) const {
    bool is_equivalent = shape.IsEquivalentTo(expected_shape_);
    *listener << Repr(shape)
              << (is_equivalent ? " which is equivalent"
                                : " which is not equivalent");
    return is_equivalent;
  }
  void DescribeTo(::std::ostream* os) const {
    *os << "is equivalent to " << Repr(expected_shape_);
  }
  void DescribeNegationTo(::std::ostream* os) const {
    *os << "is not equivalent to " << Repr(expected_shape_);
  }
 private:
  JaggedShape<Edge> expected_shape_;
};
}  
template <typename Edge>
auto IsEquivalentTo(const JaggedShape<Edge>& expected_shape) {
  return matchers_impl::JaggedShapeEquivalentToMatcher<Edge>(expected_shape);
}
}  
#endif  