#ifndef THIRD_PARTY_CEL_CPP_INTERNAL_NUMBER_H_
#define THIRD_PARTY_CEL_CPP_INTERNAL_NUMBER_H_
#include <cmath>
#include <cstdint>
#include <limits>
#include "absl/types/variant.h"
namespace cel::internal {
constexpr int64_t kInt64Max = std::numeric_limits<int64_t>::max();
constexpr int64_t kInt64Min = std::numeric_limits<int64_t>::lowest();
constexpr uint64_t kUint64Max = std::numeric_limits<uint64_t>::max();
constexpr uint64_t kUintToIntMax = static_cast<uint64_t>(kInt64Max);
constexpr double kDoubleToIntMax = static_cast<double>(kInt64Max);
constexpr double kDoubleToIntMin = static_cast<double>(kInt64Min);
constexpr double kDoubleToUintMax = static_cast<double>(kUint64Max);
template <typename T>
constexpr int RoundingError() {
  return 1 << (std::numeric_limits<T>::digits -
               std::numeric_limits<double>::digits - 1);
}
constexpr double kMaxDoubleRepresentableAsInt =
    static_cast<double>(kInt64Max - RoundingError<int64_t>());
constexpr double kMaxDoubleRepresentableAsUint =
    static_cast<double>(kUint64Max - RoundingError<uint64_t>());
#define CEL_ABSL_VISIT_CONSTEXPR
using NumberVariant = absl::variant<double, uint64_t, int64_t>;
enum class ComparisonResult {
  kLesser,
  kEqual,
  kGreater,
  kNanInequal
};
constexpr ComparisonResult Invert(ComparisonResult result) {
  switch (result) {
    case ComparisonResult::kLesser:
      return ComparisonResult::kGreater;
    case ComparisonResult::kGreater:
      return ComparisonResult::kLesser;
    case ComparisonResult::kEqual:
      return ComparisonResult::kEqual;
    case ComparisonResult::kNanInequal:
      return ComparisonResult::kNanInequal;
  }
}
template <typename OutType>
struct ConversionVisitor {
  template <typename InType>
  constexpr OutType operator()(InType v) {
    return static_cast<OutType>(v);
  }
};
template <typename T>
constexpr ComparisonResult Compare(T a, T b) {
  return (a > b)    ? ComparisonResult::kGreater
         : (a == b) ? ComparisonResult::kEqual
                    : ComparisonResult::kLesser;
}
constexpr ComparisonResult DoubleCompare(double a, double b) {
  if (!(a == a) || !(b == b)) {
    return ComparisonResult::kNanInequal;
  }
  return Compare(a, b);
}
struct DoubleCompareVisitor {
  constexpr explicit DoubleCompareVisitor(double v) : v(v) {}
  constexpr ComparisonResult operator()(double other) const {
    return DoubleCompare(v, other);
  }
  constexpr ComparisonResult operator()(uint64_t other) const {
    if (v > kDoubleToUintMax) {
      return ComparisonResult::kGreater;
    } else if (v < 0) {
      return ComparisonResult::kLesser;
    } else {
      return DoubleCompare(v, static_cast<double>(other));
    }
  }
  constexpr ComparisonResult operator()(int64_t other) const {
    if (v > kDoubleToIntMax) {
      return ComparisonResult::kGreater;
    } else if (v < kDoubleToIntMin) {
      return ComparisonResult::kLesser;
    } else {
      return DoubleCompare(v, static_cast<double>(other));
    }
  }
  double v;
};
struct UintCompareVisitor {
  constexpr explicit UintCompareVisitor(uint64_t v) : v(v) {}
  constexpr ComparisonResult operator()(double other) const {
    return Invert(DoubleCompareVisitor(other)(v));
  }
  constexpr ComparisonResult operator()(uint64_t other) const {
    return Compare(v, other);
  }
  constexpr ComparisonResult operator()(int64_t other) const {
    if (v > kUintToIntMax || other < 0) {
      return ComparisonResult::kGreater;
    } else {
      return Compare(v, static_cast<uint64_t>(other));
    }
  }
  uint64_t v;
};
struct IntCompareVisitor {
  constexpr explicit IntCompareVisitor(int64_t v) : v(v) {}
  constexpr ComparisonResult operator()(double other) {
    return Invert(DoubleCompareVisitor(other)(v));
  }
  constexpr ComparisonResult operator()(uint64_t other) {
    return Invert(UintCompareVisitor(other)(v));
  }
  constexpr ComparisonResult operator()(int64_t other) {
    return Compare(v, other);
  }
  int64_t v;
};
struct CompareVisitor {
  explicit constexpr CompareVisitor(NumberVariant rhs) : rhs(rhs) {}
  CEL_ABSL_VISIT_CONSTEXPR ComparisonResult operator()(double v) {
    return absl::visit(DoubleCompareVisitor(v), rhs);
  }
  CEL_ABSL_VISIT_CONSTEXPR ComparisonResult operator()(uint64_t v) {
    return absl::visit(UintCompareVisitor(v), rhs);
  }
  CEL_ABSL_VISIT_CONSTEXPR ComparisonResult operator()(int64_t v) {
    return absl::visit(IntCompareVisitor(v), rhs);
  }
  NumberVariant rhs;
};
struct LosslessConvertibleToIntVisitor {
  constexpr bool operator()(double value) const {
    return value >= kDoubleToIntMin && value <= kMaxDoubleRepresentableAsInt &&
           value == static_cast<double>(static_cast<int64_t>(value));
  }
  constexpr bool operator()(uint64_t value) const {
    return value <= kUintToIntMax;
  }
  constexpr bool operator()(int64_t value) const { return true; }
};
struct LosslessConvertibleToUintVisitor {
  constexpr bool operator()(double value) const {
    return value >= 0 && value <= kMaxDoubleRepresentableAsUint &&
           value == static_cast<double>(static_cast<uint64_t>(value));
  }
  constexpr bool operator()(uint64_t value) const { return true; }
  constexpr bool operator()(int64_t value) const { return value >= 0; }
};
class Number {
 public:
  static constexpr Number FromInt64(int64_t value) { return Number(value); }
  static constexpr Number FromUint64(uint64_t value) { return Number(value); }
  static constexpr Number FromDouble(double value) { return Number(value); }
  constexpr explicit Number(double double_value) : value_(double_value) {}
  constexpr explicit Number(int64_t int_value) : value_(int_value) {}
  constexpr explicit Number(uint64_t uint_value) : value_(uint_value) {}
  CEL_ABSL_VISIT_CONSTEXPR double AsDouble() const {
    return absl::visit(internal::ConversionVisitor<double>(), value_);
  }
  CEL_ABSL_VISIT_CONSTEXPR int64_t AsInt() const {
    return absl::visit(internal::ConversionVisitor<int64_t>(), value_);
  }
  CEL_ABSL_VISIT_CONSTEXPR uint64_t AsUint() const {
    return absl::visit(internal::ConversionVisitor<uint64_t>(), value_);
  }
  CEL_ABSL_VISIT_CONSTEXPR bool LosslessConvertibleToInt() const {
    return absl::visit(internal::LosslessConvertibleToIntVisitor(), value_);
  }
  CEL_ABSL_VISIT_CONSTEXPR bool LosslessConvertibleToUint() const {
    return absl::visit(internal::LosslessConvertibleToUintVisitor(), value_);
  }
  CEL_ABSL_VISIT_CONSTEXPR bool operator<(Number other) const {
    return Compare(other) == internal::ComparisonResult::kLesser;
  }
  CEL_ABSL_VISIT_CONSTEXPR bool operator<=(Number other) const {
    internal::ComparisonResult cmp = Compare(other);
    return cmp != internal::ComparisonResult::kGreater &&
           cmp != internal::ComparisonResult::kNanInequal;
  }
  CEL_ABSL_VISIT_CONSTEXPR bool operator>(Number other) const {
    return Compare(other) == internal::ComparisonResult::kGreater;
  }
  CEL_ABSL_VISIT_CONSTEXPR bool operator>=(Number other) const {
    internal::ComparisonResult cmp = Compare(other);
    return cmp != internal::ComparisonResult::kLesser &&
           cmp != internal::ComparisonResult::kNanInequal;
  }
  CEL_ABSL_VISIT_CONSTEXPR bool operator==(Number other) const {
    return Compare(other) == internal::ComparisonResult::kEqual;
  }
  CEL_ABSL_VISIT_CONSTEXPR bool operator!=(Number other) const {
    return Compare(other) != internal::ComparisonResult::kEqual;
  }
  template <typename T, typename Op>
  T visit(Op&& op) const {
    return absl::visit(std::forward<Op>(op), value_);
  }
 private:
  internal::NumberVariant value_;
  CEL_ABSL_VISIT_CONSTEXPR internal::ComparisonResult Compare(
      Number other) const {
    return absl::visit(internal::CompareVisitor(other.value_), value_);
  }
};
}  
#endif  