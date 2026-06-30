#ifndef AROLLA_OPERATORS_CORE_CAST_OPERATOR_H_
#define AROLLA_OPERATORS_CORE_CAST_OPERATOR_H_
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "arolla/memory/optional_value.h"
#include "arolla/util/meta.h"
#include "arolla/util/repr.h"
namespace arolla {
template <typename DST>
struct CastOp {
  using run_on_missing = std::true_type;
  using DstTypes =
      meta::type_list<bool, int32_t, int64_t, uint64_t, float, double>;
  using SrcTypes =
      meta::type_list<bool, int32_t, int64_t, uint64_t, float, double>;
  static_assert(meta::contains_v<DstTypes, DST>);
  template <typename SRC>
  static constexpr SRC max_float_to_int_safe_value() {
    using dst_limits = std::numeric_limits<DST>;
    using src_limits = std::numeric_limits<SRC>;
    static_assert(dst_limits::is_integer);
    static_assert(std::is_floating_point_v<SRC>);
    SRC result = 0;
    int i = 0;
    for (; i < src_limits::digits; ++i) {
      result *= 2;
      result += 1;
    }
    for (; i < dst_limits::digits; ++i) {
      result *= 2;
    }
    for (; i > dst_limits::digits; --i) {
      result /= 2;
    }
    return result;
  }
  template <typename SRC>
  static constexpr SRC min_float_to_int_safe_value() {
    using dst_limits = std::numeric_limits<DST>;
    using src_limits = std::numeric_limits<SRC>;
    static_assert(dst_limits::is_integer);
    static_assert(std::is_floating_point_v<SRC>);
    if constexpr (!dst_limits::is_signed) {
      return 0.0;
    } else {
      SRC result = 1;
      int i = 0;
      for (; i < src_limits::digits; ++i) {
        result *= 2;
      }
      for (; i < dst_limits::digits; ++i) {
        result *= 2;
      }
      for (; i > dst_limits::digits; --i) {
        result += 1;
        result /= 2;
      }
      return -result;
    }
  }
  template <typename SRC>
  static constexpr auto safe_range() {
    static_assert(meta::contains_v<SrcTypes, SRC>);
    using dst_limits = std::numeric_limits<DST>;
    using src_limits = std::numeric_limits<SRC>;
    if constexpr (std::is_same_v<SRC, DST>) {
      return std::make_tuple();
    } else if constexpr (std::is_integral_v<DST> && std::is_integral_v<SRC>) {
      constexpr SRC safe_min =
          std::max<int64_t>(dst_limits::min(), src_limits::min());
      constexpr SRC safe_max =
          std::min<uint64_t>(dst_limits::max(), src_limits::max());
      if constexpr (safe_min <= src_limits::min() &&
                    safe_max >= src_limits::max()) {
        return std::make_tuple();
      } else {
        return std::tuple<SRC, SRC>(safe_min, safe_max);
      }
    } else if constexpr (std::is_integral_v<DST> &&
                         std::is_floating_point_v<SRC>) {
      return std::tuple<SRC, SRC>(min_float_to_int_safe_value<SRC>(),
                                  max_float_to_int_safe_value<SRC>());
    } else if constexpr (std::is_floating_point_v<DST> &&
                         std::is_floating_point_v<SRC>) {
      constexpr bool ub_check =
          (src_limits::max() <= dst_limits::max() ||
           static_cast<DST>(src_limits::max()) == dst_limits::max() ||
           static_cast<DST>(src_limits::max()) == dst_limits::infinity());
      static_assert(ub_check);
      return std::make_tuple();
    } else {
      return std::make_tuple();
    }
  }
  template <typename SRC>
  auto operator()(SRC src) const {
    constexpr auto src_range = safe_range<SRC>();
    if constexpr (std::tuple_size_v<decltype(src_range)> == 0) {
      return static_cast<DST>(src);
    } else {
      using ReturnType = absl::StatusOr<DST>;
      const auto& [range_min, range_max] = src_range;
      if (range_min <= src && src <= range_max) {
        return ReturnType(static_cast<DST>(src));
      } else {
        return ReturnType(absl::InvalidArgumentError(absl::StrCat(
            "cannot cast ", ::arolla::Repr(src), " to ",
            std::is_unsigned_v<DST> ? "u" : "", "int", 8 * sizeof(DST))));
      }
    }
  }
};
struct ToBoolOp {
  using run_on_missing = std::true_type;
  template <typename T>
  bool operator()(const T& x) const {
    return x != 0;
  }
};
struct ToOptionalOp {
  using run_on_missing = std::true_type;
  template <typename T>
  OptionalValue<T> operator()(const T& x) const {
    return OptionalValue<T>(x);
  }
};
struct GetOptionalValueOp {
  template <typename T>
  absl::StatusOr<T> operator()(const OptionalValue<T>& x) const {
    if (!x.present) {
      return absl::FailedPreconditionError(
          "core.get_optional_value expects present value, got missing");
    }
    return x.value;
  }
};
}  
#endif  