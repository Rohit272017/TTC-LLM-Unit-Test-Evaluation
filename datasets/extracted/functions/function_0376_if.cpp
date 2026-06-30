#ifndef AROLLA_QEXPR_OPERATORS_BOOL_LOGIC_H_
#define AROLLA_QEXPR_OPERATORS_BOOL_LOGIC_H_
#include <type_traits>
#include "arolla/dense_array/dense_array.h"
#include "arolla/dense_array/ops/dense_ops.h"
#include "arolla/dense_array/qtype/types.h"
#include "arolla/memory/optional_value.h"
#include "arolla/qexpr/eval_context.h"
namespace arolla {
struct LogicalAndOp {
  using run_on_missing = std::true_type;
  bool operator()(bool lhs, bool rhs) const { return lhs && rhs; }
  OptionalValue<bool> operator()(const OptionalValue<bool>& lhs,
                                 const OptionalValue<bool>& rhs) const {
    if (lhs.present) {
      return !lhs.value ? false : rhs;
    } else if (rhs.present) {
      return !rhs.value ? false : lhs;
    } else {
      return OptionalValue<bool>{};
    }
  }
};
struct LogicalOrOp {
  using run_on_missing = std::true_type;
  bool operator()(bool lhs, bool rhs) const { return lhs || rhs; }
  OptionalValue<bool> operator()(const OptionalValue<bool>& lhs,
                                 const OptionalValue<bool>& rhs) const {
    if (lhs.present) {
      return lhs.value ? true : rhs;
    } else if (rhs.present) {
      return rhs.value ? true : lhs;
    } else {
      return OptionalValue<bool>{};
    }
  }
};
struct LogicalNotOp {
  using run_on_missing = std::true_type;
  bool operator()(bool arg) const { return !arg; }
};
struct LogicalIfOp {
  using run_on_missing = std::true_type;
  template <typename T, typename = std::enable_if_t<is_scalar_type_v<T>>>
  const T& operator()(const OptionalValue<bool>& condition, const T& true_value,
                      const T& false_value, const T& missing_value) const {
    if (condition.present) {
      return condition.value ? true_value : false_value;
    } else {
      return missing_value;
    }
  }
  template <typename T>
  OptionalValue<T> operator()(const OptionalValue<bool>& condition,
                              const OptionalValue<T>& true_value,
                              const OptionalValue<T>& false_value,
                              const OptionalValue<T>& missing_value) const {
    if (condition.present) {
      return condition.value ? true_value : false_value;
    } else {
      return missing_value;
    }
  }
  template <typename T>
  DenseArray<T> operator()(EvaluationContext* ctx,
                           const DenseArray<bool>& condition,
                           const OptionalValue<T>& true_value,
                           const OptionalValue<T>& false_value,
                           const OptionalValue<T>& missing_value) const {
    auto fn = [&true_value, &false_value, &missing_value](
                  OptionalValue<bool> condition) -> OptionalValue<T> {
      return LogicalIfOp()(condition, true_value, false_value, missing_value);
    };
    auto op = CreateDenseOp<DenseOpFlags::kRunOnMissing, decltype(fn), T>(
        fn, &ctx->buffer_factory());
    return op(condition);
  }
  template <typename TrueFn, typename FalseFn, typename MissingFn,
            std::enable_if_t<std::is_invocable_v<TrueFn> ||
                                 std::is_invocable_v<FalseFn> ||
                                 std::is_invocable_v<MissingFn>,
                             bool> = true>
  auto operator()(const OptionalValue<bool>& condition,
                  const TrueFn& true_value, const FalseFn& false_value,
                  const MissingFn& missing_value) const {
    auto unwrap = [](const auto& fn) {
      if constexpr (std::is_invocable_v<decltype(fn)>) {
        return fn();
      } else {
        return fn;
      }
    };
    return condition.present
               ? (condition.value ? unwrap(true_value) : unwrap(false_value))
               : unwrap(missing_value);
  }
};
}  
#endif  