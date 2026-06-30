#ifndef AROLLA_QEXPR_LIFT_ACCUMULATOR_TO_SCALAR_OPERATOR_H_
#define AROLLA_QEXPR_LIFT_ACCUMULATOR_TO_SCALAR_OPERATOR_H_
#include <type_traits>
#include "arolla/memory/optional_value.h"
#include "arolla/qexpr/aggregation_ops_interface.h"
#include "arolla/qexpr/eval_context.h"
#include "arolla/qtype/array_like/array_like_qtype.h"
#include "arolla/util/meta.h"
namespace arolla {
template <typename Accumulator, typename ParentTypes, typename ChildTypes>
class ScalarToScalarGroupLifter;
template <typename Accumulator, typename... ParentTs, typename... ChildTs>
class ScalarToScalarGroupLifter<Accumulator, meta::type_list<ParentTs...>,
                                meta::type_list<ChildTs...>> {
 public:
  template <typename... Ts>
  std::conditional_t<Accumulator::IsAggregator(),
                     typename Accumulator::result_type,
                     wrap_with_optional_t<typename Accumulator::result_type>>
  operator()(EvaluationContext* ctx, const ParentTs&... p_args,
             const wrap_with_optional_t<ChildTs>&... c_args,
             const ScalarToScalarEdge&, const Ts&... init_args) const {
    auto accumulator_or_status = CreateAccumulator<Accumulator>(init_args...);
    if (!accumulator_or_status.ok()) {
      ctx->set_status(std::move(accumulator_or_status).status());
      return typename Accumulator::result_type();
    }
    Accumulator& accumulator = *accumulator_or_status;
    accumulator.Reset(p_args...);
    bool child_args_present =
        (is_present_or_not_required<ChildTs>(c_args) && ... && true);
    if (child_args_present) {
      accumulator.Add(value<ChildTs>(c_args)...);
    }
    if constexpr (Accumulator::IsFull()) {
      accumulator.FinalizeFullGroup();
    }
    ctx->set_status(accumulator.GetStatus());
    if (Accumulator::IsAggregator() || child_args_present) {
      return typename Accumulator::result_type(accumulator.GetResult());
    } else {
      return {};
    }
  }
 private:
  template <typename T>
  bool is_present_or_not_required(const wrap_with_optional_t<T>& arg) const {
    if constexpr (is_optional_v<T>) {
      return true;
    } else {
      return arg.present;
    }
  }
  template <typename T>
  T value(const wrap_with_optional_t<T>& arg) const {
    if constexpr (is_optional_v<T>) {
      return arg;
    } else {
      return arg.value;
    }
  }
};
}  
#endif  