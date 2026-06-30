#ifndef AROLLA_QEXPR_LIFT_TO_OPTIONAL_OPERATOR_H_
#define AROLLA_QEXPR_LIFT_TO_OPTIONAL_OPERATOR_H_
#include <type_traits>
#include "arolla/memory/optional_value.h"
#include "arolla/qexpr/lifting.h"
#include "arolla/util/meta.h"
namespace arolla {
template <typename Op, typename ArgList>
class OptionalLiftedOperator;
template <typename Op, typename... Args>
class OptionalLiftedOperator<Op, meta::type_list<Args...>> {
  template <class T>
  using LiftedType =
      std::conditional_t<meta::is_wrapped_with_v<DoNotLiftTag, T>,
                         meta::strip_template_t<DoNotLiftTag, T>,
                         wrap_with_optional_t<T>>;
  template <class T>
  using LiftedTypeView = const T&;
 public:
  template <class... Ts>
  auto CreateOptionalOpWithCapturedScalars(const Ts&... args) const {
    using Tools = LiftingTools<Args...>;
    return WrapFnToAcceptOptionalArgs(
        Tools::template CreateFnWithDontLiftCaptured<LiftedTypeView>(Op(),
                                                                     args...));
  }
  auto operator()(const LiftedType<Args>&... args) const {
    using Tools = LiftingTools<Args...>;
    return Tools::CallOnLiftedArgs(CreateOptionalOpWithCapturedScalars(args...),
                                   args...);
  }
};
}  
#endif  