#ifndef TENSORSTORE_UTIL_EXECUTOR_H_
#define TENSORSTORE_UTIL_EXECUTOR_H_
#include <functional>
#include <type_traits>
#include <utility>
#include "absl/base/attributes.h"
#include "absl/functional/any_invocable.h"
#include "absl/meta/type_traits.h"
#include "tensorstore/internal/poly/poly.h"
#include "tensorstore/internal/type_traits.h"
namespace tensorstore {
using ExecutorTask = absl::AnyInvocable<void() &&>;
using Executor = poly::Poly<0, true, void(ExecutorTask) const>;
class InlineExecutor {
 public:
  template <typename Func>
  void operator()(Func&& func) const {
    std::forward<Func>(func)();
  }
};
template <typename ExecutorType, typename FunctionType>
class ExecutorBoundFunction {
 public:
  using Executor = ExecutorType;
  using Function = FunctionType;
  template <typename... T>
  std::enable_if_t<std::is_invocable_v<Function&, T...>>  
  operator()(T&&... arg) {
    executor(std::bind(std::move(function), std::forward<T>(arg)...));
  }
  template <typename... T>
  std::enable_if_t<std::is_invocable_v<const Function&, T...>> operator()(
      T&&... arg) const {
    executor(std::bind(function, std::forward<T>(arg)...));
  }
  ABSL_ATTRIBUTE_NO_UNIQUE_ADDRESS Executor executor;
  ABSL_ATTRIBUTE_NO_UNIQUE_ADDRESS Function function;
};
template <typename Executor, typename Function>
std::enable_if_t<
    !std::is_same_v<absl::remove_cvref_t<Executor>, InlineExecutor>,
    ExecutorBoundFunction<absl::remove_cvref_t<Executor>,
                          absl::remove_cvref_t<Function>>>
WithExecutor(Executor&& executor, Function&& function) {
  return {std::forward<Executor>(executor), std::forward<Function>(function)};
}
template <typename Executor, typename Function>
std::enable_if_t<std::is_same_v<absl::remove_cvref_t<Executor>, InlineExecutor>,
                 Function&&>
WithExecutor(Executor&& executor, Function&& function) {
  return std::forward<Function>(function);
}
}  
#endif  