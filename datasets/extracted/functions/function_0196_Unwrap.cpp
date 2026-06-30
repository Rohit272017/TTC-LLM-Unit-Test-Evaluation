#ifndef TENSORSTORE_INTERNAL_VOID_WRAPPER_H_
#define TENSORSTORE_INTERNAL_VOID_WRAPPER_H_
#include <type_traits>
#include <utility>
#include "absl/status/status.h"
namespace tensorstore {
namespace internal {
struct Void {
  explicit operator bool() const { return true; }
  static void Unwrap(Void) {}
  template <typename T>
  static T Unwrap(T x) {
    return x;
  }
  template <typename T>
  using WrappedType = std::conditional_t<std::is_void_v<T>, Void, T>;
  template <typename T>
  using UnwrappedType = std::conditional_t<std::is_same_v<T, Void>, void, T>;
  template <typename Func, typename... Args>
  static std::enable_if_t<std::is_void_v<std::invoke_result_t<Func, Args...>>,
                          Void>
  CallAndWrap(Func&& func, Args&&... args) {
    std::forward<Func>(func)(std::forward<Args>(args)...);
    return {};
  }
  template <typename Func, typename... Args>
  static std::enable_if_t<!std::is_void_v<std::invoke_result_t<Func, Args...>>,
                          std::invoke_result_t<Func, Args...>>
  CallAndWrap(Func&& func, Args&&... args) {
    return std::forward<Func>(func)(std::forward<Args>(args)...);
  }
};
}  
}  
#endif  