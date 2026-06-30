#ifndef TENSORSTORE_UTIL_EXECUTION_RESULT_SENDER_H_
#define TENSORSTORE_UTIL_EXECUTION_RESULT_SENDER_H_
#include <functional>
#include <type_traits>
#include <utility>
#include "absl/status/status.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/result.h"
namespace tensorstore {
namespace internal_result {
template <typename Receiver, typename = void, typename = void, typename = void,
          typename = void>
struct IsResultReceiver : public std::false_type {};
template <typename Receiver, typename T>
struct IsResultReceiver<
    Receiver, T,
    decltype(execution::set_value(std::declval<Receiver&>(),
                                  std::declval<T>())),
    decltype(execution::set_error(std::declval<Receiver&>(),
                                  std::declval<absl::Status>())),
    decltype(execution::set_cancel(std::declval<Receiver&>()))>
    : public std::true_type {};
}  
template <typename T, typename... V>
std::enable_if_t<((std::is_same_v<void, T> && sizeof...(V) == 0) ||
                  std::is_constructible_v<T, V&&...>)>
set_value(Result<T>& r, V&&... v) {
  r.emplace(std::forward<V>(v)...);
}
template <typename T, typename... V>
std::enable_if_t<((std::is_same_v<void, T> && sizeof...(V) == 0) ||
                  std::is_constructible_v<T, V&&...>)>
set_value(std::reference_wrapper<Result<T>> r, V&&... v) {
  set_value(r.get(), std::forward<V>(v)...);
}
template <typename T>
void set_error(Result<T>& r, absl::Status status) {
  r = std::move(status);
}
template <typename T>
void set_error(std::reference_wrapper<Result<T>> r, absl::Status status) {
  set_error(r.get(), std::move(status));
}
template <typename T>
void set_cancel(Result<T>& r) {
  r = absl::CancelledError("");
}
template <typename T>
void set_cancel(std::reference_wrapper<Result<T>> r) {
  set_cancel(r.get());
}
template <typename T, typename Receiver>
std::enable_if_t<internal_result::IsResultReceiver<Receiver, T>::value>  
submit(Result<T>& r, Receiver&& receiver) {
  if (r.has_value()) {
    execution::set_value(receiver, r.value());
  } else {
    auto status = r.status();
    if (status.code() == absl::StatusCode::kCancelled) {
      execution::set_cancel(receiver);
    } else {
      execution::set_error(receiver, std::move(status));
    }
  }
}
template <typename T, typename Receiver>
std::enable_if_t<internal_result::IsResultReceiver<Receiver, T>::value>  
submit(std::reference_wrapper<Result<T>> r, Receiver&& receiver) {
  submit(r.get(), std::forward<Receiver>(receiver));
}
}  
#endif  