#ifndef TENSORSTORE_UTIL_EXECUTION_FUTURE_SENDER_H_
#define TENSORSTORE_UTIL_EXECUTION_FUTURE_SENDER_H_
#include <functional>
#include <type_traits>
#include <utility>
#include "absl/status/status.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/future.h"
namespace tensorstore {
namespace internal_future {
template <typename Receiver, typename = void, typename = void, typename = void,
          typename = void>
struct IsFutureReceiver : public std::false_type {};
template <typename Receiver, typename T>
struct IsFutureReceiver<
    Receiver, T,
    decltype(execution::set_value(std::declval<Receiver&>(),
                                  std::declval<T>())),
    decltype(execution::set_error(std::declval<Receiver&>(),
                                  std::declval<absl::Status>())),
    decltype(execution::set_cancel(std::declval<Receiver&>()))>
    : public std::true_type {};
}  
template <typename T, typename... V>
std::enable_if_t<(!std::is_const_v<T> &&
                  std::is_constructible_v<typename Promise<T>::result_type,
                                          std::in_place_t, V...>)>
set_value(const Promise<T>& promise, V&&... v) {
  promise.SetResult(std::in_place, std::forward<V>(v)...);
}
template <typename T, typename... V>
std::enable_if_t<(!std::is_const_v<T> &&
                  std::is_constructible_v<typename Promise<T>::result_type,
                                          std::in_place_t, V...>)>
set_value(std::reference_wrapper<const Promise<T>> promise, V&&... v) {
  set_value(promise.get(), std::forward<V>(v)...);
}
template <typename T>
void set_error(const Promise<T>& promise, absl::Status error) {
  promise.SetResult(std::move(error));
}
template <typename T>
void set_error(std::reference_wrapper<const Promise<T>> promise,
               absl::Status error) {
  set_error(promise.get(), std::move(error));
}
template <typename T>
void set_cancel(const Promise<T>& promise) {
  promise.SetResult(absl::CancelledError(""));
}
template <typename T>
void set_cancel(std::reference_wrapper<const Promise<T>> promise) {
  set_cancel(promise.get());
}
template <typename T, typename Receiver>
std::enable_if_t<internal_future::IsFutureReceiver<Receiver, T>::value>  
submit(Future<T>& f, Receiver receiver) {
  f.Force();
  f.ExecuteWhenReady([r = std::move(receiver)](ReadyFuture<T> ready) mutable {
    auto& result = ready.result();
    if (result.has_value()) {
      execution::set_value(r, result.value());
    } else {
      auto status = ready.status();
      if (status.code() == absl::StatusCode::kCancelled) {
        execution::set_cancel(r);
      } else {
        execution::set_error(r, std::move(status));
      }
    }
  });
}
template <typename T, typename Receiver>
std::enable_if_t<internal_future::IsFutureReceiver<Receiver, T>::value>  
submit(std::reference_wrapper<Future<T>> f, Receiver&& receiver) {
  submit(f.get(), std::forward<Receiver>(receiver));
}
template <typename T, typename Sender>
Future<T> MakeSenderFuture(Sender sender) {
  auto pair = PromiseFuturePair<T>::Make();
  struct Callback {
    Sender sender;
    void operator()(Promise<T> promise) { execution::submit(sender, promise); }
  };
  pair.promise.ExecuteWhenForced(Callback{std::move(sender)});
  return pair.future;
}
}  
#endif  