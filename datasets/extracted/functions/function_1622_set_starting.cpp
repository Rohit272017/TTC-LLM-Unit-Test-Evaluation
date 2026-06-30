#ifndef TENSORSTORE_UTIL_EXECUTION_SENDER_H_
#define TENSORSTORE_UTIL_EXECUTION_SENDER_H_
#include <cstddef>
#include <tuple>
#include <utility>
#include "tensorstore/util/execution/execution.h"
namespace tensorstore {
class NullReceiver {
 public:
  template <typename CancelReceiver>
  friend void set_starting(NullReceiver&, CancelReceiver) {}
  template <typename... V>
  friend void set_value(NullReceiver&, V...) {}
  friend void set_done(NullReceiver&) {}
  template <typename E>
  friend void set_error(NullReceiver&, E e) {}
  friend void set_cancel(NullReceiver&) {}
  friend void set_stopping(NullReceiver&) {}
};
class NullSender {
  template <typename R>
  friend void submit(NullSender&, R&&) {}
};
struct CancelSender {
  template <typename Receiver>
  friend void submit(CancelSender, Receiver&& receiver) {
    execution::set_cancel(receiver);
  }
};
template <typename E>
struct ErrorSender {
  E error;
  template <typename Receiver>
  friend void submit(ErrorSender& sender, Receiver&& receiver) {
    execution::set_error(receiver, std::move(sender.error));
  }
};
template <typename E>
ErrorSender(E error) -> ErrorSender<E>;
template <typename... V>
struct ValueSender {
  ValueSender(V... v) : value(std::move(v)...) {}
  std::tuple<V...> value;
  template <typename Receiver>
  friend void submit(ValueSender& sender, Receiver&& receiver) {
    sender.SubmitHelper(std::forward<Receiver>(receiver),
                        std::make_index_sequence<sizeof...(V)>{});
  }
 private:
  template <typename Receiver, size_t... Is>
  void SubmitHelper(Receiver&& receiver, std::index_sequence<Is...>) {
    execution::set_value(receiver, std::move(std::get<Is>(value))...);
  }
};
template <typename... V>
ValueSender(V... v) -> ValueSender<V...>;
}  
#endif  