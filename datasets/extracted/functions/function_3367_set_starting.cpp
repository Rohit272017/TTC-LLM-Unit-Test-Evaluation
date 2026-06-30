#ifndef TENSORSTORE_UTIL_EXECUTION_COLLECTING_SENDER_H_
#define TENSORSTORE_UTIL_EXECUTION_COLLECTING_SENDER_H_
#include <utility>
#include "tensorstore/util/execution/execution.h"
namespace tensorstore {
namespace internal {
template <typename Container, typename SingleReceiver>
struct CollectingReceiver {
  SingleReceiver receiver;
  Container container;
  template <typename CancelReceiver>
  friend void set_starting(CollectingReceiver& self, CancelReceiver cancel) {
  }
  template <typename... V>
  friend void set_value(CollectingReceiver& self, V... v) {
    self.container.emplace_back(std::move(v)...);
  }
  template <typename E>
  friend void set_error(CollectingReceiver& self, E e) {
    execution::set_error(self.receiver, std::move(e));
  }
  friend void set_done(CollectingReceiver& self) {
    execution::set_value(self.receiver, std::move(self.container));
  }
  friend void set_stopping(CollectingReceiver& self) {}
};
template <typename Container, typename Sender>
struct CollectingSender {
  Sender sender;
  template <typename Receiver>
  friend void submit(CollectingSender& self, Receiver receiver) {
    execution::submit(self.sender, CollectingReceiver<Container, Receiver>{
                                       std::move(receiver)});
  }
};
template <typename Container, typename Sender>
CollectingSender<Container, Sender> MakeCollectingSender(Sender sender) {
  return {std::move(sender)};
}
}  
}  
#endif  