#ifndef TENSORSTORE_UTIL_EXECUTION_FUTURE_COLLECTING_RECEIVER_H_
#define TENSORSTORE_UTIL_EXECUTION_FUTURE_COLLECTING_RECEIVER_H_
#include <utility>
#include "absl/status/status.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/future.h"
namespace tensorstore {
template <typename Container>
struct FutureCollectingReceiver {
  Promise<Container> promise;
  Container container;
  FutureCallbackRegistration cancel_registration;
  template <typename... V>
  void set_value(V&&... v) {
    container.emplace_back(std::forward<V>(v)...);
  }
  void set_error(absl::Status status) { promise.SetResult(std::move(status)); }
  void set_done() { promise.SetResult(std::move(container)); }
  template <typename Cancel>
  void set_starting(Cancel cancel) {
    cancel_registration = promise.ExecuteWhenNotNeeded(std::move(cancel));
  }
  void set_stopping() { cancel_registration.Unregister(); }
};
template <typename Container, typename Sender>
Future<Container> CollectFlowSenderIntoFuture(Sender sender) {
  auto [promise, future] = PromiseFuturePair<Container>::Make();
  execution::submit(std::move(sender),
                    FutureCollectingReceiver<Container>{std::move(promise)});
  return std::move(future);
}
}  
#endif  