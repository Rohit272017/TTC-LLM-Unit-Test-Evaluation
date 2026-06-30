#ifndef TENSORSTORE_UTIL_EXECUTION_SYNC_FLOW_SENDER_H_
#define TENSORSTORE_UTIL_EXECUTION_SYNC_FLOW_SENDER_H_
#include <utility>
#include "absl/synchronization/mutex.h"
#include "tensorstore/util/execution/execution.h"
namespace tensorstore {
template <typename Receiver>
struct SyncFlowReceiver {
  SyncFlowReceiver() = default;
  SyncFlowReceiver(Receiver receiver) : receiver(std::move(receiver)) {}
  SyncFlowReceiver(SyncFlowReceiver&& other)
      : receiver(std::move(other.receiver)) {}
  SyncFlowReceiver& operator=(SyncFlowReceiver&& other) {
    receiver = std::move(other.receiver);
    return *this;
  }
  template <typename CancelReceiver>
  friend void set_starting(SyncFlowReceiver& self, CancelReceiver cancel) {
    execution::set_starting(self.receiver, std::move(cancel));
  }
  template <typename... V>
  friend void set_value(SyncFlowReceiver& self, V... v) {
    absl::MutexLock lock(&self.mutex);
    execution::set_value(self.receiver, std::move(v)...);
  }
  friend void set_done(SyncFlowReceiver& self) {
    execution::set_done(self.receiver);
  }
  template <typename E>
  friend void set_error(SyncFlowReceiver& self, E e) {
    execution::set_error(self.receiver, std::move(e));
  }
  friend void set_stopping(SyncFlowReceiver& self) {
    execution::set_stopping(self.receiver);
  }
  Receiver receiver;
  absl::Mutex mutex;
};
template <typename Sender>
struct SyncFlowSender {
  Sender sender;
  template <typename Receiver>
  friend void submit(SyncFlowSender& self, Receiver receiver) {
    execution::submit(self.sender,
                      SyncFlowReceiver<Receiver>{std::move(receiver)});
  }
};
template <typename Sender>
SyncFlowSender<Sender> MakeSyncFlowSender(Sender sender) {
  return {std::move(sender)};
}
}  
#endif  