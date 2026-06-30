#include "xla/tsl/distributed_runtime/preemption/preemption_notifier.h"
#include <atomic>
#include <csignal>
#include <functional>
#include <memory>
#include <utility>
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/statusor.h"
#if defined(PLATFORM_GOOGLE)
#include "thread/executor.h"
#include "thread/signal.h"
#endif
namespace tsl {
namespace {
constexpr absl::Duration kListenInterval = absl::Seconds(1);
constexpr absl::Time kUnsetDeathTime = absl::InfinitePast();
static std::atomic_bool sigterm_received(false);
class SigtermNotifier : public PreemptionNotifier {
 public:
  explicit SigtermNotifier(Env* env);
  ~SigtermNotifier() override {
    shutdown_notification_.Notify();
  }
 private:
  void StartListenerThread();
  absl::Notification shutdown_notification_;
  std::unique_ptr<Thread> preempt_listener_thread_;
};
SigtermNotifier::SigtermNotifier(Env* env) : PreemptionNotifier(env) {
  sigterm_received.store(false);
  StartListenerThread();
#if defined(PLATFORM_GOOGLE)
  thread::signal::Token unused_token;
  thread::signal::AddHandler(
      SIGTERM, thread::Executor::DefaultExecutor(),
      []() { sigterm_received.store(true); },
      0,  
      &unused_token);
#else
  std::signal(SIGTERM, [](int signal) { sigterm_received.store(true); });
#endif
}
void SigtermNotifier::StartListenerThread() {
  preempt_listener_thread_.reset(
      GetEnv()->StartThread({}, "PreemptionNotifier_Listen", [this]() {
        while (!sigterm_received.load()) {
          if (shutdown_notification_.WaitForNotificationWithTimeout(
                  kListenInterval)) {
            NotifyRegisteredListeners(
                errors::Cancelled("Preemption notifier is being deleted."));
            return;
          }
        }
        const absl::Time death_time = absl::Now();
        LOG(WARNING) << "SIGTERM caught at " << death_time;
        NotifyRegisteredListeners(death_time);
      }));
}
}  
absl::StatusOr<absl::Time> PreemptionNotifier::WillBePreemptedAt() {
  absl::Notification n;
  absl::StatusOr<absl::Time> result;
  WillBePreemptedAtAsync(
      [&n, &result](absl::StatusOr<absl::Time> async_result) {
        result = async_result;
        n.Notify();
      });
  n.WaitForNotification();
  return result;
}
void PreemptionNotifier::WillBePreemptedAtAsync(PreemptTimeCallback callback) {
  mutex_lock l(mu_);
  if (death_time_ == kUnsetDeathTime) {
    callbacks_.push_back(std::move(callback));
  } else {
    callback(death_time_);
  }
}
void PreemptionNotifier::NotifyRegisteredListeners(
    absl::StatusOr<absl::Time> death_time) {
  mutex_lock l(mu_);
  if (death_time.ok()) {
    death_time_ = death_time.value();
  }
  for (const auto& callback : callbacks_) {
    callback(death_time);
  }
  callbacks_.clear();
}
REGISTER_PREEMPTION_NOTIFIER(
    "sigterm", [](Env* env) -> std::unique_ptr<PreemptionNotifier> {
      return std::make_unique<SigtermNotifier>(env);
    });
}  