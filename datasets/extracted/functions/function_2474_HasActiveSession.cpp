#include "tsl/profiler/lib/profiler_lock.h"
#include <atomic>
#include "absl/status/statusor.h"
#include "xla/tsl/util/env_var.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/macros.h"
namespace tsl {
namespace profiler {
namespace {
std::atomic<int> g_session_active = ATOMIC_VAR_INIT(0);
static_assert(ATOMIC_INT_LOCK_FREE == 2, "Assumed atomic<int> was lock free");
}  
 bool ProfilerLock::HasActiveSession() {
  return g_session_active.load(std::memory_order_relaxed) != 0;
}
 absl::StatusOr<ProfilerLock> ProfilerLock::Acquire() {
  static bool tf_profiler_disabled = [] {
    bool disabled = false;
    ReadBoolFromEnvVar("TF_DISABLE_PROFILING", false, &disabled).IgnoreError();
    return disabled;
  }();
  if (TF_PREDICT_FALSE(tf_profiler_disabled)) {
    return errors::AlreadyExists(
        "TensorFlow Profiler is permanently disabled by env var "
        "TF_DISABLE_PROFILING.");
  }
  int already_active = g_session_active.exchange(1, std::memory_order_acq_rel);
  if (already_active) {
    return errors::AlreadyExists(kProfilerLockContention);
  }
  return ProfilerLock(true);
}
void ProfilerLock::ReleaseIfActive() {
  if (active_) {
    g_session_active.store(0, std::memory_order_release);
    active_ = false;
  }
}
}  
}  