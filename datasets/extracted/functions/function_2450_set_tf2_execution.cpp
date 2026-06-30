#include "tensorflow/core/platform/enable_tf2_utils.h"
#include <atomic>
#include "tensorflow/core/util/env_var.h"
namespace tensorflow {
enum Enablement : uint8 { kFalse = 0, kTrue = 1, undefined = 2 };
static std::atomic<Enablement> tf2_enabled{undefined};
void set_tf2_execution(bool enabled) {
  tf2_enabled = (enabled) ? Enablement::kTrue : Enablement::kFalse;
}
bool tf2_execution_enabled() {
  if (tf2_enabled == Enablement::undefined) {
    static bool tf2_behavior_env_enabled = [] {
      string tf2_env;
      TF_CHECK_OK(ReadStringFromEnvVar("TF2_BEHAVIOR", "0", &tf2_env));
      return tf2_env != "0";
    }();
    tf2_enabled =
        (tf2_behavior_env_enabled) ? Enablement::kTrue : Enablement::kFalse;
  }
  return tf2_enabled;
}
}  