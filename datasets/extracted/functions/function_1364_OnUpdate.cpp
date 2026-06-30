#include "tensorstore/internal/log/verbose_flag.h"
#include <stddef.h>
#include <atomic>
#include <cassert>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_log.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/env.h"
ABSL_FLAG(std::string, tensorstore_verbose_logging, {},
          "comma-separated list of tensorstore verbose logging flags")
    .OnUpdate([]() {
      if (!absl::GetFlag(FLAGS_tensorstore_verbose_logging).empty()) {
        tensorstore::internal_log::UpdateVerboseLogging(
            absl::GetFlag(FLAGS_tensorstore_verbose_logging), true);
      }
    });
namespace tensorstore {
namespace internal_log {
namespace {
ABSL_CONST_INIT absl::Mutex g_mutex(absl::kConstInit);
ABSL_CONST_INIT VerboseFlag* g_list_head ABSL_GUARDED_BY(g_mutex) = nullptr;
struct LoggingLevelConfig {
  int default_level = -1;
  absl::flat_hash_map<std::string, int> levels;
};
void UpdateLoggingLevelConfig(LoggingLevelConfig& config,
                              std::string_view input) {
  auto& levels = config.levels;
  for (std::string_view flag : absl::StrSplit(input, ',', absl::SkipEmpty())) {
    const size_t eq = flag.rfind('=');
    if (eq == flag.npos) {
      levels.insert_or_assign(std::string(flag), 0);
      continue;
    }
    if (eq == 0) continue;
    int level;
    if (!absl::SimpleAtoi(flag.substr(eq + 1), &level)) continue;
    if (level < -1) {
      level = -1;
    } else if (level > 1000) {
      level = 1000;
    }
    levels.insert_or_assign(std::string(flag.substr(0, eq)), level);
  }
  config.default_level = -1;
  if (auto it = levels.find("all"); it != levels.end()) {
    config.default_level = it->second;
  }
}
int GetLevelForVerboseFlag(const LoggingLevelConfig& config,
                           std::string_view name) {
  while (!name.empty()) {
    auto it = config.levels.find(name);
    if (it != config.levels.end()) {
      return it->second;
    }
    auto pos = name.rfind('.');
    if (pos == name.npos) {
      break;
    }
    name = name.substr(0, pos);
  }
  return config.default_level;
}
LoggingLevelConfig& GetLoggingLevelConfig()
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(g_mutex) {
  static absl::NoDestructor<LoggingLevelConfig> flags{[] {
    LoggingLevelConfig config;
    if (auto env = internal::GetEnv("TENSORSTORE_VERBOSE_LOGGING"); env) {
      UpdateLoggingLevelConfig(config, *env);
    }
    return config;
  }()};
  return *flags;
}
}  
void UpdateVerboseLogging(std::string_view input, bool overwrite)
    ABSL_LOCKS_EXCLUDED(g_mutex) {
  ABSL_LOG(INFO) << "--tensorstore_verbose_logging=" << input;
  LoggingLevelConfig config;
  UpdateLoggingLevelConfig(config, input);
  absl::MutexLock lock(&g_mutex);
  VerboseFlag* slist = g_list_head;
  LoggingLevelConfig& global_config = GetLoggingLevelConfig();
  std::swap(global_config.levels, config.levels);
  std::swap(global_config.default_level, config.default_level);
  if (!overwrite) {
    if (global_config.levels.count("all")) {
      global_config.default_level = config.default_level;
    }
    global_config.levels.merge(config.levels);
  }
  int vlevel = GetLevelForVerboseFlag(global_config, "verbose_logging");
  while (slist != nullptr) {
    int value = GetLevelForVerboseFlag(global_config, slist->name_);
    ABSL_LOG_IF(INFO, vlevel >= 1) << slist->name_ << "=" << value;
    slist->value_.store(value, std::memory_order_seq_cst);
    slist = slist->next_;
  }
}
int VerboseFlag::RegisterVerboseFlag(VerboseFlag* flag) {
  absl::MutexLock lock(&g_mutex);
  int old_v = flag->value_.load(std::memory_order_relaxed);
  if (old_v == kValueUninitialized) {
    const auto& config = GetLoggingLevelConfig();
    old_v = GetLevelForVerboseFlag(config, flag->name_);
    flag->value_.store(old_v, std::memory_order_relaxed);
    flag->next_ = std::exchange(g_list_head, flag);
  }
  return old_v;
}
bool VerboseFlag::VerboseFlagSlowPath(VerboseFlag* flag, int old_v, int level) {
  if (ABSL_PREDICT_TRUE(old_v != kValueUninitialized)) {
    return old_v >= level;
  }
  old_v = RegisterVerboseFlag(flag);
  return ABSL_PREDICT_FALSE(old_v >= level);
}
static_assert(std::is_trivially_destructible<VerboseFlag>::value,
              "VerboseFlag must be trivially destructible");
}  
}  