#include "tsl/platform/default/logging.h"
#include "absl/base/internal/cycleclock.h"
#include "absl/base/internal/sysinfo.h"
#include "absl/base/log_severity.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/env_time.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/mutex.h"
#if defined(PLATFORM_POSIX_ANDROID)
#include <android/log.h>
#include <iostream>
#include <sstream>
#endif
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <queue>
#include <unordered_map>
namespace tsl {
namespace internal {
namespace {
class TFLogSinks {
 public:
  static TFLogSinks& Instance();
  void Add(TFLogSink* sink);
  void Remove(TFLogSink* sink);
  std::vector<TFLogSink*> GetSinks() const;
  void Send(const TFLogEntry& entry);
 private:
  TFLogSinks();
  void SendToSink(TFLogSink& sink, const TFLogEntry& entry);
  std::queue<TFLogEntry> log_entry_queue_;
  static const size_t kMaxLogEntryQueueSize = 128;
  mutable tsl::mutex mutex_;
  std::vector<TFLogSink*> sinks_;
};
TFLogSinks::TFLogSinks() {
#ifndef NO_DEFAULT_LOGGER
  static TFDefaultLogSink* default_sink = new TFDefaultLogSink();
  sinks_.emplace_back(default_sink);
#endif
}
TFLogSinks& TFLogSinks::Instance() {
  static TFLogSinks* instance = new TFLogSinks();
  return *instance;
}
void TFLogSinks::Add(TFLogSink* sink) {
  assert(sink != nullptr && "The sink must not be a nullptr");
  tsl::mutex_lock lock(mutex_);
  sinks_.emplace_back(sink);
  if (sinks_.size() == 1) {
    while (!log_entry_queue_.empty()) {
      for (const auto& sink : sinks_) {
        SendToSink(*sink, log_entry_queue_.front());
      }
      log_entry_queue_.pop();
    }
  }
}
void TFLogSinks::Remove(TFLogSink* sink) {
  assert(sink != nullptr && "The sink must not be a nullptr");
  tsl::mutex_lock lock(mutex_);
  auto it = std::find(sinks_.begin(), sinks_.end(), sink);
  if (it != sinks_.end()) sinks_.erase(it);
}
std::vector<TFLogSink*> TFLogSinks::GetSinks() const {
  tsl::mutex_lock lock(mutex_);
  return sinks_;
}
void TFLogSinks::Send(const TFLogEntry& entry) {
  tsl::mutex_lock lock(mutex_);
  if (sinks_.empty()) {
    while (log_entry_queue_.size() >= kMaxLogEntryQueueSize) {
      log_entry_queue_.pop();
    }
    log_entry_queue_.push(entry);
    return;
  }
  while (!log_entry_queue_.empty()) {
    for (const auto& sink : sinks_) {
      SendToSink(*sink, log_entry_queue_.front());
    }
    log_entry_queue_.pop();
  }
  for (const auto& sink : sinks_) {
    SendToSink(*sink, entry);
  }
}
void TFLogSinks::SendToSink(TFLogSink& sink, const TFLogEntry& entry) {
  sink.Send(entry);
  sink.WaitTillSent();
}
class VlogFileMgr {
 public:
  VlogFileMgr();
  ~VlogFileMgr();
  FILE* FilePtr() const;
 private:
  FILE* vlog_file_ptr;
  char* vlog_file_name;
};
VlogFileMgr::VlogFileMgr() {
  vlog_file_name = getenv("TF_CPP_VLOG_FILENAME");
  vlog_file_ptr =
      vlog_file_name == nullptr ? nullptr : fopen(vlog_file_name, "w");
  if (vlog_file_ptr == nullptr) {
    vlog_file_ptr = stderr;
  }
}
VlogFileMgr::~VlogFileMgr() {
  if (vlog_file_ptr != stderr) {
    fclose(vlog_file_ptr);
  }
}
FILE* VlogFileMgr::FilePtr() const { return vlog_file_ptr; }
int ParseInteger(const char* str, size_t size) {
  string integer_str(str, size);
  std::istringstream ss(integer_str);
  int level = 0;
  ss >> level;
  return level;
}
int64_t LogLevelStrToInt(const char* tf_env_var_val) {
  if (tf_env_var_val == nullptr) {
    return 0;
  }
  return ParseInteger(tf_env_var_val, strlen(tf_env_var_val));
}
struct StringData {
  struct Hasher {
    size_t operator()(const StringData& sdata) const {
      size_t hash = 5381;
      const char* data = sdata.data;
      for (const char* top = data + sdata.size; data < top; ++data) {
        hash = ((hash << 5) + hash) + (*data);
      }
      return hash;
    }
  };
  StringData() = default;
  StringData(const char* data, size_t size) : data(data), size(size) {}
  bool operator==(const StringData& rhs) const {
    return size == rhs.size && memcmp(data, rhs.data, size) == 0;
  }
  const char* data = nullptr;
  size_t size = 0;
};
using VmoduleMap = std::unordered_map<StringData, int, StringData::Hasher>;
VmoduleMap* VmodulesMapFromEnv() {
  const char* env = getenv("TF_CPP_VMODULE");
  if (env == nullptr) {
    return nullptr;
  }
  const char* env_data = strdup(env);
  VmoduleMap* result = new VmoduleMap();
  while (true) {
    const char* eq = strchr(env_data, '=');
    if (eq == nullptr) {
      break;
    }
    const char* after_eq = eq + 1;
    const char* comma = strchr(after_eq, ',');
    const char* new_env_data;
    if (comma == nullptr) {
      comma = strchr(after_eq, '\0');
      new_env_data = comma;
    } else {
      new_env_data = comma + 1;
    }
    (*result)[StringData(env_data, eq - env_data)] =
        ParseInteger(after_eq, comma - after_eq);
    env_data = new_env_data;
  }
  return result;
}
bool EmitThreadIdFromEnv() {
  const char* tf_env_var_val = getenv("TF_CPP_LOG_THREAD_ID");
  return tf_env_var_val == nullptr
             ? false
             : ParseInteger(tf_env_var_val, strlen(tf_env_var_val)) != 0;
}
}  
absl::LogSeverityAtLeast MinLogLevelFromEnv() {
#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
  return absl::LogSeverityAtLeast::kInfinity;
#else
  const char* tf_env_var_val = getenv("TF_CPP_MIN_LOG_LEVEL");
  return static_cast<absl::LogSeverityAtLeast>(
      LogLevelStrToInt(tf_env_var_val));
#endif
}
int MaxVLogLevelFromEnv() {
#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
  return 0;
#else
  const char* tf_env_var_val = getenv("TF_CPP_MAX_VLOG_LEVEL");
  return LogLevelStrToInt(tf_env_var_val);
#endif
}
LogMessage::LogMessage(const char* fname, int line, absl::LogSeverity severity)
    : fname_(fname), line_(line), severity_(severity) {}
LogMessage& LogMessage::AtLocation(const char* fname, int line) {
  fname_ = fname;
  line_ = line;
  return *this;
}
LogMessage::~LogMessage() {
  static absl::LogSeverityAtLeast min_log_level = MinLogLevelFromEnv();
  if (severity_ >= min_log_level) {
    GenerateLogMessage();
  }
}
void LogMessage::GenerateLogMessage() {
  TFLogSinks::Instance().Send(TFLogEntry(severity_, fname_, line_, str()));
}
int LogMessage::MaxVLogLevel() {
  static int max_vlog_level = MaxVLogLevelFromEnv();
  return max_vlog_level;
}
bool LogMessage::VmoduleActivated(const char* fname, int level) {
  if (level <= MaxVLogLevel()) {
    return true;
  }
  static VmoduleMap* vmodules = VmodulesMapFromEnv();
  if (TF_PREDICT_TRUE(vmodules == nullptr)) {
    return false;
  }
  const char* last_slash = strrchr(fname, '/');
  const char* module_start = last_slash == nullptr ? fname : last_slash + 1;
  const char* dot_after = strchr(module_start, '.');
  const char* module_limit =
      dot_after == nullptr ? strchr(fname, '\0') : dot_after;
  StringData module(module_start, module_limit - module_start);
  auto it = vmodules->find(module);
  return it != vmodules->end() && it->second >= level;
}
LogMessageFatal::LogMessageFatal(const char* file, int line)
    : LogMessage(file, line, absl::LogSeverity::kFatal) {}
LogMessageFatal::~LogMessageFatal() {
  GenerateLogMessage();
  abort();
}
void LogString(const char* fname, int line, absl::LogSeverity severity,
               const string& message) {
  LogMessage(fname, line, severity) << message;
}
template <>
void MakeCheckOpValueString(std::ostream* os, const char& v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "char value " << static_cast<int16>(v);
  }
}
template <>
void MakeCheckOpValueString(std::ostream* os, const signed char& v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "signed char value " << static_cast<int16>(v);
  }
}
template <>
void MakeCheckOpValueString(std::ostream* os, const unsigned char& v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "unsigned char value " << static_cast<uint16>(v);
  }
}
#if LANG_CXX11
template <>
void MakeCheckOpValueString(std::ostream* os, const std::nullptr_t& v) {
  (*os) << "nullptr";
}
#endif
CheckOpMessageBuilder::CheckOpMessageBuilder(const char* exprtext)
    : stream_(new std::ostringstream) {
  *stream_ << "Check failed: " << exprtext << " (";
}
CheckOpMessageBuilder::~CheckOpMessageBuilder() { delete stream_; }
std::ostream* CheckOpMessageBuilder::ForVar2() {
  *stream_ << " vs. ";
  return stream_;
}
string* CheckOpMessageBuilder::NewString() {
  *stream_ << ")";
  return new string(stream_->str());
}
namespace {
uint32 LossyIncrement(std::atomic<uint32>* counter) {
  const uint32 value = counter->load(std::memory_order_relaxed);
  counter->store(value + 1, std::memory_order_relaxed);
  return value;
}
}  
bool LogEveryNState::ShouldLog(int n) {
  return n != 0 && (LossyIncrement(&counter_) % n) == 0;
}
bool LogFirstNState::ShouldLog(int n) {
  const int counter_value =
      static_cast<int>(counter_.load(std::memory_order_relaxed));
  if (counter_value < n) {
    counter_.store(counter_value + 1, std::memory_order_relaxed);
    return true;
  }
  return false;
}
bool LogEveryPow2State::ShouldLog(int ignored) {
  const uint32 new_value = LossyIncrement(&counter_) + 1;
  return (new_value & (new_value - 1)) == 0;
}
bool LogEveryNSecState::ShouldLog(double seconds) {
  LossyIncrement(&counter_);
  const int64_t now_cycles = absl::base_internal::CycleClock::Now();
  int64_t next_cycles = next_log_time_cycles_.load(std::memory_order_relaxed);
  do {
    if (now_cycles <= next_cycles) return false;
  } while (!next_log_time_cycles_.compare_exchange_weak(
      next_cycles,
      now_cycles + seconds * absl::base_internal::CycleClock::Frequency(),
      std::memory_order_relaxed, std::memory_order_relaxed));
  return true;
}
}  
void TFAddLogSink(TFLogSink* sink) {
  internal::TFLogSinks::Instance().Add(sink);
}
void TFRemoveLogSink(TFLogSink* sink) {
  internal::TFLogSinks::Instance().Remove(sink);
}
std::vector<TFLogSink*> TFGetLogSinks() {
  return internal::TFLogSinks::Instance().GetSinks();
}
void TFDefaultLogSink::Send(const TFLogEntry& entry) {
#ifdef PLATFORM_POSIX_ANDROID
  int android_log_level;
  switch (entry.log_severity()) {
    case absl::LogSeverity::kInfo:
      android_log_level = ANDROID_LOG_INFO;
      break;
    case absl::LogSeverity::kWarning:
      android_log_level = ANDROID_LOG_WARN;
      break;
    case absl::LogSeverity::kError:
      android_log_level = ANDROID_LOG_ERROR;
      break;
    case absl::LogSeverity::kFatal:
      android_log_level = ANDROID_LOG_FATAL;
      break;
    default:
      if (entry.log_severity() < absl::LogSeverity::kInfo) {
        android_log_level = ANDROID_LOG_VERBOSE;
      } else {
        android_log_level = ANDROID_LOG_ERROR;
      }
      break;
  }
  std::stringstream ss;
  const auto& fname = entry.FName();
  auto pos = fname.find("/");
  ss << (pos != std::string::npos ? fname.substr(pos + 1) : fname) << ":"
     << entry.Line() << " " << entry.ToString();
  __android_log_write(android_log_level, "native", ss.str().c_str());
  fprintf(stderr, "native : %s\n", ss.str().c_str());
  if (entry.log_severity() == absl::LogSeverity::kFatal) {
    abort();
  }
#else   
  static const internal::VlogFileMgr vlog_file;
  static bool log_thread_id = internal::EmitThreadIdFromEnv();
  uint64 now_micros = EnvTime::NowMicros();
  time_t now_seconds = static_cast<time_t>(now_micros / 1000000);
  int32_t micros_remainder = static_cast<int32>(now_micros % 1000000);
  const size_t time_buffer_size = 30;
  char time_buffer[time_buffer_size];
  strftime(time_buffer, time_buffer_size, "%Y-%m-%d %H:%M:%S",
           localtime(&now_seconds));
  const size_t tid_buffer_size = 10;
  char tid_buffer[tid_buffer_size] = "";
  if (log_thread_id) {
    snprintf(tid_buffer, sizeof(tid_buffer), " %7u",
             absl::base_internal::GetTID());
  }
  char sev;
  switch (entry.log_severity()) {
    case absl::LogSeverity::kInfo:
      sev = 'I';
      break;
    case absl::LogSeverity::kWarning:
      sev = 'W';
      break;
    case absl::LogSeverity::kError:
      sev = 'E';
      break;
    case absl::LogSeverity::kFatal:
      sev = 'F';
      break;
    default:
      assert(false && "Unknown logging severity");
      sev = '?';
      break;
  }
  fprintf(vlog_file.FilePtr(), "%s.%06d: %c%s %s:%d] %s\n", time_buffer,
          micros_remainder, sev, tid_buffer, entry.FName().c_str(),
          entry.Line(), entry.ToString().c_str());
  fflush(vlog_file.FilePtr());  
#endif  
}
void UpdateLogVerbosityIfDefined(const char* env_var) {}
}  