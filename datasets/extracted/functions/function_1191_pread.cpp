#define _GNU_SOURCE 1  
#include "glog/logging.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iterator>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include "config.h"
#include "glog/platform.h"
#include "glog/raw_logging.h"
#include "stacktrace.h"
#include "utilities.h"
#ifdef GLOG_OS_WINDOWS
#  include "windows/dirent.h"
#else
#  include <dirent.h>  
#endif
#include <fcntl.h>
#include <sys/stat.h>
#include <cctype>  
#include <cerrno>  
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <regex>
#include <sstream>
#include <vector>
#ifdef HAVE__CHSIZE_S
#  include <io.h>  
#endif
#ifdef HAVE_PWD_H
#  include <pwd.h>
#endif
#ifdef HAVE_SYS_UTSNAME_H
#  include <sys/utsname.h>  
#endif
#ifdef HAVE_SYSLOG_H
#  include <syslog.h>
#endif
#ifdef HAVE_SYS_TYPES_H
#  include <sys/types.h>
#endif
#ifdef HAVE_UNISTD_H
#  include <unistd.h>
#endif
#ifndef HAVE_MODE_T
typedef int mode_t;
#endif
using std::dec;
using std::hex;
using std::min;
using std::ostream;
using std::ostringstream;
using std::setfill;
using std::setw;
using std::string;
using std::vector;
using std::fclose;
using std::fflush;
using std::FILE;
using std::fprintf;
using std::fwrite;
using std::perror;
#ifdef __QNX__
using std::fdopen;
#endif
#define EXCLUSIVE_LOCKS_REQUIRED(mu)
enum { PATH_SEPARATOR = '/' };
#ifndef HAVE_PREAD
static ssize_t pread(int fd, void* buf, size_t count, off_t offset) {
  off_t orig_offset = lseek(fd, 0, SEEK_CUR);
  if (orig_offset == (off_t)-1) return -1;
  if (lseek(fd, offset, SEEK_CUR) == (off_t)-1) return -1;
  ssize_t len = read(fd, buf, count);
  if (len < 0) return len;
  if (lseek(fd, orig_offset, SEEK_SET) == (off_t)-1) return -1;
  return len;
}
#endif  
#ifndef HAVE_PWRITE
static ssize_t pwrite(int fd, void* buf, size_t count, off_t offset) {
  off_t orig_offset = lseek(fd, 0, SEEK_CUR);
  if (orig_offset == (off_t)-1) return -1;
  if (lseek(fd, offset, SEEK_CUR) == (off_t)-1) return -1;
  ssize_t len = write(fd, buf, count);
  if (len < 0) return len;
  if (lseek(fd, orig_offset, SEEK_SET) == (off_t)-1) return -1;
  return len;
}
#endif  
static void GetHostName(string* hostname) {
#if defined(HAVE_SYS_UTSNAME_H)
  struct utsname buf;
  if (uname(&buf) < 0) {
    *buf.nodename = '\0';
  }
  *hostname = buf.nodename;
#elif defined(GLOG_OS_WINDOWS)
  char buf[MAX_COMPUTERNAME_LENGTH + 1];
  DWORD len = MAX_COMPUTERNAME_LENGTH + 1;
  if (GetComputerNameA(buf, &len)) {
    *hostname = buf;
  } else {
    hostname->clear();
  }
#else
#  warning There is no way to retrieve the host name.
  *hostname = "(unknown)";
#endif
}
static bool TerminalSupportsColor() {
  bool term_supports_color = false;
#ifdef GLOG_OS_WINDOWS
  term_supports_color = true;
#else
  const char* const term = getenv("TERM");
  if (term != nullptr && term[0] != '\0') {
    term_supports_color =
        !strcmp(term, "xterm") || !strcmp(term, "xterm-color") ||
        !strcmp(term, "xterm-256color") || !strcmp(term, "screen-256color") ||
        !strcmp(term, "konsole") || !strcmp(term, "konsole-16color") ||
        !strcmp(term, "konsole-256color") || !strcmp(term, "screen") ||
        !strcmp(term, "linux") || !strcmp(term, "cygwin");
  }
#endif
  return term_supports_color;
}
#if defined(__cpp_lib_unreachable) && (__cpp_lib_unreachable >= 202202L)
#  define GLOG_UNREACHABLE std::unreachable()
#elif !defined(NDEBUG)
#  define GLOG_UNREACHABLE assert(false)
#else
#  if defined(_MSC_VER)
#    define GLOG_UNREACHABLE __assume(false)
#  elif defined(__has_builtin)
#    if __has_builtin(unreachable)
#      define GLOG_UNREACHABLE __builtin_unreachable()
#    endif
#  endif
#  if !defined(GLOG_UNREACHABLE) && defined(__GNUG__)
#    define GLOG_UNREACHABLE __builtin_unreachable()
#  endif
#  if !defined(GLOG_UNREACHABLE)
#    define GLOG_UNREACHABLE
#  endif
#endif
namespace google {
GLOG_NO_EXPORT
std::string StrError(int err);
enum GLogColor { COLOR_DEFAULT, COLOR_RED, COLOR_GREEN, COLOR_YELLOW };
static GLogColor SeverityToColor(LogSeverity severity) {
  switch (severity) {
    case GLOG_INFO:
      return COLOR_DEFAULT;
    case GLOG_WARNING:
      return COLOR_YELLOW;
    case GLOG_ERROR:
    case GLOG_FATAL:
      return COLOR_RED;
  }
  GLOG_UNREACHABLE;
}
#ifdef GLOG_OS_WINDOWS
static WORD GetColorAttribute(GLogColor color) {
  switch (color) {
    case COLOR_RED:
      return FOREGROUND_RED;
    case COLOR_GREEN:
      return FOREGROUND_GREEN;
    case COLOR_YELLOW:
      return FOREGROUND_RED | FOREGROUND_GREEN;
    case COLOR_DEFAULT:
      break;
  }
  return 0;
}
#else
static const char* GetAnsiColorCode(GLogColor color) {
  switch (color) {
    case COLOR_RED:
      return "1";
    case COLOR_GREEN:
      return "2";
    case COLOR_YELLOW:
      return "3";
    case COLOR_DEFAULT:
      return "";
  };
  return nullptr;  
}
#endif  
static uint32 MaxLogSize() {
  return (FLAGS_max_log_size > 0 && FLAGS_max_log_size < 4096
              ? FLAGS_max_log_size
              : 1);
}
const size_t LogMessage::kMaxLogMessageLen = 30000;
namespace logging {
namespace internal {
struct LogMessageData {
  LogMessageData();
  int preserved_errno_;  
  char message_text_[LogMessage::kMaxLogMessageLen + 1];
  LogMessage::LogStream stream_;
  LogSeverity severity_;  
  int line_;              
  void (LogMessage::*send_method_)();  
  union {  
    LogSink* sink_;  
    std::vector<std::string>*
        outvec_;            
    std::string* message_;  
  };
  size_t num_prefix_chars_;     
  size_t num_chars_to_log_;     
  size_t num_chars_to_syslog_;  
  const char* basename_;        
  const char* fullname_;        
  bool has_been_flushed_;       
  bool first_fatal_;            
  std::thread::id thread_id_;
  LogMessageData(const LogMessageData&) = delete;
  LogMessageData& operator=(const LogMessageData&) = delete;
};
}  
}  
static std::mutex log_mutex;
int64 LogMessage::num_messages_[NUM_SEVERITIES] = {0, 0, 0, 0};
static bool stop_writing = false;
const char* const LogSeverityNames[] = {"INFO", "WARNING", "ERROR", "FATAL"};
static bool exit_on_dfatal = true;
const char* GetLogSeverityName(LogSeverity severity) {
  return LogSeverityNames[severity];
}
static bool SendEmailInternal(const char* dest, const char* subject,
                              const char* body, bool use_logging);
base::Logger::~Logger() = default;
namespace {
constexpr std::intmax_t kSecondsInDay = 60 * 60 * 24;
constexpr std::intmax_t kSecondsInWeek = kSecondsInDay * 7;
class PrefixFormatter {
 public:
  PrefixFormatter(PrefixFormatterCallback callback, void* data) noexcept
      : version{V2}, callback_v2{callback}, data{data} {}
  void operator()(std::ostream& s, const LogMessage& message) const {
    switch (version) {
      case V2:
        callback_v2(s, message, data);
        break;
    }
  }
  PrefixFormatter(const PrefixFormatter& other) = delete;
  PrefixFormatter& operator=(const PrefixFormatter& other) = delete;
 private:
  enum Version { V2 } version;
  union {
    PrefixFormatterCallback callback_v2;
  };
  void* data;
};
std::unique_ptr<PrefixFormatter> g_prefix_formatter;
class LogFileObject : public base::Logger {
 public:
  LogFileObject(LogSeverity severity, const char* base_filename);
  ~LogFileObject() override;
  void Write(bool force_flush,  
             const std::chrono::system_clock::time_point&
                 timestamp,  
             const char* message, size_t message_len) override;
  void SetBasename(const char* basename);
  void SetExtension(const char* ext);
  void SetSymlinkBasename(const char* symlink_basename);
  void Flush() override;
  uint32 LogSize() override {
    std::lock_guard<std::mutex> l{mutex_};
    return file_length_;
  }
  void FlushUnlocked(const std::chrono::system_clock::time_point& now);
 private:
  static const uint32 kRolloverAttemptFrequency = 0x20;
  std::mutex mutex_;
  bool base_filename_selected_;
  string base_filename_;
  string symlink_basename_;
  string filename_extension_;  
  std::unique_ptr<FILE> file_;
  LogSeverity severity_;
  uint32 bytes_since_flush_{0};
  uint32 dropped_mem_length_{0};
  uint32 file_length_{0};
  unsigned int rollover_attempt_;
  std::chrono::system_clock::time_point
      next_flush_time_;  
  std::chrono::system_clock::time_point start_time_;
  bool CreateLogfile(const string& time_pid_string);
};
class LogCleaner {
 public:
  LogCleaner();
  void Enable(const std::chrono::minutes& overdue);
  void Disable();
  void Run(const std::chrono::system_clock::time_point& current_time,
           bool base_filename_selected, const string& base_filename,
           const string& filename_extension);
  bool enabled() const { return enabled_; }
 private:
  vector<string> GetOverdueLogNames(
      string log_directory,
      const std::chrono::system_clock::time_point& current_time,
      const string& base_filename, const string& filename_extension) const;
  bool IsLogFromCurrentProject(const string& filepath,
                               const string& base_filename,
                               const string& filename_extension) const;
  bool IsLogLastModifiedOver(
      const string& filepath,
      const std::chrono::system_clock::time_point& current_time) const;
  bool enabled_{false};
  std::chrono::minutes overdue_{
      std::chrono::duration<int, std::ratio<kSecondsInWeek>>{1}};
  std::chrono::system_clock::time_point
      next_cleanup_time_;  
};
LogCleaner log_cleaner;
}  
class LogDestination {
 public:
  friend class LogMessage;
  friend void ReprintFatalMessage();
  friend base::Logger* base::GetLogger(LogSeverity);
  friend void base::SetLogger(LogSeverity, base::Logger*);
  static void SetLogDestination(LogSeverity severity,
                                const char* base_filename);
  static void SetLogSymlink(LogSeverity severity, const char* symlink_basename);
  static void AddLogSink(LogSink* destination);
  static void RemoveLogSink(LogSink* destination);
  static void SetLogFilenameExtension(const char* filename_extension);
  static void SetStderrLogging(LogSeverity min_severity);
  static void SetEmailLogging(LogSeverity min_severity, const char* addresses);
  static void LogToStderr();
  static void FlushLogFiles(int min_severity);
  static void FlushLogFilesUnsafe(int min_severity);
  static const int kNetworkBytes = 1400;
  static const string& hostname();
  static const bool& terminal_supports_color() {
    return terminal_supports_color_;
  }
  static void DeleteLogDestinations();
  LogDestination(LogSeverity severity, const char* base_filename);
 private:
#if defined(__cpp_lib_shared_mutex) && (__cpp_lib_shared_mutex >= 201505L)
  using SinkMutex = std::shared_mutex;
  using SinkLock = std::lock_guard<SinkMutex>;
#else  
  using SinkMutex = std::shared_timed_mutex;
  using SinkLock = std::unique_lock<SinkMutex>;
#endif  
  friend std::default_delete<LogDestination>;
  ~LogDestination();
  static void MaybeLogToStderr(LogSeverity severity, const char* message,
                               size_t message_len, size_t prefix_len);
  static void MaybeLogToEmail(LogSeverity severity, const char* message,
                              size_t len);
  static void MaybeLogToLogfile(
      LogSeverity severity,
      const std::chrono::system_clock::time_point& timestamp,
      const char* message, size_t len);
  static void LogToAllLogfiles(
      LogSeverity severity,
      const std::chrono::system_clock::time_point& timestamp,
      const char* message, size_t len);
  static void LogToSinks(LogSeverity severity, const char* full_filename,
                         const char* base_filename, int line,
                         const LogMessageTime& time, const char* message,
                         size_t message_len);
  static void WaitForSinks(logging::internal::LogMessageData* data);
  static LogDestination* log_destination(LogSeverity severity);
  base::Logger* GetLoggerImpl() const { return logger_; }
  void SetLoggerImpl(base::Logger* logger);
  void ResetLoggerImpl() { SetLoggerImpl(&fileobject_); }
  LogFileObject fileobject_;
  base::Logger* logger_;  
  static std::unique_ptr<LogDestination> log_destinations_[NUM_SEVERITIES];
  static std::underlying_type_t<LogSeverity> email_logging_severity_;
  static string addresses_;
  static string hostname_;
  static bool terminal_supports_color_;
  static std::unique_ptr<vector<LogSink*>> sinks_;
  static SinkMutex sink_mutex_;
  LogDestination(const LogDestination&) = delete;
  LogDestination& operator=(const LogDestination&) = delete;
};
std::underlying_type_t<LogSeverity> LogDestination::email_logging_severity_ =
    99999;
string LogDestination::addresses_;
string LogDestination::hostname_;
std::unique_ptr<vector<LogSink*>> LogDestination::sinks_;
LogDestination::SinkMutex LogDestination::sink_mutex_;
bool LogDestination::terminal_supports_color_ = TerminalSupportsColor();
const string& LogDestination::hostname() {
  if (hostname_.empty()) {
    GetHostName(&hostname_);
    if (hostname_.empty()) {
      hostname_ = "(unknown)";
    }
  }
  return hostname_;
}
LogDestination::LogDestination(LogSeverity severity, const char* base_filename)
    : fileobject_(severity, base_filename), logger_(&fileobject_) {}
LogDestination::~LogDestination() { ResetLoggerImpl(); }
void LogDestination::SetLoggerImpl(base::Logger* logger) {
  if (logger_ == logger) {
    return;
  }
  if (logger_ && logger_ != &fileobject_) {
    delete logger_;
  }
  logger_ = logger;
}
inline void LogDestination::FlushLogFilesUnsafe(int min_severity) {
  std::for_each(std::next(std::begin(log_destinations_), min_severity),
                std::end(log_destinations_),
                [now = std::chrono::system_clock::now()](
                    std::unique_ptr<LogDestination>& log) {
                  if (log != nullptr) {
                    log->fileobject_.FlushUnlocked(now);
                  }
                });
}
inline void LogDestination::FlushLogFiles(int min_severity) {
  std::lock_guard<std::mutex> l{log_mutex};
  for (int i = min_severity; i < NUM_SEVERITIES; i++) {
    LogDestination* log = log_destination(static_cast<LogSeverity>(i));
    if (log != nullptr) {
      log->logger_->Flush();
    }
  }
}
inline void LogDestination::SetLogDestination(LogSeverity severity,
                                              const char* base_filename) {
  std::lock_guard<std::mutex> l{log_mutex};
  log_destination(severity)->fileobject_.SetBasename(base_filename);
}
inline void LogDestination::SetLogSymlink(LogSeverity severity,
                                          const char* symlink_basename) {
  CHECK_GE(severity, 0);
  CHECK_LT(severity, NUM_SEVERITIES);
  std::lock_guard<std::mutex> l{log_mutex};
  log_destination(severity)->fileobject_.SetSymlinkBasename(symlink_basename);
}
inline void LogDestination::AddLogSink(LogSink* destination) {
  SinkLock l{sink_mutex_};
  if (sinks_ == nullptr) sinks_ = std::make_unique<std::vector<LogSink*>>();
  sinks_->push_back(destination);
}
inline void LogDestination::RemoveLogSink(LogSink* destination) {
  SinkLock l{sink_mutex_};
  if (sinks_) {
    sinks_->erase(std::remove(sinks_->begin(), sinks_->end(), destination),
                  sinks_->end());
  }
}
inline void LogDestination::SetLogFilenameExtension(const char* ext) {
  std::lock_guard<std::mutex> l{log_mutex};
  for (int severity = 0; severity < NUM_SEVERITIES; ++severity) {
    log_destination(static_cast<LogSeverity>(severity))
        ->fileobject_.SetExtension(ext);
  }
}
inline void LogDestination::SetStderrLogging(LogSeverity min_severity) {
  std::lock_guard<std::mutex> l{log_mutex};
  FLAGS_stderrthreshold = min_severity;
}
inline void LogDestination::LogToStderr() {
  SetStderrLogging(GLOG_INFO);  
  for (int i = 0; i < NUM_SEVERITIES; ++i) {
    SetLogDestination(static_cast<LogSeverity>(i),
                      "");  
  }
}
inline void LogDestination::SetEmailLogging(LogSeverity min_severity,
                                            const char* addresses) {
  std::lock_guard<std::mutex> l{log_mutex};
  LogDestination::email_logging_severity_ = min_severity;
  LogDestination::addresses_ = addresses;
}
static void ColoredWriteToStderrOrStdout(FILE* output, LogSeverity severity,
                                         const char* message, size_t len) {
  bool is_stdout = (output == stdout);
  const GLogColor color = (LogDestination::terminal_supports_color() &&
                           ((!is_stdout && FLAGS_colorlogtostderr) ||
                            (is_stdout && FLAGS_colorlogtostdout)))
                              ? SeverityToColor(severity)
                              : COLOR_DEFAULT;
  if (COLOR_DEFAULT == color) {
    fwrite(message, len, 1, output);
    return;
  }
#ifdef GLOG_OS_WINDOWS
  const HANDLE output_handle =
      GetStdHandle(is_stdout ? STD_OUTPUT_HANDLE : STD_ERROR_HANDLE);
  CONSOLE_SCREEN_BUFFER_INFO buffer_info;
  GetConsoleScreenBufferInfo(output_handle, &buffer_info);
  const WORD old_color_attrs = buffer_info.wAttributes;
  fflush(output);
  SetConsoleTextAttribute(output_handle,
                          GetColorAttribute(color) | FOREGROUND_INTENSITY);
  fwrite(message, len, 1, output);
  fflush(output);
  SetConsoleTextAttribute(output_handle, old_color_attrs);
#else
  fprintf(output, "\033[0;3%sm", GetAnsiColorCode(color));
  fwrite(message, len, 1, output);
  fprintf(output, "\033[m");  
#endif  
}
static void ColoredWriteToStdout(LogSeverity severity, const char* message,
                                 size_t len) {
  FILE* output = stdout;
  if (severity >= FLAGS_stderrthreshold) {
    output = stderr;
  }
  ColoredWriteToStderrOrStdout(output, severity, message, len);
}
static void ColoredWriteToStderr(LogSeverity severity, const char* message,
                                 size_t len) {
  ColoredWriteToStderrOrStdout(stderr, severity, message, len);
}
static void WriteToStderr(const char* message, size_t len) {
  fwrite(message, len, 1, stderr);
}
inline void LogDestination::MaybeLogToStderr(LogSeverity severity,
                                             const char* message,
                                             size_t message_len,
                                             size_t prefix_len) {
  if ((severity >= FLAGS_stderrthreshold) || FLAGS_alsologtostderr) {
    ColoredWriteToStderr(severity, message, message_len);
    AlsoErrorWrite(severity,
                   glog_internal_namespace_::ProgramInvocationShortName(),
                   message + prefix_len);
  }
}
inline void LogDestination::MaybeLogToEmail(LogSeverity severity,
                                            const char* message, size_t len) {
  if (severity >= email_logging_severity_ || severity >= FLAGS_logemaillevel) {
    string to(FLAGS_alsologtoemail);
    if (!addresses_.empty()) {
      if (!to.empty()) {
        to += ",";
      }
      to += addresses_;
    }
    const string subject(
        string("[LOG] ") + LogSeverityNames[severity] + ": " +
        glog_internal_namespace_::ProgramInvocationShortName());
    string body(hostname());
    body += "\n\n";
    body.append(message, len);
    SendEmailInternal(to.c_str(), subject.c_str(), body.c_str(), false);
  }
}
inline void LogDestination::MaybeLogToLogfile(
    LogSeverity severity,
    const std::chrono::system_clock::time_point& timestamp, const char* message,
    size_t len) {
  const bool should_flush = severity > FLAGS_logbuflevel;
  LogDestination* destination = log_destination(severity);
  destination->logger_->Write(should_flush, timestamp, message, len);
}
inline void LogDestination::LogToAllLogfiles(
    LogSeverity severity,
    const std::chrono::system_clock::time_point& timestamp, const char* message,
    size_t len) {
  if (FLAGS_logtostdout) {  
    ColoredWriteToStdout(severity, message, len);
  } else if (FLAGS_logtostderr) {  
    ColoredWriteToStderr(severity, message, len);
  } else {
    for (int i = severity; i >= 0; --i) {
      LogDestination::MaybeLogToLogfile(static_cast<LogSeverity>(i), timestamp,
                                        message, len);
    }
  }
}
inline void LogDestination::LogToSinks(LogSeverity severity,
                                       const char* full_filename,
                                       const char* base_filename, int line,
                                       const LogMessageTime& time,
                                       const char* message,
                                       size_t message_len) {
  std::shared_lock<SinkMutex> l{sink_mutex_};
  if (sinks_) {
    for (size_t i = sinks_->size(); i-- > 0;) {
      (*sinks_)[i]->send(severity, full_filename, base_filename, line, time,
                         message, message_len);
    }
  }
}
inline void LogDestination::WaitForSinks(
    logging::internal::LogMessageData* data) {
  std::shared_lock<SinkMutex> l{sink_mutex_};
  if (sinks_) {
    for (size_t i = sinks_->size(); i-- > 0;) {
      (*sinks_)[i]->WaitTillSent();
    }
  }
  const bool send_to_sink =
      (data->send_method_ == &LogMessage::SendToSink) ||
      (data->send_method_ == &LogMessage::SendToSinkAndLog);
  if (send_to_sink && data->sink_ != nullptr) {
    data->sink_->WaitTillSent();
  }
}
std::unique_ptr<LogDestination>
    LogDestination::log_destinations_[NUM_SEVERITIES];
inline LogDestination* LogDestination::log_destination(LogSeverity severity) {
  if (log_destinations_[severity] == nullptr) {
    log_destinations_[severity] =
        std::make_unique<LogDestination>(severity, nullptr);
  }
  return log_destinations_[severity].get();
}
void LogDestination::DeleteLogDestinations() {
  for (auto& log_destination : log_destinations_) {
    log_destination.reset();
  }
  SinkLock l{sink_mutex_};
  sinks_.reset();
}
namespace {
std::string g_application_fingerprint;
}  
void SetApplicationFingerprint(const std::string& fingerprint) {
  g_application_fingerprint = fingerprint;
}
namespace {
#ifdef GLOG_OS_WINDOWS
const char possible_dir_delim[] = {'\\', '/'};
#else
const char possible_dir_delim[] = {'/'};
#endif
string PrettyDuration(const std::chrono::duration<int>& secs) {
  std::stringstream result;
  int mins = secs.count() / 60;
  int hours = mins / 60;
  mins = mins % 60;
  int s = secs.count() % 60;
  result.fill('0');
  result << hours << ':' << setw(2) << mins << ':' << setw(2) << s;
  return result.str();
}
LogFileObject::LogFileObject(LogSeverity severity, const char* base_filename)
    : base_filename_selected_(base_filename != nullptr),
      base_filename_((base_filename != nullptr) ? base_filename : ""),
      symlink_basename_(glog_internal_namespace_::ProgramInvocationShortName()),
      filename_extension_(),
      severity_(severity),
      rollover_attempt_(kRolloverAttemptFrequency - 1),
      start_time_(std::chrono::system_clock::now()) {}
LogFileObject::~LogFileObject() {
  std::lock_guard<std::mutex> l{mutex_};
  file_ = nullptr;
}
void LogFileObject::SetBasename(const char* basename) {
  std::lock_guard<std::mutex> l{mutex_};
  base_filename_selected_ = true;
  if (base_filename_ != basename) {
    if (file_ != nullptr) {
      file_ = nullptr;
      rollover_attempt_ = kRolloverAttemptFrequency - 1;
    }
    base_filename_ = basename;
  }
}
void LogFileObject::SetExtension(const char* ext) {
  std::lock_guard<std::mutex> l{mutex_};
  if (filename_extension_ != ext) {
    if (file_ != nullptr) {
      file_ = nullptr;
      rollover_attempt_ = kRolloverAttemptFrequency - 1;
    }
    filename_extension_ = ext;
  }
}
void LogFileObject::SetSymlinkBasename(const char* symlink_basename) {
  std::lock_guard<std::mutex> l{mutex_};
  symlink_basename_ = symlink_basename;
}
void LogFileObject::Flush() {
  std::lock_guard<std::mutex> l{mutex_};
  FlushUnlocked(std::chrono::system_clock::now());
}
void LogFileObject::FlushUnlocked(
    const std::chrono::system_clock::time_point& now) {
  if (file_ != nullptr) {
    fflush(file_.get());
    bytes_since_flush_ = 0;
  }
  next_flush_time_ =
      now + std::chrono::duration_cast<std::chrono::system_clock::duration>(
                std::chrono::duration<int32>{FLAGS_logbufsecs});
}
bool LogFileObject::CreateLogfile(const string& time_pid_string) {
  string string_filename = base_filename_;
  if (FLAGS_timestamp_in_logfile_name) {
    string_filename += time_pid_string;
  }
  string_filename += filename_extension_;
  const char* filename = string_filename.c_str();
  int flags = O_WRONLY | O_CREAT;
  if (FLAGS_timestamp_in_logfile_name) {
    flags = flags | O_EXCL;
  }
  FileDescriptor fd{
      open(filename, flags, static_cast<mode_t>(FLAGS_logfile_mode))};
  if (!fd) return false;
#ifdef HAVE_FCNTL
  fcntl(fd.get(), F_SETFD, FD_CLOEXEC);
  static struct flock w_lock;
  w_lock.l_type = F_WRLCK;
  w_lock.l_start = 0;
  w_lock.l_whence = SEEK_SET;
  w_lock.l_len = 0;
  int wlock_ret = fcntl(fd.get(), F_SETLK, &w_lock);
  if (wlock_ret == -1) {
    return false;
  }
#endif
  file_.reset(fdopen(fd.release(), "a"));  
  if (file_ == nullptr) {                  
    if (FLAGS_timestamp_in_logfile_name) {
      unlink(filename);  
    }
    return false;
  }
#ifdef GLOG_OS_WINDOWS
  if (!FLAGS_timestamp_in_logfile_name) {
    if (fseek(file_.get(), 0, SEEK_END) != 0) {
      return false;
    }
  }
#endif
  if (!symlink_basename_.empty()) {
    const char* slash = strrchr(filename, PATH_SEPARATOR);
    const string linkname =
        symlink_basename_ + '.' + LogSeverityNames[severity_];
    string linkpath;
    if (slash)
      linkpath = string(
          filename, static_cast<size_t>(slash - filename + 1));  
    linkpath += linkname;
    unlink(linkpath.c_str());  
#if defined(GLOG_OS_WINDOWS)
#elif defined(HAVE_UNISTD_H)
    const char* linkdest = slash ? (slash + 1) : filename;
    if (symlink(linkdest, linkpath.c_str()) != 0) {
    }
    if (!FLAGS_log_link.empty()) {
      linkpath = FLAGS_log_link + "/" + linkname;
      unlink(linkpath.c_str());  
      if (symlink(filename, linkpath.c_str()) != 0) {
      }
    }
#endif
  }
  return true;  
}
void LogFileObject::Write(
    bool force_flush, const std::chrono::system_clock::time_point& timestamp,
    const char* message, size_t message_len) {
  std::lock_guard<std::mutex> l{mutex_};
  if (base_filename_selected_ && base_filename_.empty()) {
    return;
  }
  auto cleanupLogs = [this, current_time = timestamp] {
    if (log_cleaner.enabled()) {
      log_cleaner.Run(current_time, base_filename_selected_, base_filename_,
                      filename_extension_);
    }
  };
  ScopedExit<decltype(cleanupLogs)> cleanupAtEnd{cleanupLogs};
  if (file_length_ >> 20U >= MaxLogSize() || PidHasChanged()) {
    file_ = nullptr;
    file_length_ = bytes_since_flush_ = dropped_mem_length_ = 0;
    rollover_attempt_ = kRolloverAttemptFrequency - 1;
  }
  if (file_ == nullptr) {
    if (++rollover_attempt_ != kRolloverAttemptFrequency) return;
    rollover_attempt_ = 0;
    struct ::tm tm_time;
    std::time_t t = std::chrono::system_clock::to_time_t(timestamp);
    if (FLAGS_log_utc_time) {
      gmtime_r(&t, &tm_time);
    } else {
      localtime_r(&t, &tm_time);
    }
    ostringstream time_pid_stream;
    time_pid_stream.fill('0');
    time_pid_stream << 1900 + tm_time.tm_year << setw(2) << 1 + tm_time.tm_mon
                    << setw(2) << tm_time.tm_mday << '-' << setw(2)
                    << tm_time.tm_hour << setw(2) << tm_time.tm_min << setw(2)
                    << tm_time.tm_sec << '.' << GetMainThreadPid();
    const string& time_pid_string = time_pid_stream.str();
    if (base_filename_selected_) {
      if (!CreateLogfile(time_pid_string)) {
        perror("Could not create log file");
        fprintf(stderr, "COULD NOT CREATE LOGFILE '%s'!\n",
                time_pid_string.c_str());
        return;
      }
    } else {
      string stripped_filename(
          glog_internal_namespace_::ProgramInvocationShortName());
      string hostname;
      GetHostName(&hostname);
      string uidname = MyUserName();
      if (uidname.empty()) uidname = "invalid-user";
      stripped_filename = stripped_filename + '.' + hostname + '.' + uidname +
                          ".log." + LogSeverityNames[severity_] + '.';
      const vector<string>& log_dirs = GetLoggingDirectories();
      bool success = false;
      for (const auto& log_dir : log_dirs) {
        base_filename_ = log_dir + "/" + stripped_filename;
        if (CreateLogfile(time_pid_string)) {
          success = true;
          break;
        }
      }
      if (success == false) {
        perror("Could not create logging file");
        fprintf(stderr, "COULD NOT CREATE A LOGGINGFILE %s!",
                time_pid_string.c_str());
        return;
      }
    }
    if (FLAGS_log_file_header) {
      ostringstream file_header_stream;
      file_header_stream.fill('0');
      file_header_stream << "Log file created at: " << 1900 + tm_time.tm_year
                         << '/' << setw(2) << 1 + tm_time.tm_mon << '/'
                         << setw(2) << tm_time.tm_mday << ' ' << setw(2)
                         << tm_time.tm_hour << ':' << setw(2) << tm_time.tm_min
                         << ':' << setw(2) << tm_time.tm_sec
                         << (FLAGS_log_utc_time ? " UTC\n" : "\n")
                         << "Running on machine: " << LogDestination::hostname()
                         << '\n';
      if (!g_application_fingerprint.empty()) {
        file_header_stream << "Application fingerprint: "
                           << g_application_fingerprint << '\n';
      }
      const char* const date_time_format = FLAGS_log_year_in_prefix
                                               ? "yyyymmdd hh:mm:ss.uuuuuu"
                                               : "mmdd hh:mm:ss.uuuuuu";
      file_header_stream
          << "Running duration (h:mm:ss): "
          << PrettyDuration(
                 std::chrono::duration_cast<std::chrono::duration<int>>(
                     timestamp - start_time_))
          << '\n'
          << "Log line format: [IWEF]" << date_time_format << " "
          << "threadid file:line] msg" << '\n';
      const string& file_header_string = file_header_stream.str();
      const size_t header_len = file_header_string.size();
      fwrite(file_header_string.data(), 1, header_len, file_.get());
      file_length_ += header_len;
      bytes_since_flush_ += header_len;
    }
  }
  if (!stop_writing) {
    errno = 0;
    fwrite(message, 1, message_len, file_.get());
    if (FLAGS_stop_logging_if_full_disk &&
        errno == ENOSPC) {  
      stop_writing = true;  
      return;
    } else {
      file_length_ += message_len;
      bytes_since_flush_ += message_len;
    }
  } else {
    if (timestamp >= next_flush_time_) {
      stop_writing = false;  
    }
    return;  
  }
  if (force_flush || (bytes_since_flush_ >= 1000000) ||
      (timestamp >= next_flush_time_)) {
    FlushUnlocked(timestamp);
#ifdef GLOG_OS_LINUX
    if (FLAGS_drop_log_memory && file_length_ >= (3U << 20U)) {
      uint32 total_drop_length =
          (file_length_ & ~((1U << 20U) - 1U)) - (1U << 20U);
      uint32 this_drop_length = total_drop_length - dropped_mem_length_;
      if (this_drop_length >= (2U << 20U)) {
#  if defined(HAVE_POSIX_FADVISE)
        posix_fadvise(
            fileno(file_.get()), static_cast<off_t>(dropped_mem_length_),
            static_cast<off_t>(this_drop_length), POSIX_FADV_DONTNEED);
#  endif
        dropped_mem_length_ = total_drop_length;
      }
    }
#endif
  }
}
LogCleaner::LogCleaner() = default;
void LogCleaner::Enable(const std::chrono::minutes& overdue) {
  enabled_ = true;
  overdue_ = overdue;
}
void LogCleaner::Disable() { enabled_ = false; }
void LogCleaner::Run(const std::chrono::system_clock::time_point& current_time,
                     bool base_filename_selected, const string& base_filename,
                     const string& filename_extension) {
  assert(enabled_);
  assert(!base_filename_selected || !base_filename.empty());
  if (current_time < next_cleanup_time_) {
    return;
  }
  next_cleanup_time_ =
      current_time +
      std::chrono::duration_cast<std::chrono::system_clock::duration>(
          std::chrono::duration<int32>{FLAGS_logcleansecs});
  vector<string> dirs;
  if (!base_filename_selected) {
    dirs = GetLoggingDirectories();
  } else {
    size_t pos = base_filename.find_last_of(possible_dir_delim, string::npos,
                                            sizeof(possible_dir_delim));
    if (pos != string::npos) {
      string dir = base_filename.substr(0, pos + 1);
      dirs.push_back(dir);
    } else {
      dirs.emplace_back(".");
    }
  }
  for (const std::string& dir : dirs) {
    vector<string> logs = GetOverdueLogNames(dir, current_time, base_filename,
                                             filename_extension);
    for (const std::string& log : logs) {
      int result = unlink(log.c_str());
      if (result != 0) {
        perror(("Could not remove overdue log " + log).c_str());
      }
    }
  }
}
vector<string> LogCleaner::GetOverdueLogNames(
    string log_directory,
    const std::chrono::system_clock::time_point& current_time,
    const string& base_filename, const string& filename_extension) const {
  vector<string> overdue_log_names;
  DIR* dir;
  struct dirent* ent;
  if ((dir = opendir(log_directory.c_str()))) {
    while ((ent = readdir(dir))) {
      if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
        continue;
      }
      string filepath = ent->d_name;
      const char* const dir_delim_end =
          possible_dir_delim + sizeof(possible_dir_delim);
      if (!log_directory.empty() &&
          std::find(possible_dir_delim, dir_delim_end,
                    log_directory[log_directory.size() - 1]) != dir_delim_end) {
        filepath = log_directory + filepath;
      }
      if (IsLogFromCurrentProject(filepath, base_filename,
                                  filename_extension) &&
          IsLogLastModifiedOver(filepath, current_time)) {
        overdue_log_names.push_back(filepath);
      }
    }
    closedir(dir);
  }
  return overdue_log_names;
}
bool LogCleaner::IsLogFromCurrentProject(
    const string& filepath, const string& base_filename,
    const string& filename_extension) const {
  string cleaned_base_filename;
  const char* const dir_delim_end =
      possible_dir_delim + sizeof(possible_dir_delim);
  size_t real_filepath_size = filepath.size();
  for (char c : base_filename) {
    if (cleaned_base_filename.empty()) {
      cleaned_base_filename += c;
    } else if (std::find(possible_dir_delim, dir_delim_end, c) ==
                   dir_delim_end ||
               (!cleaned_base_filename.empty() &&
                c != cleaned_base_filename[cleaned_base_filename.size() - 1])) {
      cleaned_base_filename += c;
    }
  }
  if (filepath.find(cleaned_base_filename) != 0) {
    return false;
  }
  if (!filename_extension.empty()) {
    if (cleaned_base_filename.size() >= real_filepath_size) {
      return false;
    }
    string ext = filepath.substr(cleaned_base_filename.size(),
                                 filename_extension.size());
    if (ext == filename_extension) {
      cleaned_base_filename += filename_extension;
    } else {
      if (filename_extension.size() >= real_filepath_size) {
        return false;
      }
      real_filepath_size = filepath.size() - filename_extension.size();
      if (filepath.substr(real_filepath_size) != filename_extension) {
        return false;
      }
    }
  }
  for (size_t i = cleaned_base_filename.size(); i < real_filepath_size; i++) {
    const char& c = filepath[i];
    if (i <= cleaned_base_filename.size() + 7) {  
      if (c < '0' || c > '9') {
        return false;
      }
    } else if (i == cleaned_base_filename.size() + 8) {  
      if (c != '-') {
        return false;
      }
    } else if (i <= cleaned_base_filename.size() + 14) {  
      if (c < '0' || c > '9') {
        return false;
      }
    } else if (i == cleaned_base_filename.size() + 15) {  
      if (c != '.') {
        return false;
      }
    } else if (i >= cleaned_base_filename.size() + 16) {  
      if (c < '0' || c > '9') {
        return false;
      }
    }
  }
  return true;
}
bool LogCleaner::IsLogLastModifiedOver(
    const string& filepath,
    const std::chrono::system_clock::time_point& current_time) const {
  struct stat file_stat;
  if (stat(filepath.c_str(), &file_stat) == 0) {
    const auto last_modified_time =
        std::chrono::system_clock::from_time_t(file_stat.st_mtime);
    const auto diff = current_time - last_modified_time;
    return diff >= overdue_;
  }
  return false;
}
}  
static std::mutex fatal_msg_lock;
static logging::internal::CrashReason crash_reason;
static bool fatal_msg_exclusive = true;
static logging::internal::LogMessageData fatal_msg_data_exclusive;
static logging::internal::LogMessageData fatal_msg_data_shared;
#ifdef GLOG_THREAD_LOCAL_STORAGE
static thread_local bool thread_data_available = true;
#  if defined(__cpp_lib_byte) && __cpp_lib_byte >= 201603L
alignas(logging::internal::LogMessageData) static thread_local std::byte
    thread_msg_data[sizeof(logging::internal::LogMessageData)];
#  else   
static thread_local std::aligned_storage<
    sizeof(logging::internal::LogMessageData),
    alignof(logging::internal::LogMessageData)>::type thread_msg_data;
#  endif  
#endif    
logging::internal::LogMessageData::LogMessageData()
    : stream_(message_text_, LogMessage::kMaxLogMessageLen, 0) {}
LogMessage::LogMessage(const char* file, int line, LogSeverity severity,
                       int64 ctr, void (LogMessage::*send_method)())
    : allocated_(nullptr) {
  Init(file, line, severity, send_method);
  data_->stream_.set_ctr(ctr);
}
LogMessage::LogMessage(const char* file, int line,
                       const logging::internal::CheckOpString& result)
    : allocated_(nullptr) {
  Init(file, line, GLOG_FATAL, &LogMessage::SendToLog);
  stream() << "Check failed: " << (*result.str_) << " ";
}
LogMessage::LogMessage(const char* file, int line) : allocated_(nullptr) {
  Init(file, line, GLOG_INFO, &LogMessage::SendToLog);
}
LogMessage::LogMessage(const char* file, int line, LogSeverity severity)
    : allocated_(nullptr) {
  Init(file, line, severity, &LogMessage::SendToLog);
}
LogMessage::LogMessage(const char* file, int line, LogSeverity severity,
                       LogSink* sink, bool also_send_to_log)
    : allocated_(nullptr) {
  Init(file, line, severity,
       also_send_to_log ? &LogMessage::SendToSinkAndLog
                        : &LogMessage::SendToSink);
  data_->sink_ = sink;  
}
LogMessage::LogMessage(const char* file, int line, LogSeverity severity,
                       vector<string>* outvec)
    : allocated_(nullptr) {
  Init(file, line, severity, &LogMessage::SaveOrSendToLog);
  data_->outvec_ = outvec;  
}
LogMessage::LogMessage(const char* file, int line, LogSeverity severity,
                       string* message)
    : allocated_(nullptr) {
  Init(file, line, severity, &LogMessage::WriteToStringAndLog);
  data_->message_ = message;  
}
void LogMessage::Init(const char* file, int line, LogSeverity severity,
                      void (LogMessage::*send_method)()) {
  allocated_ = nullptr;
  if (severity != GLOG_FATAL || !exit_on_dfatal) {
#ifdef GLOG_THREAD_LOCAL_STORAGE
    if (thread_data_available) {
      thread_data_available = false;
      data_ = new (&thread_msg_data) logging::internal::LogMessageData;
    } else {
      allocated_ = new logging::internal::LogMessageData();
      data_ = allocated_;
    }
#else   
    allocated_ = new logging::internal::LogMessageData();
    data_ = allocated_;
#endif  
    data_->first_fatal_ = false;
  } else {
    std::lock_guard<std::mutex> l{fatal_msg_lock};
    if (fatal_msg_exclusive) {
      fatal_msg_exclusive = false;
      data_ = &fatal_msg_data_exclusive;
      data_->first_fatal_ = true;
    } else {
      data_ = &fatal_msg_data_shared;
      data_->first_fatal_ = false;
    }
  }
  data_->preserved_errno_ = errno;
  data_->severity_ = severity;
  data_->line_ = line;
  data_->send_method_ = send_method;
  data_->sink_ = nullptr;
  data_->outvec_ = nullptr;
  const auto now = std::chrono::system_clock::now();
  time_ = LogMessageTime(now);
  data_->num_chars_to_log_ = 0;
  data_->num_chars_to_syslog_ = 0;
  data_->basename_ = const_basename(file);
  data_->fullname_ = file;
  data_->has_been_flushed_ = false;
  data_->thread_id_ = std::this_thread::get_id();
  if (FLAGS_log_prefix && (line != kNoLogPrefix)) {
    std::ios saved_fmt(nullptr);
    saved_fmt.copyfmt(stream());
    stream().fill('0');
    if (g_prefix_formatter == nullptr) {
      stream() << LogSeverityNames[severity][0];
      if (FLAGS_log_year_in_prefix) {
        stream() << setw(4) << 1900 + time_.year();
      }
      stream() << setw(2) << 1 + time_.month() << setw(2) << time_.day() << ' '
               << setw(2) << time_.hour() << ':' << setw(2) << time_.min()
               << ':' << setw(2) << time_.sec() << "." << setw(6)
               << time_.usec() << ' ' << setfill(' ') << setw(5)
               << data_->thread_id_ << setfill('0') << ' ' << data_->basename_
               << ':' << data_->line_ << "] ";
    } else {
      (*g_prefix_formatter)(stream(), *this);
      stream() << " ";
    }
    stream().copyfmt(saved_fmt);
  }
  data_->num_prefix_chars_ = data_->stream_.pcount();
  if (!FLAGS_log_backtrace_at.empty()) {
    char fileline[128];
    std::snprintf(fileline, sizeof(fileline), "%s:%d", data_->basename_, line);
#ifdef HAVE_STACKTRACE
    if (FLAGS_log_backtrace_at == fileline) {
      string stacktrace = GetStackTrace();
      stream() << " (stacktrace:\n" << stacktrace << ") ";
    }
#endif
  }
}
LogSeverity LogMessage::severity() const noexcept { return data_->severity_; }
int LogMessage::line() const noexcept { return data_->line_; }
const std::thread::id& LogMessage::thread_id() const noexcept {
  return data_->thread_id_;
}
const char* LogMessage::fullname() const noexcept { return data_->fullname_; }
const char* LogMessage::basename() const noexcept { return data_->basename_; }
const LogMessageTime& LogMessage::time() const noexcept { return time_; }
LogMessage::~LogMessage() noexcept(false) {
  Flush();
  bool fail = data_->severity_ == GLOG_FATAL && exit_on_dfatal;
#ifdef GLOG_THREAD_LOCAL_STORAGE
  if (data_ == static_cast<void*>(&thread_msg_data)) {
    data_->~LogMessageData();
    thread_data_available = true;
  } else {
    delete allocated_;
  }
#else   
  delete allocated_;
#endif  
  if (fail) {
    const char* message = "*** Check failure stack trace: ***\n";
    if (write(fileno(stderr), message, strlen(message)) < 0) {
    }
    AlsoErrorWrite(GLOG_FATAL,
                   glog_internal_namespace_::ProgramInvocationShortName(),
                   message);
#if defined(__cpp_lib_uncaught_exceptions) && \
    (__cpp_lib_uncaught_exceptions >= 201411L)
    if (std::uncaught_exceptions() == 0)
#else
    if (!std::uncaught_exception())
#endif
    {
      Fail();
    }
  }
}
int LogMessage::preserved_errno() const { return data_->preserved_errno_; }
ostream& LogMessage::stream() { return data_->stream_; }
void LogMessage::Flush() {
  if (data_->has_been_flushed_ || data_->severity_ < FLAGS_minloglevel) {
    return;
  }
  data_->num_chars_to_log_ = data_->stream_.pcount();
  data_->num_chars_to_syslog_ =
      data_->num_chars_to_log_ - data_->num_prefix_chars_;
  bool append_newline =
      (data_->message_text_[data_->num_chars_to_log_ - 1] != '\n');
  char original_final_char = '\0';
  if (append_newline) {
    original_final_char = data_->message_text_[data_->num_chars_to_log_];
    data_->message_text_[data_->num_chars_to_log_++] = '\n';
  }
  data_->message_text_[data_->num_chars_to_log_] = '\0';
  {
    std::lock_guard<std::mutex> l{log_mutex};
    (this->*(data_->send_method_))();
    ++num_messages_[static_cast<int>(data_->severity_)];
  }
  LogDestination::WaitForSinks(data_);
  if (append_newline) {
    data_->message_text_[data_->num_chars_to_log_ - 1] = original_final_char;
  }
  if (data_->preserved_errno_ != 0) {
    errno = data_->preserved_errno_;
  }
  data_->has_been_flushed_ = true;
}
static std::chrono::system_clock::time_point fatal_time;
static char fatal_message[256];
void ReprintFatalMessage() {
  if (fatal_message[0]) {
    const size_t n = strlen(fatal_message);
    if (!FLAGS_logtostderr) {
      WriteToStderr(fatal_message, n);
    }
    LogDestination::LogToAllLogfiles(GLOG_ERROR, fatal_time, fatal_message, n);
  }
}
void LogMessage::SendToLog() EXCLUSIVE_LOCKS_REQUIRED(log_mutex) {
  static bool already_warned_before_initgoogle = false;
  RAW_DCHECK(data_->num_chars_to_log_ > 0 &&
                 data_->message_text_[data_->num_chars_to_log_ - 1] == '\n',
             "");
  if (!already_warned_before_initgoogle && !IsGoogleLoggingInitialized()) {
    const char w[] =
        "WARNING: Logging before InitGoogleLogging() is "
        "written to STDERR\n";
    WriteToStderr(w, strlen(w));
    already_warned_before_initgoogle = true;
  }
  if (FLAGS_logtostderr || FLAGS_logtostdout || !IsGoogleLoggingInitialized()) {
    if (FLAGS_logtostdout) {
      ColoredWriteToStdout(data_->severity_, data_->message_text_,
                           data_->num_chars_to_log_);
    } else {
      ColoredWriteToStderr(data_->severity_, data_->message_text_,
                           data_->num_chars_to_log_);
    }
    LogDestination::LogToSinks(
        data_->severity_, data_->fullname_, data_->basename_, data_->line_,
        time_, data_->message_text_ + data_->num_prefix_chars_,
        (data_->num_chars_to_log_ - data_->num_prefix_chars_ - 1));
  } else {
    LogDestination::LogToAllLogfiles(data_->severity_, time_.when(),
                                     data_->message_text_,
                                     data_->num_chars_to_log_);
    LogDestination::MaybeLogToStderr(data_->severity_, data_->message_text_,
                                     data_->num_chars_to_log_,
                                     data_->num_prefix_chars_);
    LogDestination::MaybeLogToEmail(data_->severity_, data_->message_text_,
                                    data_->num_chars_to_log_);
    LogDestination::LogToSinks(
        data_->severity_, data_->fullname_, data_->basename_, data_->line_,
        time_, data_->message_text_ + data_->num_prefix_chars_,
        (data_->num_chars_to_log_ - data_->num_prefix_chars_ - 1));
  }
  if (data_->severity_ == GLOG_FATAL && exit_on_dfatal) {
    if (data_->first_fatal_) {
      RecordCrashReason(&crash_reason);
      SetCrashReason(&crash_reason);
      const size_t copy =
          min(data_->num_chars_to_log_, sizeof(fatal_message) - 1);
      memcpy(fatal_message, data_->message_text_, copy);
      fatal_message[copy] = '\0';
      fatal_time = time_.when();
    }
    if (!FLAGS_logtostderr && !FLAGS_logtostdout) {
      for (auto& log_destination : LogDestination::log_destinations_) {
        if (log_destination) {
          log_destination->logger_->Write(
              true, std::chrono::system_clock::time_point{}, "", 0);
        }
      }
    }
    LogDestination::WaitForSinks(data_);
  }
}
void LogMessage::RecordCrashReason(logging::internal::CrashReason* reason) {
  reason->filename = fatal_msg_data_exclusive.fullname_;
  reason->line_number = fatal_msg_data_exclusive.line_;
  reason->message = fatal_msg_data_exclusive.message_text_ +
                    fatal_msg_data_exclusive.num_prefix_chars_;
#ifdef HAVE_STACKTRACE
  reason->depth = GetStackTrace(reason->stack, ARRAYSIZE(reason->stack), 4);
#else
  reason->depth = 0;
#endif
}
GLOG_NO_EXPORT logging_fail_func_t g_logging_fail_func =
    reinterpret_cast<logging_fail_func_t>(&abort);
NullStream::NullStream() : LogMessage::LogStream(message_buffer_, 2, 0) {}
NullStream::NullStream(const char* , int ,
                       const logging::internal::CheckOpString& )
    : LogMessage::LogStream(message_buffer_, 2, 0) {}
NullStream& NullStream::stream() { return *this; }
NullStreamFatal::~NullStreamFatal() {
  std::abort();
}
logging_fail_func_t InstallFailureFunction(logging_fail_func_t fail_func) {
  return std::exchange(g_logging_fail_func, fail_func);
}
void LogMessage::Fail() { g_logging_fail_func(); }
void LogMessage::SendToSink() EXCLUSIVE_LOCKS_REQUIRED(log_mutex) {
  if (data_->sink_ != nullptr) {
    RAW_DCHECK(data_->num_chars_to_log_ > 0 &&
                   data_->message_text_[data_->num_chars_to_log_ - 1] == '\n',
               "");
    data_->sink_->send(
        data_->severity_, data_->fullname_, data_->basename_, data_->line_,
        time_, data_->message_text_ + data_->num_prefix_chars_,
        (data_->num_chars_to_log_ - data_->num_prefix_chars_ - 1));
  }
}
void LogMessage::SendToSinkAndLog() EXCLUSIVE_LOCKS_REQUIRED(log_mutex) {
  SendToSink();
  SendToLog();
}
void LogMessage::SaveOrSendToLog() EXCLUSIVE_LOCKS_REQUIRED(log_mutex) {
  if (data_->outvec_ != nullptr) {
    RAW_DCHECK(data_->num_chars_to_log_ > 0 &&
                   data_->message_text_[data_->num_chars_to_log_ - 1] == '\n',
               "");
    const char* start = data_->message_text_ + data_->num_prefix_chars_;
    size_t len = data_->num_chars_to_log_ - data_->num_prefix_chars_ - 1;
    data_->outvec_->push_back(string(start, len));
  } else {
    SendToLog();
  }
}
void LogMessage::WriteToStringAndLog() EXCLUSIVE_LOCKS_REQUIRED(log_mutex) {
  if (data_->message_ != nullptr) {
    RAW_DCHECK(data_->num_chars_to_log_ > 0 &&
                   data_->message_text_[data_->num_chars_to_log_ - 1] == '\n',
               "");
    const char* start = data_->message_text_ + data_->num_prefix_chars_;
    size_t len = data_->num_chars_to_log_ - data_->num_prefix_chars_ - 1;
    data_->message_->assign(start, len);
  }
  SendToLog();
}
void LogMessage::SendToSyslogAndLog() {
#ifdef HAVE_SYSLOG_H
  static bool openlog_already_called = false;
  if (!openlog_already_called) {
    openlog(glog_internal_namespace_::ProgramInvocationShortName(),
            LOG_CONS | LOG_NDELAY | LOG_PID, LOG_USER);
    openlog_already_called = true;
  }
  const int SEVERITY_TO_LEVEL[] = {LOG_INFO, LOG_WARNING, LOG_ERR, LOG_EMERG};
  syslog(LOG_USER | SEVERITY_TO_LEVEL[static_cast<int>(data_->severity_)],
         "%.*s", static_cast<int>(data_->num_chars_to_syslog_),
         data_->message_text_ + data_->num_prefix_chars_);
  SendToLog();
#else
  LOG(ERROR) << "No syslog support: message=" << data_->message_text_;
#endif
}
base::Logger* base::GetLogger(LogSeverity severity) {
  std::lock_guard<std::mutex> l{log_mutex};
  return LogDestination::log_destination(severity)->GetLoggerImpl();
}
void base::SetLogger(LogSeverity severity, base::Logger* logger) {
  std::lock_guard<std::mutex> l{log_mutex};
  LogDestination::log_destination(severity)->SetLoggerImpl(logger);
}
int64 LogMessage::num_messages(int severity) {
  std::lock_guard<std::mutex> l{log_mutex};
  return num_messages_[severity];
}
ostream& operator<<(ostream& os, const Counter_t&) {
#ifdef DISABLE_RTTI
  LogMessage::LogStream* log = static_cast<LogMessage::LogStream*>(&os);
#else
  auto* log = dynamic_cast<LogMessage::LogStream*>(&os);
#endif
  CHECK(log && log == log->self())
      << "You must not use COUNTER with non-glog ostream";
  os << log->ctr();
  return os;
}
ErrnoLogMessage::ErrnoLogMessage(const char* file, int line,
                                 LogSeverity severity, int64 ctr,
                                 void (LogMessage::*send_method)())
    : LogMessage(file, line, severity, ctr, send_method) {}
ErrnoLogMessage::~ErrnoLogMessage() {
  stream() << ": " << StrError(preserved_errno()) << " [" << preserved_errno()
           << "]";
}
void FlushLogFiles(LogSeverity min_severity) {
  LogDestination::FlushLogFiles(min_severity);
}
void FlushLogFilesUnsafe(LogSeverity min_severity) {
  LogDestination::FlushLogFilesUnsafe(min_severity);
}
void SetLogDestination(LogSeverity severity, const char* base_filename) {
  LogDestination::SetLogDestination(severity, base_filename);
}
void SetLogSymlink(LogSeverity severity, const char* symlink_basename) {
  LogDestination::SetLogSymlink(severity, symlink_basename);
}
LogSink::~LogSink() = default;
void LogSink::WaitTillSent() {
}
string LogSink::ToString(LogSeverity severity, const char* file, int line,
                         const LogMessageTime& time, const char* message,
                         size_t message_len) {
  ostringstream stream;
  stream.fill('0');
  stream << LogSeverityNames[severity][0];
  if (FLAGS_log_year_in_prefix) {
    stream << setw(4) << 1900 + time.year();
  }
  stream << setw(2) << 1 + time.month() << setw(2) << time.day() << ' '
         << setw(2) << time.hour() << ':' << setw(2) << time.min() << ':'
         << setw(2) << time.sec() << '.' << setw(6) << time.usec() << ' '
         << setfill(' ') << setw(5) << std::this_thread::get_id()
         << setfill('0') << ' ' << file << ':' << line << "] ";
  (stream.write)(message, static_cast<std::streamsize>(message_len));
  return stream.str();
}
void AddLogSink(LogSink* destination) {
  LogDestination::AddLogSink(destination);
}
void RemoveLogSink(LogSink* destination) {
  LogDestination::RemoveLogSink(destination);
}
void SetLogFilenameExtension(const char* ext) {
  LogDestination::SetLogFilenameExtension(ext);
}
void SetStderrLogging(LogSeverity min_severity) {
  LogDestination::SetStderrLogging(min_severity);
}
void SetEmailLogging(LogSeverity min_severity, const char* addresses) {
  LogDestination::SetEmailLogging(min_severity, addresses);
}
void LogToStderr() { LogDestination::LogToStderr(); }
namespace base {
namespace internal {
bool GetExitOnDFatal();
bool GetExitOnDFatal() {
  std::lock_guard<std::mutex> l{log_mutex};
  return exit_on_dfatal;
}
void SetExitOnDFatal(bool value);
void SetExitOnDFatal(bool value) {
  std::lock_guard<std::mutex> l{log_mutex};
  exit_on_dfatal = value;
}
}  
}  
#ifndef GLOG_OS_EMSCRIPTEN
static const char kDontNeedShellEscapeChars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+-_.=/:,@";
static string ShellEscape(const string& src) {
  string result;
  if (!src.empty() &&  
      src.find_first_not_of(kDontNeedShellEscapeChars) == string::npos) {
    result.assign(src);
  } else if (src.find_first_of('\'') == string::npos) {
    result.assign("'");
    result.append(src);
    result.append("'");
  } else {
    result.assign("\"");
    for (size_t i = 0; i < src.size(); ++i) {
      switch (src[i]) {
        case '\\':
        case '$':
        case '"':
        case '`':
          result.append("\\");
      }
      result.append(src, i, 1);
    }
    result.append("\"");
  }
  return result;
}
static inline void trim(std::string& s) {
  const auto toRemove = [](char ch) { return std::isspace(ch) == 0; };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), toRemove));
  s.erase(std::find_if(s.rbegin(), s.rend(), toRemove).base(), s.end());
}
#endif
static bool SendEmailInternal(const char* dest, const char* subject,
                              const char* body, bool use_logging) {
#ifndef GLOG_OS_EMSCRIPTEN
  if (dest && *dest) {
    std::istringstream ss(dest);
    std::ostringstream sanitized_dests;
    std::string s;
    while (std::getline(ss, s, ',')) {
      trim(s);
      if (s.empty()) {
        continue;
      }
      if (!std::regex_match(
              s,
              std::regex("^[a-zA-Z0-9]"
                         "[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]*@[a-zA-Z0-9]"
                         "(?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\\.[a-zA-Z0-9]"
                         "(?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"))) {
        if (use_logging) {
          VLOG(1) << "Invalid destination email address:" << s;
        } else {
          fprintf(stderr, "Invalid destination email address: %s\n", s.c_str());
        }
        return false;
      }
      if (!sanitized_dests.str().empty()) {
        sanitized_dests << ",";
      }
      sanitized_dests << s;
    }
    const std::string& tmp = sanitized_dests.str();
    dest = tmp.c_str();
    if (use_logging) {
      VLOG(1) << "Trying to send TITLE:" << subject << " BODY:" << body
              << " to " << dest;
    } else {
      fprintf(stderr, "Trying to send TITLE: %s BODY: %s to %s\n", subject,
              body, dest);
    }
    string logmailer;
    if (FLAGS_logmailer.empty()) {
      logmailer = "/bin/mail";
    } else {
      logmailer = ShellEscape(FLAGS_logmailer);
    }
    string cmd =
        logmailer + " -s" + ShellEscape(subject) + " " + ShellEscape(dest);
    if (use_logging) {
      VLOG(4) << "Mailing command: " << cmd;
    }
    FILE* pipe = popen(cmd.c_str(), "w");
    if (pipe != nullptr) {
      if (body) {
        fwrite(body, sizeof(char), strlen(body), pipe);
      }
      bool ok = pclose(pipe) != -1;
      if (!ok) {
        if (use_logging) {
          LOG(ERROR) << "Problems sending mail to " << dest << ": "
                     << StrError(errno);
        } else {
          fprintf(stderr, "Problems sending mail to %s: %s\n", dest,
                  StrError(errno).c_str());
        }
      }
      return ok;
    } else {
      if (use_logging) {
        LOG(ERROR) << "Unable to send mail to " << dest;
      } else {
        fprintf(stderr, "Unable to send mail to %s\n", dest);
      }
    }
  }
#else
  (void)dest;
  (void)subject;
  (void)body;
  (void)use_logging;
  LOG(WARNING) << "Email support not available; not sending message";
#endif
  return false;
}
bool SendEmail(const char* dest, const char* subject, const char* body) {
  return SendEmailInternal(dest, subject, body, true);
}
static void GetTempDirectories(vector<string>& list) {
  list.clear();
#ifdef GLOG_OS_WINDOWS
  char tmp[MAX_PATH];
  if (GetTempPathA(MAX_PATH, tmp)) list.push_back(tmp);
  list.push_back("C:\\TMP\\");
  list.push_back("C:\\TEMP\\");
#else
  const char* candidates[] = {
      getenv("TEST_TMPDIR"),
      getenv("TMPDIR"),
      getenv("TMP"),
      "/tmp",
  };
  for (auto d : candidates) {
    if (!d) continue;  
    string dstr = d;
    if (dstr[dstr.size() - 1] != '/') {
      dstr += "/";
    }
    list.push_back(dstr);
    struct stat statbuf;
    if (!stat(d, &statbuf) && S_ISDIR(statbuf.st_mode)) {
      return;
    }
  }
#endif
}
static std::unique_ptr<std::vector<std::string>> logging_directories_list;
const vector<string>& GetLoggingDirectories() {
  if (logging_directories_list == nullptr) {
    logging_directories_list = std::make_unique<std::vector<std::string>>();
    if (!FLAGS_log_dir.empty()) {
      if (std::find(std::begin(possible_dir_delim),
                    std::end(possible_dir_delim),
                    FLAGS_log_dir.back()) == std::end(possible_dir_delim)) {
        logging_directories_list->push_back(FLAGS_log_dir + "/");
      } else {
        logging_directories_list->push_back(FLAGS_log_dir);
      }
    } else {
      GetTempDirectories(*logging_directories_list);
#ifdef GLOG_OS_WINDOWS
      char tmp[MAX_PATH];
      if (GetWindowsDirectoryA(tmp, MAX_PATH))
        logging_directories_list->push_back(tmp);
      logging_directories_list->push_back(".\\");
#else
      logging_directories_list->push_back("./");
#endif
    }
  }
  return *logging_directories_list;
}
GLOG_NO_EXPORT
void GetExistingTempDirectories(vector<string>& list) {
  GetTempDirectories(list);
  auto i_dir = list.begin();
  while (i_dir != list.end()) {
    if (access(i_dir->c_str(), 0)) {
      i_dir = list.erase(i_dir);
    } else {
      ++i_dir;
    }
  }
}
void TruncateLogFile(const char* path, uint64 limit, uint64 keep) {
#if defined(HAVE_UNISTD_H) || defined(HAVE__CHSIZE_S)
  struct stat statbuf;
  const int kCopyBlockSize = 8 << 10;
  char copybuf[kCopyBlockSize];
  off_t read_offset, write_offset;
  int flags = O_RDWR;
#  ifdef GLOG_OS_LINUX
  const char* procfd_prefix = "/proc/self/fd/";
  if (strncmp(procfd_prefix, path, strlen(procfd_prefix))) flags |= O_NOFOLLOW;
#  endif
  FileDescriptor fd{open(path, flags)};
  if (!fd) {
    if (errno == EFBIG) {
#  ifdef HAVE__CHSIZE_S
      if (_chsize_s(fd.get(), 0) != 0) {
#  else
      if (truncate(path, 0) == -1) {
#  endif
        PLOG(ERROR) << "Unable to truncate " << path;
      } else {
        LOG(ERROR) << "Truncated " << path << " due to EFBIG error";
      }
    } else {
      PLOG(ERROR) << "Unable to open " << path;
    }
    return;
  }
  if (fstat(fd.get(), &statbuf) == -1) {
    PLOG(ERROR) << "Unable to fstat()";
    return;
  }
  if (!S_ISREG(statbuf.st_mode)) return;
  if (statbuf.st_size <= static_cast<off_t>(limit)) return;
  if (statbuf.st_size <= static_cast<off_t>(keep)) return;
  LOG(INFO) << "Truncating " << path << " to " << keep << " bytes";
  read_offset = statbuf.st_size - static_cast<off_t>(keep);
  write_offset = 0;
  ssize_t bytesin, bytesout;
  while ((bytesin = pread(fd.get(), copybuf, sizeof(copybuf), read_offset)) >
         0) {
    bytesout =
        pwrite(fd.get(), copybuf, static_cast<size_t>(bytesin), write_offset);
    if (bytesout == -1) {
      PLOG(ERROR) << "Unable to write to " << path;
      break;
    } else if (bytesout != bytesin) {
      LOG(ERROR) << "Expected to write " << bytesin << ", wrote " << bytesout;
    }
    read_offset += bytesin;
    write_offset += bytesout;
  }
  if (bytesin == -1) PLOG(ERROR) << "Unable to read from " << path;
#  ifdef HAVE__CHSIZE_S
  if (_chsize_s(fd.get(), write_offset) != 0) {
#  else
  if (ftruncate(fd.get(), write_offset) == -1) {
#  endif
    PLOG(ERROR) << "Unable to truncate " << path;
  }
#else
  LOG(ERROR) << "No log truncation support.";
#endif
}
void TruncateStdoutStderr() {
#ifdef HAVE_UNISTD_H
  uint64 limit = MaxLogSize() << 20U;
  uint64 keep = 1U << 20U;
  TruncateLogFile("/proc/self/fd/1", limit, keep);
  TruncateLogFile("/proc/self/fd/2", limit, keep);
#else
  LOG(ERROR) << "No log truncation support.";
#endif
}
namespace logging {
namespace internal {
#define DEFINE_CHECK_STROP_IMPL(name, func, expected)                         \
  std::unique_ptr<string> Check##func##expected##Impl(                        \
      const char* s1, const char* s2, const char* names) {                    \
    bool equal = s1 == s2 || (s1 && s2 && !func(s1, s2));                     \
    if (equal == (expected))                                                  \
      return nullptr;                                                         \
    else {                                                                    \
      ostringstream ss;                                                       \
      if (!s1) s1 = "";                                                       \
      if (!s2) s2 = "";                                                       \
      ss << #name " failed: " << names << " (" << s1 << " vs. " << s2 << ")"; \
      return std::make_unique<std::string>(ss.str());                         \
    }                                                                         \
  }
DEFINE_CHECK_STROP_IMPL(CHECK_STREQ, strcmp, true)
DEFINE_CHECK_STROP_IMPL(CHECK_STRNE, strcmp, false)
DEFINE_CHECK_STROP_IMPL(CHECK_STRCASEEQ, strcasecmp, true)
DEFINE_CHECK_STROP_IMPL(CHECK_STRCASENE, strcasecmp, false)
#undef DEFINE_CHECK_STROP_IMPL
}  
}  
GLOG_NO_EXPORT
int posix_strerror_r(int err, char* buf, size_t len) {
  if (buf == nullptr || len <= 0) {
    errno = EINVAL;
    return -1;
  }
  buf[0] = '\000';
  int old_errno = errno;
  errno = 0;
  char* rc = reinterpret_cast<char*>(strerror_r(err, buf, len));
  if (errno) {
    buf[0] = '\000';
    return -1;
  }
  errno = old_errno;
  buf[len - 1] = '\000';
  if (!rc) {
    return 0;
  } else {
    if (rc == buf) {
      return 0;
    } else {
      buf[0] = '\000';
#if defined(GLOG_OS_MACOSX) || defined(GLOG_OS_FREEBSD) || \
    defined(GLOG_OS_OPENBSD)
      if (reinterpret_cast<intptr_t>(rc) < sys_nerr) {
        return -1;
      }
#endif
      strncat(buf, rc, len - 1);
      return 0;
    }
  }
}
string StrError(int err) {
  char buf[100];
  int rc = posix_strerror_r(err, buf, sizeof(buf));
  if ((rc < 0) || (buf[0] == '\000')) {
    std::snprintf(buf, sizeof(buf), "Error number %d", err);
  }
  return buf;
}
LogMessageFatal::LogMessageFatal(const char* file, int line)
    : LogMessage(file, line, GLOG_FATAL) {}
LogMessageFatal::LogMessageFatal(const char* file, int line,
                                 const logging::internal::CheckOpString& result)
    : LogMessage(file, line, result) {}
LogMessageFatal::~LogMessageFatal() noexcept(false) {
  Flush();
  LogMessage::Fail();
}
namespace logging {
namespace internal {
CheckOpMessageBuilder::CheckOpMessageBuilder(const char* exprtext)
    : stream_(new ostringstream) {
  *stream_ << exprtext << " (";
}
CheckOpMessageBuilder::~CheckOpMessageBuilder() { delete stream_; }
ostream* CheckOpMessageBuilder::ForVar2() {
  *stream_ << " vs. ";
  return stream_;
}
std::unique_ptr<string> CheckOpMessageBuilder::NewString() {
  *stream_ << ")";
  return std::make_unique<std::string>(stream_->str());
}
template <>
void MakeCheckOpValueString(std::ostream* os, const char& v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "char value " << static_cast<short>(v);
  }
}
template <>
void MakeCheckOpValueString(std::ostream* os, const signed char& v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "signed char value " << static_cast<short>(v);
  }
}
template <>
void MakeCheckOpValueString(std::ostream* os, const unsigned char& v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "unsigned char value " << static_cast<unsigned short>(v);
  }
}
template <>
void MakeCheckOpValueString(std::ostream* os, const std::nullptr_t& ) {
  (*os) << "nullptr";
}
}  
}  
void InitGoogleLogging(const char* argv0) { InitGoogleLoggingUtilities(argv0); }
void InstallPrefixFormatter(PrefixFormatterCallback callback, void* data) {
  if (callback != nullptr) {
    g_prefix_formatter = std::make_unique<PrefixFormatter>(callback, data);
  } else {
    g_prefix_formatter = nullptr;
  }
}
void ShutdownGoogleLogging() {
  ShutdownGoogleLoggingUtilities();
  LogDestination::DeleteLogDestinations();
  logging_directories_list = nullptr;
  g_prefix_formatter = nullptr;
}
void EnableLogCleaner(unsigned int overdue_days) {
  log_cleaner.Enable(std::chrono::duration_cast<std::chrono::minutes>(
      std::chrono::duration<unsigned, std::ratio<kSecondsInDay>>{
          overdue_days}));
}
void EnableLogCleaner(const std::chrono::minutes& overdue) {
  log_cleaner.Enable(overdue);
}
void DisableLogCleaner() { log_cleaner.Disable(); }
LogMessageTime::LogMessageTime() = default;
namespace {
template <class... Args>
struct void_impl {
  using type = void;
};
template <class... Args>
using void_t = typename void_impl<Args...>::type;
template <class T, class E = void>
struct has_member_tm_gmtoff : std::false_type {};
template <class T>
struct has_member_tm_gmtoff<T, void_t<decltype(&T::tm_gmtoff)>>
    : std::true_type {};
template <class T = std::tm>
auto Breakdown(const std::chrono::system_clock::time_point& now)
    -> std::enable_if_t<!has_member_tm_gmtoff<T>::value,
                        std::tuple<std::tm, std::time_t, std::chrono::hours>> {
  std::time_t timestamp = std::chrono::system_clock::to_time_t(now);
  std::tm tm_local;
  std::tm tm_utc;
  int isdst = 0;
  if (FLAGS_log_utc_time) {
    gmtime_r(&timestamp, &tm_local);
    localtime_r(&timestamp, &tm_utc);
    isdst = tm_utc.tm_isdst;
    tm_utc = tm_local;
  } else {
    localtime_r(&timestamp, &tm_local);
    isdst = tm_local.tm_isdst;
    gmtime_r(&timestamp, &tm_utc);
  }
  std::time_t gmt_sec = std::mktime(&tm_utc);
  using namespace std::chrono_literals;
  const auto gmtoffset = std::chrono::duration_cast<std::chrono::hours>(
      now - std::chrono::system_clock::from_time_t(gmt_sec) +
      (isdst ? 1h : 0h));
  return std::make_tuple(tm_local, timestamp, gmtoffset);
}
template <class T = std::tm>
auto Breakdown(const std::chrono::system_clock::time_point& now)
    -> std::enable_if_t<has_member_tm_gmtoff<T>::value,
                        std::tuple<std::tm, std::time_t, std::chrono::hours>> {
  std::time_t timestamp = std::chrono::system_clock::to_time_t(now);
  T tm;
  if (FLAGS_log_utc_time) {
    gmtime_r(&timestamp, &tm);
  } else {
    localtime_r(&timestamp, &tm);
  }
  const auto gmtoffset = std::chrono::duration_cast<std::chrono::hours>(
      std::chrono::seconds{tm.tm_gmtoff});
  return std::make_tuple(tm, timestamp, gmtoffset);
}
}  
LogMessageTime::LogMessageTime(std::chrono::system_clock::time_point now)
    : timestamp_{now} {
  std::time_t timestamp;
  std::tie(tm_, timestamp, gmtoffset_) = Breakdown(now);
  usecs_ = std::chrono::duration_cast<std::chrono::microseconds>(
      now - std::chrono::system_clock::from_time_t(timestamp));
}
}  