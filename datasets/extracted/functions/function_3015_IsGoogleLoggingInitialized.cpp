#define _GNU_SOURCE 1
#include "utilities.h"
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include "base/googleinit.h"
#include "config.h"
#include "glog/flags.h"
#include "glog/logging.h"
#include "stacktrace.h"
#include "symbolize.h"
#ifdef GLOG_OS_ANDROID
#  include <android/log.h>
#endif
#ifdef HAVE_SYS_TIME_H
#  include <sys/time.h>
#endif
#if defined(HAVE_SYSCALL_H)
#  include <syscall.h>  
#elif defined(HAVE_SYS_SYSCALL_H)
#  include <sys/syscall.h>  
#endif
#ifdef HAVE_SYSLOG_H
#  include <syslog.h>
#endif
#ifdef HAVE_UNISTD_H
#  include <unistd.h>  
#endif
#ifdef HAVE_PWD_H
#  include <pwd.h>
#endif
#if defined(HAVE___PROGNAME)
extern char* __progname;
#endif
using std::string;
namespace google {
static const char* g_program_invocation_short_name = nullptr;
bool IsGoogleLoggingInitialized() {
  return g_program_invocation_short_name != nullptr;
}
inline namespace glog_internal_namespace_ {
constexpr int FileDescriptor::InvalidHandle;
void AlsoErrorWrite(LogSeverity severity, const char* tag,
                    const char* message) noexcept {
#if defined(GLOG_OS_WINDOWS)
  (void)severity;
  (void)tag;
  ::OutputDebugStringA(message);
#elif defined(GLOG_OS_ANDROID)
  constexpr int android_log_levels[] = {
      ANDROID_LOG_INFO,
      ANDROID_LOG_WARN,
      ANDROID_LOG_ERROR,
      ANDROID_LOG_FATAL,
  };
  __android_log_write(android_log_levels[severity], tag, message);
#else
  (void)severity;
  (void)tag;
  (void)message;
#endif
}
}  
}  
#ifdef HAVE_STACKTRACE
#  include "base/commandlineflags.h"
#  include "stacktrace.h"
#  include "symbolize.h"
namespace google {
using DebugWriter = void(const char*, void*);
static const int kPrintfPointerFieldWidth = 2 + 2 * sizeof(void*);
static void DebugWriteToStderr(const char* data, void*) {
  if (write(fileno(stderr), data, strlen(data)) < 0) {
  }
  AlsoErrorWrite(GLOG_FATAL,
                 glog_internal_namespace_::ProgramInvocationShortName(), data);
}
static void DebugWriteToString(const char* data, void* arg) {
  reinterpret_cast<string*>(arg)->append(data);
}
#  ifdef HAVE_SYMBOLIZE
static void DumpPCAndSymbol(DebugWriter* writerfn, void* arg, void* pc,
                            const char* const prefix) {
  char tmp[1024];
  const char* symbol = "(unknown)";
  if (Symbolize(reinterpret_cast<char*>(pc) - 1, tmp, sizeof(tmp))) {
    symbol = tmp;
  }
  char buf[1024];
  std::snprintf(buf, sizeof(buf), "%s@ %*p  %s\n", prefix,
                kPrintfPointerFieldWidth, pc, symbol);
  writerfn(buf, arg);
}
#  endif
static void DumpPC(DebugWriter* writerfn, void* arg, void* pc,
                   const char* const prefix) {
  char buf[100];
  std::snprintf(buf, sizeof(buf), "%s@ %*p\n", prefix, kPrintfPointerFieldWidth,
                pc);
  writerfn(buf, arg);
}
static void DumpStackTrace(int skip_count, DebugWriter* writerfn, void* arg) {
  void* stack[32];
  int depth = GetStackTrace(stack, ARRAYSIZE(stack), skip_count + 1);
  for (int i = 0; i < depth; i++) {
#  if defined(HAVE_SYMBOLIZE)
    if (FLAGS_symbolize_stacktrace) {
      DumpPCAndSymbol(writerfn, arg, stack[i], "    ");
    } else {
      DumpPC(writerfn, arg, stack[i], "    ");
    }
#  else
    DumpPC(writerfn, arg, stack[i], "    ");
#  endif
  }
}
#  ifdef __GNUC__
__attribute__((noreturn))
#  endif
static void
DumpStackTraceAndExit() {
  DumpStackTrace(1, DebugWriteToStderr, nullptr);
  if (IsFailureSignalHandlerInstalled()) {
#  ifdef HAVE_SIGACTION
    struct sigaction sig_action;
    memset(&sig_action, 0, sizeof(sig_action));
    sigemptyset(&sig_action.sa_mask);
    sig_action.sa_handler = SIG_DFL;
    sigaction(SIGABRT, &sig_action, nullptr);
#  elif defined(GLOG_OS_WINDOWS)
    signal(SIGABRT, SIG_DFL);
#  endif  
  }
  abort();
}
}  
#endif  
namespace google {
inline namespace glog_internal_namespace_ {
const char* const_basename(const char* filepath) {
  const char* base = strrchr(filepath, '/');
#ifdef GLOG_OS_WINDOWS  
  if (!base) base = strrchr(filepath, '\\');
#endif
  return base ? (base + 1) : filepath;
}
const char* ProgramInvocationShortName() {
  if (g_program_invocation_short_name != nullptr) {
    return g_program_invocation_short_name;
  }
#if defined(HAVE_PROGRAM_INVOCATION_SHORT_NAME)
  return program_invocation_short_name;
#elif defined(HAVE_GETPROGNAME)
  return getprogname();
#elif defined(HAVE___PROGNAME)
  return __progname;
#elif defined(HAVE___ARGV)
  return const_basename(__argv[0]);
#else
  return "UNKNOWN";
#endif
}
static int32 g_main_thread_pid = getpid();
int32 GetMainThreadPid() { return g_main_thread_pid; }
bool PidHasChanged() {
  int32 pid = getpid();
  if (g_main_thread_pid == pid) {
    return false;
  }
  g_main_thread_pid = pid;
  return true;
}
static string g_my_user_name;
const string& MyUserName() { return g_my_user_name; }
static void MyUserNameInitializer() {
#if defined(GLOG_OS_WINDOWS)
  const char* user = getenv("USERNAME");
#else
  const char* user = getenv("USER");
#endif
  if (user != nullptr) {
    g_my_user_name = user;
  } else {
#if defined(HAVE_PWD_H) && defined(HAVE_UNISTD_H)
    struct passwd pwd;
    struct passwd* result = nullptr;
    char buffer[1024] = {'\0'};
    uid_t uid = geteuid();
    int pwuid_res = getpwuid_r(uid, &pwd, buffer, sizeof(buffer), &result);
    if (pwuid_res == 0 && result) {
      g_my_user_name = pwd.pw_name;
    } else {
      std::snprintf(buffer, sizeof(buffer), "uid%d", uid);
      g_my_user_name = buffer;
    }
#endif
    if (g_my_user_name.empty()) {
      g_my_user_name = "invalid-user";
    }
  }
}
REGISTER_MODULE_INITIALIZER(utilities, MyUserNameInitializer())
static std::atomic<const logging::internal::CrashReason*> g_reason{nullptr};
void SetCrashReason(const logging::internal::CrashReason* r) {
  const logging::internal::CrashReason* expected = nullptr;
  g_reason.compare_exchange_strong(expected, r);
}
void InitGoogleLoggingUtilities(const char* argv0) {
  CHECK(!IsGoogleLoggingInitialized())
      << "You called InitGoogleLogging() twice!";
  g_program_invocation_short_name = const_basename(argv0);
#ifdef HAVE_STACKTRACE
  InstallFailureFunction(&DumpStackTraceAndExit);
#endif
}
void ShutdownGoogleLoggingUtilities() {
  CHECK(IsGoogleLoggingInitialized())
      << "You called ShutdownGoogleLogging() without calling "
         "InitGoogleLogging() first!";
  g_program_invocation_short_name = nullptr;
#ifdef HAVE_SYSLOG_H
  closelog();
#endif
}
}  
#ifdef HAVE_STACKTRACE
std::string GetStackTrace() {
  std::string stacktrace;
  DumpStackTrace(1, DebugWriteToString, &stacktrace);
  return stacktrace;
}
#endif
}  