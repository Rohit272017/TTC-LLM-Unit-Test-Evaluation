#include <algorithm>
#include <csignal>
#include <cstring>
#include <ctime>
#include <mutex>
#include <sstream>
#include <thread>
#include "config.h"
#include "glog/logging.h"
#include "glog/platform.h"
#include "stacktrace.h"
#include "symbolize.h"
#include "utilities.h"
#ifdef HAVE_UCONTEXT_H
#  include <ucontext.h>
#endif
#ifdef HAVE_SYS_UCONTEXT_H
#  include <sys/ucontext.h>
#endif
#ifdef HAVE_UNISTD_H
#  include <unistd.h>
#endif
namespace google {
namespace {
const struct {
  int number;
  const char* name;
} kFailureSignals[] = {
    {SIGSEGV, "SIGSEGV"}, {SIGILL, "SIGILL"},
    {SIGFPE, "SIGFPE"},   {SIGABRT, "SIGABRT"},
#if !defined(GLOG_OS_WINDOWS)
    {SIGBUS, "SIGBUS"},
#endif
    {SIGTERM, "SIGTERM"},
};
static bool kFailureSignalHandlerInstalled = false;
#if !defined(GLOG_OS_WINDOWS)
void* GetPC(void* ucontext_in_void) {
#  if (defined(HAVE_UCONTEXT_H) || defined(HAVE_SYS_UCONTEXT_H)) && \
      defined(PC_FROM_UCONTEXT)
  if (ucontext_in_void != nullptr) {
    ucontext_t* context = reinterpret_cast<ucontext_t*>(ucontext_in_void);
    return (void*)context->PC_FROM_UCONTEXT;
  }
#  else
  (void)ucontext_in_void;
#  endif
  return nullptr;
}
#endif
class MinimalFormatter {
 public:
  MinimalFormatter(char* buffer, size_t size)
      : buffer_(buffer), cursor_(buffer), end_(buffer + size) {}
  std::size_t num_bytes_written() const {
    return static_cast<std::size_t>(cursor_ - buffer_);
  }
  void AppendString(const char* str) {
    ptrdiff_t i = 0;
    while (str[i] != '\0' && cursor_ + i < end_) {
      cursor_[i] = str[i];
      ++i;
    }
    cursor_ += i;
  }
  void AppendUint64(uint64 number, unsigned radix) {
    unsigned i = 0;
    while (cursor_ + i < end_) {
      const uint64 tmp = number % radix;
      number /= radix;
      cursor_[i] = static_cast<char>(tmp < 10 ? '0' + tmp : 'a' + tmp - 10);
      ++i;
      if (number == 0) {
        break;
      }
    }
    std::reverse(cursor_, cursor_ + i);
    cursor_ += i;
  }
  void AppendHexWithPadding(uint64 number, int width) {
    char* start = cursor_;
    AppendString("0x");
    AppendUint64(number, 16);
    if (cursor_ < start + width) {
      const int64 delta = start + width - cursor_;
      std::copy(start, cursor_, start + delta);
      std::fill(start, start + delta, ' ');
      cursor_ = start + width;
    }
  }
 private:
  char* buffer_;
  char* cursor_;
  const char* const end_;
};
void WriteToStderr(const char* data, size_t size) {
  if (write(fileno(stderr), data, size) < 0) {
  }
}
void (*g_failure_writer)(const char* data, size_t size) = WriteToStderr;
void DumpTimeInfo() {
  time_t time_in_sec = time(nullptr);
  char buf[256];  
  MinimalFormatter formatter(buf, sizeof(buf));
  formatter.AppendString("*** Aborted at ");
  formatter.AppendUint64(static_cast<uint64>(time_in_sec), 10);
  formatter.AppendString(" (unix time)");
  formatter.AppendString(" try \"date -d @");
  formatter.AppendUint64(static_cast<uint64>(time_in_sec), 10);
  formatter.AppendString("\" if you are using GNU date ***\n");
  g_failure_writer(buf, formatter.num_bytes_written());
}
#if defined(HAVE_STACKTRACE) && defined(HAVE_SIGACTION)
void DumpSignalInfo(int signal_number, siginfo_t* siginfo) {
  const char* signal_name = nullptr;
  for (auto kFailureSignal : kFailureSignals) {
    if (signal_number == kFailureSignal.number) {
      signal_name = kFailureSignal.name;
    }
  }
  char buf[256];  
  MinimalFormatter formatter(buf, sizeof(buf));
  formatter.AppendString("*** ");
  if (signal_name) {
    formatter.AppendString(signal_name);
  } else {
    formatter.AppendString("Signal ");
    formatter.AppendUint64(static_cast<uint64>(signal_number), 10);
  }
  formatter.AppendString(" (@0x");
  formatter.AppendUint64(reinterpret_cast<uintptr_t>(siginfo->si_addr), 16);
  formatter.AppendString(")");
  formatter.AppendString(" received by PID ");
  formatter.AppendUint64(static_cast<uint64>(getpid()), 10);
  formatter.AppendString(" (TID ");
  std::ostringstream oss;
  oss << std::showbase << std::hex << std::this_thread::get_id();
  formatter.AppendString(oss.str().c_str());
  formatter.AppendString(") ");
#  ifdef GLOG_OS_LINUX
  formatter.AppendString("from PID ");
  formatter.AppendUint64(static_cast<uint64>(siginfo->si_pid), 10);
  formatter.AppendString("; ");
#  endif
  formatter.AppendString("stack trace: ***\n");
  g_failure_writer(buf, formatter.num_bytes_written());
}
#endif  
void DumpStackFrameInfo(const char* prefix, void* pc) {
  const char* symbol = "(unknown)";
#if defined(HAVE_SYMBOLIZE)
  char symbolized[1024];  
  if (Symbolize(reinterpret_cast<char*>(pc) - 1, symbolized,
                sizeof(symbolized))) {
    symbol = symbolized;
  }
#else
#  pragma message( \
          "Symbolize functionality is not available for target platform: stack dump will contain empty frames.")
#endif  
  char buf[1024];  
  MinimalFormatter formatter(buf, sizeof(buf));
  formatter.AppendString(prefix);
  formatter.AppendString("@ ");
  const int width = 2 * sizeof(void*) + 2;  
  formatter.AppendHexWithPadding(reinterpret_cast<uintptr_t>(pc), width);
  formatter.AppendString(" ");
  formatter.AppendString(symbol);
  formatter.AppendString("\n");
  g_failure_writer(buf, formatter.num_bytes_written());
}
void InvokeDefaultSignalHandler(int signal_number) {
#ifdef HAVE_SIGACTION
  struct sigaction sig_action;
  memset(&sig_action, 0, sizeof(sig_action));
  sigemptyset(&sig_action.sa_mask);
  sig_action.sa_handler = SIG_DFL;
  sigaction(signal_number, &sig_action, nullptr);
  kill(getpid(), signal_number);
#elif defined(GLOG_OS_WINDOWS)
  signal(signal_number, SIG_DFL);
  raise(signal_number);
#endif
}
static std::once_flag signaled;
static void HandleSignal(int signal_number
#if !defined(GLOG_OS_WINDOWS)
                         ,
                         siginfo_t* signal_info, void* ucontext
#endif
) {
  DumpTimeInfo();
#if !defined(GLOG_OS_WINDOWS)
  void* pc = GetPC(ucontext);
  DumpStackFrameInfo("PC: ", pc);
#endif
#ifdef HAVE_STACKTRACE
  void* stack[32];
  const int depth = GetStackTrace(stack, ARRAYSIZE(stack), 1);
#  ifdef HAVE_SIGACTION
  DumpSignalInfo(signal_number, signal_info);
#  elif !defined(GLOG_OS_WINDOWS)
  (void)signal_info;
#  endif
  for (int i = 0; i < depth; ++i) {
    DumpStackFrameInfo("    ", stack[i]);
  }
#elif !defined(GLOG_OS_WINDOWS)
  (void)signal_info;
#endif
  FlushLogFilesUnsafe(GLOG_INFO);
  InvokeDefaultSignalHandler(signal_number);
}
#if defined(GLOG_OS_WINDOWS)
void FailureSignalHandler(int signal_number)
#else
void FailureSignalHandler(int signal_number, siginfo_t* signal_info,
                          void* ucontext)
#endif
{
  std::call_once(signaled, &HandleSignal, signal_number
#if !defined(GLOG_OS_WINDOWS)
                 ,
                 signal_info, ucontext
#endif
  );
}
}  
bool IsFailureSignalHandlerInstalled() {
#ifdef HAVE_SIGACTION
  struct sigaction sig_action;
  memset(&sig_action, 0, sizeof(sig_action));
  sigemptyset(&sig_action.sa_mask);
  sigaction(SIGABRT, nullptr, &sig_action);
  if (sig_action.sa_sigaction == &FailureSignalHandler) {
    return true;
  }
#elif defined(GLOG_OS_WINDOWS)
  return kFailureSignalHandlerInstalled;
#endif  
  return false;
}
void InstallFailureSignalHandler() {
#ifdef HAVE_SIGACTION
  struct sigaction sig_action;
  memset(&sig_action, 0, sizeof(sig_action));
  sigemptyset(&sig_action.sa_mask);
  sig_action.sa_flags |= SA_SIGINFO;
  sig_action.sa_sigaction = &FailureSignalHandler;
  for (auto kFailureSignal : kFailureSignals) {
    CHECK_ERR(sigaction(kFailureSignal.number, &sig_action, nullptr));
  }
  kFailureSignalHandlerInstalled = true;
#elif defined(GLOG_OS_WINDOWS)
  for (size_t i = 0; i < ARRAYSIZE(kFailureSignals); ++i) {
    CHECK_NE(signal(kFailureSignals[i].number, &FailureSignalHandler), SIG_ERR);
  }
  kFailureSignalHandlerInstalled = true;
#endif  
}
void InstallFailureWriter(void (*writer)(const char* data, size_t size)) {
#if defined(HAVE_SIGACTION) || defined(GLOG_OS_WINDOWS)
  g_failure_writer = writer;
#endif  
}
}  