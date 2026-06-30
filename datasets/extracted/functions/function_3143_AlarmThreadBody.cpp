#include "tsl/platform/stacktrace_handler.h"
#include <windows.h>  
#include <dbghelp.h>
#include <errno.h>
#include <io.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>  
#include "tsl/platform/mutex.h"
#include "tsl/platform/stacktrace.h"
#include "tsl/platform/types.h"
namespace tsl {
static mutex alarm_mu(LINKER_INITIALIZED);
static bool alarm_activated = false;
static void AlarmThreadBody() {
  alarm_mu.lock();
  alarm_mu.Await(Condition(&alarm_activated));
  alarm_mu.unlock();
  Sleep(60000);
  signal(SIGABRT, SIG_DFL);
  abort();
}
static bool PtrToString(uintptr_t ptr, char* buf, size_t size) {
  static constexpr char kHexCharacters[] = "0123456789abcdef";
  static constexpr int kHexBase = 16;
  size_t num_hex_chars = 2 * sizeof(uintptr_t);
  if (size < (num_hex_chars + 4)) {
    return false;
  }
  buf[0] = '0';
  buf[1] = 'x';
  int start_index = 2;
  for (int i = num_hex_chars - 1 + start_index; i >= start_index; --i) {
    buf[i] = kHexCharacters[ptr % kHexBase];
    ptr /= kHexBase;
  }
  int current_index = start_index + num_hex_chars;
  buf[current_index] = '\n';
  buf[current_index + 1] = '\0';
  return true;
}
static inline void SafePrintStackTracePointers() {
  static constexpr char begin_msg[] = "*** BEGIN STACK TRACE POINTERS ***\n";
  (void)_write(_fileno(stderr), begin_msg, strlen(begin_msg));
  static constexpr int kMaxStackFrames = 64;
  void* trace[kMaxStackFrames];
  int num_frames = CaptureStackBackTrace(0, kMaxStackFrames, trace, NULL);
  for (int i = 0; i < num_frames; ++i) {
    char buffer[32] = "unsuccessful ptr conversion";
    PtrToString(reinterpret_cast<uintptr_t>(trace[i]), buffer, sizeof(buffer));
    (void)_write(_fileno(stderr), buffer, strlen(buffer));
  }
  static constexpr char end_msg[] = "*** END STACK TRACE POINTERS ***\n\n";
  (void)_write(_fileno(stderr), end_msg, strlen(end_msg));
}
static void StacktraceHandler(int sig) {
  alarm_mu.lock();
  alarm_activated = true;
  alarm_mu.unlock();
  char buf[128];
  snprintf(buf, sizeof(buf), "*** Received signal %d ***\n", sig);
  (void)write(_fileno(stderr), buf, strlen(buf));
  SafePrintStackTracePointers();
  std::string stacktrace = CurrentStackTrace();
  (void)write(_fileno(stderr), stacktrace.c_str(), stacktrace.length());
  signal(SIGABRT, SIG_DFL);
  abort();
}
namespace testing {
void InstallStacktraceHandler() {
  int handled_signals[] = {SIGSEGV, SIGABRT, SIGILL, SIGFPE};
  std::thread alarm_thread(AlarmThreadBody);
  alarm_thread.detach();
  typedef void (*SignalHandlerPointer)(int);
  for (int sig : handled_signals) {
    SignalHandlerPointer previousHandler = signal(sig, StacktraceHandler);
    if (previousHandler == SIG_ERR) {
      char buf[128];
      snprintf(buf, sizeof(buf),
               "tensorflow::InstallStackTraceHandler: Warning, can't install "
               "backtrace signal handler for signal %d, errno:%d \n",
               sig, errno);
      (void)write(_fileno(stderr), buf, strlen(buf));
    } else if (previousHandler != SIG_DFL) {
      char buf[128];
      snprintf(buf, sizeof(buf),
               "tensorflow::InstallStackTraceHandler: Warning, backtrace "
               "signal handler for signal %d overwrote previous handler.\n",
               sig);
      (void)write(_fileno(stderr), buf, strlen(buf));
    }
  }
}
}  
}  