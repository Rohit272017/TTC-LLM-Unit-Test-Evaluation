#include "tsl/platform/windows/stacktrace.h"
#include <windows.h>  
#include <dbghelp.h>
#include <string>
#include "tsl/platform/mutex.h"
#pragma comment(lib, "dbghelp.lib")
namespace tsl {
static bool SymbolsAreAvailableInit() {
  SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
  return SymInitialize(GetCurrentProcess(), NULL, true);
}
static bool SymbolsAreAvailable() {
  static bool kSymbolsAvailable = SymbolsAreAvailableInit();  
  return kSymbolsAvailable;
}
std::string CurrentStackTrace() {
  HANDLE current_process = GetCurrentProcess();
  static constexpr int kMaxStackFrames = 64;
  void* trace[kMaxStackFrames];
  int num_frames = CaptureStackBackTrace(0, kMaxStackFrames, trace, NULL);
  static mutex mu(tsl::LINKER_INITIALIZED);
  std::string stacktrace;
  for (int i = 0; i < num_frames; ++i) {
    const char* symbol = "(unknown)";
    if (SymbolsAreAvailable()) {
      char symbol_info_buffer[sizeof(SYMBOL_INFO) +
                              MAX_SYM_NAME * sizeof(TCHAR)];
      SYMBOL_INFO* symbol_ptr =
          reinterpret_cast<SYMBOL_INFO*>(symbol_info_buffer);
      symbol_ptr->SizeOfStruct = sizeof(SYMBOL_INFO);
      symbol_ptr->MaxNameLen = MAX_SYM_NAME;
      mutex_lock lock(mu);
      if (SymFromAddr(current_process, reinterpret_cast<DWORD64>(trace[i]), 0,
                      symbol_ptr)) {
        symbol = symbol_ptr->Name;
      }
    }
    char buffer[256];
    snprintf(buffer, sizeof(buffer), "0x%p\t%s", trace[i], symbol);
    stacktrace += buffer;
    stacktrace += "\n";
  }
  return stacktrace;
}
}  