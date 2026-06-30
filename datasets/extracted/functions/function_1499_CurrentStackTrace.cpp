#ifndef QUICHE_COMMON_PLATFORM_API_QUICHE_STACK_TRACE_H_
#define QUICHE_COMMON_PLATFORM_API_QUICHE_STACK_TRACE_H_
#include <string>
#include <vector>
#include "absl/types/span.h"
#include "quiche_platform_impl/quiche_stack_trace_impl.h"
namespace quiche {
inline std::vector<void*> CurrentStackTrace() {
  return CurrentStackTraceImpl();
}
inline std::string SymbolizeStackTrace(absl::Span<void* const> stacktrace) {
  return SymbolizeStackTraceImpl(stacktrace);
}
inline std::string QuicheStackTrace() { return QuicheStackTraceImpl(); }
inline bool QuicheShouldRunStackTraceTest() {
  return QuicheShouldRunStackTraceTestImpl();
}
}  
#endif  