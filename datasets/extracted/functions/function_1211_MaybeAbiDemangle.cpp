#include "tsl/platform/abi.h"
#include "tsl/platform/types.h"
#if defined(_MSC_VER)
#include <windows.h>
#include <cstring>
#else
#include <cxxabi.h>
#include <cstdlib>
#endif
#include <memory>
#include <string>
#if defined(_MSC_VER)
extern "C" char* __unDName(char* output_string, const char* name,
                           int max_string_length, void* (*p_alloc)(std::size_t),
                           void (*p_free)(void*), unsigned short disable_flags);
#endif  
namespace tsl {
namespace port {
string MaybeAbiDemangle(const char* name) {
#if defined(_MSC_VER)
  std::unique_ptr<char> demangled{__unDName(nullptr, name, 0, std::malloc,
                                            std::free,
                                            static_cast<unsigned short>(0))};
  return string(demangled.get() != nullptr ? demangled.get() : name);
#else
  int status = 0;
  std::unique_ptr<char, void (*)(void*)> res{
      abi::__cxa_demangle(name, nullptr, nullptr, &status), std::free};
  return (status == 0) ? res.get() : name;
#endif
}
}  
}  