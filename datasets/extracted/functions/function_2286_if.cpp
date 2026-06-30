#include "common/native_type.h"
#include <cstddef>
#include <cstdint>  
#include <cstdlib>
#include <memory>
#include <string>
#include "absl/base/casts.h"       
#include "absl/strings/str_cat.h"  
#ifdef CEL_INTERNAL_HAVE_RTTI
#ifdef _WIN32
extern "C" char* __unDName(char*, const char*, int, void* (*)(size_t),
                           void (*)(void*), int);
#else
#include <cxxabi.h>
#endif
#endif
namespace cel {
namespace {
#ifdef CEL_INTERNAL_HAVE_RTTI
struct FreeDeleter {
  void operator()(char* ptr) const { std::free(ptr); }
};
#endif
}  
std::string NativeTypeId::DebugString() const {
  if (rep_ == nullptr) {
    return std::string();
  }
#ifdef CEL_INTERNAL_HAVE_RTTI
#ifdef _WIN32
  std::unique_ptr<char, FreeDeleter> demangled(
      __unDName(nullptr, rep_->raw_name(), 0, std::malloc, std::free, 0x2800));
  if (demangled == nullptr) {
    return std::string(rep_->name());
  }
  return std::string(demangled.get());
#else
  size_t length = 0;
  int status = 0;
  std::unique_ptr<char, FreeDeleter> demangled(
      abi::__cxa_demangle(rep_->name(), nullptr, &length, &status));
  if (status != 0 || demangled == nullptr) {
    return std::string(rep_->name());
  }
  while (length != 0 && demangled.get()[length - 1] == '\0') {
    --length;
  }
  return std::string(demangled.get(), length);
#endif
#else
  return absl::StrCat("0x", absl::Hex(absl::bit_cast<uintptr_t>(rep_)));
#endif
}
}  