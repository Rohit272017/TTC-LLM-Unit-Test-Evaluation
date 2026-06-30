#include "tensorstore/internal/env.h"
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <processenv.h>
#endif
#include <stddef.h>
#include <cstdlib>
#include <cstring>
#include <memory>  
#include <optional>
#include <string>
#include "absl/container/flat_hash_map.h"
#ifndef _WIN32
extern char** environ;
#endif
namespace tensorstore {
namespace internal {
absl::flat_hash_map<std::string, std::string> GetEnvironmentMap() {
  absl::flat_hash_map<std::string, std::string> result;
#if _WIN32
  char* envblock = GetEnvironmentStrings();
  for (auto p = envblock; *p; ) {
    if (const char* eq = strchr(p, '=')) {
      result[std::string(p, eq - p)] = eq + 1;
    }
    p += strlen(p) + 1;
  }
  FreeEnvironmentStrings(envblock);
#else
  for (auto p = environ; *p; ++p) {
    if (const char* eq = strchr(*p, '=')) {
      result[std::string(*p, eq - *p)] = eq + 1;
    }
  }
#endif
  return result;
}
std::optional<std::string> GetEnv(char const* variable) {
#if _WIN32
  char* buffer;
  size_t size;
  _dupenv_s(&buffer, &size, variable);
  std::unique_ptr<char, decltype(&free)> release(buffer, &free);
#else
  char* buffer = std::getenv(variable);
#endif  
  if (buffer == nullptr) {
    return std::optional<std::string>();
  }
  return std::optional<std::string>(std::string{buffer});
}
void SetEnv(const char* variable, const char* value) {
#if _WIN32
  ::_putenv_s(variable, value);
#else
  ::setenv(variable, value, 1);
#endif
}
void UnsetEnv(const char* variable) {
#if _WIN32
  ::_putenv_s(variable, "");
#else
  ::unsetenv(variable);
#endif
}
}  
}  