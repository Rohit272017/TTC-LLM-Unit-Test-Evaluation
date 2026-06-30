#include "tensorstore/internal/path.h"
#include <cstddef>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
namespace {
#ifdef _WIN32
constexpr inline bool IsDirSeparator(char c) { return c == '\\' || c == '/'; }
#else
constexpr inline bool IsDirSeparator(char c) { return c == '/'; }
#endif
}  
namespace tensorstore {
namespace internal_path {
std::string JoinPathImpl(std::initializer_list<std::string_view> paths) {
  size_t s = 0;
  for (std::string_view path : paths) {
    s += path.size() + 1;
  }
  std::string result;
  result.reserve(s);
  for (std::string_view path : paths) {
    internal::AppendPathComponent(result, path);
  }
  return result;
}
}  
namespace internal {
std::pair<std::string_view, std::string_view> PathDirnameBasename(
    std::string_view path) {
  size_t pos = path.size();
  while (pos != 0 && !IsDirSeparator(path[pos - 1])) {
    --pos;
  }
  size_t basename = pos;
  --pos;
  if (pos == std::string_view::npos) {
    return {"", path};
  }
  while (pos != 0 && IsDirSeparator(path[pos - 1])) {
    --pos;
  }
  if (pos == 0) {
    return {"/", path.substr(basename)};
  }
  return {path.substr(0, pos), path.substr(basename)};
}
void EnsureDirectoryPath(std::string& path) {
  if (path.size() == 1 && path[0] == '/') {
    path.clear();
  } else if (!path.empty() && path.back() != '/') {
    path += '/';
  }
}
void EnsureNonDirectoryPath(std::string& path) {
  size_t size = path.size();
  while (size > 0 && path[size - 1] == '/') {
    --size;
  }
  path.resize(size);
}
void AppendPathComponent(std::string& path, std::string_view component) {
  if (!path.empty() && path.back() != '/' && !component.empty() &&
      component.front() != '/') {
    absl::StrAppend(&path, "/", component);
  } else {
    path += component;
  }
}
std::string LexicalNormalizePath(std::string path) {
  if (path.empty()) return path;
  const char* src = path.c_str();
  auto dst = path.begin();
  const bool is_absolute_path = (*src == '/');
  if (is_absolute_path) {
    dst++;
    src++;
    while (*src == '/') ++src;
  }
  auto limit = dst;
  while (*src) {
    bool parsed = false;
    if (src[0] == '.') {
      if (src[1] == '/' || src[1] == '\\' || !src[1]) {
        if (*++src) {
          ++src;
        }
        parsed = true;
      } else if (src[1] == '.' &&
                 (src[2] == '/' || src[2] == '\\' || !src[2])) {
        src += 2;
        if (dst != limit) {
          for (--dst; dst != limit && dst[-1] != '/'; --dst) {
          }
        } else if (!is_absolute_path) {
          src -= 2;
          *dst++ = *src++;
          *dst++ = *src++;
          if (*src) {
            *dst++ = *src;
          }
          limit = dst;
        }
        if (*src) {
          ++src;
        }
        parsed = true;
      }
    }
    if (!parsed) {
      while (*src && *src != '/' && *src != '\\') {
        *dst++ = *src++;
      }
      if (*src) {  
        *dst++ = '/';
        src++;
      }
    }
    while (*src == '/' || *src == '\\') {
      ++src;
    }
  }
  path.resize(dst - path.begin());
  return path;
}
}  
}  