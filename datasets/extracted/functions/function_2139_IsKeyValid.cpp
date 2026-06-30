#include "tensorstore/kvstore/file/util.h"
#include <stddef.h>
#include <string_view>
#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "tensorstore/kvstore/key_range.h"
namespace tensorstore {
namespace internal_file_util {
bool IsKeyValid(std::string_view key, std::string_view lock_suffix) {
  if (absl::StrContains(key, '\0')) return false;
  if (key.empty()) return false;
  if (key.back() == '/' || key.back() == '\\') {
    return false;
  }
  if (key.front() == '/' || key.front() == '\\') {
    key = key.substr(1);
  }
  for (std::string_view component :
       absl::StrSplit(key, absl::ByAnyChar("/\\"))) {
    if (component.empty()) return false;
    if (component == ".") return false;
    if (component == "..") return false;
    if (!lock_suffix.empty() && component.size() >= lock_suffix.size() &&
        absl::EndsWith(component, lock_suffix)) {
      return false;
    }
  }
  return true;
}
std::string_view LongestDirectoryPrefix(const KeyRange& range) {
  std::string_view prefix = tensorstore::LongestPrefix(range);
  const size_t i = prefix.rfind('/');
  if (i == std::string_view::npos) return {};
  return prefix.substr(0, i);
}
}  
}  