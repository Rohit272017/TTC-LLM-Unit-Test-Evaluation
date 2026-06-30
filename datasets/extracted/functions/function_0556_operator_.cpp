#include "tensorstore/resize_options.h"
#include <stddef.h>
#include <ostream>
#include "absl/base/macros.h"
namespace tensorstore {
std::ostream& operator<<(std::ostream& os, ResolveBoundsMode mode) {
  constexpr const char* kModeNames[] = {
      "fix_resizable_bounds",
  };
  const char* sep = "";
  constexpr const char* kSep = "|";
  for (size_t i = 0; i < ABSL_ARRAYSIZE(kModeNames); ++i) {
    if (static_cast<int>(mode) & (1 << i)) {
      os << sep << kModeNames[i];
      sep = kSep;
    }
  }
  return os;
}
std::ostream& operator<<(std::ostream& os, ResizeMode mode) {
  constexpr const char* kModeNames[] = {
      "resize_metadata_only",
      "resize_tied_bounds",
      "expand_only",
      "shrink_only",
  };
  const char* sep = "";
  constexpr const char* kSep = "|";
  for (size_t i = 0; i < ABSL_ARRAYSIZE(kModeNames); ++i) {
    if (static_cast<int>(mode) & (1 << i)) {
      os << sep << kModeNames[i];
      sep = kSep;
    }
  }
  return os;
}
}  