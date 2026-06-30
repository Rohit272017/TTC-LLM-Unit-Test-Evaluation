#include "tensorstore/internal/riegeli/find.h"
#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <cstring>
#include <optional>
#include <string_view>
#include "riegeli/bytes/reader.h"
namespace tensorstore {
namespace internal {
bool StartsWith(riegeli::Reader &reader, std::string_view needle) {
  return reader.ok() &&  
         reader.Pull(needle.size()) &&
         memcmp(reader.cursor(), needle.data(), needle.size()) == 0;
}
bool FindFirst(riegeli::Reader &reader, std::string_view needle) {
  while (true) {
    if (!reader.Pull(needle.size())) break;
    auto end = reader.cursor() + reader.available();
    auto pos = std::search(reader.cursor(), end, needle.begin(), needle.end());
    if (pos != end) {
      reader.move_cursor(pos - reader.cursor());
      return true;
    }
    reader.move_cursor(1 + reader.available() - needle.size());
  }
  return false;
}
bool FindLast(riegeli::Reader &reader, std::string_view needle) {
  if (reader.SupportsSize()) {
    auto size = reader.Size();
    if (size && reader.Pull(*size)) {
      auto found_pos = std::string_view(reader.cursor(), *size).rfind(needle);
      if (found_pos == std::string_view::npos) return false;
      return reader.Seek(found_pos + reader.pos());
    }
  }
  std::optional<uint64_t> found;
  while (reader.ok()) {
    for (size_t available = reader.available(); available > needle.size();
         available = reader.available()) {
      if (memcmp(reader.cursor(), needle.data(), needle.size()) == 0) {
        found = reader.pos();
      }
      const char *pos = static_cast<const char *>(
          memchr(reader.cursor() + 1, needle[0], available - 1));
      if (pos == nullptr) {
        reader.move_cursor(available);
        break;
      }
      reader.move_cursor(pos - reader.cursor());
    }
    if (!reader.Pull(needle.size() - reader.available())) break;
  }
  return found.has_value() && reader.Seek(*found);
}
}  
}  