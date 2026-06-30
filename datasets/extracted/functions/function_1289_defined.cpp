#include "xla/tsl/lib/hash/crc32c.h"
#include <stdint.h>
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/types.h"
namespace tsl {
namespace crc32c {
#if defined(TF_CORD_SUPPORT)
uint32 Extend(uint32 crc, const absl::Cord &cord) {
  for (absl::string_view fragment : cord.Chunks()) {
    crc = Extend(crc, fragment.data(), fragment.size());
  }
  return crc;
}
#endif
}  
}  