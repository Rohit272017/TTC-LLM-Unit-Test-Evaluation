#include "tensorstore/internal/digest/sha256.h"
#include <string_view>
#include "absl/strings/cord.h"
namespace tensorstore {
namespace internal {
void SHA256Digester::Write(const absl::Cord& cord) {
  for (std::string_view chunk : cord.Chunks()) {
    Write(chunk);
  }
}
}  
}  