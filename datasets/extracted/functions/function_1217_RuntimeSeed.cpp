#include "arolla/util/fingerprint.h"
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include "absl/hash/hash.h"
#include "absl/numeric/int128.h"
#include "absl/random/random.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "cityhash/city.h"
#include "arolla/util/types.h"
namespace arolla {
namespace {
uint32_t RuntimeSeed() {
  static uint32_t result = absl::Hash<int>{}(501816262);
  return result;
}
}  
std::string Fingerprint::AsString() const {
  return absl::StrFormat("%032x", value);
}
signed_size_t Fingerprint::PythonHash() const {
  return absl::Hash<Fingerprint>()(*this);
}
std::ostream& operator<<(std::ostream& ostream,
                         const Fingerprint& fingerprint) {
  return ostream << absl::StreamFormat("%032x", fingerprint.value);
}
Fingerprint RandomFingerprint() {
  absl::BitGen bitgen;
  return Fingerprint{absl::MakeUint128(absl::Uniform<uint64_t>(bitgen),
                                       absl::Uniform<uint64_t>(bitgen))};
}
FingerprintHasher::FingerprintHasher(absl::string_view salt)
    : state_{3102879407, 2758948377}  
{
  Combine(RuntimeSeed(), salt);
}
Fingerprint FingerprintHasher::Finish() && {
  return Fingerprint{absl::MakeUint128(state_.second, state_.first)};
}
void FingerprintHasher::CombineRawBytes(const void* data, size_t size) {
  state_ = cityhash::CityHash128WithSeed(
      static_cast<const char*>(data), size, state_);
}
}  