#include "arolla/util/unit.h"
#include "absl/strings/string_view.h"
#include "arolla/util/fingerprint.h"
#include "arolla/util/repr.h"
namespace arolla {
ReprToken ReprTraits<Unit>::operator()(const Unit&) const {
  return ReprToken{"unit"};
}
void FingerprintHasherTraits<Unit>::operator()(FingerprintHasher* hasher,
                                               const Unit& value) const {
  hasher->Combine(absl::string_view("unit"));
}
}  