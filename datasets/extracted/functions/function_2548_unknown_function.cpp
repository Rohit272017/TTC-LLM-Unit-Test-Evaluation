#include "arolla/array/array.h"
#include "absl/strings/str_cat.h"
#include "arolla/util/fingerprint.h"
#include "arolla/util/repr.h"
namespace arolla {
void FingerprintHasherTraits<ArrayShape>::operator()(
    FingerprintHasher* hasher, const ArrayShape& value) const {
  hasher->Combine(value.size);
}
ReprToken ReprTraits<ArrayShape>::operator()(const ArrayShape& value) const {
  return ReprToken{absl::StrCat("array_shape{size=", value.size, "}")};
}
}  