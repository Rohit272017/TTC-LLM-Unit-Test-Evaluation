#include "arolla/dense_array/dense_array.h"
#include "arolla/util/fingerprint.h"
namespace arolla {
void FingerprintHasherTraits<DenseArrayShape>::operator()(
    FingerprintHasher* hasher, const DenseArrayShape& value) const {
  hasher->Combine(value.size);
}
}  