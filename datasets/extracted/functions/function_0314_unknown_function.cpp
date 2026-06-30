#include "py/arolla/examples/my_complex/my_complex_type.h"
#include "absl/strings/str_format.h"
#include "arolla/qtype/simple_qtype.h"
#include "arolla/util/fingerprint.h"
#include "arolla/util/repr.h"
namespace arolla {
void FingerprintHasherTraits<my_complex::MyComplex>::operator()(
    FingerprintHasher* hasher, const my_complex::MyComplex& value) const {
  hasher->Combine(value.im, value.re);
}
ReprToken ReprTraits<my_complex::MyComplex>::operator()(
    const my_complex::MyComplex& value) const {
  return ReprToken{absl::StrFormat("%v + %vi", value.re, value.im)};
}
AROLLA_DEFINE_SIMPLE_QTYPE(MY_COMPLEX, my_complex::MyComplex);
}  