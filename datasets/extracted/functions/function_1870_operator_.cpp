#include "arolla/expr/expr_attributes.h"
#include <ostream>
#include "arolla/util/fingerprint.h"
namespace arolla::expr {
std::ostream& operator<<(std::ostream& ostream, const ExprAttributes& attr) {
  if (attr.qvalue()) {
    ostream << "Attr(qvalue=" << attr.qvalue()->Repr() << ")";
  } else if (attr.qtype()) {
    ostream << "Attr(qtype=" << attr.qtype()->name() << ")";
  } else {
    ostream << "Attr{}";
  }
  return ostream;
}
}  
namespace arolla {
void FingerprintHasherTraits<expr::ExprAttributes>::operator()(
    FingerprintHasher* hasher, const expr::ExprAttributes& attr) const {
  hasher->Combine(attr.qtype());
  hasher->Combine(attr.qvalue().has_value() ? attr.qvalue()->GetFingerprint()
                                            : Fingerprint{});
}
}  