#include "arolla/sequence/sequence.h"
#include <algorithm>
#include <cstddef>
#include <sstream>
#include <utility>
#include "arolla/qtype/qtype.h"
#include "arolla/util/fingerprint.h"
#include "arolla/util/repr.h"
namespace arolla {
void FingerprintHasherTraits<Sequence>::operator()(
    FingerprintHasher* hasher, const Sequence& sequence) const {
  const QTypePtr value_qtype = sequence.value_qtype();
  const size_t value_byte_size = value_qtype->type_layout().AllocSize();
  hasher->Combine(value_qtype, sequence.size());
  for (size_t i = 0; i < sequence.size(); ++i) {
    value_qtype->UnsafeCombineToFingerprintHasher(
        sequence.RawAt(i, value_byte_size), hasher);
  }
}
ReprToken ReprTraits<Sequence>::operator()(const Sequence& sequence) const {
  std::ostringstream result;
  result << "sequence(";
  const auto n = std::min<size_t>(sequence.size(), 10);
  for (size_t i = 0; i < n; ++i) {
    result << sequence.GetRef(i).Repr() << ", ";
  }
  if (n < sequence.size()) {
    result << "..., size=" << sequence.size() << ", ";
  }
  result << "value_qtype=" << sequence.value_qtype()->name() << ")";
  return ReprToken{std::move(result).str()};
}
}  