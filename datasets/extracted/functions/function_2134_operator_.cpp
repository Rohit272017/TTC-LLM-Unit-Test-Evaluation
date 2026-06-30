#include "tensorstore/kvstore/byte_range.h"
#include <cassert>
#include <optional>
#include <ostream>
#include <string>
#include "absl/status/status.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/std_optional.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
std::ostream& operator<<(std::ostream& os, const OptionalByteRangeRequest& r) {
  os << "[" << r.inclusive_min << ", ";
  if (r.exclusive_max != -1) {
    os << r.exclusive_max;
  } else {
    os << "?";
  }
  os << ")";
  return os;
}
std::ostream& operator<<(std::ostream& os, const ByteRange& r) {
  return os << "[" << r.inclusive_min << ", " << r.exclusive_max << ")";
}
Result<ByteRange> OptionalByteRangeRequest::Validate(int64_t size) const {
  assert(SatisfiesInvariants());
  int64_t inclusive_min = this->inclusive_min;
  int64_t exclusive_max = this->exclusive_max;
  if (exclusive_max == -1) exclusive_max = size;
  if (inclusive_min < 0) {
    inclusive_min += size;
  }
  if (inclusive_min < 0 || exclusive_max > size ||
      inclusive_min > exclusive_max) {
    return absl::OutOfRangeError(
        tensorstore::StrCat("Requested byte range ", *this,
                            " is not valid for value of size ", size));
  }
  return ByteRange{inclusive_min, exclusive_max};
}
}  
TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::ByteRange, tensorstore::serialization::ApplyMembersSerializer<
                                tensorstore::ByteRange>())
TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::OptionalByteRangeRequest,
    tensorstore::serialization::ApplyMembersSerializer<
        tensorstore::OptionalByteRangeRequest>())