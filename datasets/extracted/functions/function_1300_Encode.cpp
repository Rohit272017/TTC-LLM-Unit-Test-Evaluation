#include "tensorstore/serialization/absl_time.h"
#include <cstdint>
#include <limits>
#include "absl/time/time.h"
#include "tensorstore/serialization/serialization.h"
namespace tensorstore {
namespace serialization {
bool Serializer<absl::Duration>::Encode(EncodeSink& sink,
                                        const absl::Duration& value) {
  int64_t rep_hi = absl::time_internal::GetRepHi(value);
  uint32_t rep_lo = absl::time_internal::GetRepLo(value);
  return serialization::EncodeTuple(sink, rep_hi, rep_lo);
}
bool Serializer<absl::Duration>::Decode(DecodeSource& source,
                                        absl::Duration& value) {
  int64_t rep_hi;
  uint32_t rep_lo;
  using absl::time_internal::kTicksPerSecond;
  if (!serialization::DecodeTuple(source, rep_hi, rep_lo)) return false;
  if (rep_lo >= kTicksPerSecond &&
      (rep_lo != std::numeric_limits<uint32_t>::max() ||
       (rep_hi != std::numeric_limits<int64_t>::min() &&
        rep_hi != std::numeric_limits<int64_t>::max()))) {
    source.Fail(serialization::DecodeError("Invalid time representation"));
    return false;
  }
  value = absl::time_internal::MakeDuration(rep_hi, rep_lo);
  return true;
}
bool Serializer<absl::Time>::Encode(EncodeSink& sink, const absl::Time& value) {
  return serialization::Encode(sink, value - absl::UnixEpoch());
}
bool Serializer<absl::Time>::Decode(DecodeSource& source, absl::Time& value) {
  absl::Duration d;
  if (!serialization::Decode(source, d)) return false;
  value = absl::UnixEpoch() + d;
  return true;
}
}  
}  