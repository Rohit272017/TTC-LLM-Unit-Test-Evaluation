#include "tensorstore/serialization/serialization.h"
#include <cstring>
#include <string_view>
#include "absl/status/status.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
namespace serialization {
namespace internal_serialization {
void FailNonNull(DecodeSource& source) {
  source.Fail(serialization::DecodeError("Expected non-null value"));
}
void FailEof(DecodeSource& source) {
  source.Fail(serialization::DecodeError("Unexpected end of input"));
}
}  
void EncodeSink::Fail(absl::Status status) {
  assert(!status.ok());
  writer().Fail(std::move(status));
}
void DecodeSource::Fail(absl::Status status) {
  assert(!status.ok());
  reader().Fail(std::move(status));
}
absl::Status DecodeError() {
  return absl::DataLossError("Failed to decode value");
}
absl::Status DecodeError(std::string_view message) {
  return absl::DataLossError(tensorstore::StrCat("Error decoding: ", message));
}
namespace internal_serialization {
absl::Status NonSerializableError() {
  return absl::InvalidArgumentError("Serialization not supported");
}
}  
}  
}  