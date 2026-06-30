#include "absl/status/status.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/status.h"
namespace tensorstore {
namespace serialization {
bool ErrorStatusSerializer::Encode(EncodeSink& sink,
                                   const absl::Status& status) {
  assert(!status.ok());
  return serialization::Encode(sink, status);
}
bool ErrorStatusSerializer::Decode(DecodeSource& source, absl::Status& status) {
  if (!serialization::Decode(source, status)) return false;
  if (status.ok()) {
    source.Fail(absl::DataLossError("Expected error status"));
    return false;
  }
  return true;
}
bool Serializer<absl::Status>::Encode(EncodeSink& sink,
                                      const absl::Status& value) {
  if (!serialization::Encode(sink, value.code())) return false;
  if (value.ok()) return true;
  if (!serialization::Encode(sink, value.message())) return false;
  bool ok = true;
  value.ForEachPayload([&](std::string_view url, const absl::Cord& payload) {
    if (!ok) return;
    ok = serialization::EncodeTuple(sink, true, payload, url);
  });
  if (!ok) return false;
  return serialization::Encode(sink, false);
}
bool Serializer<absl::Status>::Decode(DecodeSource& source,
                                      absl::Status& value) {
  absl::StatusCode code;
  if (!serialization::Decode(source, code)) return false;
  if (code == absl::StatusCode::kOk) {
    value = absl::OkStatus();
    return true;
  }
  std::string_view message;
  if (!serialization::Decode(source, message)) return false;
  value = absl::Status(code, message);
  while (true) {
    bool has_payload;
    if (!serialization::Decode(source, has_payload)) return false;
    if (!has_payload) break;
    absl::Cord payload;
    std::string_view url;
    if (!serialization::DecodeTuple(source, payload, url)) return false;
    value.SetPayload(url, payload);
  }
  return true;
}
}  
}  