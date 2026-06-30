#include "quiche/http2/hpack/decoder/hpack_string_decoder.h"
#include <ostream>
#include <string>
#include "absl/strings/str_cat.h"
namespace http2 {
std::string HpackStringDecoder::DebugString() const {
  return absl::StrCat("HpackStringDecoder(state=", StateToString(state_),
                      ", length=", length_decoder_.DebugString(),
                      ", remaining=", remaining_,
                      ", huffman=", huffman_encoded_ ? "true)" : "false)");
}
std::string HpackStringDecoder::StateToString(StringDecoderState v) {
  switch (v) {
    case kStartDecodingLength:
      return "kStartDecodingLength";
    case kDecodingString:
      return "kDecodingString";
    case kResumeDecodingLength:
      return "kResumeDecodingLength";
  }
  return absl::StrCat("UNKNOWN_STATE(", static_cast<uint32_t>(v), ")");
}
std::ostream& operator<<(std::ostream& out, const HpackStringDecoder& v) {
  return out << v.DebugString();
}
}  