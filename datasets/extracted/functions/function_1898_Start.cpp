#include "quiche/http2/hpack/varint/hpack_varint_decoder.h"
#include <limits>
#include <string>
#include "absl/strings/str_cat.h"
namespace http2 {
DecodeStatus HpackVarintDecoder::Start(uint8_t prefix_value,
                                       uint8_t prefix_length,
                                       DecodeBuffer* db) {
  QUICHE_DCHECK_LE(3u, prefix_length);
  QUICHE_DCHECK_LE(prefix_length, 8u);
  const uint8_t prefix_mask = (1 << prefix_length) - 1;
  value_ = prefix_value & prefix_mask;
  if (value_ < prefix_mask) {
    MarkDone();
    return DecodeStatus::kDecodeDone;
  }
  offset_ = 0;
  return Resume(db);
}
DecodeStatus HpackVarintDecoder::StartExtended(uint8_t prefix_length,
                                               DecodeBuffer* db) {
  QUICHE_DCHECK_LE(3u, prefix_length);
  QUICHE_DCHECK_LE(prefix_length, 8u);
  value_ = (1 << prefix_length) - 1;
  offset_ = 0;
  return Resume(db);
}
DecodeStatus HpackVarintDecoder::Resume(DecodeBuffer* db) {
  const uint8_t kMaxOffset = 63;
  CheckNotDone();
  while (offset_ < kMaxOffset) {
    if (db->Empty()) {
      return DecodeStatus::kDecodeInProgress;
    }
    uint8_t byte = db->DecodeUInt8();
    uint64_t summand = byte & 0x7f;
    QUICHE_DCHECK_LE(offset_, 56);
    QUICHE_DCHECK_LE(summand, std::numeric_limits<uint64_t>::max() >> offset_);
    summand <<= offset_;
    QUICHE_DCHECK_LE(value_, std::numeric_limits<uint64_t>::max() - summand);
    value_ += summand;
    if ((byte & 0x80) == 0) {
      MarkDone();
      return DecodeStatus::kDecodeDone;
    }
    offset_ += 7;
  }
  if (db->Empty()) {
    return DecodeStatus::kDecodeInProgress;
  }
  QUICHE_DCHECK_EQ(kMaxOffset, offset_);
  uint8_t byte = db->DecodeUInt8();
  if ((byte & 0x80) == 0) {
    uint64_t summand = byte & 0x7f;
    if (summand <= std::numeric_limits<uint64_t>::max() >> offset_) {
      summand <<= offset_;
      if (value_ <= std::numeric_limits<uint64_t>::max() - summand) {
        value_ += summand;
        MarkDone();
        return DecodeStatus::kDecodeDone;
      }
    }
  }
  QUICHE_DLOG(WARNING)
      << "Variable length int encoding is too large or too long. "
      << DebugString();
  MarkDone();
  return DecodeStatus::kDecodeError;
}
uint64_t HpackVarintDecoder::value() const {
  CheckDone();
  return value_;
}
void HpackVarintDecoder::set_value(uint64_t v) {
  MarkDone();
  value_ = v;
}
std::string HpackVarintDecoder::DebugString() const {
  return absl::StrCat("HpackVarintDecoder(value=", value_, ", offset=", offset_,
                      ")");
}
DecodeStatus HpackVarintDecoder::StartForTest(uint8_t prefix_value,
                                              uint8_t prefix_length,
                                              DecodeBuffer* db) {
  return Start(prefix_value, prefix_length, db);
}
DecodeStatus HpackVarintDecoder::StartExtendedForTest(uint8_t prefix_length,
                                                      DecodeBuffer* db) {
  return StartExtended(prefix_length, db);
}
DecodeStatus HpackVarintDecoder::ResumeForTest(DecodeBuffer* db) {
  return Resume(db);
}
}  