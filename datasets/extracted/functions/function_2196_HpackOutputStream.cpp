#include "quiche/http2/hpack/hpack_output_stream.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include "absl/strings/string_view.h"
#include "quiche/http2/hpack/hpack_constants.h"
#include "quiche/common/platform/api/quiche_logging.h"
namespace spdy {
HpackOutputStream::HpackOutputStream() : bit_offset_(0) {}
HpackOutputStream::~HpackOutputStream() = default;
void HpackOutputStream::AppendBits(uint8_t bits, size_t bit_size) {
  QUICHE_DCHECK_GT(bit_size, 0u);
  QUICHE_DCHECK_LE(bit_size, 8u);
  QUICHE_DCHECK_EQ(bits >> bit_size, 0);
  size_t new_bit_offset = bit_offset_ + bit_size;
  if (bit_offset_ == 0) {
    QUICHE_DCHECK_LE(bit_size, 8u);
    buffer_.append(1, bits << (8 - bit_size));
  } else if (new_bit_offset <= 8) {
    buffer_.back() |= bits << (8 - new_bit_offset);
  } else {
    buffer_.back() |= bits >> (new_bit_offset - 8);
    buffer_.append(1, bits << (16 - new_bit_offset));
  }
  bit_offset_ = new_bit_offset % 8;
}
void HpackOutputStream::AppendPrefix(HpackPrefix prefix) {
  AppendBits(prefix.bits, prefix.bit_size);
}
void HpackOutputStream::AppendBytes(absl::string_view buffer) {
  QUICHE_DCHECK_EQ(bit_offset_, 0u);
  buffer_.append(buffer.data(), buffer.size());
}
void HpackOutputStream::AppendUint32(uint32_t I) {
  size_t N = 8 - bit_offset_;
  uint8_t max_first_byte = static_cast<uint8_t>((1 << N) - 1);
  if (I < max_first_byte) {
    AppendBits(static_cast<uint8_t>(I), N);
  } else {
    AppendBits(max_first_byte, N);
    I -= max_first_byte;
    while ((I & ~0x7f) != 0) {
      buffer_.append(1, (I & 0x7f) | 0x80);
      I >>= 7;
    }
    AppendBits(static_cast<uint8_t>(I), 8);
  }
  QUICHE_DCHECK_EQ(bit_offset_, 0u);
}
std::string* HpackOutputStream::MutableString() {
  QUICHE_DCHECK_EQ(bit_offset_, 0u);
  return &buffer_;
}
std::string HpackOutputStream::TakeString() {
  QUICHE_DCHECK_EQ(bit_offset_, 0u);
  std::string out = std::move(buffer_);
  buffer_ = {};
  bit_offset_ = 0;
  return out;
}
std::string HpackOutputStream::BoundedTakeString(size_t max_size) {
  if (buffer_.size() > max_size) {
    std::string overflow = buffer_.substr(max_size);
    buffer_.resize(max_size);
    std::string out = std::move(buffer_);
    buffer_ = std::move(overflow);
    return out;
  } else {
    return TakeString();
  }
}
}  