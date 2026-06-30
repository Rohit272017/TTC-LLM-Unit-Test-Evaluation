#include "quiche/http2/hpack/varint/hpack_varint_encoder.h"
#include <limits>
#include <string>
#include "quiche/common/platform/api/quiche_logging.h"
namespace http2 {
void HpackVarintEncoder::Encode(uint8_t high_bits, uint8_t prefix_length,
                                uint64_t varint, std::string* output) {
  QUICHE_DCHECK_LE(1u, prefix_length);
  QUICHE_DCHECK_LE(prefix_length, 8u);
  const uint8_t prefix_mask = (1 << prefix_length) - 1;
  QUICHE_DCHECK_EQ(0, high_bits & prefix_mask);
  if (varint < prefix_mask) {
    unsigned char first_byte = high_bits | static_cast<unsigned char>(varint);
    output->push_back(first_byte);
    return;
  }
  unsigned char first_byte = high_bits | prefix_mask;
  output->push_back(first_byte);
  varint -= prefix_mask;
  while (varint >= 128) {
    output->push_back(0b10000000 | (varint % 128));
    varint >>= 7;
  }
  output->push_back(varint);
}
}  