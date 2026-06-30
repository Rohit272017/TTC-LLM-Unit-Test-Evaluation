#include "quiche/http2/hpack/huffman/hpack_huffman_encoder.h"
#include <string>
#include "quiche/http2/hpack/huffman/huffman_spec_tables.h"
#include "quiche/common/platform/api/quiche_logging.h"
namespace http2 {
size_t HuffmanSize(absl::string_view plain) {
  size_t bits = 0;
  for (const uint8_t c : plain) {
    bits += HuffmanSpecTables::kCodeLengths[c];
  }
  return (bits + 7) / 8;
}
void HuffmanEncode(absl::string_view input, size_t encoded_size,
                   std::string* output) {
  const size_t original_size = output->size();
  const size_t final_size = original_size + encoded_size;
  output->resize(final_size + 4, 0);
  char* const first = &*output->begin() + original_size;
  size_t bit_counter = 0;
  for (uint8_t c : input) {
    uint64_t code = static_cast<uint64_t>(HuffmanSpecTables::kLeftCodes[c])
                    << (8 - (bit_counter % 8));
    char* const current = first + (bit_counter / 8);
    bit_counter += HuffmanSpecTables::kCodeLengths[c];
    *current |= code >> 32;
    *(current + 1) |= (code >> 24) & 0xff;
    if ((code & 0xff0000) == 0) {
      continue;
    }
    *(current + 2) |= (code >> 16) & 0xff;
    if ((code & 0xff00) == 0) {
      continue;
    }
    *(current + 3) |= (code >> 8) & 0xff;
    *(current + 4) |= code & 0xff;
  }
  QUICHE_DCHECK_EQ(encoded_size, (bit_counter + 7) / 8);
  if (bit_counter % 8 != 0) {
    *(first + encoded_size - 1) |= 0xff >> (bit_counter & 7);
  }
  output->resize(final_size);
}
}  