#include "quiche/http2/hpack/huffman/hpack_huffman_decoder.h"
#include <bitset>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include "quiche/common/platform/api/quiche_logging.h"
namespace http2 {
namespace {
typedef uint32_t HuffmanCode;
typedef uint16_t HuffmanCodeBitCount;
typedef std::bitset<32> HuffmanCodeBitSet;
typedef std::bitset<64> HuffmanAccumulatorBitSet;
static constexpr HuffmanCodeBitCount kMinCodeBitCount = 5;
static constexpr HuffmanCodeBitCount kMaxCodeBitCount = 30;
static constexpr HuffmanCodeBitCount kHuffmanCodeBitCount =
    std::numeric_limits<HuffmanCode>::digits;
static_assert(std::numeric_limits<HuffmanCode>::digits >= kMaxCodeBitCount,
              "HuffmanCode isn't big enough.");
static_assert(std::numeric_limits<HuffmanAccumulator>::digits >=
                  kMaxCodeBitCount,
              "HuffmanAccumulator isn't big enough.");
static constexpr HuffmanAccumulatorBitCount kHuffmanAccumulatorBitCount =
    std::numeric_limits<HuffmanAccumulator>::digits;
static constexpr HuffmanAccumulatorBitCount kExtraAccumulatorBitCount =
    kHuffmanAccumulatorBitCount - kHuffmanCodeBitCount;
struct PrefixInfo {
  uint32_t DecodeToCanonical(HuffmanCode bits) const {
    HuffmanCode ordinal_in_length =
        ((bits - first_code) >> (kHuffmanCodeBitCount - code_length));
    return first_canonical + ordinal_in_length;
  }
  const HuffmanCode first_code;  
  const uint16_t code_length;    
  const uint16_t first_canonical;  
};
inline std::ostream& operator<<(std::ostream& out, const PrefixInfo& v) {
  return out << "{first_code: " << HuffmanCodeBitSet(v.first_code)
             << ", code_length: " << v.code_length
             << ", first_canonical: " << v.first_canonical << "}";
}
PrefixInfo PrefixToInfo(HuffmanCode value) {
  if (value < 0b10111000000000000000000000000000) {
    if (value < 0b01010000000000000000000000000000) {
      return {0b00000000000000000000000000000000, 5, 0};
    } else {
      return {0b01010000000000000000000000000000, 6, 10};
    }
  } else {
    if (value < 0b11111110000000000000000000000000) {
      if (value < 0b11111000000000000000000000000000) {
        return {0b10111000000000000000000000000000, 7, 36};
      } else {
        return {0b11111000000000000000000000000000, 8, 68};
      }
    } else {
      if (value < 0b11111111110000000000000000000000) {
        if (value < 0b11111111101000000000000000000000) {
          if (value < 0b11111111010000000000000000000000) {
            return {0b11111110000000000000000000000000, 10, 74};
          } else {
            return {0b11111111010000000000000000000000, 11, 79};
          }
        } else {
          return {0b11111111101000000000000000000000, 12, 82};
        }
      } else {
        if (value < 0b11111111111111100000000000000000) {
          if (value < 0b11111111111110000000000000000000) {
            if (value < 0b11111111111100000000000000000000) {
              return {0b11111111110000000000000000000000, 13, 84};
            } else {
              return {0b11111111111100000000000000000000, 14, 90};
            }
          } else {
            return {0b11111111111110000000000000000000, 15, 92};
          }
        } else {
          if (value < 0b11111111111111110100100000000000) {
            if (value < 0b11111111111111101110000000000000) {
              if (value < 0b11111111111111100110000000000000) {
                return {0b11111111111111100000000000000000, 19, 95};
              } else {
                return {0b11111111111111100110000000000000, 20, 98};
              }
            } else {
              return {0b11111111111111101110000000000000, 21, 106};
            }
          } else {
            if (value < 0b11111111111111111110101000000000) {
              if (value < 0b11111111111111111011000000000000) {
                return {0b11111111111111110100100000000000, 22, 119};
              } else {
                return {0b11111111111111111011000000000000, 23, 145};
              }
            } else {
              if (value < 0b11111111111111111111101111000000) {
                if (value < 0b11111111111111111111100000000000) {
                  if (value < 0b11111111111111111111011000000000) {
                    return {0b11111111111111111110101000000000, 24, 174};
                  } else {
                    return {0b11111111111111111111011000000000, 25, 186};
                  }
                } else {
                  return {0b11111111111111111111100000000000, 26, 190};
                }
              } else {
                if (value < 0b11111111111111111111111111110000) {
                  if (value < 0b11111111111111111111111000100000) {
                    return {0b11111111111111111111101111000000, 27, 205};
                  } else {
                    return {0b11111111111111111111111000100000, 28, 224};
                  }
                } else {
                  return {0b11111111111111111111111111110000, 30, 253};
                }
              }
            }
          }
        }
      }
    }
  }
}
constexpr unsigned char kCanonicalToSymbol[] = {
    '0',  '1',  '2',  'a',  'c',  'e',  'i',  'o',
    's',  't',  0x20, '%',  '-',  '.',  '/',  '3',
    '4',  '5',  '6',  '7',  '8',  '9',  '=',  'A',
    '_',  'b',  'd',  'f',  'g',  'h',  'l',  'm',
    'n',  'p',  'r',  'u',  ':',  'B',  'C',  'D',
    'E',  'F',  'G',  'H',  'I',  'J',  'K',  'L',
    'M',  'N',  'O',  'P',  'Q',  'R',  'S',  'T',
    'U',  'V',  'W',  'Y',  'j',  'k',  'q',  'v',
    'w',  'x',  'y',  'z',  '&',  '*',  ',',  ';',
    'X',  'Z',  '!',  '\"', '(',  ')',  '?',  '\'',
    '+',  '|',  '#',  '>',  0x00, '$',  '@',  '[',
    ']',  '~',  '^',  '}',  '<',  '`',  '{',  '\\',
    0xc3, 0xd0, 0x80, 0x82, 0x83, 0xa2, 0xb8, 0xc2,
    0xe0, 0xe2, 0x99, 0xa1, 0xa7, 0xac, 0xb0, 0xb1,
    0xb3, 0xd1, 0xd8, 0xd9, 0xe3, 0xe5, 0xe6, 0x81,
    0x84, 0x85, 0x86, 0x88, 0x92, 0x9a, 0x9c, 0xa0,
    0xa3, 0xa4, 0xa9, 0xaa, 0xad, 0xb2, 0xb5, 0xb9,
    0xba, 0xbb, 0xbd, 0xbe, 0xc4, 0xc6, 0xe4, 0xe8,
    0xe9, 0x01, 0x87, 0x89, 0x8a, 0x8b, 0x8c, 0x8d,
    0x8f, 0x93, 0x95, 0x96, 0x97, 0x98, 0x9b, 0x9d,
    0x9e, 0xa5, 0xa6, 0xa8, 0xae, 0xaf, 0xb4, 0xb6,
    0xb7, 0xbc, 0xbf, 0xc5, 0xe7, 0xef, 0x09, 0x8e,
    0x90, 0x91, 0x94, 0x9f, 0xab, 0xce, 0xd7, 0xe1,
    0xec, 0xed, 0xc7, 0xcf, 0xea, 0xeb, 0xc0, 0xc1,
    0xc8, 0xc9, 0xca, 0xcd, 0xd2, 0xd5, 0xda, 0xdb,
    0xee, 0xf0, 0xf2, 0xf3, 0xff, 0xcb, 0xcc, 0xd3,
    0xd4, 0xd6, 0xdd, 0xde, 0xdf, 0xf1, 0xf4, 0xf5,
    0xf6, 0xf7, 0xf8, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe,
    0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x0b,
    0x0c, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14,
    0x15, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
    0x1e, 0x1f, 0x7f, 0xdc, 0xf9, 0x0a, 0x0d, 0x16,
};
constexpr size_t kShortCodeTableSize = 124;
struct ShortCodeInfo {
  uint8_t symbol;
  uint8_t length;
} kShortCodeTable[kShortCodeTableSize] = {
    {0x30, 5},  
    {0x30, 5},  
    {0x30, 5},  
    {0x30, 5},  
    {0x31, 5},  
    {0x31, 5},  
    {0x31, 5},  
    {0x31, 5},  
    {0x32, 5},  
    {0x32, 5},  
    {0x32, 5},  
    {0x32, 5},  
    {0x61, 5},  
    {0x61, 5},  
    {0x61, 5},  
    {0x61, 5},  
    {0x63, 5},  
    {0x63, 5},  
    {0x63, 5},  
    {0x63, 5},  
    {0x65, 5},  
    {0x65, 5},  
    {0x65, 5},  
    {0x65, 5},  
    {0x69, 5},  
    {0x69, 5},  
    {0x69, 5},  
    {0x69, 5},  
    {0x6f, 5},  
    {0x6f, 5},  
    {0x6f, 5},  
    {0x6f, 5},  
    {0x73, 5},  
    {0x73, 5},  
    {0x73, 5},  
    {0x73, 5},  
    {0x74, 5},  
    {0x74, 5},  
    {0x74, 5},  
    {0x74, 5},  
    {0x20, 6},  
    {0x20, 6},  
    {0x25, 6},  
    {0x25, 6},  
    {0x2d, 6},  
    {0x2d, 6},  
    {0x2e, 6},  
    {0x2e, 6},  
    {0x2f, 6},  
    {0x2f, 6},  
    {0x33, 6},  
    {0x33, 6},  
    {0x34, 6},  
    {0x34, 6},  
    {0x35, 6},  
    {0x35, 6},  
    {0x36, 6},  
    {0x36, 6},  
    {0x37, 6},  
    {0x37, 6},  
    {0x38, 6},  
    {0x38, 6},  
    {0x39, 6},  
    {0x39, 6},  
    {0x3d, 6},  
    {0x3d, 6},  
    {0x41, 6},  
    {0x41, 6},  
    {0x5f, 6},  
    {0x5f, 6},  
    {0x62, 6},  
    {0x62, 6},  
    {0x64, 6},  
    {0x64, 6},  
    {0x66, 6},  
    {0x66, 6},  
    {0x67, 6},  
    {0x67, 6},  
    {0x68, 6},  
    {0x68, 6},  
    {0x6c, 6},  
    {0x6c, 6},  
    {0x6d, 6},  
    {0x6d, 6},  
    {0x6e, 6},  
    {0x6e, 6},  
    {0x70, 6},  
    {0x70, 6},  
    {0x72, 6},  
    {0x72, 6},  
    {0x75, 6},  
    {0x75, 6},  
    {0x3a, 7},  
    {0x42, 7},  
    {0x43, 7},  
    {0x44, 7},  
    {0x45, 7},  
    {0x46, 7},  
    {0x47, 7},  
    {0x48, 7},  
    {0x49, 7},  
    {0x4a, 7},  
    {0x4b, 7},  
    {0x4c, 7},  
    {0x4d, 7},  
    {0x4e, 7},  
    {0x4f, 7},  
    {0x50, 7},  
    {0x51, 7},  
    {0x52, 7},  
    {0x53, 7},  
    {0x54, 7},  
    {0x55, 7},  
    {0x56, 7},  
    {0x57, 7},  
    {0x59, 7},  
    {0x6a, 7},  
    {0x6b, 7},  
    {0x71, 7},  
    {0x76, 7},  
    {0x77, 7},  
    {0x78, 7},  
    {0x79, 7},  
    {0x7a, 7},  
};
}  
HuffmanBitBuffer::HuffmanBitBuffer() { Reset(); }
void HuffmanBitBuffer::Reset() {
  accumulator_ = 0;
  count_ = 0;
}
size_t HuffmanBitBuffer::AppendBytes(absl::string_view input) {
  HuffmanAccumulatorBitCount free_cnt = free_count();
  size_t bytes_available = input.size();
  if (free_cnt < 8 || bytes_available == 0) {
    return 0;
  }
  size_t bytes_used = 0;
  auto* ptr = reinterpret_cast<const uint8_t*>(input.data());
  do {
    auto b = static_cast<HuffmanAccumulator>(*ptr++);
    free_cnt -= 8;
    accumulator_ |= (b << free_cnt);
    ++bytes_used;
  } while (free_cnt >= 8 && bytes_used < bytes_available);
  count_ += (bytes_used * 8);
  return bytes_used;
}
HuffmanAccumulatorBitCount HuffmanBitBuffer::free_count() const {
  return kHuffmanAccumulatorBitCount - count_;
}
void HuffmanBitBuffer::ConsumeBits(HuffmanAccumulatorBitCount code_length) {
  QUICHE_DCHECK_LE(code_length, count_);
  accumulator_ <<= code_length;
  count_ -= code_length;
}
bool HuffmanBitBuffer::InputProperlyTerminated() const {
  auto cnt = count();
  if (cnt < 8) {
    if (cnt == 0) {
      return true;
    }
    HuffmanAccumulator expected = ~(~HuffmanAccumulator() >> cnt);
    QUICHE_DCHECK_EQ(accumulator_ & ~expected, 0u)
        << "\n  expected: " << HuffmanAccumulatorBitSet(expected) << "\n  "
        << *this;
    return accumulator_ == expected;
  }
  return false;
}
std::string HuffmanBitBuffer::DebugString() const {
  std::stringstream ss;
  ss << "{accumulator: " << HuffmanAccumulatorBitSet(accumulator_)
     << "; count: " << count_ << "}";
  return ss.str();
}
HpackHuffmanDecoder::HpackHuffmanDecoder() = default;
HpackHuffmanDecoder::~HpackHuffmanDecoder() = default;
bool HpackHuffmanDecoder::Decode(absl::string_view input, std::string* output) {
  QUICHE_DVLOG(1) << "HpackHuffmanDecoder::Decode";
  input.remove_prefix(bit_buffer_.AppendBytes(input));
  while (true) {
    QUICHE_DVLOG(3) << "Enter Decode Loop, bit_buffer_: " << bit_buffer_;
    if (bit_buffer_.count() >= 7) {
      uint8_t short_code =
          bit_buffer_.value() >> (kHuffmanAccumulatorBitCount - 7);
      QUICHE_DCHECK_LT(short_code, 128);
      if (short_code < kShortCodeTableSize) {
        ShortCodeInfo info = kShortCodeTable[short_code];
        bit_buffer_.ConsumeBits(info.length);
        output->push_back(static_cast<char>(info.symbol));
        continue;
      }
    } else {
      size_t byte_count = bit_buffer_.AppendBytes(input);
      if (byte_count > 0) {
        input.remove_prefix(byte_count);
        continue;
      }
    }
    HuffmanCode code_prefix = bit_buffer_.value() >> kExtraAccumulatorBitCount;
    QUICHE_DVLOG(3) << "code_prefix: " << HuffmanCodeBitSet(code_prefix);
    PrefixInfo prefix_info = PrefixToInfo(code_prefix);
    QUICHE_DVLOG(3) << "prefix_info: " << prefix_info;
    QUICHE_DCHECK_LE(kMinCodeBitCount, prefix_info.code_length);
    QUICHE_DCHECK_LE(prefix_info.code_length, kMaxCodeBitCount);
    if (prefix_info.code_length <= bit_buffer_.count()) {
      uint32_t canonical = prefix_info.DecodeToCanonical(code_prefix);
      if (canonical < 256) {
        char c = kCanonicalToSymbol[canonical];
        output->push_back(c);
        bit_buffer_.ConsumeBits(prefix_info.code_length);
        continue;
      }
      QUICHE_DLOG(ERROR) << "EOS explicitly encoded!\n " << bit_buffer_ << "\n "
                         << prefix_info;
      return false;
    }
    size_t byte_count = bit_buffer_.AppendBytes(input);
    if (byte_count == 0) {
      QUICHE_DCHECK_EQ(input.size(), 0u);
      return true;
    }
    input.remove_prefix(byte_count);
  }
}
std::string HpackHuffmanDecoder::DebugString() const {
  return bit_buffer_.DebugString();
}
}  