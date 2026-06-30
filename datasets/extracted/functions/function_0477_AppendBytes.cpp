#include "tensorflow/core/lib/strings/ordered_code.h"
#include <assert.h>
#include <stddef.h>
#include "xla/tsl/lib/core/bits.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stringpiece.h"
namespace tensorflow {
namespace strings {
static const char kEscape1 = '\000';
static const char kNullCharacter = '\xff';  
static const char kSeparator = '\001';      
static const char kEscape2 = '\xff';
static const char kFFCharacter = '\000';  
static const char kEscape1_Separator[2] = {kEscape1, kSeparator};
inline static void AppendBytes(string* dest, const char* src, size_t len) {
  dest->append(src, len);
}
inline bool IsSpecialByte(char c) {
  return (static_cast<unsigned char>(c + 1)) < 2;
}
inline const char* SkipToNextSpecialByte(const char* start, const char* limit) {
  DCHECK_EQ(kEscape1, 0);
  DCHECK_EQ(kEscape2 & 0xffu, 255u);
  const char* p = start;
  while (p < limit && !IsSpecialByte(*p)) {
    p++;
  }
  return p;
}
const char* OrderedCode::TEST_SkipToNextSpecialByte(const char* start,
                                                    const char* limit) {
  return SkipToNextSpecialByte(start, limit);
}
inline static void EncodeStringFragment(string* dest, StringPiece s) {
  const char* p = s.data();
  const char* limit = p + s.size();
  const char* copy_start = p;
  while (true) {
    p = SkipToNextSpecialByte(p, limit);
    if (p >= limit) break;  
    char c = *(p++);
    DCHECK(IsSpecialByte(c));
    if (c == kEscape1) {
      AppendBytes(dest, copy_start, p - copy_start - 1);
      dest->push_back(kEscape1);
      dest->push_back(kNullCharacter);
      copy_start = p;
    } else {
      assert(c == kEscape2);
      AppendBytes(dest, copy_start, p - copy_start - 1);
      dest->push_back(kEscape2);
      dest->push_back(kFFCharacter);
      copy_start = p;
    }
  }
  if (p > copy_start) {
    AppendBytes(dest, copy_start, p - copy_start);
  }
}
void OrderedCode::WriteString(string* dest, StringPiece s) {
  EncodeStringFragment(dest, s);
  AppendBytes(dest, kEscape1_Separator, 2);
}
void OrderedCode::WriteNumIncreasing(string* dest, uint64 val) {
  unsigned char buf[9];  
  int len = 0;
  while (val > 0) {
    len++;
    buf[9 - len] = (val & 0xff);
    val >>= 8;
  }
  buf[9 - len - 1] = len;
  len++;
  AppendBytes(dest, reinterpret_cast<const char*>(buf + 9 - len), len);
}
inline static bool ReadStringInternal(StringPiece* src, string* result) {
  const char* start = src->data();
  const char* string_limit = src->data() + src->size();
  const char* limit = string_limit - 1;
  const char* copy_start = start;
  while (true) {
    start = SkipToNextSpecialByte(start, limit);
    if (start >= limit) break;  
    const char c = *(start++);
    DCHECK(IsSpecialByte(c));
    if (c == kEscape1) {
      if (result) {
        AppendBytes(result, copy_start, start - copy_start - 1);
      }
      const char next = *(start++);
      if (next == kSeparator) {
        src->remove_prefix(start - src->data());
        return true;
      } else if (next == kNullCharacter) {
        if (result) {
          *result += '\0';
        }
      } else {
        return false;
      }
      copy_start = start;
    } else {
      assert(c == kEscape2);
      if (result) {
        AppendBytes(result, copy_start, start - copy_start - 1);
      }
      const char next = *(start++);
      if (next == kFFCharacter) {
        if (result) {
          *result += '\xff';
        }
      } else {
        return false;
      }
      copy_start = start;
    }
  }
  return false;
}
bool OrderedCode::ReadString(StringPiece* src, string* result) {
  return ReadStringInternal(src, result);
}
bool OrderedCode::ReadNumIncreasing(StringPiece* src, uint64* result) {
  if (src->empty()) {
    return false;  
  }
  const size_t len = static_cast<unsigned char>((*src)[0]);
  DCHECK(0 == len || src->size() == 1 || (*src)[1] != '\0')
      << "invalid encoding";
  if (len + 1 > src->size() || len > 8) {
    return false;  
  }
  if (result) {
    uint64 tmp = 0;
    for (size_t i = 0; i < len; i++) {
      tmp <<= 8;
      tmp |= static_cast<unsigned char>((*src)[1 + i]);
    }
    *result = tmp;
  }
  src->remove_prefix(len + 1);
  return true;
}
void OrderedCode::TEST_Corrupt(string* str, int k) {
  int seen_seps = 0;
  for (size_t i = 0; i + 1 < str->size(); i++) {
    if ((*str)[i] == kEscape1 && (*str)[i + 1] == kSeparator) {
      seen_seps++;
      if (seen_seps == k) {
        (*str)[i + 1] = kSeparator + 1;
        return;
      }
    }
  }
}
static const int kMaxSigned64Length = 10;
static const char kLengthToHeaderBits[1 + kMaxSigned64Length][2] = {
    {0, 0},      {'\x80', 0},      {'\xc0', 0},     {'\xe0', 0},
    {'\xf0', 0}, {'\xf8', 0},      {'\xfc', 0},     {'\xfe', 0},
    {'\xff', 0}, {'\xff', '\x80'}, {'\xff', '\xc0'}};
static const uint64 kLengthToMask[1 + kMaxSigned64Length] = {
    0ULL,
    0x80ULL,
    0xc000ULL,
    0xe00000ULL,
    0xf0000000ULL,
    0xf800000000ULL,
    0xfc0000000000ULL,
    0xfe000000000000ULL,
    0xff00000000000000ULL,
    0x8000000000000000ULL,
    0ULL};
static const int8 kBitsToLength[1 + 63] = {
    1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4,
    4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7,
    7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10};
static inline int SignedEncodingLength(int64_t n) {
  return kBitsToLength[tsl::Log2Floor64(n < 0 ? ~n : n) + 1];
}
static void StoreBigEndian64(char* dst, uint64 v) {
  for (int i = 0; i < 8; i++) {
    dst[i] = (v >> (56 - 8 * i)) & 0xff;
  }
}
static uint64 LoadBigEndian64(const char* src) {
  uint64 result = 0;
  for (int i = 0; i < 8; i++) {
    unsigned char c = static_cast<unsigned char>(src[i]);
    result |= static_cast<uint64>(c) << (56 - 8 * i);
  }
  return result;
}
void OrderedCode::WriteSignedNumIncreasing(string* dest, int64_t val) {
  const uint64 x = val < 0 ? ~val : val;
  if (x < 64) {  
    *dest += kLengthToHeaderBits[1][0] ^ val;
    return;
  }
  const char sign_byte = val < 0 ? '\xff' : '\0';
  char buf[10] = {
      sign_byte,
      sign_byte,
  };
  StoreBigEndian64(buf + 2, val);
  static_assert(sizeof(buf) == kMaxSigned64Length, "max length size mismatch");
  const int len = SignedEncodingLength(x);
  DCHECK_GE(len, 2);
  char* const begin = buf + sizeof(buf) - len;
  begin[0] ^= kLengthToHeaderBits[len][0];
  begin[1] ^= kLengthToHeaderBits[len][1];  
  dest->append(begin, len);
}
bool OrderedCode::ReadSignedNumIncreasing(StringPiece* src, int64_t* result) {
  if (src->empty()) return false;
  const uint64 xor_mask = (!((*src)[0] & 0x80)) ? ~0ULL : 0ULL;
  const unsigned char first_byte = (*src)[0] ^ (xor_mask & 0xff);
  int len;
  uint64 x;
  if (first_byte != 0xff) {
    len = 7 - tsl::Log2Floor64(first_byte ^ 0xff);
    if (src->size() < static_cast<size_t>(len)) return false;
    x = xor_mask;  
    for (int i = 0; i < len; ++i)
      x = (x << 8) | static_cast<unsigned char>((*src)[i]);
  } else {
    len = 8;
    if (src->size() < static_cast<size_t>(len)) return false;
    const unsigned char second_byte = (*src)[1] ^ (xor_mask & 0xff);
    if (second_byte >= 0x80) {
      if (second_byte < 0xc0) {
        len = 9;
      } else {
        const unsigned char third_byte = (*src)[2] ^ (xor_mask & 0xff);
        if (second_byte == 0xc0 && third_byte < 0x80) {
          len = 10;
        } else {
          return false;  
        }
      }
      if (src->size() < static_cast<size_t>(len)) return false;
    }
    x = LoadBigEndian64(src->data() + len - 8);
  }
  x ^= kLengthToMask[len];  
  DCHECK_EQ(len, SignedEncodingLength(x)) << "invalid encoding";
  if (result) *result = x;
  src->remove_prefix(len);
  return true;
}
}  
}  