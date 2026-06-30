#include <algorithm>
#include <sstream>
#include <cassert>
#include <cstdio>
#include "phonenumbers/default_logger.h"
#include "phonenumbers/utf/unicodetext.h"
#include "phonenumbers/utf/stringpiece.h"
#include "phonenumbers/utf/utf.h"
#include "phonenumbers/utf/unilib.h"
namespace i18n {
namespace phonenumbers {
using std::string;
using std::stringstream;
using std::max;
using std::hex;
using std::dec;
static int CodepointDistance(const char* start, const char* end) {
  int n = 0;
  for (const char* p = start; p < end; ++p) {
    n += (*reinterpret_cast<const signed char*>(p) >= -0x40);
  }
  return n;
}
static int CodepointCount(const char* utf8, int len) {
  return CodepointDistance(utf8, utf8 + len);
}
UnicodeText::const_iterator::difference_type
distance(const UnicodeText::const_iterator& first,
         const UnicodeText::const_iterator& last) {
  return CodepointDistance(first.it_, last.it_);
}
static int ConvertToInterchangeValid(char* start, int len) {
  char* const in = start;
  char* out = start;
  char* const end = start + len;
  while (start < end) {
    int good = UniLib::SpanInterchangeValid(start, static_cast<int>(end - start));
    if (good > 0) {
      if (out != start) {
        memmove(out, start, good);
      }
      out += good;
      start += good;
      if (start == end) {
        break;
      }
    }
    Rune rune;
    int n;
    if (isvalidcharntorune(start, static_cast<int>(end - start), &rune, &n)) {
      start += n;  
    } else {  
      start += 1;  
    }
    *out++ = ' ';
  }
  return static_cast<int>(out - in);
}
void UnicodeText::Repr::reserve(int new_capacity) {
  if (capacity_ >= new_capacity && ours_) return;
  capacity_ = max(new_capacity, (3 * capacity_) / 2 + 20);
  char* new_data = new char[capacity_];
  if (data_) {
    memcpy(new_data, data_, size_);
    if (ours_) delete[] data_;  
  }
  data_ = new_data;
  ours_ = true;  
}
void UnicodeText::Repr::resize(int new_size) {
  if (new_size == 0) {
    clear();
  } else {
    if (!ours_ || new_size > capacity_) reserve(new_size);
    if (size_ < new_size) memset(data_ + size_, 0, new_size - size_);
    size_ = new_size;
    ours_ = true;
  }
}
void UnicodeText::Repr::clear() {
  if (ours_) delete[] data_;
  data_ = NULL;
  size_ = capacity_ = 0;
  ours_ = true;
}
void UnicodeText::Repr::Copy(const char* data, int size) {
  resize(size);
  memcpy(data_, data, size);
}
void UnicodeText::Repr::TakeOwnershipOf(char* data, int size, int capacity) {
  if (data == data_) return;  
  if (ours_ && data_) delete[] data_;  
  data_ = data;
  size_ = size;
  capacity_ = capacity;
  ours_ = true;
}
void UnicodeText::Repr::PointTo(const char* data, int size) {
  if (ours_ && data_) delete[] data_;  
  data_ = const_cast<char*>(data);
  size_ = size;
  capacity_ = size;
  ours_ = false;
}
void UnicodeText::Repr::append(const char* bytes, int byte_length) {
  reserve(size_ + byte_length);
  memcpy(data_ + size_, bytes, byte_length);
  size_ += byte_length;
}
string UnicodeText::Repr::DebugString() const {
  stringstream ss;
  ss << "{Repr " << hex << this << " data=" << data_ << " size=" << dec
     << size_ << " capacity=" << capacity_ << " "
     << (ours_ ? "Owned" : "Alias") << "}";
  string result;
  ss >> result;
  return result;
}
UnicodeText::UnicodeText() {
}
UnicodeText::UnicodeText(const UnicodeText& src) {
  Copy(src);
}
UnicodeText::UnicodeText(const UnicodeText::const_iterator& first,
                         const UnicodeText::const_iterator& last) {
  assert(first <= last && "Incompatible iterators");
  repr_.append(first.it_, static_cast<int>(last.it_ - first.it_));
}
string UnicodeText::UTF8Substring(const const_iterator& first,
                                  const const_iterator& last) {
  assert(first <= last && "Incompatible iterators");
  return string(first.it_, last.it_ - first.it_);
}
UnicodeText& UnicodeText::operator=(const UnicodeText& src) {
  if (this != &src) {
    Copy(src);
  }
  return *this;
}
UnicodeText& UnicodeText::Copy(const UnicodeText& src) {
  repr_.Copy(src.repr_.data_, src.repr_.size_);
  return *this;
}
UnicodeText& UnicodeText::CopyUTF8(const char* buffer, int byte_length) {
  repr_.Copy(buffer, byte_length);
  repr_.utf8_was_valid_ = UniLib:: IsInterchangeValid(buffer, byte_length);
  if (!repr_.utf8_was_valid_) {
    LOG(WARNING) << "UTF-8 buffer is not interchange-valid.";
    repr_.size_ = ConvertToInterchangeValid(repr_.data_, byte_length);
  }
  return *this;
}
UnicodeText& UnicodeText::UnsafeCopyUTF8(const char* buffer,
                                           int byte_length) {
  repr_.Copy(buffer, byte_length);
  return *this;
}
UnicodeText& UnicodeText::TakeOwnershipOfUTF8(char* buffer,
                                              int byte_length,
                                              int byte_capacity) {
  repr_.TakeOwnershipOf(buffer, byte_length, byte_capacity);
  repr_.utf8_was_valid_ = UniLib:: IsInterchangeValid(buffer, byte_length);
  if (!repr_.utf8_was_valid_) {
    LOG(WARNING) << "UTF-8 buffer is not interchange-valid.";
    repr_.size_ = ConvertToInterchangeValid(repr_.data_, byte_length);
  }
  return *this;
}
UnicodeText& UnicodeText::UnsafeTakeOwnershipOfUTF8(char* buffer,
                                                    int byte_length,
                                                    int byte_capacity) {
  repr_.TakeOwnershipOf(buffer, byte_length, byte_capacity);
  return *this;
}
UnicodeText& UnicodeText::PointToUTF8(const char* buffer, int byte_length) {
  repr_.utf8_was_valid_ = UniLib:: IsInterchangeValid(buffer, byte_length);
  if (repr_.utf8_was_valid_) {
    repr_.PointTo(buffer, byte_length);
  } else {
    LOG(WARNING) << "UTF-8 buffer is not interchange-valid.";
    repr_.Copy(buffer, byte_length);
    repr_.size_ = ConvertToInterchangeValid(repr_.data_, byte_length);
  }
  return *this;
}
UnicodeText& UnicodeText::UnsafePointToUTF8(const char* buffer,
                                          int byte_length) {
  repr_.PointTo(buffer, byte_length);
  return *this;
}
UnicodeText& UnicodeText::PointTo(const UnicodeText& src) {
  repr_.PointTo(src.repr_.data_, src.repr_.size_);
  return *this;
}
UnicodeText& UnicodeText::PointTo(const const_iterator &first,
                                  const const_iterator &last) {
  assert(first <= last && " Incompatible iterators");
  repr_.PointTo(first.utf8_data(), static_cast<int>(last.utf8_data() - first.utf8_data()));
  return *this;
}
UnicodeText& UnicodeText::append(const UnicodeText& u) {
  repr_.append(u.repr_.data_, u.repr_.size_);
  return *this;
}
UnicodeText& UnicodeText::append(const const_iterator& first,
                                 const const_iterator& last) {
  assert(first <= last && "Incompatible iterators");
  repr_.append(first.it_, static_cast<int>(last.it_ - first.it_));
  return *this;
}
UnicodeText& UnicodeText::UnsafeAppendUTF8(const char* utf8, int len) {
  repr_.append(utf8, len);
  return *this;
}
UnicodeText::const_iterator UnicodeText::find(const UnicodeText& look,
                                              const_iterator start_pos) const {
  assert(start_pos.utf8_data() >= utf8_data());
  assert(start_pos.utf8_data() <= utf8_data() + utf8_length());
  return UnsafeFind(look, start_pos);
}
UnicodeText::const_iterator UnicodeText::find(const UnicodeText& look) const {
  return UnsafeFind(look, begin());
}
UnicodeText::const_iterator UnicodeText::UnsafeFind(
    const UnicodeText& look, const_iterator start_pos) const {
  StringPiece searching(utf8_data(), utf8_length());
  StringPiece look_piece(look.utf8_data(), look.utf8_length());
  StringPiece::size_type found =
      searching.find(look_piece, start_pos.utf8_data() - utf8_data());
  if (found == StringPiece::npos) return end();
  return const_iterator(utf8_data() + found);
}
bool UnicodeText::HasReplacementChar() const {
  StringPiece searching(utf8_data(), utf8_length());
  StringPiece looking_for("\xEF\xBF\xBD", 3);
  return searching.find(looking_for) != StringPiece::npos;
}
void UnicodeText::clear() {
  repr_.clear();
}
UnicodeText::~UnicodeText() {}
void UnicodeText::push_back(char32 c) {
  if (UniLib::IsValidCodepoint(c)) {
    char buf[UTFmax];
    Rune rune = c;
    int len = runetochar(buf, &rune);
    if (UniLib::IsInterchangeValid(buf, len)) {
      repr_.append(buf, len);
    } else {
      fprintf(stderr, "Unicode value 0x%x is not valid for interchange\n", c);
      repr_.append(" ", 1);
    }
  } else {
    fprintf(stderr, "Illegal Unicode value: 0x%x\n", c);
    repr_.append(" ", 1);
  }
}
int UnicodeText::size() const {
  return CodepointCount(repr_.data_, repr_.size_);
}
bool operator==(const UnicodeText& lhs, const UnicodeText& rhs) {
  if (&lhs == &rhs) return true;
  if (lhs.repr_.size_ != rhs.repr_.size_) return false;
  return memcmp(lhs.repr_.data_, rhs.repr_.data_, lhs.repr_.size_) == 0;
}
string UnicodeText::DebugString() const {
  stringstream ss;
  ss << "{UnicodeText " << hex << this << dec << " chars="
     << size() << " repr=" << repr_.DebugString() << "}";
#if 0
  return StringPrintf("{UnicodeText %p chars=%d repr=%s}",
                      this,
                      size(),
                      repr_.DebugString().c_str());
#endif
  string result;
  ss >> result;
  return result;
}
UnicodeText::const_iterator::const_iterator() : it_(0) {}
UnicodeText::const_iterator::const_iterator(const const_iterator& other)
    : it_(other.it_) {
}
UnicodeText::const_iterator&
UnicodeText::const_iterator::operator=(const const_iterator& other) {
  if (&other != this)
    it_ = other.it_;
  return *this;
}
UnicodeText::const_iterator UnicodeText::begin() const {
  return const_iterator(repr_.data_);
}
UnicodeText::const_iterator UnicodeText::end() const {
  return const_iterator(repr_.data_ + repr_.size_);
}
bool operator<(const UnicodeText::const_iterator& lhs,
               const UnicodeText::const_iterator& rhs) {
  return lhs.it_ < rhs.it_;
}
char32 UnicodeText::const_iterator::operator*() const {
  uint8 byte1 = static_cast<uint8>(it_[0]);
  if (byte1 < 0x80)
    return byte1;
  uint8 byte2 = static_cast<uint8>(it_[1]);
  if (byte1 < 0xE0)
    return ((byte1 & 0x1F) << 6)
          | (byte2 & 0x3F);
  uint8 byte3 = static_cast<uint8>(it_[2]);
  if (byte1 < 0xF0)
    return ((byte1 & 0x0F) << 12)
         | ((byte2 & 0x3F) << 6)
         |  (byte3 & 0x3F);
  uint8 byte4 = static_cast<uint8>(it_[3]);
  return ((byte1 & 0x07) << 18)
       | ((byte2 & 0x3F) << 12)
       | ((byte3 & 0x3F) << 6)
       |  (byte4 & 0x3F);
}
UnicodeText::const_iterator& UnicodeText::const_iterator::operator++() {
  it_ += UniLib::OneCharLen(it_);
  return *this;
}
UnicodeText::const_iterator& UnicodeText::const_iterator::operator--() {
  while (UniLib::IsTrailByte(*--it_)) { }
  return *this;
}
int UnicodeText::const_iterator::get_utf8(char* utf8_output) const {
  utf8_output[0] = it_[0];
  if (static_cast<unsigned char>(it_[0]) < 0x80)
    return 1;
  utf8_output[1] = it_[1];
  if (static_cast<unsigned char>(it_[0]) < 0xE0)
    return 2;
  utf8_output[2] = it_[2];
  if (static_cast<unsigned char>(it_[0]) < 0xF0)
    return 3;
  utf8_output[3] = it_[3];
  return 4;
}
UnicodeText::const_iterator UnicodeText::MakeIterator(const char* p) const {
#ifndef NDEBUG
  assert(p != NULL);
  const char* start = utf8_data();
  int len = utf8_length();
  const char* end = start + len;
  assert(p >= start);
  assert(p <= end);
  assert(p == end || !UniLib::IsTrailByte(*p));
#endif
  return const_iterator(p);
}
string UnicodeText::const_iterator::DebugString() const {
  stringstream ss;
  ss << "{iter " << hex << it_ << "}";
  string result;
  ss >> result;
  return result;
}
}  
}  