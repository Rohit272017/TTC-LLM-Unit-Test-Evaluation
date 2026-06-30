#include "xla/tsl/lib/core/bitmap.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include "absl/numeric/bits.h"
namespace tsl {
namespace core {
void Bitmap::Reset(size_t n) {
  const size_t num_words = NumWords(n);
  if (num_words != NumWords(nbits_)) {
    Word* w = new Word[num_words];
    delete[] word_;
    word_ = w;
  }
  memset(word_, 0, sizeof(word_[0]) * num_words);
  nbits_ = n;
}
static size_t FindFirstSet(uint32_t w) {
  return w == 0 ? 0 : absl::countr_zero(w) + 1;
}
size_t Bitmap::FirstUnset(size_t start) const {
  if (start >= nbits_) {
    return nbits_;
  }
  size_t mask = (1ull << (start % kBits)) - 1;
  const size_t nwords = NumWords(nbits_);
  for (size_t i = start / kBits; i < nwords; i++) {
    Word word = word_[i] | mask;
    mask = 0;  
    size_t r = FindFirstSet(~word);
    if (r) {
      size_t result = i * kBits + (r - 1);
      if (result > nbits_) result = nbits_;
      return result;
    }
  }
  return nbits_;
}
std::string Bitmap::ToString() const {
  std::string result;
  result.resize(bits());
  for (size_t i = 0; i < nbits_; i++) {
    result[i] = get(i) ? '1' : '0';
  }
  return result;
}
}  
}  