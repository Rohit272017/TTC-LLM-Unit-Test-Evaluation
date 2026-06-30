#ifndef TENSORSTORE_UTIL_SMALL_BIT_SET_H_
#define TENSORSTORE_UTIL_SMALL_BIT_SET_H_
#include <stddef.h>
#include <cassert>
#include <iterator>
#include <ostream>
#include <type_traits>
#include "absl/base/attributes.h"
#include "absl/numeric/bits.h"
#include "tensorstore/internal/integer_types.h"
namespace tensorstore {
template <typename T>
class BitRef {
  static_assert(std::is_unsigned_v<T>, "Storage type T must be unsigned.");
 public:
  friend class BitRef<const T>;
  using block_type = T;
  using value_type = bool;
  using element_type = bool;
  constexpr static ptrdiff_t kBitsPerBlock = sizeof(T) * 8;
  constexpr BitRef(T* block ABSL_ATTRIBUTE_LIFETIME_BOUND, ptrdiff_t offset)
      : block_(block), mask_(static_cast<T>(1) << (offset % kBitsPerBlock)) {
    assert(offset >= 0);
  }
  constexpr operator bool() const { return *block_ & mask_; }
  const BitRef& operator=(bool value) const {
    *block_ = value ? (*block_ | mask_) : (*block_ & ~mask_);
    return *this;
  }
  const BitRef& operator=(BitRef value) const {
    return (*this = static_cast<bool>(value));
  }
  friend void swap(BitRef a, bool& x) {
    bool temp = a;
    a = x;
    x = temp;
  }
  friend void swap(bool& x, BitRef a) {
    bool temp = a;
    a = x;
    x = temp;
  }
 private:
  T* block_;
  T mask_;
};
template <typename T, typename U>
std::enable_if_t<(!std::is_const_v<T> && !std::is_const_v<U>)> swap(
    BitRef<T> a, BitRef<U> b) {
  bool temp = a;
  a = b;
  b = temp;
}
template <typename T>
std::enable_if_t<(!std::is_const_v<T>)> swap(BitRef<T> a, BitRef<T> b) {
  bool temp = a;
  a = b;
  b = temp;
}
template <typename T>
class BitIterator {
  static_assert(std::is_unsigned_v<T>, "Storage type T must be unsigned.");
 public:
  using pointer = BitIterator<T>;
  using const_pointer = BitIterator<const T>;
  using reference = BitRef<T>;
  using const_reference = BitRef<const T>;
  using difference_type = ptrdiff_t;
  using value_type = bool;
  using iterator_category = std::random_access_iterator_tag;
  constexpr static ptrdiff_t kBitsPerBlock = sizeof(T) * 8;
  constexpr BitIterator() : base_(nullptr), offset_(0) {}
  constexpr BitIterator(T* base ABSL_ATTRIBUTE_LIFETIME_BOUND, ptrdiff_t offset)
      : base_(base), offset_(offset) {}
  template <typename U, std::enable_if_t<std::is_same_v<const U, T>>* = nullptr>
  constexpr BitIterator(BitIterator<U> other)
      : base_(other.base()), offset_(other.offset()) {}
  constexpr T* base() const { return base_; }
  constexpr ptrdiff_t offset() const { return offset_; }
  constexpr BitRef<T> operator*() const {
    return BitRef<T>(base() + offset() / kBitsPerBlock, offset());
  }
  constexpr BitRef<T> operator[](ptrdiff_t offset) const {
    return *(*this + offset);
  }
  BitIterator& operator++() {
    ++offset_;
    return *this;
  }
  BitIterator& operator--() {
    --offset_;
    return *this;
  }
  BitIterator operator++(int) {
    BitIterator temp = *this;
    ++offset_;
    return temp;
  }
  BitIterator operator--(int) {
    BitIterator temp = *this;
    --offset_;
    return temp;
  }
  friend BitIterator operator+(BitIterator it, ptrdiff_t offset) {
    it += offset;
    return it;
  }
  friend BitIterator operator+(ptrdiff_t offset, BitIterator it) {
    it += offset;
    return it;
  }
  BitIterator& operator+=(ptrdiff_t x) {
    offset_ += x;
    return *this;
  }
  friend BitIterator operator-(BitIterator it, ptrdiff_t offset) {
    it -= offset;
    return it;
  }
  BitIterator& operator-=(ptrdiff_t x) {
    offset_ -= x;
    return *this;
  }
  friend constexpr ptrdiff_t operator-(BitIterator a, BitIterator b) {
    assert(a.base() == b.base());
    return a.offset() - b.offset();
  }
  friend constexpr bool operator==(BitIterator a, BitIterator b) {
    assert(a.base() == b.base());
    return a.offset() == b.offset();
  }
  friend constexpr bool operator!=(BitIterator a, BitIterator b) {
    assert(a.base() == b.base());
    return a.offset() != b.offset();
  }
  friend constexpr bool operator<(BitIterator a, BitIterator b) {
    assert(a.base() == b.base());
    return a.offset() < b.offset();
  }
  friend constexpr bool operator<=(BitIterator a, BitIterator b) {
    assert(a.base() == b.base());
    return a.offset() <= b.offset();
  }
  friend constexpr bool operator>(BitIterator a, BitIterator b) {
    assert(a.base() == b.base());
    return a.offset() > b.offset();
  }
  friend constexpr bool operator>=(BitIterator a, BitIterator b) {
    assert(a.base() == b.base());
    return a.offset() >= b.offset();
  }
 private:
  T* base_;
  ptrdiff_t offset_;
};
namespace bitset_impl {
template <typename Iterator, size_t N>
class BoolsView {
 public:
  using iterator = Iterator;
  using value_type = typename iterator::value_type;
  using difference_type = typename iterator::difference_type;
  using reference = typename iterator::reference;
  explicit BoolsView(iterator it) : it_(std::move(it)) {}
  constexpr iterator begin() const { return it_; }
  constexpr iterator end() const { return iterator(it_.base(), N); }
 private:
  iterator it_;
};
template <typename Uint>
class OneBitsIterator {
 public:
  using value_type = int;
  using difference_type = int;
  using reference = int;
  OneBitsIterator() : value_(0) {}
  explicit OneBitsIterator(Uint value) : value_(value) {}
  friend constexpr bool operator==(OneBitsIterator a, OneBitsIterator b) {
    return a.value_ == b.value_;
  }
  friend constexpr bool operator!=(OneBitsIterator a, OneBitsIterator b) {
    return !(a == b);
  }
  constexpr int operator*() const { return absl::countr_zero(value_); }
  constexpr OneBitsIterator& operator++() {
    Uint t = value_ & -value_;
    value_ ^= t;
    return *this;
  }
  constexpr OneBitsIterator operator++(int) {
    auto copy = *this;
    ++*this;
    return copy;
  }
 private:
  Uint value_;
};
template <typename Uint>
class IndexView {
 public:
  IndexView(Uint bits) : bits_(bits) {}
  using const_iterator = OneBitsIterator<Uint>;
  using value_type = typename const_iterator::value_type;
  using difference_type = typename const_iterator::difference_type;
  using reference = typename const_iterator::reference;
  constexpr const_iterator begin() const { return const_iterator(bits_); }
  constexpr const_iterator end() const { return const_iterator(); }
  constexpr int front() const { return *begin(); }
 private:
  Uint bits_;
};
}  
template <size_t N>
class SmallBitSet {
 public:
  using Uint = typename internal::uint_type<N>::type;
  using value_type = bool;
  using reference = BitRef<Uint>;
  constexpr SmallBitSet() : bits_(0) {}
  template <typename T,
            typename = std::enable_if_t<std::is_same_v<T, bool>>>
  constexpr SmallBitSet(T value) : bits_(value * ~Uint(0)) {}
  static constexpr SmallBitSet FromUint(Uint bits) {
    SmallBitSet v;
    v.bits_ = bits;
    return v;
  }
  template <size_t NumBits, typename = std::enable_if_t<(NumBits <= N)>>
  static constexpr SmallBitSet FromIndices(const int (&positions)[NumBits]) {
    return FromIndexRange(std::begin(positions), std::end(positions));
  }
  template <typename Range>
  static constexpr SmallBitSet FromIndexRange(Range&& range) {
    return FromIndexRange(range.begin(), range.end());
  }
  template <typename Iterator>
  static constexpr SmallBitSet FromIndexRange(Iterator begin, Iterator end) {
    SmallBitSet set;
    while (begin != end) set.set(*begin++);
    return set;
  }
  template <size_t NumBits, typename = std::enable_if_t<(NumBits <= N)>>
  static constexpr SmallBitSet FromBools(const bool (&bits)[NumBits]) {
    return FromBoolRange(std::begin(bits), std::end(bits));
  }
  template <typename Range>
  static constexpr SmallBitSet FromBoolRange(Range&& range) {
    return FromBoolRange(range.begin(), range.end());
  }
  template <typename Iterator>
  static constexpr SmallBitSet FromBoolRange(Iterator begin, Iterator end) {
    SmallBitSet set;
    size_t i = 0;
    while (begin != end) {
      set.bits_ |= (*begin++ ? Uint(1) : Uint(0)) << i;
      i++;
    }
    assert(i <= N);
    return set;
  }
  static constexpr SmallBitSet UpTo(size_t k) {
    assert(k <= N);
    return k == 0 ? SmallBitSet()
                  : SmallBitSet::FromUint(~Uint(0) << (N - k) >> (N - k));
  }
  template <typename T,
            typename = std::enable_if_t<std::is_same_v<T, bool>>>
  constexpr SmallBitSet& operator=(T value) {
    bits_ = ~Uint(0) * value;
    return *this;
  }
  using BoolsView = bitset_impl::BoolsView<BitIterator<Uint>, N>;
  constexpr BoolsView bools_view() ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return BoolsView(BitIterator<Uint>(&bits_, 0));
  }
  using ConstBoolsView = bitset_impl::BoolsView<BitIterator<const Uint>, N>;
  constexpr ConstBoolsView bools_view() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return ConstBoolsView(BitIterator<const Uint>(&bits_, 0));
  }
  using IndexView = bitset_impl::IndexView<Uint>;
  constexpr IndexView index_view() const { return IndexView(bits_); }
  constexpr static size_t size() { return N; }
  constexpr size_t count() const { return absl::popcount(bits_); }
  constexpr bool none() const { return bits_ == 0; }
  constexpr bool any() const { return bits_ != 0; }
  constexpr bool all() const { return bits_ == ~Uint(0); }
  explicit operator bool() const { return any(); }
  constexpr SmallBitSet& set() noexcept {
    bits_ = ~Uint(0);
    return *this;
  }
  constexpr SmallBitSet& reset() noexcept {
    bits_ = 0;
    return *this;
  }
  constexpr SmallBitSet& flip() noexcept {
    bits_ = ~bits_;
    return *this;
  }
  constexpr bool test(int pos) const noexcept {
    assert(pos >= 0 && pos < N);
    return (bits_ >> pos) & 1;
  }
  constexpr SmallBitSet& set(int pos) noexcept {
    assert(pos >= 0 && pos < N);
    bits_ |= (static_cast<Uint>(1) << pos);
    return *this;
  }
  constexpr SmallBitSet& reset(int pos) noexcept {
    assert(pos >= 0 && pos < N);
    bits_ &= ~(static_cast<Uint>(1) << pos);
    return *this;
  }
  constexpr SmallBitSet& flip(int pos) noexcept {
    assert(pos >= 0 && pos < N);
    bits_ ^= (static_cast<Uint>(1) << pos);
    return *this;
  }
  constexpr reference operator[](size_t offset) ABSL_ATTRIBUTE_LIFETIME_BOUND {
    assert(offset >= 0 && offset < N);
    return reference(&bits_, offset);
  }
  constexpr bool operator[](size_t offset) const {
    assert(offset >= 0 && offset < N);
    return test(offset);
  }
  constexpr Uint to_uint() const { return bits_; }
  friend constexpr SmallBitSet operator~(SmallBitSet v) {
    return SmallBitSet::FromUint(~v.bits_);
  }
  friend constexpr SmallBitSet operator&(SmallBitSet a, SmallBitSet b) {
    return SmallBitSet::FromUint(a.bits_ & b.bits_);
  }
  friend constexpr SmallBitSet& operator&=(SmallBitSet& a, SmallBitSet b) {
    a.bits_ &= b.bits_;
    return a;
  }
  friend constexpr SmallBitSet operator^(SmallBitSet a, SmallBitSet b) {
    return SmallBitSet::FromUint(a.bits_ ^ b.bits_);
  }
  friend constexpr SmallBitSet& operator^=(SmallBitSet& a, SmallBitSet b) {
    a.bits_ ^= b.bits_;
    return a;
  }
  friend constexpr SmallBitSet operator|(SmallBitSet a, SmallBitSet b) {
    return SmallBitSet::FromUint(a.bits_ | b.bits_);
  }
  friend constexpr SmallBitSet& operator|=(SmallBitSet& a, SmallBitSet b) {
    a.bits_ |= b.bits_;
    return a;
  }
  friend constexpr bool operator==(SmallBitSet a, SmallBitSet b) {
    return a.bits_ == b.bits_;
  }
  friend constexpr bool operator!=(SmallBitSet a, SmallBitSet b) {
    return !(a == b);
  }
  friend std::ostream& operator<<(std::ostream& os, SmallBitSet v) {
    for (size_t i = 0; i < N; ++i) {
      os << (static_cast<bool>(v[i]) ? '1' : '0');
    }
    return os;
  }
 private:
  Uint bits_;
};
}  
#endif  