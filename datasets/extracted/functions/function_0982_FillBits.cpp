#ifndef TENSORSTORE_UTIL_BIT_SPAN_H_
#define TENSORSTORE_UTIL_BIT_SPAN_H_
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include "absl/base/attributes.h"
#include "tensorstore/util/small_bit_set.h"
#include "tensorstore/util/span.h"
namespace tensorstore {
namespace internal_bit_span {
template <bool FillValue, typename T>
void FillBits(T* base, std::ptrdiff_t offset, std::ptrdiff_t size) {
  constexpr std::ptrdiff_t kBitsPerBlock = sizeof(T) * 8;
  constexpr const T kAllOnes = ~static_cast<T>(0);
  assert(offset >= 0);
  std::ptrdiff_t end;
  for (base += offset / kBitsPerBlock, offset %= kBitsPerBlock,
       end = size + offset;
       end >= kBitsPerBlock; ++base, offset = 0, end -= kBitsPerBlock) {
    const T mask = kAllOnes << offset;
    if (FillValue) {
      *base |= mask;
    } else {
      *base &= ~mask;
    }
  }
  if (end) {
    const T mask = (kAllOnes << offset) ^ (kAllOnes << (end % kBitsPerBlock));
    if (FillValue) {
      *base |= mask;
    } else {
      *base &= ~mask;
    }
  }
}
template <typename T, typename U>
void CopyBits(const U* source, std::ptrdiff_t source_offset, T* dest,
              std::ptrdiff_t dest_offset, std::ptrdiff_t size) {
  std::copy(BitIterator<const U>(source, source_offset),
            BitIterator<const U>(source, source_offset + size),
            BitIterator<T>(dest, dest_offset));
}
}  
template <typename T, std::ptrdiff_t Extent = dynamic_extent>
class BitSpan {
  static_assert(std::is_unsigned_v<T>, "Storage type T must be unsigned.");
  static_assert(Extent == dynamic_extent || Extent >= 0,
                "Extent must be dynamic_extent or >= 0.");
 public:
  using ExtentType =
      std::conditional_t<Extent == dynamic_extent, std::ptrdiff_t,
                         std::integral_constant<std::ptrdiff_t, Extent>>;
  using size_type = std::ptrdiff_t;
  using difference_type = std::ptrdiff_t;
  using iterator = BitIterator<T>;
  using const_iterator = BitIterator<const T>;
  using pointer = BitIterator<T>;
  using const_pointer = BitIterator<T>;
  using value_type = bool;
  using reference = BitRef<T>;
  using base_type = T;
  using element_type = std::conditional_t<std::is_const_v<T>, const bool, bool>;
  constexpr static std::ptrdiff_t kBitsPerBlock = sizeof(T) * 8;
  constexpr static std::ptrdiff_t static_extent = Extent;
  constexpr BitSpan(T* base ABSL_ATTRIBUTE_LIFETIME_BOUND,
                    std::ptrdiff_t offset, std::ptrdiff_t size)
      : BitSpan(BitIterator<T>(base, offset), size) {}
  constexpr BitSpan(BitIterator<T> begin, std::ptrdiff_t size) : begin_(begin) {
    if constexpr (Extent == dynamic_extent) {
      assert(size >= 0);
      size_ = size;
    } else {
      assert(size == Extent);
    }
  }
  template <
      typename U, std::ptrdiff_t E,
      std::enable_if_t<((std::is_same_v<T, U> || std::is_same_v<T, const U>)&&(
          E == Extent || Extent == dynamic_extent))>* = nullptr>
  constexpr BitSpan(BitSpan<U, E> other)
      : begin_(other.begin()), size_(other.size()) {}
  constexpr T* base() const { return begin().base(); }
  constexpr std::ptrdiff_t offset() const { return begin().offset(); }
  constexpr ExtentType size() const { return size_; }
  BitIterator<T> begin() const { return begin_; }
  BitIterator<T> end() const { return begin_ + size_; }
  constexpr BitRef<T> operator[](std::ptrdiff_t i) const {
    assert(i >= 0 && i <= size());
    return *(begin() + i);
  }
  template <bool FillValue, int&... ExplicitArgumentBarrier, typename X = T>
  std::enable_if_t<!std::is_const_v<X>> fill() const {
    internal_bit_span::FillBits<FillValue>(base(), offset(), size());
  }
  template <int&... ExplicitArgumentBarrier, typename X = T>
  std::enable_if_t<!std::is_const_v<X>> fill(bool value) const {
    if (value) {
      fill<true>();
    } else {
      fill<false>();
    }
  }
  template <typename U, std::ptrdiff_t E, int&... ExplicitArgumentBarrier,
            typename X = T>
  std::enable_if_t<!std::is_const_v<X> &&
                   (E == Extent || Extent == dynamic_extent ||
                    E == dynamic_extent)>
  DeepAssign(BitSpan<U, E> other) {
    assert(other.size() == size());
    internal_bit_span::CopyBits(other.base(), other.offset(), base(), offset(),
                                size());
  }
 private:
  BitIterator<T> begin_;
  ABSL_ATTRIBUTE_NO_UNIQUE_ADDRESS ExtentType size_;
};
template <typename Block>
inline constexpr std::ptrdiff_t BitVectorSizeInBlocks(std::ptrdiff_t length) {
  return (length + sizeof(Block) * 8 - 1) / (sizeof(Block) * 8);
}
}  
#endif  