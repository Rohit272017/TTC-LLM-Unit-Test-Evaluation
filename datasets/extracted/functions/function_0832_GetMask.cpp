#ifndef AROLLA_DENSE_ARRAY_OPS_UTIL_H_
#define AROLLA_DENSE_ARRAY_OPS_UTIL_H_
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include "absl/base/attributes.h"
#include "absl/log/check.h"
#include "arolla/dense_array/bitmap.h"
#include "arolla/memory/buffer.h"
#include "arolla/memory/optional_value.h"
#include "arolla/util/meta.h"
#include "arolla/util/view_types.h"
namespace arolla::dense_ops_internal {
template <bool AllowBitmapOffset = true, class ArrayT>
bitmap::Word GetMask(const ArrayT& array, int64_t word_id) {
  if constexpr (AllowBitmapOffset) {
    return bitmap::GetWordWithOffset(array.bitmap, word_id,
                                     array.bitmap_bit_offset);
  } else {
    DCHECK_EQ(array.bitmap_bit_offset, 0);
    return bitmap::GetWord(array.bitmap, word_id);
  }
}
template <class T, bool AllowBitmapOffset = true, class ArrayT>
bitmap::Word GetOptionalMask(const ArrayT& array,
                             int64_t word_id ABSL_ATTRIBUTE_UNUSED) {
  if constexpr (meta::is_wrapped_with<OptionalValue, T>::value) {
    return bitmap::kFullWord;
  } else {
    return GetMask<AllowBitmapOffset>(array, word_id);
  }
}
template <class T, class ArrayT, bool AllowBitmapOffset = true>
struct Getter {
  Getter(const ArrayT& array, int64_t word_id) {
    iter = array.values.begin() + word_id * bitmap::kWordBitCount;
  }
  view_type_t<T> operator()(int i) const { return *(iter + i); }
  typename Buffer<T>::const_iterator iter;
};
template <class T, class ArrayT, bool AllowBitmapOffset>
struct Getter<OptionalValue<T>, ArrayT, AllowBitmapOffset> {
  Getter(const ArrayT& array, int64_t word_id) {
    mask = GetMask<AllowBitmapOffset>(array, word_id);
    iter = array.values.begin() + word_id * bitmap::kWordBitCount;
  }
  OptionalValue<view_type_t<T>> operator()(int i) const {
    return {bitmap::GetBit(mask, i), *(iter + i)};
  }
  bitmap::Word mask;
  typename Buffer<T>::const_iterator iter;
};
template <class ArgList, bool AllowBitmapOffset = true>
struct DenseOpsUtil {};
template <class... Ts, bool AllowBitmapOffset>
struct DenseOpsUtil<meta::type_list<Ts...>, AllowBitmapOffset> {
  template <class... As>
  static bitmap::Word IntersectMasks(int64_t word_id ABSL_ATTRIBUTE_UNUSED,
                                     const As&... arrays) {
    return (GetOptionalMask<Ts, AllowBitmapOffset>(arrays, word_id) & ... &
            bitmap::kFullWord);
  }
  template <class... As>
  static std::tuple<Getter<Ts, As, AllowBitmapOffset>...> CreateGetters(
      int64_t word_id ABSL_ATTRIBUTE_UNUSED, const As&... arrays) {
    return {Getter<Ts, As, AllowBitmapOffset>(arrays, word_id)...};
  }
  template <class Fn, class... As>
  static void Iterate(Fn&& fn, int64_t from, int64_t to, const As&... args) {
    DCHECK_GE(from, 0);
    DCHECK_GE(to, from);
    return Iterate(fn, std::index_sequence_for<As...>{}, from, to, args...);
  }
  template <class Fn, class... As>
  static void IterateFromZero(Fn&& fn, int64_t to, const As&... args) {
    return IterateFromZero(fn, std::index_sequence_for<As...>{}, to, args...);
  }
 private:
  template <class Fn, class... As, size_t... Is>
  static void Iterate(Fn&& fn, std::index_sequence<Is...>, uint64_t from,
                      uint64_t to, const As&... args) {
    auto group_fn = [&](int64_t word_id, int local_from, int local_to) {
      bitmap::Word mask = IntersectMasks(word_id, args...);
      auto getters ABSL_ATTRIBUTE_UNUSED = CreateGetters(word_id, args...);
      for (int i = local_from; i < local_to; ++i) {
        fn(word_id * bitmap::kWordBitCount + i, bitmap::GetBit(mask, i),
           std::get<Is>(getters)(i)...);
      }
    };
    uint64_t word_id = from / bitmap::kWordBitCount;
    int local_from = from & (bitmap::kWordBitCount - 1);
    if (local_from > 0) {
      int local_to =
          std::min<int64_t>(to - from + local_from, bitmap::kWordBitCount);
      group_fn(word_id, local_from, local_to);
      word_id++;
    }
    for (; word_id < to / bitmap::kWordBitCount; ++word_id) {
      group_fn(word_id, 0, bitmap::kWordBitCount);
    }
    int local_to = to - word_id * bitmap::kWordBitCount;
    if (local_to > 0) {
      group_fn(word_id, 0, local_to);
    }
  }
  template <class Fn, class... As, size_t... Is>
  static void IterateFromZero(Fn&& fn, std::index_sequence<Is...>, uint64_t to,
                              const As&... args) {
    for (uint64_t offset = 0; offset < to; offset += bitmap::kWordBitCount) {
      int group_size = std::min<int64_t>(bitmap::kWordBitCount, to - offset);
      int64_t word_id = offset / bitmap::kWordBitCount;
      bitmap::Word mask = IntersectMasks(word_id, args...);
      auto getters ABSL_ATTRIBUTE_UNUSED = CreateGetters(word_id, args...);
      for (int i = 0; i < group_size; ++i) {
        fn(word_id * bitmap::kWordBitCount + i, bitmap::GetBit(mask, i),
           std::get<Is>(getters)(i)...);
      }
    }
  }
};
}  
#endif  