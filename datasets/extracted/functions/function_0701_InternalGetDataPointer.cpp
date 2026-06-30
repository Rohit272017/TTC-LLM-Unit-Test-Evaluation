#ifndef TENSORSTORE_INTERNAL_MULTI_VECTOR_H_
#define TENSORSTORE_INTERNAL_MULTI_VECTOR_H_
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <utility>
#include "tensorstore/internal/gdb_scripting.h"
#include "tensorstore/internal/meta.h"
#include "tensorstore/internal/multi_vector_impl.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/span.h"
TENSORSTORE_GDB_AUTO_SCRIPT("multi_vector_gdb.py")
namespace tensorstore {
namespace internal {
template <ptrdiff_t Extent, ptrdiff_t InlineSize, typename... Ts>
class MultiVectorStorageImpl;
template <ptrdiff_t Extent, typename... Ts>
using MultiVectorStorage =
    MultiVectorStorageImpl<RankConstraint::FromInlineRank(Extent),
                           InlineRankLimit(Extent), Ts...>;
template <typename StorageT>
class MultiVectorAccess;
template <ptrdiff_t Extent, ptrdiff_t InlineSize, typename... Ts>
class MultiVectorStorageImpl {
 private:
  static_assert((... && std::is_trivial_v<Ts>),
                "Non-trivial types are not currently supported.");
  static_assert(InlineSize == 0,
                "InlineSize must be 0 if Extent != dynamic_extent.");
  using Offsets = internal_multi_vector::PackStorageOffsets<Ts...>;
  friend class MultiVectorAccess<MultiVectorStorageImpl>;
  void* InternalGetDataPointer(size_t array_i) {
    return data_ + Offsets::GetVectorOffset(Extent, array_i);
  }
  constexpr static StaticRank<Extent> InternalGetExtent() { return {}; }
  void InternalResize(StaticRank<Extent>) {}
  alignas(Offsets::kAlignment) char data_[Offsets::GetTotalSize(Extent)];
};
template <ptrdiff_t InlineSize, typename... Ts>
class MultiVectorStorageImpl<0, InlineSize, Ts...> {
 private:
  static_assert(InlineSize == 0,
                "InlineSize must be 0 if Extent != dynamic_extent.");
  friend class MultiVectorAccess<MultiVectorStorageImpl>;
  void* InternalGetDataPointer(size_t array_i) { return nullptr; }
  constexpr static StaticRank<0> InternalGetExtent() { return {}; }
  void InternalResize(StaticRank<0>) {}
};
template <ptrdiff_t InlineSize, typename... Ts>
class MultiVectorStorageImpl<dynamic_rank, InlineSize, Ts...> {
  static_assert((std::is_trivial_v<Ts> && ...),
                "Non-trivial types are not currently supported.");
  static_assert(InlineSize >= 0, "InlineSize must be non-negative.");
  using Offsets = internal_multi_vector::PackStorageOffsets<Ts...>;
 public:
  explicit constexpr MultiVectorStorageImpl() noexcept {}
  MultiVectorStorageImpl(MultiVectorStorageImpl&& other) {
    *this = std::move(other);
  }
  MultiVectorStorageImpl(const MultiVectorStorageImpl& other) { *this = other; }
  MultiVectorStorageImpl& operator=(MultiVectorStorageImpl&& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(extent_, other.extent_);
    return *this;
  }
  MultiVectorStorageImpl& operator=(const MultiVectorStorageImpl& other) {
    if (this == &other) return *this;
    const ptrdiff_t extent = other.extent_;
    InternalResize(extent);
    const bool use_inline = InlineSize > 0 && extent <= InlineSize;
    std::memcpy(use_inline ? data_.inline_data : data_.pointer,
                use_inline ? other.data_.inline_data : other.data_.pointer,
                Offsets::GetTotalSize(extent));
    return *this;
  }
  ~MultiVectorStorageImpl() {
    if (extent_ > InlineSize) {
      ::operator delete(data_.pointer);
    }
  }
 private:
  friend class MultiVectorAccess<MultiVectorStorageImpl>;
  ptrdiff_t InternalGetExtent() const { return extent_; }
  void* InternalGetDataPointer(ptrdiff_t array_i) {
    return (extent_ > InlineSize ? data_.pointer : data_.inline_data) +
           Offsets::GetVectorOffset(extent_, array_i);
  }
  void InternalResize(ptrdiff_t new_extent) {
    assert(new_extent >= 0);
    if (extent_ == new_extent) return;
    if (new_extent > InlineSize) {
      void* new_data = ::operator new(Offsets::GetTotalSize(new_extent));
      if (extent_ > InlineSize) ::operator delete(data_.pointer);
      data_.pointer = static_cast<char*>(new_data);
    } else if (extent_ > InlineSize) {
      ::operator delete(data_.pointer);
    }
    extent_ = new_extent;
  }
  constexpr static ptrdiff_t kAlignment =
      InlineSize == 0 ? 1 : Offsets::kAlignment;
  constexpr static ptrdiff_t kInlineBytes =
      InlineSize == 0 ? 1 : Offsets::GetTotalSize(InlineSize);
  union Data {
    char* pointer;
    alignas(kAlignment) char inline_data[kInlineBytes];
  };
  Data data_;
  ptrdiff_t extent_ = 0;
};
template <ptrdiff_t Extent, ptrdiff_t InlineSize, typename... Ts>
class MultiVectorAccess<MultiVectorStorageImpl<Extent, InlineSize, Ts...>> {
 public:
  using StorageType = MultiVectorStorageImpl<Extent, InlineSize, Ts...>;
  using ExtentType = StaticOrDynamicRank<Extent>;
  constexpr static ptrdiff_t static_extent = Extent;
  constexpr static size_t num_vectors = sizeof...(Ts);
  template <size_t I>
  using ElementType = TypePackElement<I, Ts...>;
  template <size_t I>
  using ConstElementType = const TypePackElement<I, Ts...>;
  static ExtentType GetExtent(const StorageType& storage) {
    return storage.InternalGetExtent();
  }
  template <size_t I>
  static tensorstore::span<ElementType<I>, Extent> get(StorageType* array) {
    return {static_cast<ElementType<I>*>(array->InternalGetDataPointer(I)),
            GetExtent(*array)};
  }
  template <size_t I>
  static tensorstore::span<ConstElementType<I>, Extent> get(
      const StorageType* array) {
    return get<I>(const_cast<StorageType*>(array));
  }
  template <typename... Us>
  static void Assign(StorageType* array, ExtentType extent, Us*... pointers) {
    static_assert(sizeof...(Us) == sizeof...(Ts));
    array->InternalResize(extent);
    size_t vector_i = 0;
    (std::copy_n(pointers, extent,
                 static_cast<Ts*>(array->InternalGetDataPointer(vector_i++))),
     ...);
  }
  template <typename... Us, ptrdiff_t... Extents>
  static void Assign(StorageType* array,
                     tensorstore::span<Us, Extents>... spans) {
    static_assert(sizeof...(Us) == sizeof...(Ts));
    const ExtentType extent =
        GetFirstArgument(GetStaticOrDynamicExtent(spans)...);
    assert(((spans.size() == extent) && ...));
    Assign(array, extent, spans.data()...);
  }
  static void Resize(StorageType* array, ExtentType new_extent) {
    array->InternalResize(new_extent);
  }
};
}  
}  
#endif  