#ifndef TENSORSTORE_INTERNAL_MULTI_VECTOR_VIEW_H_
#define TENSORSTORE_INTERNAL_MULTI_VECTOR_VIEW_H_
#include <cassert>
#include <cstddef>
#include "tensorstore/index.h"
#include "tensorstore/internal/gdb_scripting.h"
#include "tensorstore/internal/meta.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/span.h"
TENSORSTORE_GDB_AUTO_SCRIPT("multi_vector_gdb.py")
namespace tensorstore {
namespace internal {
template <DimensionIndex Extent, typename... Ts>
class MultiVectorViewStorage;
template <typename StorageT>
class MultiVectorAccess;
template <ptrdiff_t Extent, typename... Ts>
class MultiVectorViewStorage {
 private:
  friend class MultiVectorAccess<MultiVectorViewStorage>;
  constexpr static StaticRank<Extent> InternalGetExtent() { return {}; }
  void InternalSetExtent(StaticRank<Extent>) {}
  void* InternalGetDataPointer(size_t i) const {
    return const_cast<void*>(data_[i]);
  }
  void InternalSetDataPointer(size_t i, const void* ptr) { data_[i] = ptr; }
  const void* data_[sizeof...(Ts)]{};
};
template <typename... Ts>
class MultiVectorViewStorage<0, Ts...> {
 private:
  friend class MultiVectorAccess<MultiVectorViewStorage>;
  constexpr static StaticRank<0> InternalGetExtent() { return {}; }
  void InternalSetExtent(StaticRank<0>) {}
  void* InternalGetDataPointer(size_t i) const { return nullptr; }
  void InternalSetDataPointer(size_t i, const void* ptr) {}
};
template <typename... Ts>
class MultiVectorViewStorage<dynamic_rank, Ts...> {
 private:
  friend class MultiVectorAccess<MultiVectorViewStorage>;
  ptrdiff_t InternalGetExtent() const { return extent_; }
  void InternalSetExtent(ptrdiff_t extent) { extent_ = extent; }
  void* InternalGetDataPointer(size_t i) const {
    return const_cast<void*>(data_[i]);
  }
  void InternalSetDataPointer(size_t i, const void* ptr) { data_[i] = ptr; }
  const void* data_[sizeof...(Ts)]{};
  ptrdiff_t extent_ = 0;
};
template <DimensionIndex Extent, typename... Ts>
class MultiVectorAccess<MultiVectorViewStorage<Extent, Ts...>> {
 public:
  using StorageType = MultiVectorViewStorage<Extent, Ts...>;
  using ExtentType = StaticOrDynamicRank<Extent>;
  constexpr static ptrdiff_t static_extent = Extent;
  constexpr static size_t num_vectors = sizeof...(Ts);
  template <size_t I>
  using ElementType = TypePackElement<I, Ts...>;
  template <size_t I>
  using ConstElementType = TypePackElement<I, Ts...>;
  static ExtentType GetExtent(const StorageType& storage) {
    return storage.InternalGetExtent();
  }
  template <size_t I>
  static tensorstore::span<ElementType<I>, Extent> get(
      const StorageType* array) noexcept {
    return {static_cast<ElementType<I>*>(array->InternalGetDataPointer(I)),
            array->InternalGetExtent()};
  }
  static void Assign(StorageType* array, ExtentType extent, Ts*... pointers) {
    array->InternalSetExtent(extent);
    size_t i = 0;
    (array->InternalSetDataPointer(i++, pointers), ...);
  }
  static void Assign(StorageType* array,
                     tensorstore::span<Ts, Extent>... spans) {
    const ExtentType extent =
        GetFirstArgument(GetStaticOrDynamicExtent(spans)...);
    assert(((spans.size() == extent) && ...));
    Assign(array, extent, spans.data()...);
  }
};
}  
}  
#endif  