#include "tensorstore/internal/nditerable_data_type_conversion.h"
#include <cassert>
#include <memory>
#include <utility>
#include "tensorstore/data_type.h"
#include "tensorstore/data_type_conversion.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_elementwise_input_transform.h"
#include "tensorstore/internal/nditerable_elementwise_output_transform.h"
#include "tensorstore/internal/nditerable_util.h"
#include "tensorstore/internal/unique_with_intrusive_allocator.h"
namespace tensorstore {
namespace internal {
namespace {
template <typename Derived, typename BasePointer = NDIterable::Ptr>
class NDIterableAdapter : public NDIterable::Base<Derived> {
 public:
  NDIterableAdapter(BasePointer base) : base_(std::move(base)) {}
  const BasePointer& base() const { return base_; }
  BasePointer& base() { return base_; }
  int GetDimensionOrder(DimensionIndex dim_i,
                        DimensionIndex dim_j) const override {
    return base_->GetDimensionOrder(dim_i, dim_j);
  }
  void UpdateDirectionPrefs(NDIterable::DirectionPref* prefs) const override {
    base_->UpdateDirectionPrefs(prefs);
  }
  bool CanCombineDimensions(DimensionIndex dim_i, int dir_i,
                            DimensionIndex dim_j, int dir_j,
                            Index size_j) const override {
    return base_->CanCombineDimensions(dim_i, dir_i, dim_j, dir_j, size_j);
  }
  NDIterable::IterationBufferConstraint GetIterationBufferConstraint(
      NDIterable::IterationLayoutView layout) const override {
    return base_->GetIterationBufferConstraint(layout);
  }
  std::ptrdiff_t GetWorkingMemoryBytesPerElement(
      NDIterable::IterationLayoutView layout,
      IterationBufferKind buffer_kind) const override {
    return base_->GetWorkingMemoryBytesPerElement(layout, buffer_kind);
  }
  DataType dtype() const override { return base_->dtype(); }
  ArenaAllocator<> get_allocator() const override {
    return base_->get_allocator();
  }
  NDIterator::Ptr GetIterator(
      NDIterable::IterationBufferKindLayoutView layout) const override {
    return base_->GetIterator(layout);
  }
 private:
  BasePointer base_;
};
class ReinterpretCastNDIterable
    : public NDIterableAdapter<ReinterpretCastNDIterable> {
 public:
  ReinterpretCastNDIterable(NDIterable::Ptr base, DataType new_dtype,
                            ArenaAllocator<> allocator)
      : NDIterableAdapter<ReinterpretCastNDIterable>(std::move(base)),
        dtype_(new_dtype) {}
  DataType dtype() const override { return dtype_; }
 private:
  DataType dtype_;
};
}  
NDIterable::Ptr GetConvertedInputNDIterable(
    NDIterable::Ptr iterable, DataType target_type,
    const DataTypeConversionLookupResult& conversion) {
  assert(DataTypeConversionFlags::kSupported ==
         (conversion.flags & DataTypeConversionFlags::kSupported));
  if (DataTypeConversionFlags::kIdentity ==
      (conversion.flags & DataTypeConversionFlags::kIdentity)) {
    return iterable;
  }
  auto allocator = iterable->get_allocator();
  if (DataTypeConversionFlags::kCanReinterpretCast ==
      (conversion.flags & DataTypeConversionFlags::kCanReinterpretCast)) {
    return MakeUniqueWithVirtualIntrusiveAllocator<ReinterpretCastNDIterable>(
        allocator, std::move(iterable), target_type);
  }
  return GetElementwiseInputTransformNDIterable({{std::move(iterable)}},
                                                target_type, conversion.closure,
                                                allocator.arena());
}
NDIterable::Ptr GetConvertedOutputNDIterable(
    NDIterable::Ptr iterable, DataType source_type,
    const DataTypeConversionLookupResult& conversion) {
  assert(!!(conversion.flags & DataTypeConversionFlags::kSupported));
  if (!!(conversion.flags & DataTypeConversionFlags::kIdentity)) {
    return iterable;
  }
  auto allocator = iterable->get_allocator();
  if (!!(conversion.flags & DataTypeConversionFlags::kCanReinterpretCast)) {
    return MakeUniqueWithVirtualIntrusiveAllocator<ReinterpretCastNDIterable>(
        allocator, std::move(iterable), source_type);
  }
  return GetElementwiseOutputTransformNDIterable(
      std::move(iterable), source_type, conversion.closure, allocator.arena());
}
}  
}  