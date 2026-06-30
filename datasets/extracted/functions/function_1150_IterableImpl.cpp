#include "tensorstore/internal/nditerable_transformed_array.h"
#include <cassert>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/internal/iterate_impl.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_array.h"
#include "tensorstore/internal/nditerable_array_util.h"
#include "tensorstore/internal/nditerable_util.h"
#include "tensorstore/internal/unique_with_intrusive_allocator.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
namespace tensorstore {
namespace internal {
namespace input_dim_iter_flags =
    internal_index_space::input_dimension_iteration_flags;
namespace {
class IterableImpl : public NDIterable::Base<IterableImpl> {
 public:
  IterableImpl(IndexTransform<> transform, allocator_type allocator)
      : transform_(std::move(transform)),
        input_dimension_flags_(transform_.input_rank(),
                               input_dim_iter_flags::can_skip, allocator) {}
  allocator_type get_allocator() const override {
    return input_dimension_flags_.get_allocator();
  }
  int GetDimensionOrder(DimensionIndex dim_i,
                        DimensionIndex dim_j) const override {
    auto flags_i = input_dimension_flags_[dim_i];
    if ((flags_i & input_dim_iter_flags::array_indexed) !=
        (input_dimension_flags_[dim_j] & input_dim_iter_flags::array_indexed)) {
      return (flags_i & input_dim_iter_flags::array_indexed) ? -2 : 2;
    }
    if (flags_i & input_dim_iter_flags::array_indexed) {
      for (DimensionIndex i = 0; i < state_.num_array_indexed_output_dimensions;
           ++i) {
        const int order = GetDimensionOrderFromByteStrides(
            state_.index_array_byte_strides[i][dim_i],
            state_.index_array_byte_strides[i][dim_j]);
        if (order != 0) return order;
      }
    }
    return GetDimensionOrderFromByteStrides(state_.input_byte_strides[dim_i],
                                            state_.input_byte_strides[dim_j]);
  }
  void UpdateDirectionPrefs(NDIterable::DirectionPref* prefs) const override {
    const DimensionIndex input_rank = transform_.input_rank();
    for (DimensionIndex i = 0; i < state_.num_array_indexed_output_dimensions;
         ++i) {
      UpdateDirectionPrefsFromByteStrides(
          tensorstore::span(state_.index_array_byte_strides[i], input_rank),
          prefs);
    }
    UpdateDirectionPrefsFromByteStrides(
        tensorstore::span(&state_.input_byte_strides[0], input_rank), prefs);
  }
  bool CanCombineDimensions(DimensionIndex dim_i, int dir_i,
                            DimensionIndex dim_j, int dir_j,
                            Index size_j) const override {
    auto flags_i = input_dimension_flags_[dim_i];
    if ((flags_i & input_dim_iter_flags::array_indexed) !=
        (input_dimension_flags_[dim_j] & input_dim_iter_flags::array_indexed)) {
      return false;
    }
    if (flags_i & input_dim_iter_flags::array_indexed) {
      for (DimensionIndex i = 0; i < state_.num_array_indexed_output_dimensions;
           ++i) {
        if (!CanCombineStridedArrayDimensions(
                state_.index_array_byte_strides[i][dim_i], dir_i,
                state_.index_array_byte_strides[i][dim_j], dir_j, size_j)) {
          return false;
        }
      }
    }
    return CanCombineStridedArrayDimensions(
        state_.input_byte_strides[dim_i], dir_i,
        state_.input_byte_strides[dim_j], dir_j, size_j);
  }
  DataType dtype() const override { return dtype_; }
  IterationBufferConstraint GetIterationBufferConstraint(
      IterationLayoutView layout) const override {
    const DimensionIndex penultimate_dim =
        layout.iteration_dimensions[layout.iteration_dimensions.size() - 2];
    const DimensionIndex last_dim =
        layout.iteration_dimensions[layout.iteration_dimensions.size() - 1];
    if ((last_dim == -1 || (input_dimension_flags_[last_dim] &
                            input_dim_iter_flags::array_indexed) == 0) &&
        (penultimate_dim == -1 || (input_dimension_flags_[penultimate_dim] &
                                   input_dim_iter_flags::array_indexed) == 0)) {
      return {(last_dim == -1 || state_.input_byte_strides[last_dim] *
                                         layout.directions[last_dim] ==
                                     this->dtype_->size)
                  ? IterationBufferKind::kContiguous
                  : IterationBufferKind::kStrided,
              false};
    } else {
      return {IterationBufferKind::kIndexed, false};
    }
  }
  std::ptrdiff_t GetWorkingMemoryBytesPerElement(
      IterationLayoutView layout,
      IterationBufferKind buffer_kind) const override {
    return buffer_kind == IterationBufferKind::kIndexed ? sizeof(Index) : 0;
  }
  NDIterator::Ptr GetIterator(
      NDIterable::IterationBufferKindLayoutView layout) const override {
    return MakeUniqueWithVirtualIntrusiveAllocator<IteratorImpl>(
        get_allocator(), this, layout);
  }
  class IteratorImpl : public NDIterator::Base<IteratorImpl> {
   public:
    IteratorImpl(const IterableImpl* iterable,
                 NDIterable::IterationBufferKindLayoutView layout,
                 allocator_type allocator)
        : num_index_arrays_(
              iterable->state_.num_array_indexed_output_dimensions),
          num_index_array_iteration_dims_(0),
          iterable_(iterable),
          buffer_(
              num_index_arrays_ +
                  layout.iteration_rank() * (num_index_arrays_ + 1) +
                  ((layout.buffer_kind == IterationBufferKind::kIndexed)
                       ? layout.block_shape[0] * layout.block_shape[1]
                       : 0),
              allocator) {
      static_assert(sizeof(Index) >= sizeof(void*));
      for (DimensionIndex j = 0; j < num_index_arrays_; ++j) {
        ByteStridedPointer<const Index> index_array_pointer =
            iterable->state_.index_array_pointers[j].get();
        for (DimensionIndex dim = 0; dim < layout.full_rank(); ++dim) {
          if (layout.directions[dim] != -1) continue;
          const Index size_minus_1 = layout.shape[dim] - 1;
          const Index index_array_byte_stride =
              iterable->state_.index_array_byte_strides[j][dim];
          index_array_pointer +=
              wrap_on_overflow::Multiply(index_array_byte_stride, size_minus_1);
        }
        buffer_[j] = reinterpret_cast<Index>(index_array_pointer.get());
      }
      Index base_offset = 0;
      for (DimensionIndex dim = 0; dim < layout.full_rank(); ++dim) {
        if (layout.directions[dim] != -1) continue;
        const Index size_minus_1 = layout.shape[dim] - 1;
        const Index input_byte_stride =
            iterable->state_.input_byte_strides[dim];
        base_offset = wrap_on_overflow::Add(
            base_offset,
            wrap_on_overflow::Multiply(input_byte_stride, size_minus_1));
      }
      for (DimensionIndex i = 0; i < layout.iteration_rank(); ++i) {
        const DimensionIndex dim = layout.iteration_dimensions[i];
        if (dim == -1) {
          for (DimensionIndex j = 0; j < num_index_arrays_ + 1; ++j) {
            buffer_[num_index_arrays_ + layout.iteration_rank() * j + i] = 0;
          }
        } else {
          const Index dir = layout.directions[dim];
          const Index input_byte_stride =
              iterable->state_.input_byte_strides[dim];
          buffer_[num_index_arrays_ + i] =
              wrap_on_overflow::Multiply(input_byte_stride, dir);
          if (iterable->input_dimension_flags_[dim] &
              input_dim_iter_flags::array_indexed) {
            num_index_array_iteration_dims_ = i + 1;
            for (DimensionIndex j = 0; j < num_index_arrays_; ++j) {
              const Index index_array_byte_stride =
                  iterable->state_.index_array_byte_strides[j][dim];
              buffer_[num_index_arrays_ + layout.iteration_rank() * (j + 1) +
                      i] =
                  wrap_on_overflow::Multiply(index_array_byte_stride, dir);
            }
          }
        }
      }
      if (layout.buffer_kind == IterationBufferKind::kIndexed) {
        Index* offsets_array =
            buffer_.data() + num_index_arrays_ +
            layout.iteration_rank() * (num_index_arrays_ + 1);
        pointer_ =
            IterationBufferPointer{iterable->state_.base_pointer + base_offset,
                                   layout.block_shape[1], offsets_array};
        if (num_index_array_iteration_dims_ + 1 < layout.iteration_rank()) {
          FillOffsetsArrayFromStride(
              buffer_[num_index_arrays_ + layout.iteration_rank() - 2],
              buffer_[num_index_arrays_ + layout.iteration_rank() - 1],
              layout.block_shape[0], layout.block_shape[1], offsets_array);
        }
      } else {
        assert(num_index_array_iteration_dims_ + 1 < layout.iteration_rank());
        pointer_ = IterationBufferPointer{
            iterable->state_.base_pointer + base_offset,
            buffer_[num_index_arrays_ + layout.iteration_rank() - 2],
            buffer_[num_index_arrays_ + layout.iteration_rank() - 1]};
      }
    }
    allocator_type get_allocator() const override {
      return buffer_.get_allocator();
    }
    bool GetBlock(tensorstore::span<const Index> indices,
                  IterationBufferShape block_shape,
                  IterationBufferPointer* pointer,
                  absl::Status* status) override {
      IterationBufferPointer block_pointer = pointer_;
      block_pointer.pointer += IndexInnerProduct(
          indices.size(), indices.data(), buffer_.data() + num_index_arrays_);
      if (num_index_array_iteration_dims_ + 1 < indices.size()) {
        for (DimensionIndex j = 0; j < num_index_arrays_; ++j) {
          const Index index = ByteStridedPointer<const Index>(
              reinterpret_cast<const Index*>(buffer_[j]))[IndexInnerProduct(
              num_index_array_iteration_dims_, indices.data(),
              buffer_.data() + num_index_arrays_ + indices.size() * (j + 1))];
          block_pointer.pointer += wrap_on_overflow::Multiply(
              iterable_->state_.index_array_output_byte_strides[j], index);
        }
      } else {
        block_pointer.byte_offsets_outer_stride = block_shape[1];
        Index* offsets_array = const_cast<Index*>(block_pointer.byte_offsets);
        FillOffsetsArrayFromStride(
            buffer_[num_index_arrays_ + indices.size() - 2],
            buffer_[num_index_arrays_ + indices.size() - 1], block_shape[0],
            block_shape[1], offsets_array);
        for (DimensionIndex j = 0; j < num_index_arrays_; ++j) {
          const Index* index_array_byte_strides =
              buffer_.data() + num_index_arrays_ + indices.size() * (j + 1);
          ByteStridedPointer<const Index> index_array_pointer =
              ByteStridedPointer<const Index>(
                  reinterpret_cast<const Index*>(buffer_[j])) +
              IndexInnerProduct(indices.size() - 2, indices.data(),
                                index_array_byte_strides);
          const Index output_byte_stride =
              iterable_->state_.index_array_output_byte_strides[j];
          const Index penultimate_index_array_byte_stride =
              index_array_byte_strides[indices.size() - 2];
          const Index last_index_array_byte_stride =
              index_array_byte_strides[indices.size() - 1];
          if (last_index_array_byte_stride == 0 &&
              penultimate_index_array_byte_stride == 0) {
            block_pointer.pointer += wrap_on_overflow::Multiply(
                output_byte_stride, *index_array_pointer);
          } else {
            Index block_start0 = indices[indices.size() - 2];
            Index block_start1 = indices[indices.size() - 1];
            for (Index outer = 0; outer < block_shape[0]; ++outer) {
              for (Index inner = 0; inner < block_shape[1]; ++inner) {
                Index cur_contribution = wrap_on_overflow::Multiply(
                    output_byte_stride,
                    index_array_pointer[wrap_on_overflow::Add(
                        wrap_on_overflow::Multiply(
                            outer + block_start0,
                            penultimate_index_array_byte_stride),
                        wrap_on_overflow::Multiply(
                            inner + block_start1,
                            last_index_array_byte_stride))]);
                auto& offset = offsets_array[outer * block_shape[1] + inner];
                offset = wrap_on_overflow::Add(offset, cur_contribution);
              }
            }
          }
        }
      }
      *pointer = block_pointer;
      return true;
    }
   private:
    DimensionIndex num_index_arrays_;
    DimensionIndex num_index_array_iteration_dims_;
    const IterableImpl* iterable_;
    IterationBufferPointer pointer_;
    std::vector<Index, ArenaAllocator<Index>> buffer_;
  };
  std::shared_ptr<const void> data_owner_;
  IndexTransform<> transform_;
  internal_index_space::SingleArrayIterationState state_;
  DataType dtype_;
  std::vector<input_dim_iter_flags::Bitmask,
              ArenaAllocator<input_dim_iter_flags::Bitmask>>
      input_dimension_flags_;
};
Result<NDIterable::Ptr> MaybeConvertToArrayNDIterable(
    std::unique_ptr<IterableImpl, VirtualDestroyDeleter> impl, Arena* arena) {
  if (impl->state_.num_array_indexed_output_dimensions == 0) {
    return GetArrayNDIterable(
        SharedOffsetArrayView<const void>(
            SharedElementPointer<const void>(
                std::shared_ptr<const void>(std::move(impl->data_owner_),
                                            impl->state_.base_pointer),
                impl->dtype_),
            StridedLayoutView<>(impl->transform_.input_rank(),
                                impl->transform_.input_shape().data(),
                                &impl->state_.input_byte_strides[0])),
        arena);
  }
  return impl;
}
}  
Result<NDIterable::Ptr> GetTransformedArrayNDIterable(
    SharedOffsetArrayView<const void> array, IndexTransformView<> transform,
    Arena* arena) {
  if (!transform.valid()) {
    return GetArrayNDIterable(array, arena);
  }
  auto impl = MakeUniqueWithVirtualIntrusiveAllocator<IterableImpl>(
      ArenaAllocator<>(arena), transform);
  TENSORSTORE_RETURN_IF_ERROR(InitializeSingleArrayIterationState(
      array, internal_index_space::TransformAccess::rep(transform),
      transform.input_origin().data(), transform.input_shape().data(),
      &impl->state_, impl->input_dimension_flags_.data()));
  impl->dtype_ = array.dtype();
  impl->data_owner_ = std::move(array.element_pointer().pointer());
  return MaybeConvertToArrayNDIterable(std::move(impl), arena);
}
Result<NDIterable::Ptr> GetTransformedArrayNDIterable(
    TransformedArray<Shared<const void>> array, Arena* arena) {
  auto impl = MakeUniqueWithVirtualIntrusiveAllocator<IterableImpl>(
      ArenaAllocator<>(arena), std::move(array.transform()));
  TENSORSTORE_RETURN_IF_ERROR(InitializeSingleArrayIterationState(
      ElementPointer<const void>(array.element_pointer()),
      internal_index_space::TransformAccess::rep(impl->transform_),
      impl->transform_.input_origin().data(),
      impl->transform_.input_shape().data(), &impl->state_,
      impl->input_dimension_flags_.data()));
  impl->dtype_ = array.dtype();
  impl->data_owner_ = std::move(array.element_pointer().pointer());
  return MaybeConvertToArrayNDIterable(std::move(impl), arena);
}
}  
}  