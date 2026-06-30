#ifndef AROLLA_DENSE_ARRAY_QTYPE_COPIER_H_
#define AROLLA_DENSE_ARRAY_QTYPE_COPIER_H_
#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "arolla/dense_array/bitmap.h"
#include "arolla/dense_array/dense_array.h"
#include "arolla/memory/buffer.h"
#include "arolla/memory/frame.h"
#include "arolla/memory/optional_value.h"
#include "arolla/memory/raw_buffer_factory.h"
#include "arolla/qtype/array_like/array_like_qtype.h"
#include "arolla/qtype/optional_qtype.h"
#include "arolla/qtype/qtype.h"
#include "arolla/qtype/qtype_traits.h"
#include "arolla/qtype/typed_ref.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla {
template <class T>
class DenseArray2FramesCopier : public BatchToFramesCopier {
 public:
  absl::Status AddMapping(TypedRef array_ptr, TypedSlot scalar_slot) final {
    if (IsStarted()) {
      return absl::FailedPreconditionError(
          "can't add new mappings when started");
    }
    ASSIGN_OR_RETURN(const DenseArray<T>& array, array_ptr.As<DenseArray<T>>());
    RETURN_IF_ERROR(SetRowCount(array.size()));
    if (scalar_slot.GetType() == GetQType<T>()) {
      mappings_.push_back(Mapping{array, scalar_slot.UnsafeToSlot<T>()});
    } else {
      ASSIGN_OR_RETURN(auto slot, scalar_slot.ToSlot<OptionalValue<T>>());
      mappings_.push_back(Mapping{array, slot});
    }
    return absl::OkStatus();
  }
  void CopyNextBatch(absl::Span<FramePtr> output_buffers) final {
    if (!IsStarted()) Start();  
    for (const auto& mapping : mappings_) {
      auto iter = mapping.array.values.begin() + current_row_id_;
      if (std::holds_alternative<FrameLayout::Slot<T>>(mapping.scalar_slot)) {
        auto scalar_slot =
            *std::get_if<FrameLayout::Slot<T>>(&mapping.scalar_slot);
        for (size_t i = 0; i < output_buffers.size(); ++i) {
          output_buffers[i].Set(scalar_slot, T(iter[i]));
        }
      } else {
        DCHECK(std::holds_alternative<FrameLayout::Slot<OptionalValue<T>>>(
            mapping.scalar_slot));
        auto scalar_slot = *std::get_if<FrameLayout::Slot<OptionalValue<T>>>(
            &mapping.scalar_slot);
        if (mapping.array.bitmap.empty()) {
          for (FramePtr frame : output_buffers) {
            frame.Set(scalar_slot, {true, T(*(iter++))});
          }
        } else {
          bitmap::IterateByGroups(
              mapping.array.bitmap.begin(),
              current_row_id_ + mapping.array.bitmap_bit_offset,
              output_buffers.size(), [&](int64_t offset) {
                FramePtr* frame_group = output_buffers.begin() + offset;
                auto values = iter + offset;
                return [=](int i, bool present) {
                  frame_group[i].Set(scalar_slot, {present, T(values[i])});
                };
              });
          iter += output_buffers.size();
        }
      }
    }
    current_row_id_ += output_buffers.size();
  }
 private:
  struct Mapping {
    const DenseArray<T>& array;
    std::variant<FrameLayout::Slot<T>, FrameLayout::Slot<OptionalValue<T>>>
        scalar_slot;
  };
  std::vector<Mapping> mappings_;
  int64_t current_row_id_ = 0;
};
template <class T>
class Frames2DenseArrayCopier : public BatchFromFramesCopier {
 public:
  explicit Frames2DenseArrayCopier(
      RawBufferFactory* buffer_factory = GetHeapBufferFactory())
      : buffer_factory_(buffer_factory) {}
  absl::Status AddMapping(TypedSlot scalar_slot, TypedSlot array_slot) final {
    if (IsStarted()) {
      return absl::FailedPreconditionError(
          "can't add new mappings when started");
    }
    ASSIGN_OR_RETURN(auto dst_slot, array_slot.ToSlot<DenseArray<T>>());
    if (IsOptionalQType(scalar_slot.GetType())) {
      ASSIGN_OR_RETURN(auto src_slot, scalar_slot.ToSlot<OptionalValue<T>>());
      mappings_.push_back(Mapping{src_slot, dst_slot});
    } else {
      ASSIGN_OR_RETURN(auto src_slot, scalar_slot.ToSlot<T>());
      mappings_.push_back(Mapping{src_slot, dst_slot});
    }
    return absl::OkStatus();
  }
  absl::Status CopyNextBatch(
      absl::Span<const ConstFramePtr> input_buffers) final {
    if (!IsStarted()) {
      return absl::FailedPreconditionError(
          "start(row_count) should be called before CopyNextBatch");
    }
    for (Mapping& mapping : mappings_) {
      DCHECK(mapping.values_builder.has_value());
      if (std::holds_alternative<FrameLayout::Slot<T>>(mapping.scalar_slot)) {
        auto scalar_slot =
            *std::get_if<FrameLayout::Slot<T>>(&mapping.scalar_slot);
        const ConstFramePtr* iter = input_buffers.data();
        mapping.values_builder->SetN(
            current_row_id_, input_buffers.size(),
            [&] { return (iter++)->Get(scalar_slot); });
      } else {
        DCHECK(std::holds_alternative<FrameLayout::Slot<OptionalValue<T>>>(
            mapping.scalar_slot));
        DCHECK(mapping.bitmap_builder.has_value());
        auto scalar_slot = *std::get_if<FrameLayout::Slot<OptionalValue<T>>>(
            &mapping.scalar_slot);
        auto values_inserter =
            mapping.values_builder->GetInserter(current_row_id_);
        auto fn = [&](ConstFramePtr frame) {
          const OptionalValue<T>& v = frame.Get(scalar_slot);
          values_inserter.Add(v.value);
          return v.present;
        };
        mapping.bitmap_builder->AddForEach(input_buffers, fn);
      }
    }
    current_row_id_ += input_buffers.size();
    return absl::OkStatus();
  }
  absl::Status Finalize(FramePtr arrays_frame) final {
    if (finished_) {
      return absl::FailedPreconditionError("finalize can be called only once");
    }
    finished_ = true;
    for (Mapping& mapping : mappings_) {
      DCHECK(mapping.values_builder.has_value());
      DenseArray<T> res;
      res.values = (*std::move(mapping.values_builder)).Build();
      if (mapping.bitmap_builder.has_value()) {
        res.bitmap = (*std::move(mapping.bitmap_builder)).Build();
      }
      arrays_frame.Set(mapping.array_slot, std::move(res));
    }
    return absl::OkStatus();
  }
 private:
  struct Mapping {
    std::variant<FrameLayout::Slot<T>, FrameLayout::Slot<OptionalValue<T>>>
        scalar_slot;
    FrameLayout::Slot<DenseArray<T>> array_slot;
    std::optional<typename Buffer<T>::Builder> values_builder;
    std::optional<bitmap::Builder> bitmap_builder;
  };
  void SetArraySize(int64_t size) final {
    for (auto& mapping : mappings_) {
      mapping.values_builder.emplace(size, buffer_factory_);
      if (mapping.scalar_slot.index() == 1) {
        mapping.bitmap_builder.emplace(size, buffer_factory_);
      }
    }
  }
  std::vector<Mapping> mappings_;
  int64_t current_row_id_ = 0;
  bool finished_ = false;
  RawBufferFactory* buffer_factory_;
};
}  
#endif  