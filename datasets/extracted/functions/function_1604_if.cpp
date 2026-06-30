#ifndef AROLLA_QEXPR_OPERATORS_DENSE_ARRAY_LOGIC_OPS_H_
#define AROLLA_QEXPR_OPERATORS_DENSE_ARRAY_LOGIC_OPS_H_
#include <cstdint>
#include <cstring>
#include <utility>
#include "absl/base/optimization.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "arolla/dense_array/bitmap.h"
#include "arolla/dense_array/dense_array.h"
#include "arolla/dense_array/ops/dense_ops.h"
#include "arolla/dense_array/qtype/types.h"
#include "arolla/memory/buffer.h"
#include "arolla/memory/optional_value.h"
#include "arolla/qexpr/eval_context.h"
#include "arolla/util/unit.h"
#include "arolla/util/view_types.h"
namespace arolla {
struct DenseArrayHasOp {
  template <typename T>
  DenseArray<Unit> operator()(const DenseArray<T>& arg) const {
    return {VoidBuffer(arg.size()), arg.bitmap, arg.bitmap_bit_offset};
  }
};
struct DenseArrayPresenceAndOp {
  template <typename T>
  absl::StatusOr<DenseArray<T>> operator()(EvaluationContext* ctx,
                                           const DenseArray<T>& lhs,
                                           const DenseArray<Unit>& rhs) const {
    if (ABSL_PREDICT_FALSE(lhs.size() != rhs.size())) {
      return SizeMismatchError({lhs.size(), rhs.size()});
    }
    if (rhs.bitmap.empty()) {
      return lhs;
    } else if (lhs.bitmap.empty()) {
      return DenseArray<T>{lhs.values, rhs.bitmap, rhs.bitmap_bit_offset};
    } else {
      int64_t bitmap_size = bitmap::BitmapSize(lhs.size());
      bitmap::RawBuilder bldr(bitmap_size, &ctx->buffer_factory());
      bitmap::Intersect(lhs.bitmap, rhs.bitmap, lhs.bitmap_bit_offset,
                        rhs.bitmap_bit_offset, bldr.GetMutableSpan());
      return DenseArray<T>{
          lhs.values, std::move(bldr).Build(),
          std::min(lhs.bitmap_bit_offset, rhs.bitmap_bit_offset)};
    }
  }
};
struct DenseArrayPresenceNotOp {
  template <typename T>
  DenseArray<Unit> operator()(EvaluationContext* ctx,
                              const DenseArray<T>& arg) const {
    if (arg.bitmap.empty()) {
      return CreateEmptyDenseArray<Unit>(arg.size(), &ctx->buffer_factory());
    }
    absl::Span<const bitmap::Word> bitmap_in = arg.bitmap.span();
    int64_t first_not_zero_index = 0;
    int64_t bitmap_size = arg.bitmap.size();
    while (first_not_zero_index < bitmap_size &&
           bitmap_in[first_not_zero_index] == 0) {
      first_not_zero_index++;
    }
    if (first_not_zero_index == bitmap_size) {
      return {VoidBuffer(arg.size())};
    }
    bitmap::RawBuilder bldr(bitmap_size, &ctx->buffer_factory());
    absl::Span<bitmap::Word> bitmap_out = bldr.GetMutableSpan();
    if (first_not_zero_index > 0) {
      std::memset(bitmap_out.data(), 0xff,
                  sizeof(bitmap::Word) * first_not_zero_index);
    }
    for (int64_t i = first_not_zero_index; i < bitmap_size; ++i) {
      bitmap_out[i] = ~bitmap_in[i];
    }
    return {VoidBuffer(arg.size()), std::move(bldr).Build(),
            arg.bitmap_bit_offset};
  }
};
struct DenseArrayPresenceOrOp {
  template <typename T>
  absl::StatusOr<DenseArray<T>> operator()(EvaluationContext* ctx,
                                           const DenseArray<T>& lhs,
                                           const DenseArray<T>& rhs) const {
    if (ABSL_PREDICT_FALSE(lhs.size() != rhs.size())) {
      return SizeMismatchError({lhs.size(), rhs.size()});
    }
    if (lhs.bitmap.empty()) {
      return lhs;
    } else if (bitmap::AreAllBitsUnset(lhs.bitmap.begin(), lhs.size())) {
      return rhs;
    } else {
      auto fn = [&](OptionalValue<view_type_t<T>> a,
                    OptionalValue<view_type_t<T>> b) {
        return OptionalValue<view_type_t<T>>{a.present || b.present,
                                             a.present ? a.value : b.value};
      };
      return CreateDenseOp<DenseOpFlags::kRunOnMissing |
                               DenseOpFlags::kNoBitmapOffset |
                               DenseOpFlags::kNoSizeValidation,
                           decltype(fn), T>(fn, &ctx->buffer_factory())(lhs,
                                                                        rhs);
    }
  }
  template <typename T>
  DenseArray<T> operator()(EvaluationContext* ctx, const DenseArray<T>& lhs,
                           const OptionalValue<T>& rhs) const {
    if (!rhs.present || lhs.bitmap.empty()) {
      return lhs;
    } else if (bitmap::AreAllBitsUnset(lhs.bitmap.begin(), lhs.size())) {
      return CreateConstDenseArray<T>(lhs.size(), rhs.value,
                                      &ctx->buffer_factory());
    } else {
      auto fn = [value = rhs.value](OptionalValue<view_type_t<T>> a) {
        return a.present ? a.value : value;
      };
      return CreateDenseOp<DenseOpFlags::kRunOnMissing |
                               DenseOpFlags::kNoBitmapOffset |
                               DenseOpFlags::kNoSizeValidation,
                           decltype(fn), T>(fn, &ctx->buffer_factory())(lhs);
    }
  }
};
}  
#endif  