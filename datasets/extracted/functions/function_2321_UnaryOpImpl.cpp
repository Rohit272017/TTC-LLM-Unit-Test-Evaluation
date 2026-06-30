#ifndef AROLLA_DENSE_ARRAY_OPS_DENSE_OPS_H_
#define AROLLA_DENSE_ARRAY_OPS_DENSE_OPS_H_
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "arolla/dense_array/bitmap.h"
#include "arolla/dense_array/dense_array.h"
#include "arolla/dense_array/ops/universal_dense_op.h"  
#include "arolla/dense_array/ops/util.h"
#include "arolla/memory/buffer.h"
#include "arolla/memory/optional_value.h"
#include "arolla/memory/raw_buffer_factory.h"
#include "arolla/util/meta.h"
#include "arolla/util/unit.h"
#include "arolla/util/view_types.h"
namespace arolla {
struct DenseOpFlags {
  static constexpr int kRunOnMissing = 1 << 0;
  static constexpr int kNoBitmapOffset = 1 << 1;
  static constexpr int kNoSizeValidation = 1 << 2;
};
namespace dense_ops_internal {
template <class Fn>
struct SpanOp {
  Fn fn;
  template <class Res, class... Ts>
  void operator()(absl::Span<Res> res, absl::Span<const Ts>... args) const {
    for (size_t i = 0; i < res.size(); ++i) {
      res[i] = fn(args[i]...);
    }
  }
};
template <class ResT, class SpanOpT>
class UnaryOpImpl {
 public:
  explicit UnaryOpImpl(
      SpanOpT op, RawBufferFactory* buffer_factory = GetHeapBufferFactory())
      : op_(op), buffer_factory_(buffer_factory) {}
  template <class ArgT>
  DenseArray<ResT> operator()(const DenseArray<ArgT>& arg) const {
    typename Buffer<ResT>::Builder builder(arg.size(), buffer_factory_);
    op_(builder.GetMutableSpan(), arg.values.span());
    return {std::move(builder).Build(), arg.bitmap, arg.bitmap_bit_offset};
  }
 private:
  SpanOpT op_;
  RawBufferFactory* buffer_factory_;
};
template <class ResT, bool NoBitmapOffset, class SpanOpT>
class BinaryOpImpl {
 public:
  explicit BinaryOpImpl(
      SpanOpT op, RawBufferFactory* buffer_factory = GetHeapBufferFactory())
      : op_(op), buffer_factory_(buffer_factory) {}
  template <class Arg1T, class Arg2T>
  DenseArray<ResT> operator()(const DenseArray<Arg1T>& arg1,
                              const DenseArray<Arg2T>& arg2) const {
    DCHECK_EQ(arg1.size(), arg2.size());
    DCHECK(!NoBitmapOffset ||
           (arg1.bitmap_bit_offset == 0 && arg2.bitmap_bit_offset == 0));
    typename Buffer<ResT>::Builder builder(arg1.size(), buffer_factory_);
    op_(builder.GetMutableSpan(), arg1.values.span(), arg2.values.span());
    if (arg2.bitmap.empty()) {
      return {std::move(builder).Build(), arg1.bitmap, arg1.bitmap_bit_offset};
    } else if (arg1.bitmap.empty()) {
      return {std::move(builder).Build(), arg2.bitmap, arg2.bitmap_bit_offset};
    } else {
      bitmap::RawBuilder bitmap_builder(
          std::min(arg1.bitmap.size(), arg2.bitmap.size()), buffer_factory_);
      int res_bit_offset = 0;
      if constexpr (NoBitmapOffset) {
        bitmap::Intersect(arg1.bitmap, arg2.bitmap,
                          bitmap_builder.GetMutableSpan());
      } else {
        res_bit_offset =
            std::min(arg1.bitmap_bit_offset, arg2.bitmap_bit_offset);
        bitmap::Intersect(arg1.bitmap, arg2.bitmap, arg1.bitmap_bit_offset,
                          arg2.bitmap_bit_offset,
                          bitmap_builder.GetMutableSpan());
      }
      return {std::move(builder).Build(), std::move(bitmap_builder).Build(),
              res_bit_offset};
    }
  }
 private:
  SpanOpT op_;
  RawBufferFactory* buffer_factory_;
};
template <class ResT, class SpanOpT>
class SimpleOpImpl {
 public:
  explicit SimpleOpImpl(
      SpanOpT op, RawBufferFactory* buffer_factory = GetHeapBufferFactory())
      : op_(op), buffer_factory_(buffer_factory) {}
  template <class Arg1T, class... ArgsT>
  DenseArray<ResT> operator()(const DenseArray<Arg1T>& arg1,
                              const DenseArray<ArgsT>&... args) const {
    DCHECK(((arg1.size() == args.size()) && ... && true));
    DCHECK(arg1.bitmap_bit_offset == 0 &&
           ((args.bitmap_bit_offset == 0) && ... && true));
    typename Buffer<ResT>::Builder builder(arg1.size(), buffer_factory_);
    op_(builder.GetMutableSpan(), arg1.values.span(), args.values.span()...);
    if ((args.bitmap.empty() && ... && true)) {
      return {std::move(builder).Build(), arg1.bitmap};
    } else {
      size_t bitmap_size = bitmap::BitmapSize(arg1.size());
      bitmap::RawBuilder bitmap_builder(bitmap_size, buffer_factory_);
      bitmap::Word* bitmap = bitmap_builder.GetMutableSpan().begin();
      bool initialized = false;
      auto intersect_fn = [&](const bitmap::Bitmap& b) {
        if (b.empty()) return;
        const bitmap::Word* ptr = b.begin();
        if (initialized) {
          for (int64_t i = 0; i < bitmap_size; ++i) {
            bitmap[i] &= ptr[i];
          }
        } else {
          std::memcpy(bitmap, ptr, bitmap_size * sizeof(bitmap::Word));
          initialized = true;
        }
      };
      intersect_fn(arg1.bitmap);
      (intersect_fn(args.bitmap), ...);
      return {std::move(builder).Build(), std::move(bitmap_builder).Build()};
    }
  }
 private:
  SpanOpT op_;
  RawBufferFactory* buffer_factory_;
};
template <class ResT, class Op>
class OpWithSizeValidation {
 public:
  explicit OpWithSizeValidation(Op op) : op_(std::move(op)) {}
  template <class... Args>
  absl::StatusOr<DenseArray<ResT>> operator()(const Args&... args) const {
    if (!AreSizesEqual(args...)) {
      return SizeMismatchError({args.size()...});
    }
    return op_(args...);
  }
 private:
  template <class A, class... As>
  static bool AreSizesEqual(const A& a, const As&... as) {
    return ((a.size() == as.size()) && ...);
  }
  Op op_;
};
template <class TypeList>
struct ArgsAnalyzer {};
template <class... Ts>
struct ArgsAnalyzer<meta::type_list<Ts...>> {
  static constexpr bool kHasOptionalArg =
      (meta::is_wrapped_with<OptionalValue, Ts>::value || ...);
  static constexpr bool kHasStringArg =
      (std::is_same_v<view_type_t<Ts>, absl::string_view> || ...);
  static constexpr bool kHasUnitArg =
      (std::is_same_v<view_type_t<Ts>, Unit> || ...);
};
template <class Fn, int flags>
struct ImplChooser {
  static constexpr bool kRunOnMissing = flags & DenseOpFlags::kRunOnMissing;
  static constexpr bool kNoBitmapOffset = flags & DenseOpFlags::kNoBitmapOffset;
  static constexpr bool kNoSizeValidation =
      flags & DenseOpFlags::kNoSizeValidation;
  static constexpr int kArgCount = meta::function_traits<Fn>::arity;
  using args = ArgsAnalyzer<typename meta::function_traits<Fn>::arg_types>;
  using fn_return_t = typename meta::function_traits<Fn>::return_type;
  static constexpr bool kComplicatedFn =
      meta::is_wrapped_with<absl::StatusOr, fn_return_t>::value ||
      meta::is_wrapped_with<OptionalValue, fn_return_t>::value ||
      std::is_same_v<view_type_t<fn_return_t>, absl::string_view> ||
      args::kHasOptionalArg || args::kHasStringArg || args::kHasUnitArg ||
      !(flags & DenseOpFlags::kRunOnMissing);
  static constexpr bool kWithSizeValidationOp =
      !kNoSizeValidation && kArgCount > 1;
  static constexpr bool kCanUseUnaryOp = kArgCount == 1 && !kComplicatedFn;
  static constexpr bool kCanUseBinaryOp =
      kArgCount == 2 && !kComplicatedFn && !kWithSizeValidationOp;
  static constexpr bool kCanUseSimpleOp = kArgCount > 2 && !kComplicatedFn &&
                                          kNoBitmapOffset &&
                                          !kWithSizeValidationOp;
};
template <class Fn, class ResT, int flags, class = void>
struct ImplSwitcher {
  using Chooser = ImplChooser<Fn, flags>;
  using Impl = UniversalDenseOp<Fn, ResT, !Chooser::kRunOnMissing,
                                Chooser::kNoBitmapOffset>;
  static Impl Create(Fn fn, RawBufferFactory* buffer_factory) {
    return Impl(fn, buffer_factory);
  }
};
template <class Fn, class ResT, int flags>
struct ImplSwitcher<
    Fn, ResT, flags,
    std::enable_if_t<ImplChooser<Fn, flags>::kCanUseUnaryOp, void>> {
  using Impl = UnaryOpImpl<ResT, SpanOp<Fn>>;
  static Impl Create(Fn fn, RawBufferFactory* buffer_factory) {
    return Impl({fn}, buffer_factory);
  }
};
template <class Fn, class ResT, int flags>
struct ImplSwitcher<
    Fn, ResT, flags,
    std::enable_if_t<ImplChooser<Fn, flags>::kCanUseBinaryOp, void>> {
  using Chooser = ImplChooser<Fn, flags>;
  using Impl = BinaryOpImpl<ResT, Chooser::kNoBitmapOffset, SpanOp<Fn>>;
  static Impl Create(Fn fn, RawBufferFactory* buffer_factory) {
    return Impl({fn}, buffer_factory);
  }
};
template <class Fn, class ResT, int flags>
struct ImplSwitcher<
    Fn, ResT, flags,
    std::enable_if_t<ImplChooser<Fn, flags>::kCanUseSimpleOp, void>> {
  using Chooser = ImplChooser<Fn, flags>;
  using Impl = SimpleOpImpl<ResT, SpanOp<Fn>>;
  static Impl Create(Fn fn, RawBufferFactory* buffer_factory) {
    return Impl({fn}, buffer_factory);
  }
};
template <class Fn, class ResT, int flags>
struct ImplSwitcher<
    Fn, ResT, flags,
    std::enable_if_t<ImplChooser<Fn, flags>::kWithSizeValidationOp, void>> {
  using BaseSwitcher =
      ImplSwitcher<Fn, ResT, flags | DenseOpFlags::kNoSizeValidation>;
  using Impl = OpWithSizeValidation<ResT, typename BaseSwitcher::Impl>;
  template <class... Args>
  static Impl Create(Args&&... args) {
    return Impl(BaseSwitcher::Create(args...));
  }
};
template <class Fn>
using result_base_t = strip_optional_t<meta::strip_template_t<
    absl::StatusOr, typename meta::function_traits<Fn>::return_type>>;
}  
template <class Fn, class ResT = dense_ops_internal::result_base_t<Fn>,
          int flags = 0>
using DenseOp =
    typename dense_ops_internal::ImplSwitcher<Fn, ResT, flags>::Impl;
template <class Fn, class ResT = dense_ops_internal::result_base_t<Fn>>
DenseOp<Fn, ResT> CreateDenseOp(
    Fn fn, RawBufferFactory* buf_factory = GetHeapBufferFactory()) {
  return dense_ops_internal::ImplSwitcher<Fn, ResT, 0>::Create(fn, buf_factory);
}
template <int flags, class Fn,
          class ResT = dense_ops_internal::result_base_t<Fn>>
DenseOp<Fn, ResT, flags> CreateDenseOp(
    Fn fn, RawBufferFactory* buf_factory = GetHeapBufferFactory()) {
  return dense_ops_internal::ImplSwitcher<Fn, ResT, flags>::Create(fn,
                                                                   buf_factory);
}
template <class ResT, class SpanOpT>
auto CreateDenseUnaryOpFromSpanOp(
    SpanOpT op, RawBufferFactory* buf_factory = GetHeapBufferFactory()) {
  return dense_ops_internal::UnaryOpImpl<ResT, SpanOpT>(op, buf_factory);
}
template <class ResT, class SpanOpT>
auto CreateDenseBinaryOpFromSpanOp(
    SpanOpT op, RawBufferFactory* buf_factory = GetHeapBufferFactory()) {
  using ImplBase = dense_ops_internal::BinaryOpImpl<ResT, false, SpanOpT>;
  using Impl = dense_ops_internal::OpWithSizeValidation<ResT, ImplBase>;
  return Impl(ImplBase(op, buf_factory));
}
template <class ResT, int flags, class SpanOpT>
auto CreateDenseBinaryOpFromSpanOp(
    SpanOpT op, RawBufferFactory* buf_factory = GetHeapBufferFactory()) {
  static constexpr bool kNoBitmapOffset = flags & DenseOpFlags::kNoBitmapOffset;
  static constexpr bool kNoSizeValidation =
      flags & DenseOpFlags::kNoSizeValidation;
  using ImplBase =
      dense_ops_internal::BinaryOpImpl<ResT, kNoBitmapOffset, SpanOpT>;
  if constexpr (kNoSizeValidation) {
    return ImplBase(op, buf_factory);
  } else {
    using Impl = dense_ops_internal::OpWithSizeValidation<ResT, ImplBase>;
    return Impl(ImplBase(op, buf_factory));
  }
}
template <class Fn, class T, class... As>
absl::Status DenseArraysForEach(Fn&& fn, const DenseArray<T>& arg0,
                                const As&... args) {
  if (!((arg0.size() == args.size()) && ...)) {
    return SizeMismatchError({arg0.size(), args.size()...});
  }
  using fn_arg_types =
      typename meta::function_traits<std::decay_t<Fn>>::arg_types;
  using value_types = meta::tail_t<meta::tail_t<fn_arg_types>>;
  dense_ops_internal::DenseOpsUtil<value_types>::IterateFromZero(
      fn, arg0.size(), arg0, args...);
  return absl::OkStatus();
}
template <class Fn, class T, class... As>
absl::Status DenseArraysForEachPresent(Fn&& fn, const DenseArray<T>& arg0,
                                       const As&... args) {
  if (!((arg0.size() == args.size()) && ...)) {
    return SizeMismatchError({arg0.size(), args.size()...});
  }
  using fn_arg_types =
      typename meta::function_traits<std::decay_t<Fn>>::arg_types;
  using value_types = meta::tail_t<fn_arg_types>;
  dense_ops_internal::DenseOpsUtil<value_types>::IterateFromZero(
      [&fn](int64_t id, bool valid, auto&&... vals) {
        if (valid) {
          fn(id, std::forward<decltype(vals)>(vals)...);
        }
      },
      arg0.size(), arg0, args...);
  return absl::OkStatus();
}
}  
#endif  