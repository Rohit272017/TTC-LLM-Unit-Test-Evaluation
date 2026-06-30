#ifndef TENSORSTORE_UTIL_ELEMENTWISE_FUNCTION_H_
#define TENSORSTORE_UTIL_ELEMENTWISE_FUNCTION_H_
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>
#include "absl/meta/type_traits.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/internal/void_wrapper.h"
#include "tensorstore/util/byte_strided_pointer.h"
namespace tensorstore {
namespace internal {
enum class IterationBufferKind {
  kContiguous,
  kStrided,
  kIndexed,
};
constexpr size_t kNumIterationBufferKinds = 3;
struct IterationBufferPointer {
  IterationBufferPointer() = default;
  explicit IterationBufferPointer(ByteStridedPointer<void> pointer,
                                  Index outer_byte_stride,
                                  Index inner_byte_stride)
      : pointer(pointer),
        outer_byte_stride(outer_byte_stride),
        inner_byte_stride(inner_byte_stride) {}
  explicit IterationBufferPointer(ByteStridedPointer<void> pointer,
                                  Index byte_offsets_outer_stride,
                                  const Index* byte_offsets)
      : pointer(pointer),
        byte_offsets_outer_stride(byte_offsets_outer_stride),
        byte_offsets(byte_offsets) {}
  ByteStridedPointer<void> pointer;
  union {
    Index outer_byte_stride;
    Index byte_offsets_outer_stride;
  };
  union {
    Index inner_byte_stride;
    const Index* byte_offsets;
  };
  void AddElementOffset(IterationBufferKind kind, Index outer_offset,
                        Index inner_offset) {
    if (kind == IterationBufferKind::kIndexed) {
      byte_offsets += inner_offset;
      byte_offsets +=
          wrap_on_overflow::Multiply(byte_offsets_outer_stride, outer_offset);
    } else {
      pointer += wrap_on_overflow::Multiply(inner_byte_stride, inner_offset);
      pointer += wrap_on_overflow::Multiply(outer_byte_stride, outer_offset);
    }
  }
};
template <IterationBufferKind BufferKind>
struct IterationBufferAccessor;
template <>
struct IterationBufferAccessor<IterationBufferKind::kStrided> {
  constexpr static IterationBufferKind buffer_kind =
      IterationBufferKind::kStrided;
  template <typename Element>
  static Element* GetPointerAtPosition(IterationBufferPointer ptr, Index outer,
                                       Index inner) {
    return static_cast<Element*>(
        ptr.pointer +
        internal::wrap_on_overflow::Multiply(ptr.outer_byte_stride, outer) +
        internal::wrap_on_overflow::Multiply(ptr.inner_byte_stride, inner));
  }
};
template <>
struct IterationBufferAccessor<IterationBufferKind::kContiguous> {
  constexpr static IterationBufferKind buffer_kind =
      IterationBufferKind::kContiguous;
  template <typename Element>
  static Element* GetPointerAtPosition(IterationBufferPointer ptr, Index outer,
                                       Index inner) {
    return static_cast<Element*>(
        ptr.pointer +
        internal::wrap_on_overflow::Multiply(ptr.outer_byte_stride, outer) +
        internal::wrap_on_overflow::Multiply(
            static_cast<Index>(sizeof(Element)), inner));
  }
};
template <>
struct IterationBufferAccessor<IterationBufferKind::kIndexed> {
  constexpr static IterationBufferKind buffer_kind =
      IterationBufferKind::kIndexed;
  template <typename Element>
  static Element* GetPointerAtPosition(IterationBufferPointer ptr, Index outer,
                                       Index inner) {
    return static_cast<Element*>(
        ptr.pointer +
        ptr.byte_offsets[internal::wrap_on_overflow::Multiply(
                             ptr.byte_offsets_outer_stride, outer) +
                         inner]);
  }
};
template <size_t Arity, typename... ExtraArg>
class ElementwiseFunction;
using IterationBufferShape = std::array<Index, 2>;
}  
namespace internal_elementwise_function {
template <typename SequenceType, typename... ExtraArg>
struct ElementwiseFunctionPointerHelper;
template <size_t I>
using IterationBufferPointerHelper = internal::IterationBufferPointer;
template <size_t... Is, typename... ExtraArg>
struct ElementwiseFunctionPointerHelper<std::index_sequence<Is...>,
                                        ExtraArg...> {
  using type = bool (*)(void*, internal::IterationBufferShape,
                        IterationBufferPointerHelper<Is>..., ExtraArg...);
};
template <typename, typename SFINAE, typename...>
constexpr inline bool HasApplyContiguous = false;
template <typename Func, typename... Element, typename... ExtraArg>
constexpr inline bool HasApplyContiguous<
    Func(Element...),
    std::void_t<decltype(std::declval<Func>().ApplyContiguous(
        std::declval<Index>(), std::declval<Element*>()...,
        std::declval<ExtraArg>()...))>,
    ExtraArg...> = true;
template <typename, typename...>
struct SimpleLoopTemplate;
template <typename T, typename Func>
struct Stateless {
  static_assert(std::is_empty_v<Func>);
  using type = Func;
  using ContextType = T;
};
template <typename T>
struct StatelessTraits {
  constexpr static bool is_stateless = false;
  using type = T;
};
template <typename T, typename Func>
struct StatelessTraits<Stateless<T, Func>> {
  constexpr static bool is_stateless = true;
  using type = Func;
};
template <typename Func, typename... Element, typename... ExtraArg>
struct SimpleLoopTemplate<Func(Element...), ExtraArg...> {
  using ElementwiseFunctionType =
      internal::ElementwiseFunction<sizeof...(Element), ExtraArg...>;
  template <typename ArrayAccessor>
  static constexpr auto GetLoopFn() {
    if constexpr (ArrayAccessor::buffer_kind ==
                      internal::IterationBufferKind::kContiguous &&
                  HasApplyContiguous<Func(Element...), void,
                                     ExtraArg...>) {
      return &FastLoop<ArrayAccessor>;
    } else {
      return &Loop<ArrayAccessor>;
    }
  }
  template <typename ArrayAccessor>
  static bool FastLoop(
      void* context, internal::IterationBufferShape shape,
      internal::FirstType<internal::IterationBufferPointer, Element>... pointer,
      ExtraArg... extra_arg) {
    using Traits = StatelessTraits<Func>;
    using FuncType = typename Traits::type;
    static_assert(ArrayAccessor::buffer_kind ==
                  internal::IterationBufferKind::kContiguous);
    static_assert(
        HasApplyContiguous<Func(Element...), void, ExtraArg...>);
    internal::PossiblyEmptyObjectGetter<FuncType> func_helper;
    FuncType& func = func_helper.get(static_cast<FuncType*>(context));
    for (Index outer = 0; outer < shape[0]; ++outer) {
      if constexpr (StatelessTraits<Func>::is_stateless) {
        if (!func.ApplyContiguous(
                *static_cast<typename Func::ContextType*>(context), shape[1],
                ArrayAccessor::template GetPointerAtPosition<Element>(
                    pointer, outer, 0)...,
                extra_arg...)) {
          return false;
        }
      } else {
        if (!func.ApplyContiguous(
                shape[1],
                ArrayAccessor::template GetPointerAtPosition<Element>(
                    pointer, outer, 0)...,
                extra_arg...)) {
          return false;
        }
      }
    }
    return true;
  }
  template <typename ArrayAccessor>
  static bool Loop(
      void* context, internal::IterationBufferShape shape,
      internal::FirstType<internal::IterationBufferPointer, Element>... pointer,
      ExtraArg... extra_arg) {
    static_assert(
        !(ArrayAccessor::buffer_kind ==
              internal::IterationBufferKind::kContiguous &&
          HasApplyContiguous<Func(Element...), void, ExtraArg...>));
    using Traits = StatelessTraits<Func>;
    using FuncType = typename Traits::type;
    internal::PossiblyEmptyObjectGetter<FuncType> func_helper;
    FuncType& func = func_helper.get(static_cast<FuncType*>(context));
    for (Index outer = 0; outer < shape[0]; ++outer) {
      for (Index inner = 0; inner < shape[1]; ++inner) {
        if constexpr (StatelessTraits<Func>::is_stateless) {
          if (!static_cast<bool>(internal::Void::CallAndWrap(
                  func, *static_cast<typename Func::ContextType*>(context),
                  ArrayAccessor::template GetPointerAtPosition<Element>(
                      pointer, outer, inner)...,
                  extra_arg...))) {
            return false;
          }
        } else {
          if (!static_cast<bool>(internal::Void::CallAndWrap(
                  func,
                  ArrayAccessor::template GetPointerAtPosition<Element>(
                      pointer, outer, inner)...,
                  extra_arg...))) {
            return false;
          }
        }
      }
    }
    return true;
  }
};
}  
namespace internal {
template <size_t Arity, typename... ExtraArg>
using SpecializedElementwiseFunctionPointer =
    typename internal_elementwise_function::ElementwiseFunctionPointerHelper<
        std::make_index_sequence<Arity>, ExtraArg...>::type;
template <size_t Arity, typename... ExtraArg>
struct ElementwiseClosure {
  using Function = ElementwiseFunction<Arity, ExtraArg...>;
  constexpr static size_t arity = Arity;
  const Function* function;
  void* context;
};
template <size_t Arity, typename... ExtraArg>
class ElementwiseFunction {
 public:
  constexpr static size_t arity = Arity;
  using Closure = ElementwiseClosure<Arity, ExtraArg...>;
  using SpecializedFunctionPointer =
      SpecializedElementwiseFunctionPointer<Arity, ExtraArg...>;
  constexpr ElementwiseFunction() = default;
  template <typename LoopTemplate,
            typename = decltype(LoopTemplate::template GetLoopFn<
                                IterationBufferAccessor<
                                    IterationBufferKind::kContiguous>>())>
  constexpr explicit ElementwiseFunction(LoopTemplate)
      : functions_{
            LoopTemplate::template GetLoopFn<
                IterationBufferAccessor<IterationBufferKind::kContiguous>>(),
            LoopTemplate::template GetLoopFn<
                IterationBufferAccessor<IterationBufferKind::kStrided>>(),
            LoopTemplate::template GetLoopFn<
                IterationBufferAccessor<IterationBufferKind::kIndexed>>()} {}
  constexpr SpecializedFunctionPointer operator[](
      IterationBufferKind buffer_kind) const {
    return functions_[static_cast<size_t>(buffer_kind)];
  }
  constexpr SpecializedFunctionPointer& operator[](
      IterationBufferKind buffer_kind) {
    return functions_[static_cast<size_t>(buffer_kind)];
  }
 private:
  SpecializedFunctionPointer functions_[kNumIterationBufferKinds];
};
template <typename LoopTemplate>
struct GetElementwiseFunction {
  using ElementwiseFunctionType =
      typename LoopTemplate::ElementwiseFunctionType;
  constexpr static ElementwiseFunctionType function{LoopTemplate{}};
  constexpr operator const ElementwiseFunctionType*() const {
    return &function;
  }
  constexpr operator ElementwiseFunctionType() const { return function; }
};
template <typename LoopTemplate>
constexpr typename LoopTemplate::ElementwiseFunctionType
    GetElementwiseFunction<LoopTemplate>::function;
template <typename, typename...>
struct SimpleElementwiseFunction;
template <typename Func, typename... Element, typename... ExtraArg>
struct SimpleElementwiseFunction<Func(Element...), ExtraArg...>
    : public GetElementwiseFunction<
          internal_elementwise_function::SimpleLoopTemplate<
              std::remove_reference_t<Func>(Element...), ExtraArg...>> {
  using ElementwiseFunctionType =
      internal::ElementwiseFunction<sizeof...(Element), ExtraArg...>;
  using ClosureType =
      internal::ElementwiseClosure<sizeof...(Element), ExtraArg...>;
  constexpr static ClosureType Closure(std::remove_reference_t<Func>* func) {
    return ClosureType{SimpleElementwiseFunction{},
                       const_cast<absl::remove_cvref_t<Func>*>(func)};
  }
  template <int&... ExplicitArgumentBarrier,
            std::enable_if_t<
                (sizeof...(ExplicitArgumentBarrier) == 0 &&
                 std::is_empty<absl::remove_cvref_t<Func>>::value)>* = nullptr>
  constexpr operator ClosureType() const {
    return {SimpleElementwiseFunction{}, nullptr};
  }
};
}  
namespace internal_elementwise_function {
template <size_t Arity, typename... ExtraArg, typename Pointers, size_t... Is>
inline bool InvokeElementwiseFunctionImpl(
    std::index_sequence<Is...>,
    internal::SpecializedElementwiseFunctionPointer<Arity, ExtraArg...>
        function,
    void* context, internal::IterationBufferShape shape,
    const Pointers& pointers, ExtraArg... extra_arg) {
  using std::get;
  return function(context, shape, get<Is>(pointers)...,
                  std::forward<ExtraArg>(extra_arg)...);
}
}  
namespace internal {
template <size_t Arity, typename... ExtraArg, typename Pointers>
inline bool InvokeElementwiseClosure(
    ElementwiseClosure<Arity, ExtraArg...> closure,
    IterationBufferKind buffer_kind, internal::IterationBufferShape shape,
    const Pointers& pointers,
    internal::type_identity_t<ExtraArg>... extra_arg) {
  return internal_elementwise_function::InvokeElementwiseFunctionImpl<
      Arity, ExtraArg...>(
      std::make_index_sequence<Arity>{}, (*closure.function)[buffer_kind],
      closure.context, shape, pointers, std::forward<ExtraArg>(extra_arg)...);
}
template <size_t Arity, typename... ExtraArg, typename Pointers>
inline bool InvokeElementwiseFunction(
    SpecializedElementwiseFunctionPointer<Arity, ExtraArg...> function,
    void* context, internal::IterationBufferShape shape,
    const Pointers& pointers, ExtraArg... extra_arg) {
  return internal_elementwise_function::InvokeElementwiseFunctionImpl<
      Arity, ExtraArg...>(std::make_index_sequence<Arity>{}, function, context,
                          shape, pointers,
                          std::forward<ExtraArg>(extra_arg)...);
}
}  
}  
#endif  