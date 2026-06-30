#ifndef AROLLA_QEXPR_OPERATOR_FACTORY_H_
#define AROLLA_QEXPR_OPERATOR_FACTORY_H_
#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "absl/base/attributes.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "arolla/memory/frame.h"
#include "arolla/qexpr/bound_operators.h"
#include "arolla/qexpr/eval_context.h"
#include "arolla/qexpr/operators.h"
#include "arolla/qexpr/qexpr_operator_signature.h"
#include "arolla/qexpr/result_type_traits.h"
#include "arolla/qtype/base_types.h"
#include "arolla/qtype/qtype.h"
#include "arolla/qtype/qtype_traits.h"
#include "arolla/qtype/tuple_qtype.h"
#include "arolla/qtype/typed_slot.h"
#include "arolla/util/demangle.h"
#include "arolla/util/meta.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla {
template <typename FUNC>
absl::StatusOr<OperatorPtr> QExprOperatorFromFunction(FUNC func);
template <typename FUNC>
absl::StatusOr<OperatorPtr> QExprOperatorBuildFromFunction(
    FUNC func, const QExprOperatorSignature* signature);
template <typename FUNC, typename... ARG_Ts>
absl::StatusOr<OperatorPtr> QExprOperatorFromFunctor();
template <typename FUNC>
std::unique_ptr<arolla::OperatorFamily> MakeVariadicInputOperatorFamily(
    FUNC eval_func);
namespace operator_factory_impl {
template <typename T>
using Slot = FrameLayout::Slot<T>;
template <typename FUNC, typename... OTHER_ARGs>
struct ContextFunc : private FUNC {
  explicit ContextFunc(FUNC func) : FUNC(std::move(func)) {}
  auto operator()(EvaluationContext*, OTHER_ARGs... args) const {
    return static_cast<const FUNC&>(*this)(args...);
  }
};
template <typename FUNC, typename... OTHER_ARGs>
struct ContextFunc<FUNC, EvaluationContext*, OTHER_ARGs...> : FUNC {
  explicit ContextFunc(FUNC func) : FUNC(std::move(func)) {}
};
template <typename FUNC, typename... FUNC_ARGs>
auto WrapIntoContextFunc(FUNC func, meta::type_list<FUNC_ARGs...>) {
  if constexpr (std::is_class_v<FUNC>) {
    return ContextFunc<FUNC, FUNC_ARGs...>(std::move(func));
  } else {
    auto fn = [func = std::forward<FUNC>(func)](FUNC_ARGs... args) {
      return func(args...);
    };
    return ContextFunc<decltype(fn), FUNC_ARGs...>(std::move(fn));
  }
}
template <typename... Ts>
struct QTypesVerifier;
template <typename T, typename... Ts>
struct QTypesVerifier<T, Ts...> {
  static absl::Status Verify(absl::Span<const QTypePtr> qtypes) {
    if (qtypes.size() != sizeof...(Ts) + 1) {
      return absl::Status(
          absl::StatusCode::kInvalidArgument,
          absl::StrFormat(
              "unexpected number of types: expected %d types %s, got %d",
              qtypes.size(), FormatTypeVector(qtypes), sizeof...(Ts) + 1));
    }
    DCHECK_GT(qtypes.size(), size_t{0});
    if (qtypes[0]->type_info() != typeid(T)) {
      return absl::Status(
          absl::StatusCode::kInvalidArgument,
          absl::StrFormat(
              "unexpected type: expected %s with C++ type %s, got %s",
              qtypes[0]->name(), TypeName(qtypes[0]->type_info()),
              TypeName<T>()));
    }
    return QTypesVerifier<Ts...>::Verify(qtypes.subspan(1));
  }
};
template <>
struct QTypesVerifier<> {
  static absl::Status Verify(absl::Span<const QTypePtr> qtypes) {
    if (!qtypes.empty()) {
      return absl::Status(
          absl::StatusCode::kInvalidArgument,
          absl::StrFormat(
              "unexpected number of types: expected %d types %s, got 0",
              qtypes.size(), FormatTypeVector(qtypes)));
    }
    return absl::OkStatus();
  }
};
template <typename... Ts>
struct QTypesVerifier<meta::type_list<Ts...>> {
  static absl::Status Verify(absl::Span<const QTypePtr> qtypes) {
    return QTypesVerifier<Ts...>::Verify(qtypes);
  }
};
template <typename Slots, std::size_t... Is>
Slots UnsafeToSlotsTupleImpl(absl::Span<const TypedSlot> slots,
                             std::index_sequence<Is...>) {
  DCHECK_EQ(slots.size(), sizeof...(Is));
  return {
      slots[Is]
          .UnsafeToSlot<
              typename std::tuple_element<Is, Slots>::type::value_type>()...};
}
template <typename Slots>
Slots UnsafeToSlotsTuple(absl::Span<const TypedSlot> slots) {
  return UnsafeToSlotsTupleImpl<Slots>(
      slots, std::make_index_sequence<std::tuple_size<Slots>::value>{});
}
template <typename FUNC, typename RES, typename... ARGs>
const QExprOperatorSignature* DeduceOperatorSignatureImpl(
    meta::type_list<RES>, meta::type_list<ARGs...>) {
  return QExprOperatorSignature::Get(
      {GetQType<std::decay_t<ARGs>>()...},
      qexpr_impl::ResultTypeTraits<RES>::GetOutputType());
}
template <typename FUNC>
const QExprOperatorSignature* DeduceOperatorSignature() {
  return DeduceOperatorSignatureImpl<FUNC>(
      meta::type_list<typename meta::function_traits<FUNC>::return_type>(),
      meta::tail_t<typename meta::function_traits<FUNC>::arg_types>());
}
template <typename FUNC>
absl::Status VerifyOperatorSignature(const QExprOperatorSignature* signature) {
  RETURN_IF_ERROR(QTypesVerifier<meta::tail_t<typename meta::function_traits<
                      FUNC>::arg_types>>::Verify(signature->input_types()))
      << "in input types of " << signature << ".";
  std::vector<QTypePtr> output_types = {signature->output_type()};
  if (IsTupleQType(signature->output_type())) {
    output_types = SlotsToTypes(signature->output_type()->type_fields());
  }
  RETURN_IF_ERROR(
      QTypesVerifier<typename qexpr_impl::ResultTypeTraits<
          typename meta::function_traits<FUNC>::return_type>::Types>::
          Verify(output_types))
      << "in output types of " << signature << ".";
  return absl::OkStatus();
}
template <typename CTX_FUNC, typename RES, typename... ARGs>
class OpImpl : public QExprOperator {
 public:
  OpImpl(const QExprOperatorSignature* signature, CTX_FUNC func)
      : QExprOperator(signature), func_(std::move(func)) {}
 private:
  absl::StatusOr<std::unique_ptr<BoundOperator>> DoBind(
      absl::Span<const TypedSlot> input_slots,
      TypedSlot output_slot) const override {
    auto inputs = UnsafeToSlotsTuple<InputSlots>(input_slots);
    auto outputs =
        qexpr_impl::ResultTypeTraits<RES>::UnsafeToSlots(output_slot);
    return MakeBoundOperator(
        [data = BoundOpData(func_, std::move(inputs), std::move(outputs))](
            EvaluationContext* ctx, FramePtr frame) {
          RunImpl(data, ctx, frame, std::index_sequence_for<ARGs...>{});
        });
  }
 private:
  using InputSlots = std::tuple<Slot<absl::decay_t<ARGs>>...>;
  using OutputSlots = typename qexpr_impl::ResultTypeTraits<RES>::Slots;
  struct BoundOpData : private CTX_FUNC {
    BoundOpData(CTX_FUNC func, InputSlots input_slots, OutputSlots output_slots)
        : CTX_FUNC(std::move(func)),
          input_slots(input_slots),
          output_slots(output_slots) {}
    const CTX_FUNC& func() const { return static_cast<const CTX_FUNC&>(*this); }
    const InputSlots input_slots;
    const OutputSlots output_slots;
  };
  template <std::size_t... Is>
  static void RunImpl(const BoundOpData& data, EvaluationContext* ctx,
                      FramePtr frame, std::index_sequence<Is...>) {
    qexpr_impl::ResultTypeTraits<RES>::SaveAndReturn(
        ctx, frame, data.output_slots,
        data.func()(ctx, frame.Get(std::get<Is>(data.input_slots))...));
  }
  const CTX_FUNC func_;
};
template <typename CTX_FUNC, typename... ARGs>
absl::StatusOr<OperatorPtr> QExprOperatorFromFunctionImpl(
    CTX_FUNC func, const QExprOperatorSignature* signature,
    meta::type_list<ARGs...>) {
  return OperatorPtr(
      new operator_factory_impl::OpImpl<
          CTX_FUNC, typename meta::function_traits<CTX_FUNC>::return_type,
          ARGs...>(signature, std::move(func)));
}
template <typename T>
struct VariadicInputTypeTraits {
  using Container = nullptr_t;  
  using Slot = nullptr_t;       
  static_assert(sizeof(T) == 0,
                "unsupported input for VariadicInputOperatorFamily");
};
template <typename T>
struct VariadicInputTypeTraits<meta::type_list<absl::Span<const T* const>>> {
  using Container = absl::InlinedVector<const T*, 4>;
  using Slot = FrameLayout::Slot<T>;
  static QTypePtr GetInputType() ABSL_ATTRIBUTE_ALWAYS_INLINE {
    return GetQType<T>();
  }
  static Container GetInputs(arolla::FramePtr frame,
                             absl::Span<const Slot> input_slots) {
    Container inputs;
    inputs.reserve(input_slots.size());
    for (const auto& input_slot : input_slots) {
      inputs.push_back(&frame.Get(input_slot));
    }
    return inputs;
  }
  static Slot UnsafeToSlot(TypedSlot output_slot) ABSL_ATTRIBUTE_ALWAYS_INLINE {
    return output_slot.UnsafeToSlot<T>();
  }
};
template <typename T>
struct VariadicInputTypeTraits<meta::type_list<std::vector<T>>> {
  using Container = std::vector<T>;
  using Slot = FrameLayout::Slot<T>;
  static QTypePtr GetInputType() ABSL_ATTRIBUTE_ALWAYS_INLINE {
    return GetQType<T>();
  }
  static Container GetInputs(arolla::FramePtr frame,
                             absl::Span<const Slot> input_slots) {
    Container inputs;
    inputs.reserve(input_slots.size());
    for (const auto& input_slot : input_slots) {
      inputs.push_back(frame.Get(input_slot));
    }
    return inputs;
  }
  static Slot UnsafeToSlot(TypedSlot output_slot) ABSL_ATTRIBUTE_ALWAYS_INLINE {
    return output_slot.UnsafeToSlot<T>();
  }
};
template <typename FUNC>
struct VariadicInputFuncTraits {
  using input =
      VariadicInputTypeTraits<typename meta::function_traits<FUNC>::arg_types>;
  using result = qexpr_impl::ResultTypeTraits<
      typename meta::function_traits<FUNC>::return_type>;
};
template <typename FUNC>
class VariadicInputOperator : public arolla::QExprOperator {
  using input_traits = VariadicInputFuncTraits<FUNC>::input;
  using result_traits = VariadicInputFuncTraits<FUNC>::result;
 public:
  explicit VariadicInputOperator(FUNC eval_func,
                                 absl::Span<const arolla::QTypePtr> input_types)
      : arolla::QExprOperator(arolla::QExprOperatorSignature::Get(
            input_types, result_traits::GetOutputType())),
        eval_func_(std::move(eval_func)) {}
 private:
  absl::StatusOr<std::unique_ptr<arolla::BoundOperator>> DoBind(
      absl::Span<const arolla::TypedSlot> typed_input_slots,
      arolla::TypedSlot typed_output_slot) const final {
    std::vector<typename input_traits::Slot> input_slots;
    input_slots.reserve(typed_input_slots.size());
    for (const auto& input_slot : typed_input_slots) {
      input_slots.push_back(input_traits::UnsafeToSlot(input_slot));
    }
    return arolla::MakeBoundOperator(
        [input_slots = std::move(input_slots),
         output_slot = result_traits::UnsafeToSlots(typed_output_slot),
         eval_func = eval_func_](arolla::EvaluationContext* ctx,
                                 arolla::FramePtr frame) {
          auto inputs = input_traits::GetInputs(frame, input_slots);
          result_traits::SaveAndReturn(ctx, frame, output_slot,
                                       eval_func(std::move(inputs)));
        });
  }
  FUNC eval_func_;
};
template <typename FUNC>
class VariadicInputOperatorFamily : public arolla::OperatorFamily {
  using input_traits = VariadicInputFuncTraits<FUNC>::input;
 public:
  explicit VariadicInputOperatorFamily(FUNC eval_func)
      : eval_func_(std::move(eval_func)) {}
 private:
  absl::StatusOr<arolla::OperatorPtr> DoGetOperator(
      absl::Span<const arolla::QTypePtr> input_types,
      arolla::QTypePtr output_type) const final {
    for (const auto& input_type : input_types) {
      if (input_type != input_traits::GetInputType()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "expected only %s, got %s", input_traits::GetInputType()->name(),
            input_type->name()));
      }
    }
    return arolla::EnsureOutputQTypeMatches(
        std::make_shared<VariadicInputOperator<FUNC>>(eval_func_, input_types),
        input_types, output_type);
  }
  FUNC eval_func_;
};
}  
template <typename FUNC>
absl::StatusOr<OperatorPtr> QExprOperatorFromFunction(FUNC func) {
  auto context_func = operator_factory_impl::WrapIntoContextFunc(
      std::move(func), typename meta::function_traits<FUNC>::arg_types());
  using CtxFunc = decltype(context_func);
  const QExprOperatorSignature* signature =
      operator_factory_impl::DeduceOperatorSignature<CtxFunc>();
  return QExprOperatorFromFunctionImpl(
      std::move(context_func), signature,
      meta::tail_t<typename meta::function_traits<CtxFunc>::arg_types>());
}
template <typename FUNC>
absl::StatusOr<OperatorPtr> QExprOperatorFromFunction(
    FUNC func, const QExprOperatorSignature* signature) {
  auto context_func = operator_factory_impl::WrapIntoContextFunc(
      std::move(func), typename meta::function_traits<FUNC>::arg_types());
  using CtxFunc = decltype(context_func);
  RETURN_IF_ERROR(
      operator_factory_impl::VerifyOperatorSignature<CtxFunc>(signature));
  return QExprOperatorFromFunctionImpl(
      std::move(context_func), signature,
      meta::tail_t<typename meta::function_traits<CtxFunc>::arg_types>());
}
template <typename FUNC, typename... ARG_Ts>
absl::StatusOr<OperatorPtr> QExprOperatorFromFunctor() {
  return QExprOperatorFromFunction(
      [](EvaluationContext* ctx, const ARG_Ts&... args) {
        if constexpr (std::is_invocable_v<FUNC, ARG_Ts...>) {
          ((void)(ctx));
          return FUNC()(args...);
        } else {
          return FUNC()(ctx, args...);
        }
      });
}
template <typename FUNC>
std::unique_ptr<arolla::OperatorFamily> MakeVariadicInputOperatorFamily(
    FUNC eval_func) {
  return std::make_unique<
      operator_factory_impl::VariadicInputOperatorFamily<FUNC>>(
      std::move(eval_func));
}
}  
#endif  