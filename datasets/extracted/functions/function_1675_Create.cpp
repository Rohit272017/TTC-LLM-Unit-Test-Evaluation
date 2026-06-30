#ifndef AROLLA_QEXPR_OPERATOR_FIXTURE_H_
#define AROLLA_QEXPR_OPERATOR_FIXTURE_H_
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <typeindex>
#include <utility>
#include <vector>
#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "arolla/memory/frame.h"
#include "arolla/qexpr/eval_context.h"
#include "arolla/qexpr/operators.h"
#include "arolla/qexpr/qexpr_operator_signature.h"
#include "arolla/qtype/qtype.h"
#include "arolla/qtype/tuple_qtype.h"
#include "arolla/qtype/typed_slot.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla {
template <typename ARG_Ts, typename RES_Ts>
class OperatorFixture;
template <typename... ARG_Ts, typename... RES_Ts>
class OperatorFixture<std::tuple<ARG_Ts...>, std::tuple<RES_Ts...>> {
 public:
  static absl::StatusOr<OperatorFixture> Create(const QExprOperator& op) {
    return CreateImpl(op, std::make_index_sequence<sizeof...(ARG_Ts)>(),
                      std::make_index_sequence<sizeof...(RES_Ts)>());
  }
  OperatorFixture(OperatorFixture&& other) = default;
  OperatorFixture& operator=(OperatorFixture&& other) = default;
  absl::StatusOr<std::tuple<RES_Ts...>> Call(ARG_Ts&&... args) const {
    return CallImpl(std::forward<ARG_Ts&&>(args)...,
                    std::make_index_sequence<sizeof...(RES_Ts)>());
  }
 private:
  OperatorFixture(std::unique_ptr<BoundOperator> bound_op, FrameLayout&& layout,
                  std::tuple<FrameLayout::Slot<ARG_Ts>...> input_slots,
                  std::tuple<FrameLayout::Slot<RES_Ts>...> output_slots)
      : bound_op_(std::move(bound_op)),
        layout_(std::move(layout)),
        input_slots_(input_slots),
        output_slots_(output_slots) {}
  template <typename... Ts>
  static absl::Status VerifyTypes(absl::Span<const QTypePtr> types) {
    if (sizeof...(Ts) != types.size()) {
      return absl::FailedPreconditionError(
          absl::StrFormat("argument count mismatch; got %d expected %d",
                          types.size(), sizeof...(Ts)));
    }
    std::array<std::type_index, sizeof...(Ts)> expected_types = {
        std::type_index(typeid(Ts))...};
    for (size_t i = 0; i < types.size(); ++i) {
      if (expected_types[i] != std::type_index(types[i]->type_info())) {
        return absl::FailedPreconditionError(
            absl::StrFormat("type mismatch at position %d", i));
      }
    }
    return absl::OkStatus();
  }
  template <typename... Ts>
  static absl::Status VerifyTypes(absl::Span<const TypedSlot> slots) {
    std::vector<QTypePtr> types;
    types.reserve(slots.size());
    for (auto slot : slots) {
      types.push_back(slot.GetType());
    }
    return VerifyTypes<Ts...>(types);
  }
  template <size_t... ARG_Is, size_t... RES_Is>
  static absl::StatusOr<OperatorFixture> CreateImpl(
      const QExprOperator& op, std::index_sequence<ARG_Is...> arg_seq,
      std::index_sequence<RES_Is...> res_seq) {
    FrameLayout::Builder layout_builder;
    auto input_slots = std::make_tuple(layout_builder.AddSlot<ARG_Ts>()...);
    const QExprOperatorSignature* op_signature = op.signature();
    auto input_types = op_signature->input_types();
    RETURN_IF_ERROR(VerifyTypes<ARG_Ts...>(input_types)) << "on input types";
    auto output_type = op_signature->output_type();
    auto output_typed_slot = AddSlot(output_type, &layout_builder);
    std::vector<TypedSlot> output_typed_subslots;
    if (IsTupleQType(output_type)) {
      output_typed_subslots.reserve(output_typed_slot.SubSlotCount());
      for (int64_t i = 0; i < output_typed_slot.SubSlotCount(); ++i) {
        output_typed_subslots.push_back(output_typed_slot.SubSlot(i));
      }
    } else {
      output_typed_subslots = {output_typed_slot};
    }
    ASSIGN_OR_RETURN(auto output_slots,
                     TypedSlot::ToSlots<RES_Ts...>(output_typed_subslots));
    RETURN_IF_ERROR(VerifyTypes<RES_Ts...>(output_typed_subslots))
        << "on output types";
    ASSIGN_OR_RETURN(auto bound_op,
                     op.Bind({TypedSlot::FromSlot(std::get<ARG_Is>(input_slots),
                                                  input_types[ARG_Is])...},
                             output_typed_slot));
    auto layout = std::move(layout_builder).Build();
    return OperatorFixture(std::move(bound_op), std::move(layout), input_slots,
                           output_slots);
  }
  template <size_t... ARG_Is>
  void SetInputs(FramePtr frame
                     ABSL_ATTRIBUTE_UNUSED,  
                 ARG_Ts&&... args, std::index_sequence<ARG_Is...>) const {
    (frame.Set(std::get<ARG_Is>(input_slots_), std::move(args)), ...);
  }
  template <size_t... RES_Is>
  absl::StatusOr<std::tuple<RES_Ts...>> CallImpl(
      ARG_Ts&&... args, std::index_sequence<RES_Is...>) const {
    RootEvaluationContext root_ctx(&layout_);
    SetInputs(root_ctx.frame(), std::move(args)...,
              std::make_index_sequence<sizeof...(ARG_Ts)>());
    EvaluationContext ctx(root_ctx);
    bound_op_->Run(&ctx, root_ctx.frame());
    if (!ctx.status().ok()) {
      return std::move(ctx).status();
    }
    return std::make_tuple(
        std::move(*root_ctx.GetMutable(std::get<RES_Is>(output_slots_)))...);
  }
  std::unique_ptr<BoundOperator> bound_op_;
  FrameLayout layout_;
  std::tuple<FrameLayout::Slot<ARG_Ts>...> input_slots_;
  std::tuple<FrameLayout::Slot<RES_Ts>...> output_slots_;
};
template <template <typename...> class TYPE_LIST, typename... ARG_Ts,
          typename RES_T>
class OperatorFixture<TYPE_LIST<ARG_Ts...>, RES_T> {
 public:
  OperatorFixture(OperatorFixture&& other) = default;
  OperatorFixture& operator=(OperatorFixture&& other) = default;
  static absl::StatusOr<OperatorFixture> Create(const QExprOperator& op) {
    ASSIGN_OR_RETURN(auto delegate, DelegateT::Create(op));
    return OperatorFixture(std::move(delegate));
  }
  absl::StatusOr<RES_T> Call(ARG_Ts&&... args) const {
    ASSIGN_OR_RETURN(auto tuple,
                     delegate_.Call(std::forward<ARG_Ts&&>(args)...));
    return std::get<0>(std::move(tuple));
  }
 private:
  using DelegateT = OperatorFixture<TYPE_LIST<ARG_Ts...>, TYPE_LIST<RES_T>>;
  explicit OperatorFixture(DelegateT&& delegate)
      : delegate_(std::move(delegate)) {}
  DelegateT delegate_;
};
}  
#endif  