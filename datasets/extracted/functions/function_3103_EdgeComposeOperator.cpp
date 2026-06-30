#ifndef AROLLA_QEXPR_OPERATORS_ARRAY_LIKE_EDGE_OPS_H_
#define AROLLA_QEXPR_OPERATORS_ARRAY_LIKE_EDGE_OPS_H_
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "arolla/memory/frame.h"
#include "arolla/qexpr/bound_operators.h"
#include "arolla/qexpr/eval_context.h"
#include "arolla/qexpr/operators.h"
#include "arolla/qexpr/qexpr_operator_signature.h"
#include "arolla/qtype/qtype.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla {
template <typename EdgeT>
class EdgeComposeOperator : public InlineOperator {
 public:
  explicit EdgeComposeOperator(size_t size)
      : InlineOperator(QExprOperatorSignature::Get(
            std::vector<QTypePtr>(size, ::arolla::GetQType<EdgeT>()),
            ::arolla::GetQType<EdgeT>())) {}
 private:
  absl::StatusOr<std::unique_ptr<BoundOperator>> DoBind(
      absl::Span<const TypedSlot> input_slots,
      TypedSlot output_slot) const override {
    std::vector<Slot<EdgeT>> input_edge_slots;
    input_edge_slots.reserve(input_slots.size());
    for (const auto& input_slot : input_slots) {
      ASSIGN_OR_RETURN(Slot<EdgeT> edge_slot, input_slot.ToSlot<EdgeT>());
      input_edge_slots.push_back(std::move(edge_slot));
    }
    ASSIGN_OR_RETURN(Slot<EdgeT> output_edge_slot, output_slot.ToSlot<EdgeT>());
    return MakeBoundOperator([input_edge_slots = std::move(input_edge_slots),
                              output_edge_slot = std::move(output_edge_slot)](
                                 EvaluationContext* ctx, FramePtr frame) {
      std::vector<EdgeT> edges;
      edges.reserve(input_edge_slots.size());
      for (const auto& edge_slot : input_edge_slots) {
        edges.push_back(frame.Get(edge_slot));
      }
      ASSIGN_OR_RETURN(auto composed_edge,
                       EdgeT::ComposeEdges(edges, ctx->buffer_factory()),
                       ctx->set_status(std::move(_)));
      frame.Set(output_edge_slot, std::move(composed_edge));
    });
  }
};
}  
#endif  