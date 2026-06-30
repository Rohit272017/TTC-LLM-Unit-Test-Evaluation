#include "arolla/qexpr/operators/core/utility_operators.h"
#include <memory>
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "arolla/memory/frame.h"
#include "arolla/qexpr/bound_operators.h"
#include "arolla/qexpr/eval_context.h"
#include "arolla/qexpr/operators.h"
#include "arolla/qexpr/qexpr_operator_signature.h"
#include "arolla/qtype/qtype.h"
namespace arolla {
namespace {
class CopyOperator : public QExprOperator {
 public:
  explicit CopyOperator(QTypePtr type)
      : QExprOperator(QExprOperatorSignature::Get({type}, type)) {}
 private:
  absl::StatusOr<std::unique_ptr<BoundOperator>> DoBind(
      absl::Span<const TypedSlot> input_slots,
      TypedSlot output_slot) const final {
    return MakeBoundOperator(
        [input_slot = input_slots[0], output_slot = output_slot](
            EvaluationContext*, FramePtr frame) {
          input_slot.CopyTo(frame, output_slot, frame);
        });
  }
};
}  
OperatorPtr MakeCopyOp(QTypePtr type) {
  return OperatorPtr(std::make_unique<CopyOperator>(type));
}
}  