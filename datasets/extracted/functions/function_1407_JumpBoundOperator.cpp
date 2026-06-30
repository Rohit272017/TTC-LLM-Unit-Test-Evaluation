#include "arolla/qexpr/bound_operators.h"
#include <cstdint>
#include <memory>
#include "arolla/memory/frame.h"
#include "arolla/qexpr/eval_context.h"
#include "arolla/qexpr/operators.h"
namespace arolla {
std::unique_ptr<BoundOperator> JumpBoundOperator(int64_t jump) {
  return MakeBoundOperator([=](EvaluationContext* ctx, FramePtr frame) {
    ctx->set_requested_jump(jump);
  });
}
std::unique_ptr<BoundOperator> JumpIfNotBoundOperator(
    FrameLayout::Slot<bool> cond_slot, int64_t jump) {
  return MakeBoundOperator([=](EvaluationContext* ctx, FramePtr frame) {
    if (!frame.Get(cond_slot)) {
      ctx->set_requested_jump(jump);
    }
  });
}
}  