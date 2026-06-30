#include "arolla/qexpr/simple_executable.h"
#include <memory>
#include "absl/status/status.h"
#include "arolla/memory/frame.h"
#include "arolla/qexpr/bound_operators.h"
#include "arolla/qexpr/eval_context.h"
#include "arolla/qexpr/evaluation_engine.h"
namespace arolla {
void SimpleBoundExpr::InitializeLiterals(EvaluationContext* ctx,
                                         FramePtr frame) const {
  RunBoundOperators(init_ops_, ctx, frame);
}
void SimpleBoundExpr::Execute(EvaluationContext* ctx, FramePtr frame) const {
  RunBoundOperators(eval_ops_, ctx, frame);
}
void CombinedBoundExpr::InitializeLiterals(EvaluationContext* ctx,
                                           FramePtr frame) const {
  for (const auto& e : subexprs_) {
    if (e->InitializeLiterals(ctx, frame); !ctx->status().ok()) {
      break;
    }
  }
}
void CombinedBoundExpr::Execute(EvaluationContext* ctx, FramePtr frame) const {
  for (const auto& e : subexprs_) {
    if (e->Execute(ctx, frame); !ctx->status().ok()) {
      break;
    }
  }
}
}  