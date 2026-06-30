#include "arolla/expr/optimization/optimizer.h"
#include <memory>
#include <utility>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "arolla/expr/expr_debug_string.h"
#include "arolla/expr/expr_node.h"
#include "arolla/expr/optimization/peephole_optimizer.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla::expr {
namespace {
constexpr int kPeepholeOptimizerIterationsLimit = 100;
}  
Optimizer MakeOptimizer(std::unique_ptr<PeepholeOptimizer> peephole_optimizer) {
  return [peephole_optimizer = std::shared_ptr<PeepholeOptimizer>(
              std::move(peephole_optimizer))](
             ExprNodePtr expr) -> absl::StatusOr<ExprNodePtr> {
    ExprNodePtr previous_expr;
    int iteration = 0;
    do {
      if (++iteration > kPeepholeOptimizerIterationsLimit) {
        return absl::InternalError(absl::StrFormat(
            "too many iterations of peephole optimizer; this may indicate that "
            "the set of optimizations contains cycles, or just too big "
            "expression unsupported by the optimizer (last iterations: %s vs "
            "%s)",
            GetDebugSnippet(previous_expr), GetDebugSnippet(expr)));
      }
      previous_expr = expr;
      ASSIGN_OR_RETURN(expr, peephole_optimizer->ApplyToNode(expr));
      if (expr->qtype() != previous_expr->qtype()) {
        return absl::InternalError(absl::StrFormat(
            "expression %s was optimized into %s, which changed its output "
            "type from %s to %s; this indicates incorrect optimization",
            GetDebugSnippet(previous_expr), GetDebugSnippet(expr),
            previous_expr->qtype() != nullptr ? previous_expr->qtype()->name()
                                              : "NULL",
            expr->qtype() != nullptr ? expr->qtype()->name() : "NULL"));
      }
    } while (previous_expr->fingerprint() != expr->fingerprint());
    return expr;
  };
}
}  