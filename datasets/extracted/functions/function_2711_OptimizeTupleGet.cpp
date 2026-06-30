#include "arolla/expr/optimization/peephole_optimizations/tuple.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "arolla/expr/expr_node.h"
#include "arolla/expr/optimization/peephole_optimizer.h"
#include "arolla/expr/registered_expr_operator.h"
#include "arolla/expr/tuple_expr_operator.h"
#include "arolla/util/fast_dynamic_downcast_final.h"
#include "arolla/util/fingerprint.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla::expr {
namespace {
absl::StatusOr<ExprNodePtr> OptimizeTupleGet(ExprNodePtr expr) {
  static Fingerprint make_tuple_fingerprint = MakeTupleOperator().fingerprint();
  if (!expr->is_op()) {
    return expr;
  }
  auto get_nth_operator =
      fast_dynamic_downcast_final<const GetNthOperator*>(expr->op().get());
  if (get_nth_operator == nullptr) {
    return expr;
  }
  if (expr->node_deps().size() != 1) {
    return expr;
  }
  auto tuple_expr = expr->node_deps()[0];
  if (!tuple_expr->is_op()) {
    return expr;
  }
  ASSIGN_OR_RETURN(auto tuple_op, DecayRegisteredOperator(tuple_expr->op()));
  if (tuple_op->fingerprint() != make_tuple_fingerprint ||
      tuple_expr->node_deps().size() <= get_nth_operator->index()) {
    return expr;
  }
  return tuple_expr->node_deps()[get_nth_operator->index()];
}
absl::Status AppendGetNOptimizations(PeepholeOptimizationPack& optimizations) {
  ASSIGN_OR_RETURN(
      optimizations.emplace_back(),
      PeepholeOptimization::CreateTransformOptimization(OptimizeTupleGet));
  return absl::OkStatus();
}
}  
absl::StatusOr<PeepholeOptimizationPack> TupleOptimizations() {
  PeepholeOptimizationPack optimizations;
  RETURN_IF_ERROR(AppendGetNOptimizations(optimizations));
  return optimizations;
}
}  