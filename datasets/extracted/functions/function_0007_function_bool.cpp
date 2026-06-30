#include "arolla/expr/optimization/peephole_optimizations/short_circuit_where.h"
#include <functional>
#include <memory>
#include <utility>
#include <vector>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "arolla/expr/expr.h"
#include "arolla/expr/expr_node.h"
#include "arolla/expr/operators/type_meta_eval_strategies.h"
#include "arolla/expr/optimization/peephole_optimizer.h"
#include "arolla/memory/optional_value.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla::expr {
namespace {
using ::arolla::expr_operators::type_meta::Is;
std::function<bool(const ExprNodePtr&)> TypeMatches(
    expr_operators::type_meta::Strategy strategy) {
  return [strategy = std::move(strategy)](const ExprNodePtr& node) {
    return node->qtype() != nullptr && strategy({node->qtype()}).ok();
  };
}
absl::Status AddCoreWhereOptimizations(
    PeepholeOptimizationPack& optimizations) {
  ExprNodePtr cond = Placeholder("cond");
  ExprNodePtr x = Placeholder("x");
  ExprNodePtr y = Placeholder("y");
  {
    ASSIGN_OR_RETURN(ExprNodePtr from,
                     CallOpReference("core.where", {cond, x, y}));
    ASSIGN_OR_RETURN(
        ExprNodePtr to,
        CallOpReference("core._short_circuit_where", {cond, x, y}));
    ASSIGN_OR_RETURN(optimizations.emplace_back(),
                     PeepholeOptimization::CreatePatternOptimization(
                         from, to, {{"cond", TypeMatches(Is<OptionalUnit>)}}));
  }
  {
    ExprNodePtr shape = Placeholder("shape");
    ASSIGN_OR_RETURN(
        ExprNodePtr from,
        CallOpReference("core.where",
                        {CallOpReference("core.const_with_shape._array_shape",
                                         {shape, cond}),
                         x, y}));
    ASSIGN_OR_RETURN(
        ExprNodePtr to,
        CallOpReference("core._short_circuit_where", {cond, x, y}));
    ASSIGN_OR_RETURN(optimizations.emplace_back(),
                     PeepholeOptimization::CreatePatternOptimization(
                         from, to, {{"cond", TypeMatches(Is<OptionalUnit>)}}));
  }
  return absl::OkStatus();
}
absl::StatusOr<std::unique_ptr<PeepholeOptimization>>
AlwaysTrueConditionOptimization() {
  ASSIGN_OR_RETURN(
      ExprNodePtr short_circuit_where,
      CallOpReference("core._short_circuit_where",
                      {Literal(kPresent), Placeholder("x"), Placeholder("y")}));
  ExprNodePtr x = Placeholder("x");
  return PeepholeOptimization::CreatePatternOptimization(short_circuit_where,
                                                         x);
}
absl::StatusOr<std::unique_ptr<PeepholeOptimization>>
AlwaysFalseConditionOptimization() {
  ASSIGN_OR_RETURN(
      ExprNodePtr short_circuit_where,
      CallOpReference("core._short_circuit_where",
                      {Literal(kMissing), Placeholder("x"), Placeholder("y")}));
  ExprNodePtr y = Placeholder("y");
  return PeepholeOptimization::CreatePatternOptimization(short_circuit_where,
                                                         y);
}
}  
absl::StatusOr<PeepholeOptimizationPack> ShortCircuitWhereOptimizations() {
  std::vector<std::unique_ptr<PeepholeOptimization>> optimizations;
  RETURN_IF_ERROR(AddCoreWhereOptimizations(optimizations));
  ASSIGN_OR_RETURN(optimizations.emplace_back(),
                   AlwaysTrueConditionOptimization());
  ASSIGN_OR_RETURN(optimizations.emplace_back(),
                   AlwaysFalseConditionOptimization());
  return optimizations;
}
}  