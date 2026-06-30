#include "arolla/expr/optimization/peephole_optimizations/arithmetic.h"
#include <cstdint>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "arolla/expr/expr.h"
#include "arolla/expr/expr_node.h"
#include "arolla/expr/optimization/peephole_optimizer.h"
#include "arolla/memory/optional_value.h"
#include "arolla/util/meta.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla::expr {
namespace {
absl::Status RemoveAddOptimizationsImpl(
    const ExprNodePtr& zero, PeepholeOptimizationPack& optimizations) {
  auto same_qtype = [qtype = zero->qtype()](const ExprNodePtr& expr) {
    return expr->qtype() == qtype;
  };
  ExprNodePtr a = Placeholder("a");
  ASSIGN_OR_RETURN(ExprNodePtr from1, CallOpReference("math.add", {a, zero}));
  ASSIGN_OR_RETURN(ExprNodePtr from2, CallOpReference("math.add", {zero, a}));
  for (const auto& from : {from1, from2}) {
    ASSIGN_OR_RETURN(optimizations.emplace_back(),
                     PeepholeOptimization::CreatePatternOptimization(
                         from, a, {{"a", same_qtype}}));
  }
  return absl::OkStatus();
}
template <class T>
absl::Status RemoveAddOptimizationsImpl(
    PeepholeOptimizationPack& optimizations) {
  return RemoveAddOptimizationsImpl(Literal(T(0)), optimizations);
}
absl::Status RemoveAddOptimizations(PeepholeOptimizationPack& optimizations) {
  absl::Status res_status;
  meta::foreach_type<meta::type_list<float, double, int32_t, int64_t>>(
      [&](auto meta_type) {
        using type = typename decltype(meta_type)::type;
        auto status = RemoveAddOptimizationsImpl<type>(optimizations);
        if (!status.ok()) res_status = status;
        status = RemoveAddOptimizationsImpl<OptionalValue<type>>(optimizations);
        if (!status.ok()) res_status = status;
      });
  return res_status;
}
absl::Status RemoveMulOptimizationsImpl(
    const ExprNodePtr& one, PeepholeOptimizationPack& optimizations) {
  ExprNodePtr a = Placeholder("a");
  auto same_qtype = [qtype = one->qtype()](const ExprNodePtr& expr) {
    return expr->qtype() == qtype;
  };
  ASSIGN_OR_RETURN(ExprNodePtr to_a1,
                   CallOpReference("math.multiply", {a, one}));
  ASSIGN_OR_RETURN(ExprNodePtr to_a2,
                   CallOpReference("math.multiply", {one, a}));
  for (const auto& from : {to_a1, to_a2}) {
    ASSIGN_OR_RETURN(optimizations.emplace_back(),
                     PeepholeOptimization::CreatePatternOptimization(
                         from, a, {{"a", same_qtype}}));
  }
  return absl::OkStatus();
}
template <class T>
absl::Status RemoveMulOptimizationsImpl(
    PeepholeOptimizationPack& optimizations) {
  return RemoveMulOptimizationsImpl(Literal<T>(1), optimizations);
}
absl::Status RemoveMulOptimizations(PeepholeOptimizationPack& optimizations) {
  absl::Status res_status;
  meta::foreach_type<meta::type_list<float, double, int32_t, int64_t>>(
      [&](auto meta_type) {
        using type = typename decltype(meta_type)::type;
        auto status = RemoveMulOptimizationsImpl<type>(optimizations);
        if (!status.ok()) res_status = status;
        status = RemoveMulOptimizationsImpl<OptionalValue<type>>(optimizations);
        if (!status.ok()) res_status = status;
      });
  return res_status;
}
}  
absl::StatusOr<PeepholeOptimizationPack> ArithmeticOptimizations() {
  PeepholeOptimizationPack optimizations;
  RETURN_IF_ERROR(RemoveAddOptimizations(optimizations));
  RETURN_IF_ERROR(RemoveMulOptimizations(optimizations));
  return optimizations;
}
}  