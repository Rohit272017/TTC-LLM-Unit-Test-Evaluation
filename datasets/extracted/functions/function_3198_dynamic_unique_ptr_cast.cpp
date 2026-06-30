#include "arolla/expr/eval/dynamic_compiled_operator.h"
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "arolla/expr/eval/dynamic_compiled_expr.h"
#include "arolla/expr/eval/eval.h"
#include "arolla/expr/eval/executable_builder.h"
#include "arolla/expr/expr.h"
#include "arolla/expr/expr_node.h"
#include "arolla/expr/expr_operator.h"
#include "arolla/qtype/qtype.h"
#include "arolla/util/fingerprint.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla::expr::eval_internal {
template <typename T, typename U>
std::unique_ptr<T> dynamic_unique_ptr_cast(std::unique_ptr<U> unique) {
  T* casted = dynamic_cast<T*>(unique.get());
  if (casted != nullptr) {
    unique.release();
  }  
  return std::unique_ptr<T>(casted);
}
absl::StatusOr<DynamicCompiledOperator> DynamicCompiledOperator::Build(
    const DynamicEvaluationEngineOptions& options, const ExprOperatorPtr& op,
    std::vector<QTypePtr> input_qtypes) {
  std::vector<absl::StatusOr<ExprNodePtr>> inputs;
  std::vector<std::string> input_arg_names;
  absl::flat_hash_map<std::string, QTypePtr> input_qtypes_map;
  inputs.reserve(input_qtypes.size());
  input_qtypes_map.reserve(input_qtypes.size());
  input_arg_names.reserve(input_qtypes.size());
  for (size_t i = 0; i < input_qtypes.size(); ++i) {
    std::string name = absl::StrFormat("_%d", i);
    inputs.push_back(Leaf(name));
    input_qtypes_map.emplace(name, input_qtypes[i]);
    input_arg_names.emplace_back(std::move(name));
  }
  ASSIGN_OR_RETURN(auto expr, CallOp(op, inputs));
  ASSIGN_OR_RETURN(auto compiled_expr, CompileForDynamicEvaluation(
                                           options, expr, input_qtypes_map));
  std::unique_ptr<const DynamicCompiledExpr> dynamic_compiled_expr =
      dynamic_unique_ptr_cast<const DynamicCompiledExpr>(
          std::move(compiled_expr));
  DCHECK(dynamic_compiled_expr);
  return DynamicCompiledOperator(
      std::string(op->display_name()), std::move(input_qtypes),
      std::move(dynamic_compiled_expr), std::move(input_arg_names),
      FingerprintHasher("arolla::expr::eval_internal::DynamicCompiledOperator")
          .Combine(op->fingerprint())
          .CombineSpan(input_qtypes)
          .Finish());
}
absl::Status DynamicCompiledOperator::BindTo(
    ExecutableBuilder& executable_builder,
    absl::Span<const TypedSlot> input_slots, TypedSlot output_slot) const {
  if (input_slots.size() != input_arg_names_.size()) {
    return absl::InternalError(absl::StrFormat(
        "input count mismatch in DynamicCompiledOperator: expected %d, got %d",
        input_arg_names_.size(), input_slots.size()));
  }
  absl::flat_hash_map<std::string, TypedSlot> input_slots_map;
  input_slots_map.reserve(input_slots.size());
  for (size_t i = 0; i < input_slots.size(); ++i) {
    input_slots_map.emplace(input_arg_names_[i], input_slots[i]);
  }
  return compiled_expr_->BindToExecutableBuilder(executable_builder,
                                                 input_slots_map, output_slot);
}
DynamicCompiledOperator::DynamicCompiledOperator(
    std::string display_name, std::vector<QTypePtr> input_qtypes,
    std::unique_ptr<const DynamicCompiledExpr> compiled_expr,
    std::vector<std::string> input_arg_names, Fingerprint fingerprint)
    : display_name_(std::move(display_name)),
      input_qtypes_(input_qtypes.begin(), input_qtypes.end()),
      compiled_expr_(std::move(compiled_expr)),
      input_arg_names_(std::move(input_arg_names)),
      fingerprint_(fingerprint) {
  DCHECK_EQ(input_qtypes_.size(), input_arg_names_.size());
}
}  