#include "eval/compiler/constant_folding.h"
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/variant.h"
#include "base/ast_internal/ast_impl.h"
#include "base/ast_internal/expr.h"
#include "base/builtins.h"
#include "base/kind.h"
#include "base/type_provider.h"
#include "common/value.h"
#include "common/value_manager.h"
#include "eval/compiler/flat_expr_builder_extensions.h"
#include "eval/compiler/resolver.h"
#include "eval/eval/const_value_step.h"
#include "eval/eval/evaluator_core.h"
#include "internal/status_macros.h"
#include "runtime/activation.h"
#include "runtime/internal/convert_constant.h"
namespace cel::runtime_internal {
namespace {
using ::cel::ast_internal::AstImpl;
using ::cel::ast_internal::Call;
using ::cel::ast_internal::Comprehension;
using ::cel::ast_internal::Constant;
using ::cel::ast_internal::CreateList;
using ::cel::ast_internal::CreateStruct;
using ::cel::ast_internal::Expr;
using ::cel::ast_internal::Ident;
using ::cel::ast_internal::Select;
using ::cel::builtin::kAnd;
using ::cel::builtin::kOr;
using ::cel::builtin::kTernary;
using ::cel::runtime_internal::ConvertConstant;
using ::google::api::expr::runtime::CreateConstValueDirectStep;
using ::google::api::expr::runtime::CreateConstValueStep;
using ::google::api::expr::runtime::EvaluationListener;
using ::google::api::expr::runtime::ExecutionFrame;
using ::google::api::expr::runtime::ExecutionPath;
using ::google::api::expr::runtime::ExecutionPathView;
using ::google::api::expr::runtime::FlatExpressionEvaluatorState;
using ::google::api::expr::runtime::PlannerContext;
using ::google::api::expr::runtime::ProgramOptimizer;
using ::google::api::expr::runtime::ProgramOptimizerFactory;
using ::google::api::expr::runtime::Resolver;
class ConstantFoldingExtension : public ProgramOptimizer {
 public:
  ConstantFoldingExtension(MemoryManagerRef memory_manager,
                           const TypeProvider& type_provider)
      : memory_manager_(memory_manager),
        state_(kDefaultStackLimit, kComprehensionSlotCount, type_provider,
               memory_manager_) {}
  absl::Status OnPreVisit(google::api::expr::runtime::PlannerContext& context,
                          const Expr& node) override;
  absl::Status OnPostVisit(google::api::expr::runtime::PlannerContext& context,
                           const Expr& node) override;
 private:
  enum class IsConst {
    kConditional,
    kNonConst,
  };
  static constexpr size_t kDefaultStackLimit = 4;
  static constexpr size_t kComprehensionSlotCount = 0;
  MemoryManagerRef memory_manager_;
  Activation empty_;
  FlatExpressionEvaluatorState state_;
  std::vector<IsConst> is_const_;
};
absl::Status ConstantFoldingExtension::OnPreVisit(PlannerContext& context,
                                                  const Expr& node) {
  struct IsConstVisitor {
    IsConst operator()(const Constant&) { return IsConst::kConditional; }
    IsConst operator()(const Ident&) { return IsConst::kNonConst; }
    IsConst operator()(const Comprehension&) {
      return IsConst::kNonConst;
    }
    IsConst operator()(const CreateStruct& create_struct) {
      return IsConst::kNonConst;
    }
    IsConst operator()(const cel::MapExpr& map_expr) {
      if (map_expr.entries().empty()) {
        return IsConst::kNonConst;
      }
      return IsConst::kConditional;
    }
    IsConst operator()(const CreateList& create_list) {
      if (create_list.elements().empty()) {
        return IsConst::kNonConst;
      }
      return IsConst::kConditional;
    }
    IsConst operator()(const Select&) { return IsConst::kConditional; }
    IsConst operator()(const cel::UnspecifiedExpr&) {
      return IsConst::kNonConst;
    }
    IsConst operator()(const Call& call) {
      if (call.function() == kAnd || call.function() == kOr ||
          call.function() == kTernary) {
        return IsConst::kNonConst;
      }
      if (call.function() == "cel.@block") {
        return IsConst::kNonConst;
      }
      int arg_len = call.args().size() + (call.has_target() ? 1 : 0);
      std::vector<cel::Kind> arg_matcher(arg_len, cel::Kind::kAny);
      if (!resolver
               .FindLazyOverloads(call.function(), call.has_target(),
                                  arg_matcher)
               .empty()) {
        return IsConst::kNonConst;
      }
      return IsConst::kConditional;
    }
    const Resolver& resolver;
  };
  IsConst is_const =
      absl::visit(IsConstVisitor{context.resolver()}, node.kind());
  is_const_.push_back(is_const);
  return absl::OkStatus();
}
absl::Status ConstantFoldingExtension::OnPostVisit(PlannerContext& context,
                                                   const Expr& node) {
  if (is_const_.empty()) {
    return absl::InternalError("ConstantFoldingExtension called out of order.");
  }
  IsConst is_const = is_const_.back();
  is_const_.pop_back();
  if (is_const == IsConst::kNonConst) {
    if (!is_const_.empty()) {
      is_const_.back() = IsConst::kNonConst;
    }
    return absl::OkStatus();
  }
  ExecutionPathView subplan = context.GetSubplan(node);
  if (subplan.empty()) {
    return absl::OkStatus();
  }
  Value value;
  if (node.has_const_expr()) {
    CEL_ASSIGN_OR_RETURN(
        value, ConvertConstant(node.const_expr(), state_.value_factory()));
  } else {
    ExecutionFrame frame(subplan, empty_, context.options(), state_);
    state_.Reset();
    state_.value_stack().SetMaxSize(subplan.size());
    auto result = frame.Evaluate();
    if (!result.ok()) {
      return absl::OkStatus();
    }
    value = *result;
    if (value->Is<UnknownValue>()) {
      return absl::OkStatus();
    }
  }
  if (context.options().max_recursion_depth != 0) {
    return context.ReplaceSubplan(
        node, CreateConstValueDirectStep(std::move(value), node.id()), 1);
  }
  ExecutionPath new_plan;
  CEL_ASSIGN_OR_RETURN(
      new_plan.emplace_back(),
      CreateConstValueStep(std::move(value), node.id(), false));
  return context.ReplaceSubplan(node, std::move(new_plan));
}
}  
ProgramOptimizerFactory CreateConstantFoldingOptimizer(
    MemoryManagerRef memory_manager) {
  return [memory_manager](PlannerContext& ctx, const AstImpl&)
             -> absl::StatusOr<std::unique_ptr<ProgramOptimizer>> {
    return std::make_unique<ConstantFoldingExtension>(
        memory_manager, ctx.value_factory().type_provider());
  };
}
}  