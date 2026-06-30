#include "eval/compiler/comprehension_vulnerability_check.h"
#include <algorithm>
#include <memory>
#include <vector>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/variant.h"
#include "base/ast_internal/ast_impl.h"
#include "base/ast_internal/expr.h"
#include "base/builtins.h"
#include "eval/compiler/flat_expr_builder_extensions.h"
namespace google::api::expr::runtime {
namespace {
using ::cel::ast_internal::Comprehension;
int ComprehensionAccumulationReferences(const cel::ast_internal::Expr& expr,
                                        absl::string_view var_name) {
  struct Handler {
    const cel::ast_internal::Expr& expr;
    absl::string_view var_name;
    int operator()(const cel::ast_internal::Call& call) {
      int references = 0;
      absl::string_view function = call.function();
      if (function == cel::builtin::kTernary && call.args().size() == 3) {
        return std::max(
            ComprehensionAccumulationReferences(call.args()[1], var_name),
            ComprehensionAccumulationReferences(call.args()[2], var_name));
      }
      if (function == cel::builtin::kAdd) {
        for (int i = 0; i < call.args().size(); i++) {
          references +=
              ComprehensionAccumulationReferences(call.args()[i], var_name);
        }
        return references;
      }
      if ((function == cel::builtin::kIndex && call.args().size() == 2) ||
          (function == cel::builtin::kDyn && call.args().size() == 1)) {
        return ComprehensionAccumulationReferences(call.args()[0], var_name);
      }
      return 0;
    }
    int operator()(const cel::ast_internal::Comprehension& comprehension) {
      absl::string_view accu_var = comprehension.accu_var();
      absl::string_view iter_var = comprehension.iter_var();
      int result_references = 0;
      int loop_step_references = 0;
      int sum_of_accumulator_references = 0;
      if (accu_var != var_name && iter_var != var_name) {
        loop_step_references = ComprehensionAccumulationReferences(
            comprehension.loop_step(), var_name);
      }
      if (accu_var != var_name) {
        result_references = ComprehensionAccumulationReferences(
            comprehension.result(), var_name);
      }
      sum_of_accumulator_references = ComprehensionAccumulationReferences(
          comprehension.accu_init(), var_name);
      sum_of_accumulator_references += ComprehensionAccumulationReferences(
          comprehension.iter_range(), var_name);
      return std::max({loop_step_references, result_references,
                       sum_of_accumulator_references});
    }
    int operator()(const cel::ast_internal::CreateList& list) {
      int references = 0;
      for (int i = 0; i < list.elements().size(); i++) {
        references += ComprehensionAccumulationReferences(
            list.elements()[i].expr(), var_name);
      }
      return references;
    }
    int operator()(const cel::ast_internal::CreateStruct& map) {
      int references = 0;
      for (int i = 0; i < map.fields().size(); i++) {
        const auto& entry = map.fields()[i];
        if (entry.has_value()) {
          references +=
              ComprehensionAccumulationReferences(entry.value(), var_name);
        }
      }
      return references;
    }
    int operator()(const cel::MapExpr& map) {
      int references = 0;
      for (int i = 0; i < map.entries().size(); i++) {
        const auto& entry = map.entries()[i];
        if (entry.has_value()) {
          references +=
              ComprehensionAccumulationReferences(entry.value(), var_name);
        }
      }
      return references;
    }
    int operator()(const cel::ast_internal::Select& select) {
      if (select.test_only()) {
        return 0;
      }
      return ComprehensionAccumulationReferences(select.operand(), var_name);
    }
    int operator()(const cel::ast_internal::Ident& ident) {
      return ident.name() == var_name ? 1 : 0;
    }
    int operator()(const cel::ast_internal::Constant& constant) { return 0; }
    int operator()(const cel::UnspecifiedExpr&) { return 0; }
  } handler{expr, var_name};
  return absl::visit(handler, expr.kind());
}
bool ComprehensionHasMemoryExhaustionVulnerability(
    const Comprehension& comprehension) {
  absl::string_view accu_var = comprehension.accu_var();
  const auto& loop_step = comprehension.loop_step();
  return ComprehensionAccumulationReferences(loop_step, accu_var) >= 2;
}
class ComprehensionVulnerabilityCheck : public ProgramOptimizer {
 public:
  absl::Status OnPreVisit(PlannerContext& context,
                          const cel::ast_internal::Expr& node) override {
    if (node.has_comprehension_expr() &&
        ComprehensionHasMemoryExhaustionVulnerability(
            node.comprehension_expr())) {
      return absl::InvalidArgumentError(
          "Comprehension contains memory exhaustion vulnerability");
    }
    return absl::OkStatus();
  }
  absl::Status OnPostVisit(PlannerContext& context,
                           const cel::ast_internal::Expr& node) override {
    return absl::OkStatus();
  }
};
}  
ProgramOptimizerFactory CreateComprehensionVulnerabilityCheck() {
  return [](PlannerContext&, const cel::ast_internal::AstImpl&) {
    return std::make_unique<ComprehensionVulnerabilityCheck>();
  };
}
}  