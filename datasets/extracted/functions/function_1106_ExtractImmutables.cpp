#include "arolla/expr/operators/while_loop/while_loop_impl.h"
#include <algorithm>
#include <functional>
#include <string>
#include <utility>
#include <vector>
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "arolla/expr/expr.h"
#include "arolla/expr/expr_node.h"
#include "arolla/expr/expr_operator.h"
#include "arolla/expr/expr_visitor.h"
#include "arolla/expr/operators/while_loop/while_loop.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla::expr_operators::while_loop_impl {
using ::arolla::expr::ExprNodePtr;
using ::arolla::expr::ExprOperatorPtr;
using ::arolla::expr::Placeholder;
absl::StatusOr<std::pair<ExprNodePtr, NamedExpressions>> ExtractImmutables(
    const ExprNodePtr& expr, std::function<std::string(const ExprNodePtr& node)>
                                 immutable_naming_function) {
  NamedExpressions immutables;
  struct Visit {
    ExprNodePtr expr;
    bool has_placeholder_dep;
    bool has_leaf_dep;
  };
  ASSIGN_OR_RETURN(
      (auto [converted_expr, has_placeholder_dep, has_leaf_dep]),
      expr::PostOrderTraverse(
          expr,
          [&](const ExprNodePtr& node,
              absl::Span<const Visit* const> visits) -> absl::StatusOr<Visit> {
            if (node->is_placeholder()) {
              return Visit{.expr = node,
                           .has_placeholder_dep = true,
                           .has_leaf_dep = false};
            }
            if (node->is_leaf()) {
              return Visit{.expr = node,
                           .has_placeholder_dep = false,
                           .has_leaf_dep = true};
            }
            bool has_placeholder_dep = std::any_of(
                visits.begin(), visits.end(),
                [](const auto& v) { return v->has_placeholder_dep; });
            bool has_leaf_dep =
                std::any_of(visits.begin(), visits.end(),
                            [](const auto& v) { return v->has_leaf_dep; });
            if (!has_placeholder_dep) {
              return Visit{.expr = node,
                           .has_placeholder_dep = false,
                           .has_leaf_dep = has_leaf_dep};
            }
            std::vector<ExprNodePtr> new_deps;
            new_deps.reserve(visits.size());
            for (const auto& visit : visits) {
              if (visit->has_placeholder_dep || !visit->has_leaf_dep) {
                new_deps.push_back(visit->expr);
              } else {
                auto placeholder_key = immutable_naming_function(visit->expr);
                new_deps.emplace_back(Placeholder(placeholder_key));
                immutables.emplace(std::move(placeholder_key), visit->expr);
              }
            }
            ASSIGN_OR_RETURN(auto new_node, expr::WithNewDependencies(
                                                node, std::move(new_deps)));
            return Visit{.expr = new_node,
                         .has_placeholder_dep = true,
                         .has_leaf_dep = has_leaf_dep};
          }));
  if (!has_placeholder_dep) {
    DCHECK(immutables.empty());
    auto placeholder_key = immutable_naming_function(converted_expr);
    immutables.emplace(placeholder_key, converted_expr);
    converted_expr = Placeholder(placeholder_key);
  }
  return {{std::move(converted_expr), std::move(immutables)}};
}
}  