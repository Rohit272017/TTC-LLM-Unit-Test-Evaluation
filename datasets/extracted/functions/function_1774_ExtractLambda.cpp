#include "arolla/expr/eval/expr_utils.h"
#include <functional>
#include <stack>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "arolla/expr/expr.h"
#include "arolla/expr/expr_node.h"
#include "arolla/expr/expr_operator.h"
#include "arolla/expr/expr_operator_signature.h"
#include "arolla/expr/lambda_expr_operator.h"
#include "arolla/util/fingerprint.h"
#include "arolla/util/status_macros_backport.h"
namespace arolla::expr::eval_internal {
absl::StatusOr<ExprNodePtr> ExtractLambda(
    const ExprNodePtr& expr,
    std::function<absl::StatusOr<bool>(const ExprNodePtr&)> is_in_lambda) {
  struct Task {
    enum class Stage { kPreorder, kPostorder };
    ExprNodePtr node;
    Stage stage;
  };
  std::vector<ExprNodePtr> lambda_args;
  ExprOperatorSignature lambda_signature;
  absl::flat_hash_set<Fingerprint> previsited;
  absl::flat_hash_map<Fingerprint, ExprNodePtr> new_nodes;
  std::stack<Task> tasks;
  tasks.push(Task{.node = expr, .stage = Task::Stage::kPreorder});
  int next_placeholder = 0;
  while (!tasks.empty()) {
    auto [node, stage] = std::move(tasks.top());
    tasks.pop();
    if (stage == Task::Stage::kPreorder) {
      if (!previsited.insert(node->fingerprint()).second) {
        continue;
      }
      ASSIGN_OR_RETURN(bool in_lambda, is_in_lambda(node));
      if (in_lambda) {
        tasks.push(Task{.node = node, .stage = Task::Stage::kPostorder});
        for (auto dep = node->node_deps().rbegin();
             dep != node->node_deps().rend(); ++dep) {
          tasks.push(Task{.node = *dep, .stage = Task::Stage::kPreorder});
        }
      } else {
        auto [it, inserted] = new_nodes.insert({node->fingerprint(), nullptr});
        if (inserted) {
          it->second = Placeholder(absl::StrCat("_", next_placeholder++));
          lambda_args.emplace_back(node);
          lambda_signature.parameters.push_back(
              ExprOperatorSignature::Parameter{
                  .name = it->second->placeholder_key()});
        }
      }
    } else {
      std::vector<ExprNodePtr> new_deps;
      new_deps.reserve(node->node_deps().size());
      for (const auto& dep : node->node_deps()) {
        new_deps.push_back(new_nodes.at(dep->fingerprint()));
      }
      ASSIGN_OR_RETURN(new_nodes[node->fingerprint()],
                       WithNewDependencies(node, new_deps));
    }
  }
  ASSIGN_OR_RETURN(
      ExprOperatorPtr lambda,
      MakeLambdaOperator(lambda_signature, new_nodes.at(expr->fingerprint())));
  return MakeOpNode(lambda, lambda_args);
}
}  