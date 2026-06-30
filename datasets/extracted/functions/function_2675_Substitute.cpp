#include "arolla/expr/visitors/substitution.h"
#include <optional>
#include <string>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "arolla/expr/annotation_utils.h"
#include "arolla/expr/expr.h"
#include "arolla/expr/expr_node.h"
#include "arolla/expr/expr_visitor.h"
#include "arolla/util/fingerprint.h"
namespace arolla::expr {
namespace {
template <class Key, class KeyFn>
absl::StatusOr<ExprNodePtr> Substitute(
    ExprNodePtr expr, const absl::flat_hash_map<Key, ExprNodePtr>& subs,
    KeyFn key_fn) {
  return PostOrderTraverse(
      expr,
      [&](const ExprNodePtr& node, absl::Span<const ExprNodePtr* const> visits)
          -> absl::StatusOr<ExprNodePtr> {
        if (auto key = key_fn(node); key.has_value()) {
          if (auto it = subs.find(*key); it != subs.end()) {
            return it->second;
          }
        }
        return WithNewDependencies(node, DereferenceVisitPointers(visits));
      });
}
}  
absl::StatusOr<ExprNodePtr> SubstituteByName(
    ExprNodePtr expr,
    const absl::flat_hash_map<std::string, ExprNodePtr>& subs) {
  return Substitute(expr, subs,
                    [](const auto& expr) -> std::optional<std::string> {
                      if (IsNameAnnotation(expr)) {
                        return std::string(ReadNameAnnotation(expr));
                      }
                      return std::nullopt;
                    });
}
absl::StatusOr<ExprNodePtr> SubstituteLeaves(
    ExprNodePtr expr,
    const absl::flat_hash_map<std::string, ExprNodePtr>& subs) {
  return Substitute(expr, subs,
                    [](const auto& expr) -> std::optional<std::string> {
                      if (expr->is_leaf()) return expr->leaf_key();
                      return std::nullopt;
                    });
}
absl::StatusOr<ExprNodePtr> SubstitutePlaceholders(
    ExprNodePtr expr, const absl::flat_hash_map<std::string, ExprNodePtr>& subs,
    bool must_substitute_all) {
  return PostOrderTraverse(
      expr,
      [&](const ExprNodePtr& node, absl::Span<const ExprNodePtr* const> visits)
          -> absl::StatusOr<ExprNodePtr> {
        if (node->is_placeholder()) {
          if (subs.contains(node->placeholder_key())) {
            return subs.at(node->placeholder_key());
          } else if (must_substitute_all) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "No value was provided for P.%s, but substitution of all "
                "placeholders was requested.",
                node->placeholder_key()));
          }
        }
        return WithNewDependencies(node, DereferenceVisitPointers(visits));
      });
}
absl::StatusOr<ExprNodePtr> SubstituteByFingerprint(
    ExprNodePtr expr,
    const absl::flat_hash_map<Fingerprint, ExprNodePtr>& subs) {
  return Substitute(expr, subs,
                    [](const auto& expr) -> std::optional<Fingerprint> {
                      return expr->fingerprint();
                    });
}
}  