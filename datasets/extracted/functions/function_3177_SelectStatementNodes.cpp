#include "arolla/expr/expr_debug_string.h"
#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "arolla/expr/annotation_utils.h"
#include "arolla/expr/expr_node.h"
#include "arolla/expr/expr_operator.h"
#include "arolla/expr/expr_visitor.h"
#include "arolla/expr/operator_repr_functions.h"
#include "arolla/expr/registered_expr_operator.h"
#include "arolla/util/fingerprint.h"
#include "arolla/util/repr.h"
#include "arolla/util/string.h"
namespace arolla::expr {
namespace {
std::vector<ExprNodePtr> SelectStatementNodes(const PostOrder& post_order) {
  const size_t kCriticalDepth = 3;
  std::vector<size_t> node_parent_count(post_order.nodes_size(), 0);
  for (size_t i = 0; i < post_order.nodes_size(); ++i) {
    for (size_t j : post_order.dep_indices(i)) {
      node_parent_count[j] += 1;
    }
  }
  std::vector<ExprNodePtr> result;
  std::vector<size_t> node_depth(post_order.nodes_size());
  for (size_t i = 0; i < post_order.nodes_size(); ++i) {
    size_t depth = 1;
    for (size_t j : post_order.dep_indices(i)) {
      depth = std::max(depth, 1 + node_depth[j]);
    }
    const auto& node = post_order.node(i);
    const bool is_statement =
        IsNameAnnotation(node) ||  
        (node_parent_count[i] > 1 &&
         depth >= kCriticalDepth);  
    if (is_statement) {
      result.push_back(node);
      depth = 1;
    }
    node_depth[i] = depth;
  }
  return result;
}
constexpr bool IsSafeStatementName(absl::string_view str) {
  return IsQualifiedIdentifier(str) &&
         !(str.size() > 1 && str[0] == '_' &&
           std::find_if_not(str.begin() + 1, str.end(), IsDigit) == str.end());
}
absl::flat_hash_map<Fingerprint, std::string> GenStatementNames(
    const PostOrder& post_order) {
  const auto statement_nodes = SelectStatementNodes(post_order);
  absl::flat_hash_map<absl::string_view, size_t> name_counts;
  name_counts.reserve(statement_nodes.size());
  for (const auto& node : statement_nodes) {
    if (auto name = ReadNameAnnotation(node); IsSafeStatementName(name)) {
      name_counts[name] += 1;
    }
  }
  for (auto& [_, v] : name_counts) {
    v = (v > 1);
  }
  absl::flat_hash_map<Fingerprint, std::string> result;
  result.reserve(statement_nodes.size());
  size_t anonymous_count = 1;
  for (const auto& node : statement_nodes) {
    const auto name = ReadNameAnnotation(node);
    if (!IsSafeStatementName(name)) {
      result.emplace(node->fingerprint(), absl::StrCat("_", anonymous_count++));
      continue;
    }
    auto& name_count = name_counts[name];
    if (name_count == 0) {
      result.emplace(node->fingerprint(), name);
    } else {
      result.emplace(node->fingerprint(),
                     absl::StrCat(name, "._", name_count++));
    }
  }
  return result;
}
std::vector<const ReprToken*> GetNodeDepsTokens(
    const ExprNodePtr& node,
    const absl::flat_hash_map<Fingerprint, ReprToken>& node_tokens) {
  std::vector<const ReprToken*> inputs(node->node_deps().size());
  for (size_t i = 0; i < node->node_deps().size(); ++i) {
    inputs[i] = &node_tokens.at(node->node_deps()[i]->fingerprint());
  }
  return inputs;
}
ReprToken FormatLiteral(const ExprNodePtr& node) {
  if (auto literal = node->qvalue()) {
    return literal->GenReprToken();
  } else {
    return ReprToken{"<broken_literal>"};
  }
}
ReprToken FormatLeaf(const ExprNodePtr& node) {
  return ReprToken{absl::StrCat("L", ContainerAccessString(node->leaf_key()))};
}
ReprToken FormatPlaceholder(const ExprNodePtr& node) {
  return ReprToken{
      absl::StrCat("P", ContainerAccessString(node->placeholder_key()))};
}
ReprToken FormatOperatorCanonical(const ExprNodePtr& node,
                                  absl::Span<const ReprToken* const> inputs) {
  ReprToken result;
  if (IsRegisteredOperator(node->op())) {
    absl::StrAppend(&result.str, "M.");
  }
  absl::StrAppend(&result.str, node->op()->display_name(), "(");
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (i > 0) {
      absl::StrAppend(&result.str, ", ");
    }
    absl::StrAppend(&result.str, inputs[i]->str);
  }
  absl::StrAppend(&result.str, ")");
  return result;
}
ReprToken FormatOperatorVerbose(const ExprNodePtr& node,
                                absl::Span<const ReprToken* const> inputs) {
  ReprToken result = FormatOperatorCanonical(node, inputs);
  if (!IsQTypeAnnotation(node)) {
    if (auto* qtype = node->qtype()) {
      absl::StrAppend(&result.str, ":", qtype->name());
    }
  }
  return result;
}
ReprToken FormatOperatorPretty(
    const ExprNodePtr& node,
    const absl::flat_hash_map<Fingerprint, ReprToken>& node_tokens) {
  if (auto repr = FormatOperatorNodePretty(node, node_tokens)) {
    return *std::move(repr);
  }
  return FormatOperatorCanonical(node, GetNodeDepsTokens(node, node_tokens));
}
ReprToken FormatVerbose(const ExprNodePtr& node,
                        absl::Span<const ReprToken* const> inputs) {
  switch (node->type()) {
    case ExprNodeType::kLiteral:
      return FormatLiteral(node);
    case ExprNodeType::kLeaf:
      return FormatLeaf(node);
    case ExprNodeType::kPlaceholder:
      return FormatPlaceholder(node);
    case ExprNodeType::kOperator:
      return FormatOperatorVerbose(node, inputs);
  }
  ABSL_UNREACHABLE();
}
ReprToken FormatPretty(
    const ExprNodePtr& node,
    const absl::flat_hash_map<Fingerprint, ReprToken>& node_tokens) {
  switch (node->type()) {
    case ExprNodeType::kLiteral:
      return FormatLiteral(node);
    case ExprNodeType::kLeaf:
      return FormatLeaf(node);
    case ExprNodeType::kPlaceholder:
      return FormatPlaceholder(node);
    case ExprNodeType::kOperator:
      return FormatOperatorPretty(node, node_tokens);
  }
  ABSL_UNREACHABLE();
}
ReprToken FormatWithHiddenInputs(const ExprNodePtr& node) {
  const ReprToken kDots{.str = "..."};
  std::vector<const ReprToken*> inputs(node->node_deps().size(), &kDots);
  return FormatVerbose(node, inputs);
}
}  
std::string ToDebugString(const ExprNodePtr& root, bool verbose) {
  const PostOrder post_order(root);
  const auto statement_names = GenStatementNames(post_order);
  std::vector<std::string> result;
  absl::flat_hash_map<Fingerprint, ReprToken> node_tokens(
      post_order.nodes_size());
  auto format = verbose ? [](
      const ExprNodePtr& node,
      const absl::flat_hash_map<Fingerprint, ReprToken>& node_tokens) {
    return FormatVerbose(node, GetNodeDepsTokens(node, node_tokens));
  }: FormatPretty;
  for (const auto& node : post_order.nodes()) {
    auto it = statement_names.find(node->fingerprint());
    if (it == statement_names.end()) {
      node_tokens[node->fingerprint()] = format(node, node_tokens);
      continue;
    }
    const auto& statement_name = it->second;
    if (IsSafeStatementName(ReadNameAnnotation(node))) {
      DCHECK_EQ(node->node_deps().size(), 2);
      const auto& res = node_tokens[node->node_deps()[0]->fingerprint()];
      result.push_back(absl::StrCat(statement_name, " = ", res.str));
    } else {
      result.push_back(
          absl::StrCat(statement_name, " = ", format(node, node_tokens).str));
    }
    node_tokens[node->fingerprint()] = ReprToken{.str = statement_name};
  }
  result.push_back(std::move(node_tokens[root->fingerprint()].str));
  return absl::StrJoin(result, "\n");
}
constexpr int kMaxDebugSnippetSize = 200;
std::string GetDebugSnippet(const ExprNodePtr& node) {
  const auto& node_deps = node->node_deps();
  absl::InlinedVector<ReprToken, 4> dep_snippets(node_deps.size());
  absl::InlinedVector<const ReprToken*, 4> dep_snippet_ptrs(node_deps.size());
  for (size_t i = 0; i < node_deps.size(); ++i) {
    dep_snippets[i] = FormatWithHiddenInputs(node_deps[i]);
    dep_snippet_ptrs[i] = &dep_snippets[i];
  }
  std::string snippet = FormatVerbose(node, dep_snippet_ptrs).str;
  return Truncate(std::move(snippet), kMaxDebugSnippetSize);
}
}  