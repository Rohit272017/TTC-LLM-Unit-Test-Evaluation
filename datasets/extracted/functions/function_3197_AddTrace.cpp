#include "arolla/expr/expr_stack_trace.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "arolla/dense_array/dense_array.h"
#include "arolla/expr/expr_debug_string.h"
#include "arolla/expr/expr_node.h"
#include "arolla/expr/expr_visitor.h"
#include "arolla/util/fingerprint.h"
#include "arolla/util/text.h"
namespace arolla::expr {
void DetailedExprStackTrace::AddTrace(ExprNodePtr target_node,
                                      ExprNodePtr source_node,
                                      TransformationType t) {
  if (!target_node->is_op()) {
    return;
  }
  if (target_node->fingerprint() == source_node->fingerprint()) {
    return;
  }
  traceback_.insert(
      {target_node->fingerprint(), {source_node->fingerprint(), t}});
  if (traceback_.find(source_node->fingerprint()) == traceback_.end()) {
    repr_[source_node->fingerprint()] = source_node;
  }
  if (t != TransformationType::kUntraced) {
    repr_[target_node->fingerprint()] = target_node;
  }
}
std::optional<std::pair<Fingerprint, TransformationType>>
DetailedExprStackTrace::GetTrace(Fingerprint fp) const {
  auto it = traceback_.find(fp);
  if (it == traceback_.end()) {
    return std::nullopt;
  }
  return it->second;
}
std::string DetailedExprStackTrace::GetRepr(Fingerprint fp) const {
  if (auto it = repr_.find(fp); it != repr_.end()) {
    return GetDebugSnippet(it->second);
  } else {
    return absl::StrCat("Could not find representation for node ",
                        fp.AsString());
  }
}
std::vector<DetailedExprStackTrace::Transformation>
DetailedExprStackTrace::GetTransformations(Fingerprint fp) const {
  auto current_fp = fp;
  std::vector<Transformation> transformations;
  absl::flat_hash_set<Fingerprint> visited;
  visited.insert(current_fp);
  auto nxt = GetTrace(current_fp);
  while (nxt.has_value()) {
    if (nxt->second != TransformationType::kUntraced) {
      transformations.push_back({current_fp, nxt->first, nxt->second});
    }
    current_fp = nxt->first;
    if (!visited.insert(current_fp).second) {
      break;
    }
    nxt = GetTrace(current_fp);
  }
  std::reverse(transformations.begin(), transformations.end());
  if (!transformations.empty()) {
    transformations.begin()->source_fp = current_fp;
  }
  return transformations;
}
std::string DetailedExprStackTrace::FullTrace(Fingerprint fp) const {
  auto transformations = GetTransformations(fp);
  if (transformations.empty()) return "";
  std::string stack_trace = absl::StrCat(
      "ORIGINAL NODE: ", GetRepr(transformations.front().source_fp),
      "\nCOMPILED NODE: ", GetRepr(transformations.back().target_fp));
  if (transformations.size() == 1) return stack_trace;
  stack_trace += absl::StrCat("\nDETAILED STACK TRACE:\n",
                              GetRepr(transformations.begin()->source_fp));
  for (auto it = transformations.begin(); it != transformations.end(); ++it) {
    stack_trace += absl::StrCat("\n  ", TransformationString(it->type), "\n",
                                GetRepr(it->target_fp));
  }
  return stack_trace;
}
void LightweightExprStackTrace::AddTrace(ExprNodePtr target_node,
                                         ExprNodePtr source_node,
                                         TransformationType t) {
  if (!target_node->is_op()) {
    return;
  }
  if (target_node->fingerprint() == source_node->fingerprint()) {
    return;
  }
  auto original_it = original_node_mapping_.find(source_node->fingerprint());
  bool source_node_is_original = (original_it == original_node_mapping_.end());
  if (source_node_is_original) {
    original_node_mapping_.insert(
        {target_node->fingerprint(), source_node->fingerprint()});
  } else {
    DCHECK(!original_node_mapping_.contains(original_it->second));
    original_node_mapping_.insert(
        {target_node->fingerprint(), original_it->second});
  }
}
void LightweightExprStackTrace::AddRepresentations(ExprNodePtr compiled_node,
                                                   ExprNodePtr original_node) {
  auto compiled_post_order = PostOrder(compiled_node);
  for (const auto& node : compiled_post_order.nodes()) {
    repr_.insert({node->fingerprint(), node});
  }
  auto original_post_order = PostOrder(original_node);
  for (const auto& node : original_post_order.nodes()) {
    repr_.insert({node->fingerprint(), node});
  }
}
std::string LightweightExprStackTrace::GetRepr(Fingerprint fp) const {
  if (auto it = repr_.find(fp); it != repr_.end()) {
    return GetDebugSnippet(it->second);
  } else {
    return "?";
  }
}
std::string LightweightExprStackTrace::FullTrace(Fingerprint fp) const {
  if (auto it = original_node_mapping_.find(fp);
      it != original_node_mapping_.end()) {
    return absl::StrCat("ORIGINAL NODE: ", GetRepr(it->second),
                        "\nCOMPILED NODE: ", GetRepr(fp));
  } else {
    return absl::StrCat("NODE: ", GetRepr(fp));
  }
}
BoundExprStackTraceBuilder::BoundExprStackTraceBuilder(
    std::shared_ptr<const ExprStackTrace> stack_trace)
    : stack_trace_(stack_trace) {}
void BoundExprStackTraceBuilder::RegisterIp(int64_t ip,
                                            const ExprNodePtr& node) {
  ip_to_fingerprint_.insert({ip, node->fingerprint()});
}
DenseArray<Text> BoundExprStackTraceBuilder::Build(
    int64_t num_operators) const {
  DenseArrayBuilder<Text> traces_array_builder(num_operators);
  for (int64_t i = 0; i < num_operators; ++i) {
    if (auto it = ip_to_fingerprint_.find(i); it != ip_to_fingerprint_.end()) {
      traces_array_builder.Add(i, Text{stack_trace_->FullTrace(it->second)});
    }
  }
  return std::move(traces_array_builder).Build();
}
}  