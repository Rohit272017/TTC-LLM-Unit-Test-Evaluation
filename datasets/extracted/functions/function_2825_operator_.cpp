#include "arolla/expr/expr_node.h"
#include <cstddef>
#include <deque>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>
#include "absl/base/no_destructor.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "arolla/expr/expr_attributes.h"
#include "arolla/expr/expr_operator.h"
#include "arolla/qtype/typed_value.h"
#include "arolla/util/fingerprint.h"
namespace arolla::expr {
std::ostream& operator<<(std::ostream& os, ExprNodeType t) {
  switch (t) {
    case expr::ExprNodeType::kLiteral: {
      return os << "kLiteral";
    }
    case expr::ExprNodeType::kLeaf: {
      return os << "kLeaf";
    }
    case expr::ExprNodeType::kOperator: {
      return os << "kOperator";
    }
    case expr::ExprNodeType::kPlaceholder: {
      return os << "kPlaceholder";
    }
  }
  return os << "ExprNodeType(" << static_cast<int>(t) << ")";
}
ExprNodePtr ExprNode::MakeLiteralNode(TypedValue&& qvalue) {
  FingerprintHasher hasher("LiteralNode");
  hasher.Combine(qvalue.GetFingerprint());
  auto self = std::make_unique<ExprNode>(PrivateConstructorTag());
  self->type_ = ExprNodeType::kLiteral;
  self->attr_ = ExprAttributes(std::move(qvalue));
  self->fingerprint_ = std::move(hasher).Finish();
  return ExprNodePtr::Own(std::move(self));
}
ExprNodePtr ExprNode::MakeLeafNode(absl::string_view leaf_key) {
  auto self = std::make_unique<ExprNode>(PrivateConstructorTag());
  self->type_ = ExprNodeType::kLeaf;
  self->leaf_key_ = std::string(leaf_key);
  self->fingerprint_ = FingerprintHasher("LeafNode").Combine(leaf_key).Finish();
  return ExprNodePtr::Own(std::move(self));
}
ExprNodePtr ExprNode::MakePlaceholderNode(absl::string_view placeholder_key) {
  auto self = std::make_unique<ExprNode>(PrivateConstructorTag());
  self->type_ = ExprNodeType::kPlaceholder;
  self->placeholder_key_ = std::string(placeholder_key);
  self->fingerprint_ =
      FingerprintHasher("PlaceholderNode").Combine(placeholder_key).Finish();
  return ExprNodePtr::Own(std::move(self));
}
ExprNodePtr ExprNode::UnsafeMakeOperatorNode(
    ExprOperatorPtr&& op, std::vector<ExprNodePtr>&& node_deps,
    ExprAttributes&& attr) {
  FingerprintHasher hasher("OpNode");
  DCHECK(op);
  hasher.Combine(op->fingerprint());
  for (const auto& node_dep : node_deps) {
    DCHECK(node_dep != nullptr);
    hasher.Combine(node_dep->fingerprint());
  }
  hasher.Combine(attr);
  auto self = std::make_unique<ExprNode>(PrivateConstructorTag());
  self->type_ = ExprNodeType::kOperator;
  self->op_ = std::move(op);
  self->node_deps_ = std::move(node_deps);
  self->attr_ = std::move(attr);
  self->fingerprint_ = std::move(hasher).Finish();
  return ExprNodePtr::Own(std::move(self));
}
ExprNode::~ExprNode() {
  if (node_deps_.empty()) {
    return;
  }
  constexpr size_t kMaxDepth = 32;
  thread_local absl::NoDestructor<std::deque<std::vector<ExprNodePtr>>> deps;
  thread_local size_t destructor_depth = 0;
  if (destructor_depth > kMaxDepth) {
    deps->push_back(std::move(node_deps_));
    return;
  }
  destructor_depth++;
  absl::Cleanup decrease_depth = [&] { --destructor_depth; };
  node_deps_.clear();
  if (destructor_depth == 1 && !deps->empty()) {
    while (!deps->empty()) {
      auto tmp = std::move(deps->back());
      deps->pop_back();
    }
    deps->shrink_to_fit();
  }
}
}  