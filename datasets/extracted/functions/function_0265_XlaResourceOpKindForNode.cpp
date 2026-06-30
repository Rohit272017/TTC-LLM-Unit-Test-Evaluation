#include "tensorflow/compiler/jit/resource_operation_safety_analysis.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/tf2xla/resource_operation_table.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/hash/hash.h"
namespace tensorflow {
namespace {
Status XlaResourceOpKindForNode(
    const Node& n, const FunctionLibraryDefinition* flib_def,
    const std::function<Status(const Node&, bool*)>& resource_ops_to_ignore,
    std::optional<XlaResourceOpKind>* out_resource_op_kind) {
  bool should_ignore = false;
  if (resource_ops_to_ignore) {
    TF_RETURN_IF_ERROR(resource_ops_to_ignore(n, &should_ignore));
  }
  if (should_ignore) {
    *out_resource_op_kind = std::nullopt;
    return absl::OkStatus();
  }
  const XlaResourceOpInfo* op_info = GetResourceOpInfoForOp(n.type_string());
  if (op_info) {
    *out_resource_op_kind = op_info->kind();
    return absl::OkStatus();
  }
  if (MayCallFunction(n, flib_def)) {
    *out_resource_op_kind = XlaResourceOpKind::kReadWrite;
  } else {
    *out_resource_op_kind = std::nullopt;
  }
  return absl::OkStatus();
}
bool IsEdgeSafe(XlaResourceOpKind from, XlaResourceOpKind to) {
  return from == XlaResourceOpKind::kRead || to == XlaResourceOpKind::kWrite;
}
using ResourceOp = std::pair<int, XlaResourceOpKind>;
string ResourceOpToString(const ResourceOp& resource_op) {
  return absl::StrCat(
      resource_op.first, ": ",
      XlaResourceOpInfo::XlaResourceOpKindToString(resource_op.second));
}
class ResourceOpSet {
 private:
  using Impl = absl::flat_hash_set<ResourceOp>;
 public:
  ResourceOpSet() = default;
  void Add(const ResourceOpSet& other) {
    CHECK(!frozen_);
    if (other.impl_ == impl_) {
      other.frozen_ = true;
      return;
    }
    if (!impl_) {
      other.frozen_ = true;
      impl_ = other.impl_;
      return;
    }
    for (ResourceOp resource_op : other) {
      Add(resource_op);
    }
  }
  void Add(const ResourceOp& resource_op) {
    CHECK(!frozen_);
    if (!IsCopy() && Contains(resource_op)) {
      return;
    }
    EnsureIsCopied();
    impl_->insert(resource_op);
  }
  Impl::const_iterator begin() const {
    return impl_ ? impl_->begin() : GetEmptyImpl()->begin();
  }
  Impl::const_iterator end() const {
    return impl_ ? impl_->end() : GetEmptyImpl()->end();
  }
  bool Contains(const ResourceOp& resource_op) const {
    return impl_ != nullptr && impl_->count(resource_op);
  }
 private:
  bool IsCopy() const { return storage_ != nullptr; }
  void EnsureIsCopied() {
    if (storage_ == nullptr) {
      storage_ = std::make_unique<Impl>();
      for (ResourceOp op : *this) {
        storage_->insert(op);
      }
      impl_ = storage_.get();
    }
  }
  static Impl* GetEmptyImpl() {
    static Impl* empty_impl = new Impl;
    return empty_impl;
  }
  Impl* impl_ = nullptr;
  std::unique_ptr<Impl> storage_;
  mutable bool frozen_ = false;
  ResourceOpSet(const ResourceOpSet&) = delete;
  void operator=(const ResourceOpSet&) = delete;
};
string ResourceOpSetToString(const ResourceOpSet& resource_op_set) {
  std::vector<string> elements_debug_string;
  std::transform(resource_op_set.begin(), resource_op_set.end(),
                 std::back_inserter(elements_debug_string), ResourceOpToString);
  return absl::StrCat("{", absl::StrJoin(elements_debug_string, ","), "}");
}
string NodeToString(const Node& n, XlaResourceOpKind resource_op_kind) {
  return absl::StrCat(
      "[", n.name(), ": ", n.type_string(), "(",
      XlaResourceOpInfo::XlaResourceOpKindToString(resource_op_kind), ")", "]");
}
}  
Status ComputeIncompatibleResourceOperationPairs(
    const Graph& g, const FunctionLibraryDefinition* flib_def,
    const std::function<Status(const Node&, bool*)>& resource_ops_to_ignore,
    std::vector<std::pair<int, int>>* result) {
  CHECK(result->empty());
  std::vector<Node*> rpo;
  GetReversePostOrder(g, &rpo, NodeComparatorName(),
                      [](const Edge& edge) {
                        return !edge.src()->IsNextIteration();
                      });
  auto resource_op_set_for_node =
      std::make_unique<ResourceOpSet[]>(g.num_node_ids());
  const bool vlog = VLOG_IS_ON(2);
  for (Node* n : rpo) {
    std::optional<XlaResourceOpKind> op_kind;
    TF_RETURN_IF_ERROR(XlaResourceOpKindForNode(
        *n, flib_def, resource_ops_to_ignore, &op_kind));
    ResourceOpSet* resource_op_set = &resource_op_set_for_node[n->id()];
    for (const Edge* e : n->in_edges()) {
      if (n->IsMerge() && e->src()->IsNextIteration()) {
        continue;
      }
      const ResourceOpSet& incoming_op_set =
          resource_op_set_for_node[e->src()->id()];
      resource_op_set->Add(incoming_op_set);
    }
    if (op_kind) {
      for (ResourceOp incoming_op : *resource_op_set) {
        if (IsEdgeSafe(incoming_op.second, *op_kind)) {
          continue;
        }
        if (vlog) {
          VLOG(2) << "Unsafe edge: "
                  << NodeToString(*g.FindNodeId(incoming_op.first),
                                  incoming_op.second)
                  << " -> " << NodeToString(*n, *op_kind);
        }
        result->push_back({incoming_op.first, n->id()});
      }
      if (op_kind != XlaResourceOpKind::kRead) {
        resource_op_set->Add({n->id(), *op_kind});
      }
    }
    if (vlog) {
      VLOG(3) << n->name() << " -> " << ResourceOpSetToString(*resource_op_set);
    }
  }
  std::sort(result->begin(), result->end());
  CHECK(std::unique(result->begin(), result->end()) == result->end());
  return absl::OkStatus();
}
}  