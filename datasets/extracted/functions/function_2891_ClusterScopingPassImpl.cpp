#include "tensorflow/compiler/jit/cluster_scoping_pass.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"
namespace tensorflow {
namespace {
class ClusterScopingPassImpl {
 public:
  ClusterScopingPassImpl(Graph* graph,
                         OptimizerOptions::GlobalJitLevel global_jit_level)
      : graph_(graph),
        global_jit_level_(global_jit_level),
        unique_scope_id_(0) {}
  Status Run();
 private:
  Status ScopingForPipelineStages();
  size_t GetUniqueScopeId() { return unique_scope_id_++; }
  void AddScopeToAllTransitivePredecessors(Node* start);
  void AddScopeToAllTransitiveSuccessors(Node* start);
 private:
  Graph* graph_;
  OptimizerOptions::GlobalJitLevel global_jit_level_;
  size_t unique_scope_id_;
};
std::optional<string> GetXlaInternalScope(Node* node) {
  string scope;
  if (GetNodeAttr(node->attrs(), kXlaInternalScopeAttr, &scope).ok()) {
    return scope;
  }
  return std::nullopt;
}
void SetXlaInternalScope(Node* node, StringPiece scope) {
  node->AddAttr(kXlaInternalScopeAttr, scope);
}
void AddOrAppendXlaInternalScope(Node* node, absl::string_view suffix) {
  string updated_scope;
  std::optional<string> cur_scope = GetXlaInternalScope(node);
  if (cur_scope == std::nullopt) {
    updated_scope = std::string(suffix);
  } else {
    updated_scope = absl::StrCat(cur_scope.value(), "&", suffix);
  }
  SetXlaInternalScope(node, updated_scope);
}
void ClusterScopingPassImpl::AddScopeToAllTransitivePredecessors(Node* start) {
  const string unique_suffix = absl::StrCat("_", GetUniqueScopeId());
  std::vector<Node*> starts;
  starts.push_back(start);
  auto enter = [&](Node* n) { AddOrAppendXlaInternalScope(n, unique_suffix); };
  ReverseDFSFrom(*graph_, starts, enter, nullptr,
                 NodeComparatorName());
}
void ClusterScopingPassImpl::AddScopeToAllTransitiveSuccessors(Node* start) {
  const string unique_suffix = absl::StrCat("_", GetUniqueScopeId());
  std::vector<Node*> starts;
  starts.push_back(start);
  auto enter = [&](Node* n) { AddOrAppendXlaInternalScope(n, unique_suffix); };
  DFSFrom(*graph_, starts, enter, nullptr,
          NodeComparatorName(),
          nullptr);
}
Status ClusterScopingPassImpl::ScopingForPipelineStages() {
  for (Node* n : graph_->nodes()) {
    DCHECK(n);
    if (n->type_string() == "Unstage") {
      AddScopeToAllTransitiveSuccessors(n);
    }
    if (n->type_string() == "Stage") {
      AddScopeToAllTransitivePredecessors(n);
    }
  }
  return absl::OkStatus();
}
Status ClusterScopingPassImpl::Run() {
  if (global_jit_level_ == OptimizerOptions::OFF) {
    return absl::OkStatus();
  }
  return ScopingForPipelineStages();
}
}  
Status ClusterScopingPass::Run(const GraphOptimizationPassOptions& options) {
  Graph* graph = options.graph->get();
  return ClusterScopingPassImpl{graph, GetGlobalJitLevelForGraph(options)}
      .Run();
}
}  