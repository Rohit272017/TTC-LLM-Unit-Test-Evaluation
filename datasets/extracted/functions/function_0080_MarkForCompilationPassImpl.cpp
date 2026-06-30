#include "tensorflow/compiler/jit/mark_for_compilation_pass.h"
#include <algorithm>
#include <atomic>
#include <deque>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "absl/base/call_once.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/jit/compilability_check_util.h"
#include "tensorflow/compiler/jit/deadness_analysis.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/device_util.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/resource_operation_safety_analysis.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/tf2xla/resource_operation_table.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/service/graphcycles/graphcycles.h"
#include "xla/union_find.h"
#include "xla/util.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/dump_graph.h"
namespace tensorflow {
namespace {
using DeadnessPredicate = DeadnessAnalysis::DeadnessPredicate;
using jit::DeviceId;
using jit::DeviceSet;
const char* kXlaAlreadyClustered = "_XlaAlreadyClustered";
class MarkForCompilationPassImpl {
 public:
  struct DebugOptions {
    bool ignore_deadness_checks;
    bool ignore_resource_variable_checks;
    bool ignore_xla_compile_attr;
    bool deterministic_cluster_names;
    int max_cluster_size;
    int min_cluster_size;
    std::atomic<int64_t>* fuel;
    bool dump_graphs;
  };
  MarkForCompilationPassImpl(DebugOptions debug_options, Graph* graph,
                             FunctionLibraryDefinition* flib_def, Env* env,
                             OptimizerOptions::GlobalJitLevel global_jit_level,
                             bool cpu_global_jit,
                             std::string cluster_name_prefix)
      : debug_options_(debug_options),
        graph_(graph),
        graph_fingerprint_(0),
        flib_def_(flib_def),
        env_(env),
        global_jit_level_(global_jit_level),
        cpu_global_jit_(cpu_global_jit),
        cluster_name_prefix_(cluster_name_prefix) {}
  Status Run();
 private:
  class Cluster {
   public:
    Cluster(int tf_graph_node_id, int effective_cluster_size,
            bool has_functional_control_flow, DeviceSet devices,
            std::optional<DeviceId> resource_op_device,
            std::optional<int> resource_var_operation_node_id,
            std::optional<DeadnessPredicate> deadness_predicate,
            bool is_xla_compile_attr_true, std::optional<string> xla_scope)
        : cycles_graph_node_id_(tf_graph_node_id),
          effective_cluster_size_(effective_cluster_size),
          has_functional_control_flow_(has_functional_control_flow),
          devices_(std::move(devices)),
          resource_op_device_(resource_op_device),
          deadness_predicate_(deadness_predicate),
          is_xla_compile_attr_true_(is_xla_compile_attr_true),
          xla_scope_(std::move(xla_scope)) {
      if (resource_var_operation_node_id.has_value()) {
        resource_var_operation_node_ids_.push_back(
            *resource_var_operation_node_id);
      }
    }
    void Merge(Cluster* other);
    int GetIdOfOnlyNode() const {
      DCHECK_EQ(cluster_size(), 1);
      return cycles_graph_node_id();
    }
    int cluster_size() const { return cluster_size_; }
    int cycles_graph_node_id() const { return cycles_graph_node_id_; }
    void set_cycles_graph_node_id(int cycles_graph_node_id) {
      cycles_graph_node_id_ = cycles_graph_node_id;
    }
    int effective_cluster_size() const { return effective_cluster_size_; }
    bool has_functional_control_flow() const {
      return has_functional_control_flow_;
    }
    const DeviceSet& devices() const { return devices_; }
    const std::optional<DeviceId>& resource_op_device() const {
      return resource_op_device_;
    }
    const std::optional<DeadnessPredicate>& deadness_predicate() const {
      return deadness_predicate_;
    }
    bool is_xla_compile_attr_true() const { return is_xla_compile_attr_true_; }
    const std::optional<string>& xla_scope() const { return xla_scope_; }
    absl::Span<const int> resource_var_operation_node_ids() const {
      return resource_var_operation_node_ids_;
    }
    string DebugString(const Graph& graph) const {
      Node* node = graph.FindNodeId(cycles_graph_node_id());
      if (!node) {
        return absl::StrCat("NULL NODE IN #", cycles_graph_node_id());
      }
      if (cluster_size() == 1) {
        return absl::StrCat("<", node->name(), " #", cycles_graph_node_id(),
                            ">");
      }
      return absl::StrCat("<", node->name(), " + ", cluster_size() - 1,
                          " others #", cycles_graph_node_id(), ">");
    }
   private:
    int cluster_size_ = 1;
    int cycles_graph_node_id_;
    int effective_cluster_size_;
    bool has_functional_control_flow_;
    DeviceSet devices_;
    std::optional<DeviceId> resource_op_device_;
    std::optional<DeadnessPredicate> deadness_predicate_;
    bool is_xla_compile_attr_true_;
    std::optional<string> xla_scope_;
    std::vector<int> resource_var_operation_node_ids_;
    Cluster(const Cluster&) = delete;
    void operator=(const Cluster&) = delete;
  };
  Node* GetOnlyNodeIn(const Cluster& cluster);
  bool IsSinkLike(const Cluster& cluster);
  bool IsScalarIntegerResourceOperation(const Cluster& cluster);
  absl::StatusOr<bool> Initialize();
  template <typename FnTy>
  absl::StatusOr<bool> ForEachEdgeInPostOrder(FnTy fn);
  Status RunEdgeContractionLoop();
  Status DeclusterNodes();
  Status CreateClusters();
  Status DumpDebugInfo();
  bool IsCompilationCandidate(Node* n) const {
    return compilation_candidates_.find(n) != compilation_candidates_.end();
  }
  absl::StatusOr<bool> TryToContractEdge(Cluster* from, Cluster* to);
  Status FindCompilationCandidates();
  bool CompilationDisallowedByXlaCompileAttr(Node* node);
  Status BuildInitialClusterSet();
  absl::StatusOr<bool> ShouldCompileClusterImpl(const Cluster& cluster);
  absl::StatusOr<bool> ShouldCompileCluster(const Cluster& cluster);
  absl::StatusOr<bool> ClusteringWillIntroduceInterDeviceDependency(
      const Cluster& from, const Cluster& to);
  bool ShouldCompile(bool is_xla_compile_attr_true,
                     const DeviceType& device_type,
                     XlaOpRegistry::AutoclusteringPolicy policy) {
    return is_xla_compile_attr_true ||
           policy == XlaOpRegistry::AutoclusteringPolicy::kAlways ||
           (policy == XlaOpRegistry::AutoclusteringPolicy::kIfEnabledGlobally &&
            global_jit_level_ != OptimizerOptions::OFF) ||
           (device_type.type_string() == DEVICE_CPU &&
            policy ==
                XlaOpRegistry::AutoclusteringPolicy::kIfExplicitlyRequested &&
            cpu_global_jit_);
  }
  absl::StatusOr<bool> AreDevicesCompatible(const Cluster& cluster_a,
                                            const Cluster& cluster_b);
  void DumpPostClusteringGraphs();
  void VLogClusteringSummary();
  Cluster* MakeNewCluster(int cycles_graph_node_id, int effective_cluster_size,
                          bool has_functional_control_flow,
                          const DeviceSet& device_set,
                          std::optional<DeviceId> resource_op_device,
                          std::optional<int> resource_var_operation_node_id,
                          std::optional<DeadnessPredicate> deadness_predicate,
                          bool is_xla_compile_attr_true,
                          std::optional<string> xla_scope) {
    cluster_storage_.push_back(std::make_unique<Cluster>(
        cycles_graph_node_id, effective_cluster_size,
        has_functional_control_flow, device_set, resource_op_device,
        resource_var_operation_node_id, deadness_predicate,
        is_xla_compile_attr_true, xla_scope));
    return cluster_storage_.back().get();
  }
  std::optional<string> GetXlaScope(Node* n);
  Cluster* GetClusterForNode(Node* n) {
    return cluster_for_node_[n->id()].Get();
  }
  Cluster* GetClusterForCyclesGraphNode(int node_id) {
    if (node_id >= graph_->num_node_ids() ||
        graph_->FindNodeId(node_id) == nullptr) {
      return nullptr;
    }
    Cluster* cluster = cluster_for_node_[node_id].Get();
    if (cluster) {
      DCHECK_EQ(cluster->cycles_graph_node_id(), node_id);
    }
    return cluster;
  }
  bool LogNotContractableAndReturnFalse(Cluster* from, Cluster* to,
                                        absl::string_view reason);
  std::vector<int> FindAlternatePathForDebugging(int from, int to);
  string DebugStringForCyclesGraphNode(int node_id, bool* found_unclustered);
  string DescribePotentialCycle(int from, int to);
  bool MergeClusters(Cluster* cluster_from, Cluster* cluster_to) {
    int from = cluster_from->cycles_graph_node_id();
    int to = cluster_to->cycles_graph_node_id();
    auto optional_merged_node = cycles_graph_.ContractEdge(from, to);
    if (!optional_merged_node.has_value()) {
      VLOG(3) << "Could not contract " << cluster_from->DebugString(*graph_)
              << " -> " << cluster_to->DebugString(*graph_)
              << " because contracting the edge would create a cycle via "
              << DescribePotentialCycle(from, to) << ".";
      return false;
    }
    cluster_from->Merge(cluster_to);
    cluster_from->set_cycles_graph_node_id(optional_merged_node.value());
    cluster_for_node_[from].Merge(&cluster_for_node_[to]);
    return true;
  }
  string EdgeContractionFailureMsg(Cluster* from, Cluster* to,
                                   absl::string_view reason) {
    return absl::StrCat("Could not contract ", from->DebugString(*graph_),
                        " -> ", to->DebugString(*graph_), " because ", reason,
                        ".");
  }
  DebugOptions debug_options_;
  Graph* graph_;
  uint64 graph_fingerprint_;
  FunctionLibraryDefinition* flib_def_;
  Env* env_;
  OptimizerOptions::GlobalJitLevel global_jit_level_;
  bool cpu_global_jit_;
  const std::string cluster_name_prefix_;
  absl::flat_hash_map<const Cluster*, bool> should_compile_cluster_cache_;
  jit::DeviceInfoCache device_info_cache_;
  bool initialized_ = false;
  bool edges_contracted_ = false;
  bool clusters_created_ = false;
  std::vector<std::unique_ptr<Cluster>> cluster_storage_;
  std::vector<xla::UnionFind<Cluster*>> cluster_for_node_;
  absl::flat_hash_set<const Node*> declustered_nodes_;
  xla::GraphCycles cycles_graph_;
  OrderedNodeSet compilation_candidates_;
  std::unique_ptr<DeadnessAnalysis> deadness_analysis_;
  int64_t iteration_count_ = 0;
  absl::flat_hash_set<std::pair<int, int>> unsafe_resource_deps_;
};
std::vector<int> MarkForCompilationPassImpl::FindAlternatePathForDebugging(
    int from, int to) {
  std::vector<int> rpo = cycles_graph_.AllNodesInPostOrder();
  absl::c_reverse(rpo);
  absl::flat_hash_map<int, int> best_pred_for_node;
  best_pred_for_node[from] = -1;
  int rpo_index = 0, current_rpo_node;
  do {
    current_rpo_node = rpo[rpo_index++];
    std::optional<int> some_pred, preferred_pred;
    for (int pred : cycles_graph_.Predecessors(current_rpo_node)) {
      if (!best_pred_for_node.contains(pred)) {
        continue;
      }
      if (current_rpo_node == to && pred == from) {
        continue;
      }
      some_pred = pred;
      if (GetClusterForCyclesGraphNode(pred) == nullptr) {
        preferred_pred = pred;
      }
    }
    if (some_pred || preferred_pred) {
      best_pred_for_node[current_rpo_node] =
          preferred_pred.has_value() ? *preferred_pred : *some_pred;
    }
  } while (current_rpo_node != to);
  auto get_best_pred = [&](int n) {
    auto it = best_pred_for_node.find(n);
    CHECK(it != best_pred_for_node.end());
    return it->second;
  };
  std::vector<int> path;
  int current_path_node = get_best_pred(to);
  while (current_path_node != from) {
    path.push_back(current_path_node);
    current_path_node = get_best_pred(current_path_node);
  }
  absl::c_reverse(path);
  return path;
}
string MarkForCompilationPassImpl::DebugStringForCyclesGraphNode(
    int cycles_graph_node_id, bool* found_unclustered) {
  Cluster* cluster = GetClusterForCyclesGraphNode(cycles_graph_node_id);
  if (cluster) {
    return cluster->DebugString(*graph_);
  }
  *found_unclustered = true;
  if (cycles_graph_node_id >= graph_->num_node_ids()) {
    return absl::StrCat("<oob #", cycles_graph_node_id, ">");
  }
  Node* node = graph_->FindNodeId(cycles_graph_node_id);
  if (!node) {
    return absl::StrCat("<bad #", cycles_graph_node_id, ">");
  }
  return node->name();
}
string MarkForCompilationPassImpl::DescribePotentialCycle(int from, int to) {
  std::vector<string> path_str;
  bool found_unclustered = false;
  absl::c_transform(FindAlternatePathForDebugging(from, to),
                    std::back_inserter(path_str), [&](int node_id) {
                      return DebugStringForCyclesGraphNode(node_id,
                                                           &found_unclustered);
                    });
  return absl::StrCat(!found_unclustered ? "(all clusters) " : "", "[",
                      absl::StrJoin(path_str, ","), "]");
}
void MarkForCompilationPassImpl::Cluster::Merge(Cluster* other) {
  cluster_size_ += other->cluster_size_;
  effective_cluster_size_ += other->effective_cluster_size_;
  has_functional_control_flow_ |= other->has_functional_control_flow_;
  devices_.UnionWith(other->devices_);
  DCHECK(!(resource_op_device_.has_value() &&
           other->resource_op_device_.has_value()) ||
         *resource_op_device_ == *other->resource_op_device_)
      << "AreDevicesCompatible should have returned false otherwise!";
  if (!resource_op_device_.has_value()) {
    resource_op_device_ = other->resource_op_device_;
  }
  is_xla_compile_attr_true_ |= other->is_xla_compile_attr_true_;
  if (!xla_scope_.has_value()) {
    xla_scope_ = std::move(other->xla_scope_);
  }
  resource_var_operation_node_ids_.reserve(
      resource_var_operation_node_ids_.size() +
      other->resource_var_operation_node_ids_.size());
  absl::c_copy(other->resource_var_operation_node_ids_,
               std::back_inserter(resource_var_operation_node_ids_));
  other->resource_var_operation_node_ids_.clear();
}
Status IgnoreResourceOpForSafetyAnalysis(
    jit::DeviceInfoCache* device_info_cache, const Node& n, bool* ignore) {
  if (n.assigned_device_name().empty()) {
    *ignore = false;
    return absl::OkStatus();
  }
  TF_ASSIGN_OR_RETURN(
      const XlaOpRegistry::DeviceRegistration* registration,
      device_info_cache->GetCompilationDevice(n.assigned_device_name()));
  if (!registration) {
    *ignore = true;
  } else {
    *ignore = registration->cluster_resource_variable_ops_unsafely;
  }
  return absl::OkStatus();
}
absl::StatusOr<bool> MarkForCompilationPassImpl::Initialize() {
  TF_RET_CHECK(!initialized_ && !edges_contracted_ && !clusters_created_);
  initialized_ = true;
  TF_RETURN_IF_ERROR(FindCompilationCandidates());
  if (compilation_candidates_.empty()) {
    VLOG(2) << "No compilable candidates";
    return false;
  }
  TF_ASSIGN_OR_RETURN(bool cycle_detection_graph_ok,
                      CreateCycleDetectionGraph(graph_, &cycles_graph_));
  if (!cycle_detection_graph_ok) {
    VLOG(2) << "Could not form cycle detection graph";
    return false;
  }
  if (!debug_options_.ignore_deadness_checks) {
    XLA_SCOPED_LOGGING_TIMER_LEVEL("DeadnessAnalysis", 1);
    TF_RETURN_IF_ERROR(DeadnessAnalysis::Run(*graph_, &deadness_analysis_));
  }
  if (debug_options_.deterministic_cluster_names) {
    TF_ASSIGN_OR_RETURN(graph_fingerprint_, FingerprintGraph(*graph_));
  }
  TF_RETURN_IF_ERROR(BuildInitialClusterSet());
  return true;
}
template <typename FnTy>
absl::StatusOr<bool> MarkForCompilationPassImpl::ForEachEdgeInPostOrder(
    FnTy fn) {
  bool changed = false;
  for (int32_t node : cycles_graph_.AllNodesInPostOrder()) {
    Cluster* cluster_from = GetClusterForCyclesGraphNode(node);
    if (!cluster_from) {
      continue;
    }
    std::vector<int32> successors_copy =
        cycles_graph_.SuccessorsCopy(cluster_from->cycles_graph_node_id());
    for (int to : successors_copy) {
      iteration_count_++;
      Cluster* cluster_to = GetClusterForCyclesGraphNode(to);
      if (!cluster_to) {
        continue;
      }
      TF_ASSIGN_OR_RETURN(bool contracted_edge, fn(cluster_from, cluster_to));
      changed |= contracted_edge;
    }
  }
  return changed;
}
Node* MarkForCompilationPassImpl::GetOnlyNodeIn(const Cluster& cluster) {
  return cluster.cluster_size() == 1
             ? graph_->FindNodeId(cluster.GetIdOfOnlyNode())
             : nullptr;
}
bool MarkForCompilationPassImpl::IsSinkLike(const Cluster& cluster) {
  if (Node* n = GetOnlyNodeIn(cluster)) {
    return n->type_string() == "NoOp" && n->out_edges().size() == 1 &&
           (*n->out_edges().begin())->dst()->IsSink();
  }
  return false;
}
bool MarkForCompilationPassImpl::IsScalarIntegerResourceOperation(
    const Cluster& cluster) {
  Node* n = GetOnlyNodeIn(cluster);
  if (!n) {
    return false;
  }
  if (n->type_string() != "AssignAddVariableOp" &&
      n->type_string() != "AssignSubVariableOp") {
    return false;
  }
  DataType dtype;
  if (!TryGetNodeAttr(n->def(), "dtype", &dtype) || !DataTypeIsInteger(dtype)) {
    return false;
  }
  Node* const_input = nullptr;
  for (const Edge* e : n->in_edges()) {
    if (!e->IsControlEdge() && e->src()->IsConstant()) {
      const_input = e->src();
      break;
    }
  }
  if (!const_input) {
    return false;
  }
  const TensorProto* proto = nullptr;
  if (!TryGetNodeAttr(const_input->def(), "value", &proto)) {
    return false;
  }
  return TensorShapeUtils::IsScalar(proto->tensor_shape());
}
Status MarkForCompilationPassImpl::RunEdgeContractionLoop() {
  TF_RET_CHECK(initialized_ && !edges_contracted_ && !clusters_created_);
  edges_contracted_ = true;
  VLOG(4) << "Running phase 0";
  TF_RETURN_IF_ERROR(
      ForEachEdgeInPostOrder([&](Cluster* from,
                                 Cluster* to) -> absl::StatusOr<bool> {
        Node* n = GetOnlyNodeIn(*to);
        bool is_shape_consumer_op = n && IsShapeConsumerOp(*n);
        if (!is_shape_consumer_op) {
          return false;
        }
        return TryToContractEdge(from, to);
      }).status());
  VLOG(4) << "Running phase 1";
  TF_RETURN_IF_ERROR(
      ForEachEdgeInPostOrder([&](Cluster* from,
                                 Cluster* to) -> absl::StatusOr<bool> {
        if (IsSinkLike(*to)) {
          return false;
        }
        if (IsScalarIntegerResourceOperation(*from)) {
          return false;
        }
        return TryToContractEdge(from, to);
      }).status());
  VLOG(4) << "Running phase 2";
  TF_RETURN_IF_ERROR(ForEachEdgeInPostOrder([&](Cluster* from, Cluster* to) {
                       return TryToContractEdge(from, to);
                     }).status());
  VLOG(2) << "Checking idempotence";
  TF_ASSIGN_OR_RETURN(bool changed,
                      ForEachEdgeInPostOrder([&](Cluster* from, Cluster* to) {
                        return TryToContractEdge(from, to);
                      }));
  TF_RET_CHECK(!changed);
  return absl::OkStatus();
}
Status MarkForCompilationPassImpl::DeclusterNodes() {
  for (Node* n : compilation_candidates_) {
    Cluster* cluster = GetClusterForNode(n);
    if (cluster == nullptr) {
      continue;
    }
    if (n->op_def().name() == "Fill" &&
        n->out_nodes().begin() != n->out_nodes().end() &&
        absl::c_all_of(n->out_nodes(), [&](Node* user) {
          return GetClusterForNode(user) != cluster;
        })) {
      declustered_nodes_.insert(n);
    }
  }
  return absl::OkStatus();
}
class ClusterSequenceNumberGenerator {
 public:
  void Reset() {
    mutex_lock lock(mu_);
    sequence_numbers_.clear();
  }
  int64 GetNext(uint64 key) {
    mutex_lock lock(mu_);
    return sequence_numbers_[key]++;
  }
  static ClusterSequenceNumberGenerator& Global() {
    static ClusterSequenceNumberGenerator* gen =
        new ClusterSequenceNumberGenerator;
    return *gen;
  }
 private:
  mutex mu_;
  absl::flat_hash_map<uint64, int64> sequence_numbers_;
};
int64_t GetNextClusterSequenceNumber(uint64 fingerprint) {
  return ClusterSequenceNumberGenerator::Global().GetNext(fingerprint);
}
Status MarkForCompilationPassImpl::CreateClusters() {
  TF_RET_CHECK(initialized_ && edges_contracted_ && !clusters_created_);
  clusters_created_ = true;
  std::unordered_map<int, string> cluster_names;
  if (debug_options_.dump_graphs) {
    DumpGraphToFile("before_mark_for_compilation", *graph_, flib_def_);
  }
  for (Node* n : compilation_candidates_) {
    Cluster* cluster = GetClusterForNode(n);
    TF_ASSIGN_OR_RETURN(bool should_compile_cluster,
                        ShouldCompileCluster(*cluster));
    if (!should_compile_cluster || declustered_nodes_.contains(n)) {
      continue;
    }
    if (cluster->effective_cluster_size() >= debug_options_.min_cluster_size ||
        cluster->has_functional_control_flow() ||
        cluster->is_xla_compile_attr_true()) {
      string& name = cluster_names[cluster->cycles_graph_node_id()];
      if (name.empty()) {
        if (!cluster_name_prefix_.empty()) {
          name = absl::StrCat(cluster_name_prefix_, "_");
        } else {
          name = "cluster_";
        }
        if (debug_options_.deterministic_cluster_names) {
          absl::StrAppend(&name, graph_fingerprint_, "_");
        }
        absl::StrAppend(&name,
                        GetNextClusterSequenceNumber(graph_fingerprint_));
      }
      n->AddAttr(kXlaClusterAttr, name);
      n->AddAttr(kXlaAlreadyClustered, true);
      VLOG(3) << "Assigning node " << n->name() << " to cluster " << name;
    }
  }
  return absl::OkStatus();
}
Status MarkForCompilationPassImpl::DumpDebugInfo() {
  TF_RET_CHECK(initialized_ && edges_contracted_ && clusters_created_);
  if (debug_options_.dump_graphs) {
    DumpPostClusteringGraphs();
  }
  VLogClusteringSummary();
  return absl::OkStatus();
}
absl::StatusOr<bool>
MarkForCompilationPassImpl::ClusteringWillIntroduceInterDeviceDependency(
    const Cluster& cluster_from, const Cluster& cluster_to) {
  for (const auto& in_id :
       cycles_graph_.Predecessors(cluster_to.cycles_graph_node_id())) {
    const Cluster* cluster_in = GetClusterForCyclesGraphNode(in_id);
    if (cluster_in) {
      TF_ASSIGN_OR_RETURN(bool devices_compatible,
                          AreDevicesCompatible(cluster_to, *cluster_in));
      if (!devices_compatible) {
        return true;
      }
      TF_ASSIGN_OR_RETURN(devices_compatible,
                          AreDevicesCompatible(cluster_from, *cluster_in));
      if (!devices_compatible) {
        return true;
      }
    }
  }
  return false;
}
std::optional<string> MarkForCompilationPassImpl::GetXlaScope(Node* node) {
  if (global_jit_level_ != OptimizerOptions::OFF) {
    const string& scope =
        GetNodeAttrString(node->attrs(), kXlaInternalScopeAttr);
    if (!scope.empty()) {
      return scope;
    }
  } else {
    const string& scope = GetNodeAttrString(node->attrs(), kXlaScopeAttr);
    if (!scope.empty()) {
      return scope;
    }
  }
  return std::nullopt;
}
static bool GetNodeOrFuncAttr(Node* node, FunctionLibraryDefinition* flib_def,
                              const char* attr_name) {
  bool out = false;
  bool attr_value;
  if (TryGetNodeAttr(node->attrs(), attr_name, &attr_value)) {
    out |= attr_value;
  }
  if (flib_def->GetAttr(*node, attr_name, &attr_value).ok()) {
    out |= attr_value;
  }
  return out;
}
Status MarkForCompilationPassImpl::BuildInitialClusterSet() {
  auto ignore_resource_ops = [&](const Node& n, bool* ignore) {
    return IgnoreResourceOpForSafetyAnalysis(&device_info_cache_, n, ignore);
  };
  std::vector<std::pair<int, int>> unsafe_resource_deps_vect;
  TF_RETURN_IF_ERROR(ComputeIncompatibleResourceOperationPairs(
      *graph_, flib_def_, ignore_resource_ops, &unsafe_resource_deps_vect));
  absl::c_copy(
      unsafe_resource_deps_vect,
      std::inserter(unsafe_resource_deps_, unsafe_resource_deps_.begin()));
  cluster_for_node_.resize(graph_->num_node_ids());
  for (Node* node : graph_->nodes()) {
    if (!IsCompilationCandidate(node)) {
      cluster_for_node_[node->id()].Get() = nullptr;
      continue;
    }
    int effective_cluster_size =
        (node->IsIdentity() || node->IsConstant()) ? 0 : 1;
    bool has_functional_control_flow = node->IsWhileNode() || node->IsIfNode();
    std::optional<DeadnessPredicate> deadness_predicate;
    if (deadness_analysis_) {
      TF_ASSIGN_OR_RETURN(
          deadness_predicate,
          deadness_analysis_->GetPredicateFor(node, Graph::kControlSlot));
    }
    const string& device_name_str = !node->assigned_device_name().empty()
                                        ? node->assigned_device_name()
                                        : node->requested_device();
    TF_ASSIGN_OR_RETURN(DeviceId device,
                        device_info_cache_.GetIdFor(device_name_str));
    bool is_resource_op = HasResourceInputOrOutput(*node);
    std::optional<DeviceId> resource_op_device;
    if (is_resource_op) {
      resource_op_device = device;
    }
    std::optional<int> resource_var_operation_node_id;
    if (is_resource_op || MayCallFunction(*node, flib_def_)) {
      resource_var_operation_node_id = node->id();
    }
    bool is_xla_compile_attr_true =
        GetNodeOrFuncAttr(node, flib_def_, kXlaCompileAttr) ||
        (global_jit_level_ != OptimizerOptions::OFF &&
         GetNodeOrFuncAttr(node, flib_def_, kXlaMustCompileAttr));
    DeviceSet devices;
    devices.Insert(device);
    Cluster* new_cluster = MakeNewCluster(
        node->id(),
        effective_cluster_size,
        has_functional_control_flow, devices,
        resource_op_device, resource_var_operation_node_id, deadness_predicate,
        is_xla_compile_attr_true,
        GetXlaScope(node));
    cluster_for_node_[node->id()].Get() = new_cluster;
  }
  return absl::OkStatus();
}
absl::StatusOr<bool> IsIdentityDrivingConstsInLoop(Node* node) {
  if (!node->IsIdentity()) {
    return false;
  }
  auto it = absl::c_find_if(node->in_edges(), [](const Edge* e) {
    return e->src()->IsSwitch() && e->src_output() == 1;
  });
  if (it == node->in_edges().end()) {
    return false;
  }
  const Node* switch_node = (*it)->src();
  const Node* maybe_loop_cond;
  TF_RETURN_IF_ERROR(switch_node->input_node(1, &maybe_loop_cond));
  if (!maybe_loop_cond->IsLoopCond()) {
    return false;
  }
  bool driving_any_consts =
      absl::c_any_of(node->out_edges(), [](const Edge* e) {
        return e->dst()->IsConstant() && e->IsControlEdge();
      });
  if (!driving_any_consts) {
    return false;
  }
  return true;
}
absl::flat_hash_set<string> CreateClusterExcludeList() {
  MarkForCompilationPassFlags* flags = GetMarkForCompilationPassFlags();
  absl::flat_hash_set<string> excludelist;
  for (auto s : absl::StrSplit(flags->tf_xla_cluster_exclude_ops, ',')) {
    if (!s.empty()) {
      excludelist.insert(string(s));
    }
  }
  if (VLOG_IS_ON(2) && !excludelist.empty()) {
    std::vector<string> vexcludelist(excludelist.begin(), excludelist.end());
    absl::c_sort(vexcludelist);
    VLOG(2) << "XLA clustering will exclude following TF operations from auto "
               "clustering: "
            << absl::StrJoin(vexcludelist, " ");
  }
  return excludelist;
}
absl::flat_hash_set<string> GetOrCreateAllowlist() {
  absl::flat_hash_map<string, std::vector<string>>* allowlist_table =
      tensorflow::GetAllowlistTable();
  MarkForCompilationPassFlags* flags = GetMarkForCompilationPassFlags();
  absl::flat_hash_set<string> allowlist;
  for (auto s : absl::StrSplit(flags->tf_xla_ops_to_cluster, ',')) {
    if (s == "FUSIBLE") {
      for (auto pair : *allowlist_table) {
        allowlist.insert(pair.second.begin(), pair.second.end());
      }
    } else if (allowlist_table->contains(s)) {
      auto v = allowlist_table->at(s);
      allowlist.insert(v.begin(), v.end());
    } else if (!s.empty()) {
      allowlist.insert(string(s));
    }
  }
  if (VLOG_IS_ON(2) && !allowlist.empty()) {
    std::vector<string> vallowlist(allowlist.begin(), allowlist.end());
    absl::c_sort(vallowlist);
    VLOG(2) << "XLA clustering will only consider the following TF operations: "
            << absl::StrJoin(vallowlist, " ");
  }
  return allowlist;
}
Status MarkForCompilationPassImpl::FindCompilationCandidates() {
  OptimizerOptions opts;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(
      new ProcessFunctionLibraryRuntime(nullptr, env_, nullptr,
                                        TF_GRAPH_DEF_VERSION, flib_def_, opts));
  FunctionLibraryRuntime* lib_runtime =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);
  std::vector<bool> compile_time_const_nodes(graph_->num_node_ids(), false);
  TF_RETURN_IF_ERROR(BackwardsConstAnalysis(
      *graph_, nullptr,
      &compile_time_const_nodes, lib_runtime));
  std::vector<Node*> sorted_nodes;
  for (Node* node : graph_->op_nodes()) {
    sorted_nodes.push_back(node);
  }
  std::sort(sorted_nodes.begin(), sorted_nodes.end(), NodeComparatorID());
  if (*debug_options_.fuel >= std::numeric_limits<int64_t>::max() / 2) {
    VLOG(2) << "Starting fuel: infinity";
  } else {
    VLOG(2) << "Starting fuel: " << *debug_options_.fuel;
  }
  VLOG(2) << "sorted_nodes.size() = " << sorted_nodes.size();
  auto allowlist = GetOrCreateAllowlist();
  std::vector<string> vall_ops = XlaOpRegistry::GetAllRegisteredOps();
  absl::flat_hash_set<string> all_ops(vall_ops.begin(), vall_ops.end());
  for (const auto& s : allowlist) {
    if (!all_ops.contains(s)) {
      return errors::InvalidArgument(
          "The operation '", s,
          "' passed to --tf_xla_ops_to_cluster is not supported by XLA.");
    }
  }
  auto cluster_exclude_op_list = CreateClusterExcludeList();
  bool allow_where_op = true;
  for (const auto& s : cluster_exclude_op_list) {
    if (s == "Where") {
      allow_where_op = false;
    } else {
      return errors::InvalidArgument(
          "The operation '", s,
          "' passed to --tf_xla_cluster_exclude_ops is not supported by "
          "XLA.");
    }
  }
  for (Node* node : sorted_nodes) {
    if (*debug_options_.fuel <= 0) {
      VLOG(1)
          << "Hit fuel limit; not marking any remaining ops as clusterable.";
      break;
    }
    TF_ASSIGN_OR_RETURN(
        const DeviceType& device_type,
        device_info_cache_.GetDeviceTypeFor(node->assigned_device_name()));
    VLOG(4) << "Device type for " << node->name() << ": "
            << device_type.type_string();
    if (CompilationDisallowedByXlaCompileAttr(node)) {
      VLOG(2) << "Not clustering " << node->name()
              << ": disallowed by _XlaCompile attribute";
      continue;
    }
    const XlaOpRegistry::DeviceRegistration* registration;
    if (!XlaOpRegistry::GetCompilationDevice(device_type.type(),
                                             &registration)) {
      VLOG(2) << "Rejecting " << node->name()
              << ": could not find JIT device for " << device_type.type();
      continue;
    }
    bool is_xla_compile_attr_true =
        GetNodeOrFuncAttr(node, flib_def_, kXlaCompileAttr) ||
        (global_jit_level_ != OptimizerOptions::OFF &&
         GetNodeOrFuncAttr(node, flib_def_, kXlaMustCompileAttr));
    auto policy = registration->autoclustering_policy;
    if (!ShouldCompile(is_xla_compile_attr_true, device_type, policy)) {
      continue;
    }
    RecursiveCompilabilityChecker::OperationFilter filter =
        CreateOperationFilter(*registration);
    filter.require_always_compilable = true;
    filter.allow_string_consts = false;
    filter.allow_collective_reduce_v2 = false;
    filter.allow_unique_op = false;
    filter.allow_where_op = allow_where_op;
    RecursiveCompilabilityChecker checker(
        filter, DeviceType{registration->compilation_device_name});
    if (!checker.IsCompilableNode(*node, lib_runtime)) {
      continue;
    }
    if (node->type_string() == "Const") {
      const AttrValue* attr = node->attrs().Find("dtype");
      if (attr != nullptr && attr->type() == DT_STRING) {
        continue;
      }
    }
    if (!allowlist.empty() && !allowlist.contains(node->def().op())) {
      VLOG(1) << "Rejecting TF operation " << node->def().op()
              << " as it is not listed in --tf_xla_ops_to_cluster.";
      continue;
    }
    if (compile_time_const_nodes[node->id()]) {
      const OpDef* op_def;
      TF_RETURN_IF_ERROR(
          graph_->op_registry()->LookUpOpDef(node->type_string(), &op_def));
      if (op_def->is_stateful()) {
        const XlaResourceOpInfo* op_info =
            GetResourceOpInfoForOp(node->type_string());
        bool is_tensor_array_or_stack_op =
            op_info && op_info->resource_kind() != XlaResourceKind::kVariable;
        if (!is_tensor_array_or_stack_op) {
          VLOG(2) << "Isolating " << node->name()
                  << ": must-be-constant stateful op";
          continue;
        }
      }
    }
    TF_ASSIGN_OR_RETURN(bool is_identity_driving_consts_in_loop,
                        IsIdentityDrivingConstsInLoop(node));
    if (is_identity_driving_consts_in_loop) {
      VLOG(2) << "Rejecting " << node->name()
              << ": including it can create dependencies between while loop "
                 "condition and body computations with runtime overhead.";
      continue;
    }
    compilation_candidates_.insert(node);
    --(*debug_options_.fuel);
  }
  VLOG(2) << "compilation_candidates_.size() = "
          << compilation_candidates_.size();
  return absl::OkStatus();
}
bool MarkForCompilationPassImpl::CompilationDisallowedByXlaCompileAttr(
    Node* node) {
  if (debug_options_.ignore_xla_compile_attr) {
    return false;
  }
  bool compile = false;
  Status status = GetNodeAttr(node->attrs(), kXlaCompileAttr, &compile);
  if (status.ok()) {
    if (!compile) {
      VLOG(2) << "Rejecting " << node->name() << ": kXlaCompileAttr("
              << kXlaCompileAttr << ") is false.";
    }
    return !compile;
  }
  status = flib_def_->GetAttr(*node, kXlaCompileAttr, &compile);
  if (status.ok()) {
    if (!compile) {
      VLOG(2) << "Rejecting " << node->name() << ": kXlaCompileAttr("
              << kXlaCompileAttr << ") on callee is false.";
    }
    return !compile;
  }
  return false;
}
bool MarkForCompilationPassImpl::LogNotContractableAndReturnFalse(
    Cluster* from, Cluster* to, absl::string_view reason) {
  VLOG(3) << EdgeContractionFailureMsg(from, to, reason);
  return false;
}
absl::StatusOr<bool> MarkForCompilationPassImpl::TryToContractEdge(
    Cluster* from, Cluster* to) {
  DCHECK(from->deadness_predicate().has_value() ==
         to->deadness_predicate().has_value());
  if (from->deadness_predicate() != to->deadness_predicate()) {
    VLOG(3) << EdgeContractionFailureMsg(
        from, to,
        absl::StrCat(
            "the two nodes have mismatching deadness: ",
            deadness_analysis_->DebugString(*from->deadness_predicate()),
            " and ",
            deadness_analysis_->DebugString(*to->deadness_predicate())));
    return false;
  }
  TF_ASSIGN_OR_RETURN(bool devices_compatible,
                      AreDevicesCompatible(*from, *to));
  if (!devices_compatible) {
    return LogNotContractableAndReturnFalse(
        from, to, "the two nodes have incompatible devices");
  }
  if (from->xla_scope().has_value() && to->xla_scope().has_value() &&
      *from->xla_scope() != *to->xla_scope()) {
    return LogNotContractableAndReturnFalse(
        from, to, "the two nodes have mismatching XLA scopes");
  }
  if (from->cluster_size() + to->cluster_size() >
      debug_options_.max_cluster_size) {
    return LogNotContractableAndReturnFalse(
        from, to, "the new cluster will be larger than the max cluster size");
  }
  TF_ASSIGN_OR_RETURN(bool will_introduce_cross_device_dependency,
                      ClusteringWillIntroduceInterDeviceDependency(*from, *to));
  if (will_introduce_cross_device_dependency) {
    return LogNotContractableAndReturnFalse(
        from, to, "the new cluster will introduce a cross device dependency");
  }
  if (!debug_options_.ignore_resource_variable_checks) {
    for (int resource_var_from : from->resource_var_operation_node_ids()) {
      for (int resource_var_to : to->resource_var_operation_node_ids()) {
        if (unsafe_resource_deps_.contains(
                {resource_var_from, resource_var_to})) {
          return LogNotContractableAndReturnFalse(
              from, to,
              "the new cluster would break resource variable semantics");
        }
      }
    }
  }
  return MergeClusters(from, to);
}
Status MarkForCompilationPassImpl::Run() {
  XlaOpRegistry::RegisterCompilationKernels();
  XLA_SCOPED_LOGGING_TIMER_LEVEL("MarkForCompilationPassImpl::Run", 1);
  TF_ASSIGN_OR_RETURN(bool initialized, Initialize());
  if (!initialized) {
    return absl::OkStatus();
  }
  TF_RETURN_IF_ERROR(RunEdgeContractionLoop());
  TF_RETURN_IF_ERROR(DeclusterNodes());
  TF_RETURN_IF_ERROR(CreateClusters());
  TF_RETURN_IF_ERROR(DumpDebugInfo());
  return absl::OkStatus();
}
void MarkForCompilationPassImpl::DumpPostClusteringGraphs() {
  DumpGraphToFile("mark_for_compilation", *graph_, flib_def_);
  Graph new_graph(graph_->op_registry());
  CopyGraph(*graph_, &new_graph);
  for (Node* n : new_graph.nodes()) {
    if (std::optional<absl::string_view> cluster_name =
            GetXlaClusterForNode(*n)) {
      n->set_name(absl::StrCat(*cluster_name, "/", n->name()));
    } else if (n->type_string() == "VarHandleOp") {
      n->set_name(absl::StrCat("varhandle/", n->name()));
    } else {
      n->set_name(absl::StrCat("unclustered/", n->name()));
    }
  }
  DumpGraphToFile("mark_for_compilation_annotated", new_graph, flib_def_);
}
string RatioToString(int numerator, int denominator) {
  return absl::StrFormat("%d / %d (%.2f%%)", numerator, denominator,
                         (100.0 * numerator) / denominator);
}
void MarkForCompilationPassImpl::VLogClusteringSummary() {
  if (!VLOG_IS_ON(2)) {
    return;
  }
  XlaAutoClusteringSummary auto_clustering_info =
      GetXlaAutoClusteringSummary(*graph_);
  VLOG(2) << "*** Clustering info for graph of size " << graph_->num_nodes();
  VLOG(2) << " Built " << auto_clustering_info.clusters_size()
          << " clusters, size "
          << RatioToString(auto_clustering_info.clustered_node_count(),
                           graph_->num_nodes());
  for (const XlaAutoClusteringSummary::Cluster& cluster :
       auto_clustering_info.clusters()) {
    absl::string_view cluster_name = cluster.name();
    int size = cluster.size();
    VLOG(2) << "  " << cluster_name << " "
            << RatioToString(size, graph_->num_nodes());
    for (const XlaAutoClusteringSummary::OpAndCount& op_count :
         cluster.op_histogram()) {
      VLOG(3) << "   " << op_count.op() << ": " << op_count.count()
              << " instances";
    }
  }
  if (!auto_clustering_info.unclustered_op_histogram().empty()) {
    VLOG(2) << " Unclustered nodes: "
            << RatioToString(auto_clustering_info.unclustered_node_count(),
                             graph_->num_nodes());
    for (const XlaAutoClusteringSummary::OpAndCount& op_count :
         auto_clustering_info.unclustered_op_histogram()) {
      VLOG(3) << "  " << op_count.op() << ": " << op_count.count()
              << " instances";
    }
  }
  struct EdgeInfo {
    absl::string_view node_name;
    std::optional<absl::string_view> cluster_name;
    absl::string_view GetClusterName() const {
      return cluster_name ? *cluster_name : "[none]";
    }
    std::pair<absl::string_view, std::optional<absl::string_view>> AsPair()
        const {
      return {node_name, cluster_name};
    }
    bool operator<(const EdgeInfo& other) const {
      return AsPair() < other.AsPair();
    }
  };
  using EdgeInfoMap = std::map<absl::string_view, std::map<EdgeInfo, int64_t>>;
  EdgeInfoMap incoming_edge_infos;
  EdgeInfoMap outgoing_edge_infos;
  std::set<absl::string_view> cluster_names_to_print;
  for (const Edge* e : graph_->edges()) {
    const Node* from = e->src();
    std::optional<absl::string_view> from_cluster_name =
        GetXlaClusterForNode(*from);
    const Node* to = e->dst();
    std::optional<absl::string_view> to_cluster_name =
        GetXlaClusterForNode(*to);
    if (to_cluster_name == from_cluster_name) {
      continue;
    }
    if (to_cluster_name) {
      incoming_edge_infos[*to_cluster_name]
                         [EdgeInfo{from->name(), from_cluster_name}]++;
      cluster_names_to_print.insert(*to_cluster_name);
    }
    if (from_cluster_name) {
      outgoing_edge_infos[*from_cluster_name][{to->name(), to_cluster_name}]++;
      cluster_names_to_print.insert(*from_cluster_name);
    }
  }
  VLOG(4) << "*** Inter-Cluster edges:";
  if (cluster_names_to_print.empty()) {
    VLOG(4) << "   [none]";
  }
  auto print_edge_info_set_for_cluster = [&](absl::string_view cluster_name,
                                             const EdgeInfoMap& edge_info_map,
                                             absl::string_view desc) {
    auto it = edge_info_map.find(cluster_name);
    if (it != edge_info_map.end()) {
      VLOG(4) << "  " << it->second.size() << " " << desc << " edges";
      for (const auto& edge_info_count_pair : it->second) {
        VLOG(4) << "   " << edge_info_count_pair.first.GetClusterName() << " "
                << edge_info_count_pair.first.node_name << " # "
                << edge_info_count_pair.second;
      }
    } else {
      VLOG(4) << "  No " << desc << " edges.";
    }
  };
  for (absl::string_view cluster_name : cluster_names_to_print) {
    VLOG(4) << " ** Cluster " << cluster_name;
    print_edge_info_set_for_cluster(cluster_name, incoming_edge_infos,
                                    "incoming");
    print_edge_info_set_for_cluster(cluster_name, outgoing_edge_infos,
                                    "outgoing");
  }
}
absl::StatusOr<bool> MarkForCompilationPassImpl::AreDevicesCompatible(
    const Cluster& cluster_a, const Cluster& cluster_b) {
  DeviceSet devices = cluster_a.devices();
  devices.UnionWith(cluster_b.devices());
  TF_ASSIGN_OR_RETURN(
      std::optional<jit::DeviceId> maybe_chosen_device,
      MaybePickDeviceForXla(device_info_cache_, devices,
                            false));
  if (!maybe_chosen_device.has_value()) {
    return false;
  }
  jit::DeviceId chosen_device = *maybe_chosen_device;
  auto resource_op_device_ok = [&](std::optional<DeviceId> resource_op_device) {
    return !resource_op_device.has_value() ||
           *resource_op_device == chosen_device;
  };
  return resource_op_device_ok(cluster_a.resource_op_device()) &&
         resource_op_device_ok(cluster_b.resource_op_device());
}
absl::StatusOr<bool> MarkForCompilationPassImpl::ShouldCompileClusterImpl(
    const Cluster& cluster) {
  TF_ASSIGN_OR_RETURN(DeviceId chosen_device,
                      PickDeviceForXla(device_info_cache_, cluster.devices(),
                                       false));
  const DeviceType& device_type =
      device_info_cache_.GetDeviceTypeFor(chosen_device);
  const XlaOpRegistry::DeviceRegistration* registration =
      device_info_cache_.GetCompilationDevice(chosen_device);
  TF_RET_CHECK(registration)
      << "chosen device = " << device_info_cache_.GetNameFor(chosen_device)
      << "; device type = " << device_type.type() << "; devices ("
      << device_info_cache_.DebugString(cluster.devices());
  auto policy = registration->autoclustering_policy;
  bool should_compile =
      ShouldCompile(cluster.is_xla_compile_attr_true(), device_type, policy);
  if (!should_compile && device_type.type_string() == DEVICE_CPU &&
      global_jit_level_ > OptimizerOptions::OFF) {
    static absl::once_flag once;
    absl::call_once(once, [] {
      LOG(WARNING) << R"((One-time warning): Not using XLA:CPU for cluster.
If you want XLA:CPU, do one of the following:
 - set the TF_XLA_FLAGS to include "--tf_xla_cpu_global_jit", or
 - set cpu_global_jit to true on this session's OptimizerOptions, or
 - use experimental_jit_scope, or
 - use tf.function(jit_compile=True).
To confirm that XLA is active, pass --vmodule=xla_compilation_cache=1 (as a
proper command-line flag, not via TF_XLA_FLAGS).)";
      MarkForCompilationPassFlags* flags = GetMarkForCompilationPassFlags();
      if (flags->tf_xla_cpu_global_jit) {
        LOG(WARNING)
            << "(Although the tf_xla_cpu_global_jit flag is currently enabled, "
               "perhaps it wasn't enabled at process startup?)";
      }
    });
  }
  VLOG(3) << (should_compile ? "Compiling" : "Not compiling")
          << " cluster with device "
          << device_info_cache_.GetNameFor(chosen_device);
  return should_compile;
}
absl::StatusOr<bool> MarkForCompilationPassImpl::ShouldCompileCluster(
    const Cluster& cluster) {
  auto it = should_compile_cluster_cache_.find(&cluster);
  if (it != should_compile_cluster_cache_.end()) {
    return it->second;
  }
  TF_ASSIGN_OR_RETURN(bool should_compile, ShouldCompileClusterImpl(cluster));
  should_compile_cluster_cache_.insert({&cluster, should_compile});
  return should_compile;
}
Status MarkForCompilation(
    const GraphOptimizationPassOptions& options,
    const MarkForCompilationPassImpl::DebugOptions& debug_options) {
  Graph* graph = options.graph->get();
  FunctionLibraryDefinition* flib_def = options.flib_def;
  FixupSourceAndSinkEdges(graph);
  for (Node* n : graph->nodes()) {
    if (n->attrs().Find(kXlaAlreadyClustered)) {
      return absl::OkStatus();
    }
    if (n->type_string() == "TPUExecute" ||
        n->type_string() == "TPUExecuteAndUpdateVariables") {
      return absl::OkStatus();
    }
  }
  return MarkForCompilationPassImpl{
      debug_options,
      graph,
      flib_def,
      options.session_options != nullptr ? options.session_options->env
                                         : Env::Default(),
      GetGlobalJitLevelForGraph(options),
      options.session_options->config.graph_options()
          .optimizer_options()
          .cpu_global_jit(),
      options.session_options != nullptr
          ? options.session_options->config.experimental()
                .session_metadata()
                .name()
          : ""}
      .Run();
}
std::atomic<int64_t>* GetPointerToFuel(int64_t initial_value) {
  static std::atomic<int64_t>* fuel = [&]() {
    std::atomic<int64_t>* fuel = new std::atomic<int64_t>;
    *fuel = initial_value;
    return fuel;
  }();
  return fuel;
}
}  
Status MarkForCompilationPass::Run(
    const GraphOptimizationPassOptions& options) {
  MarkForCompilationPassFlags* flags = GetMarkForCompilationPassFlags();
  MarkForCompilationPassImpl::DebugOptions debug_options;
  debug_options.ignore_deadness_checks =
      flags->tf_xla_disable_deadness_safety_checks_for_debugging;
  debug_options.ignore_resource_variable_checks =
      flags->tf_xla_disable_resource_variable_safety_checks_for_debugging;
  debug_options.ignore_xla_compile_attr = false;
  debug_options.deterministic_cluster_names =
      flags->tf_xla_deterministic_cluster_names;
  debug_options.max_cluster_size = flags->tf_xla_max_cluster_size;
  debug_options.min_cluster_size = flags->tf_xla_min_cluster_size;
  debug_options.fuel = GetPointerToFuel(flags->tf_xla_clustering_fuel);
  debug_options.dump_graphs = flags->tf_xla_clustering_debug;
  return MarkForCompilation(options, debug_options);
}
Status MarkForCompilationPass::RunForTest(
    const GraphOptimizationPassOptions& options, bool disable_deadness_analysis,
    bool deterministic_cluster_names) {
  MarkForCompilationPassFlags* flags = GetMarkForCompilationPassFlags();
  MarkForCompilationPassImpl::DebugOptions debug_options;
  debug_options.ignore_deadness_checks = disable_deadness_analysis;
  debug_options.ignore_resource_variable_checks =
      flags->tf_xla_disable_resource_variable_safety_checks_for_debugging;
  debug_options.ignore_xla_compile_attr = true;
  debug_options.deterministic_cluster_names = deterministic_cluster_names;
  debug_options.max_cluster_size = flags->tf_xla_max_cluster_size;
  debug_options.min_cluster_size = flags->tf_xla_min_cluster_size;
  debug_options.fuel = GetPointerToFuel(flags->tf_xla_clustering_fuel);
  debug_options.dump_graphs = flags->tf_xla_clustering_debug;
  return MarkForCompilation(options, debug_options);
}
absl::flat_hash_map<string, std::vector<string>>* GetAllowlistTable() {
  static absl::flat_hash_map<string, std::vector<string>>* result =
      new absl::flat_hash_map<string, std::vector<string>>{
          {"PW",
           {"ComplexAbs", "Angle", "Conj", "Abs", "Acos", "Acosh", "Asin",
            "Atan", "Atanh", "Ceil", "Cos", "Cosh", "Sin", "Exp", "Expm1",
            "Floor", "IsFinite", "IsInf", "IsNan", "Inv", "Reciprocal", "Log",
            "Log1p", "Invert", "LogicalNot", "Ndtri", "Neg", "Rint", "Round",
            "Rsqrt", "Sigmoid", "Sign", "Sinh", "Softplus", "Softsign", "Sqrt",
            "Square", "Tan", "Tanh", "Real", "Imag", "Erf", "Erfc", "Erfinv",
            "Lgamma", "Digamma",
            "Add", "AddV2", "Sub", "Mul", "Div", "Atan2", "Complex", "DivNoNan",
            "MulNoNan", "FloorDiv", "Xlogy", "Xlog1py", "Xdivy", "FloorMod",
            "BitwiseAnd", "BitwiseOr", "BitwiseXor", "LeftShift", "RightShift",
            "LogicalAnd", "LogicalOr", "Mod", "Maximum", "Minimum", "RealDiv",
            "ReciprocalGrad", "RsqrtGrad", "SqrtGrad", "TruncateDiv",
            "TruncateMod", "Equal", "NotEqual", "Greater", "GreaterEqual",
            "Less", "LessEqual", "SigmoidGrad", "SoftplusGrad", "SoftsignGrad",
            "TanhGrad", "Pow", "SquaredDifference", "ApproximateEqual",
            "AddN", "Bitcast", "Cast", "ClipByValue", "Const", "Empty",
            "Identity", "IdentityN", "Relu", "Relu6", "ReluGrad", "Relu6Grad",
            "LeakyReluGrad", "Elu", "EluGrad", "Selu", "SeluGrad", "Select",
            "SelectV2", "Transpose", "ConjugateTranspose",
            "_UnaryOpsComposition", "CollectiveReduceV2",
            "CollectiveAssignGroupV2",
            "PlaceholderWithDefault", "PreventGradient", "StopGradient",
            "Snapshot", "_EagerConst"}},
    {"RED",
     {"All", "Any", "Min", "Max", "Mean", "Prod", "Sum"}},
          {"PWRED",
           {"ArgMax", "ArgMin", "DiagPart", "Softmax",
            "SparseSoftmaxCrossEntropyWithLogits", "LogSoftmax"}},
          {"REDUCEWINDOW",
           {"ArgMax", "ArgMin", "DiagPart", "Softmax",
            "SparseSoftmaxCrossEntropyWithLogits", "LogSoftmax"}},
          {"REDUCEWINDOWPW", {"BiasAddGrad", "LRN", "LRNGrad"}},
          {"BN",
           {"FusedBatchNorm", "FusedBatchNormV2", "FusedBatchNormV3",
            "_FusedBatchNormEx", "FusedBatchNormGrad", "FusedBatchNormGradV2",
            "FusedBatchNormGradV3"}},
          {"Conv", {"_FusedConv2D"}},
          {"SORT", {"TopKV2"}},  
          {"MISC",
     {"ApproxTopK", "BroadcastTo", "ExpandDims", "Fill", "NoOp",
      "Range", "Rank", "Reshape", "Shape", "ShapeN", "Size", "Squeeze",
      "Transpose", "ZerosLike", "OnesLike", "BiasAdd" ,
      "BroadcastArgs", "BroadcastGradientArgs", "OneHot", "Concat", "ConcatV2",
      "ConcatOffset", "Const", "MirrorPad", "MirrorPadGrad", "Pack", "Pad",
      "PadV2", "Reverse", "ReverseV2", "ReverseSequence", "Slice", "Split",
      "SplitV", "StridedSlice", "StridedSliceGrad",
      "ResourceStridedSliceAssign", "Tile", "Transpose", "InvertPermutation",
      "Unpack", "DeviceIndex", "TensorStridedSliceUpdate", "XlaConcatND",
      "XlaSplitND",
     }}};
  return result;
}
namespace testing {
void ResetClusterSequenceNumber() {
  ClusterSequenceNumberGenerator::Global().Reset();
}
absl::flat_hash_set<string> GetKnownXLAAllowlistOp() {
  absl::flat_hash_set<string> result{
      "AdjustContrastv2",
      "AdjustHue",
      "AdjustSaturation",
      "Asinh",
      "Assert",
      "AssignAddVariableOp",
      "AssignSubVariableOp",
      "AssignVariableOp",
      "AssignVariableXlaConcatND",
      "AvgPool",
      "AvgPool3D",
      "AvgPool3DGrad",
      "AvgPoolGrad",
      "BatchMatMul",
      "BatchMatMulV2",
      "BatchMatMulV3",
      "BatchToSpace",
      "BatchToSpaceND",
      "BesselI0e",
      "BesselI1e",
      "Betainc",
      "BiasAddV1",
      "Bincount",
      "Bucketize",
      "Case",
      "CheckNumerics",
      "Cholesky",
      "ControlTrigger",
      "Conv",
      "Conv2D",
      "Conv2DBackpropFilter",
      "Conv2DBackpropInput",
      "Conv3D",
      "Conv3DBackpropFilterV2",
      "Conv3DBackpropInputV2",
      "Cross",
      "Cumprod",
      "Cumsum",
      "CumulativeLogsumexp",
      "DenseBincount",
      "DataFormatDimMap",
      "DataFormatVecPermute",
      "DepthToSpace",
      "DepthwiseConv2dNative",
      "DepthwiseConv2dNativeBackpropFilter",
      "DepthwiseConv2dNativeBackpropInput",
      "Dequantize",
      "Diag",
      "DynamicInfeedEnqueueTupleOp",
      "DynamicInfeedDequeueTupleOp",
      "DynamicStitch",
      "DynamicPartition",
      "Einsum",
      "EmptyTensorList",
      "EnsureShape",
      "ExtractImagePatches",
      "Igamma",
      "IgammaGradA",
      "RandomGammaGrad",
      "Igammac",
      "FFT",
      "FFT2D",
      "FFT3D",
      "FakeParam",
      "FakeQuantWithMinMaxArgs",
      "FakeQuantWithMinMaxArgsGradient",
      "FakeQuantWithMinMaxVars",
      "FakeQuantWithMinMaxVarsGradient",
      "FakeQuantWithMinMaxVarsPerChannel",
      "FakeQuantWithMinMaxVarsPerChannelGradient",
      "Gather",
      "GatherNd",
      "GatherV2",
      "HSVToRGB",
      "IFFT",
      "IFFT2D",
      "IFFT3D",
      "IRFFT",
      "IRFFT2D",
      "IRFFT3D",
      "If",
      "InTopKV2",
      "L2Loss",
      "LeakyRelu",
      "LinSpace",
      "ListDiff",
      "LogMatrixDeterminant",
      "LowerBound",
      "MatMul",
      "MatrixBandPart",
      "MatrixDiag",
      "MatrixDiagPart",
      "MatrixDiagPartV2",
      "MatrixDiagPartV3",
      "MatrixDiagV2",
      "MatrixDiagV3",
      "MatrixInverse",
      "MatrixSetDiag",
      "MatrixSetDiagV2",
      "MatrixSetDiagV3",
      "MatrixSolve",
      "MatrixTriangularSolve",
      "MaxPool",
      "MaxPool3D",
      "MaxPool3DGrad",
      "MaxPool3DGradGrad",
      "MaxPoolGrad",
      "MaxPoolGradGrad",
      "MaxPoolGradGradV2",
      "MaxPoolGradV2",
      "MaxPoolV2",
      "Multinomial",
      "NextAfter",
      "NonMaxSuppressionV3",
      "NonMaxSuppressionV4",
      "ParallelDynamicStitch",
      "ParameterizedTruncatedNormal",
      "PartitionedCall",
      "Polygamma",
      "PopulationCount",
      "Qr",
      "QuantizeAndDequantizeV2",
      "QuantizeAndDequantizeV3",
      "QuantizeAndDequantizeV4",
      "RFFT",
      "RFFT2D",
      "RFFT3D",
      "RGBToHSV",
      "RandomShuffle",
      "RandomStandardNormal",
      "RandomUniform",
      "RandomUniformInt",
      "ReadVariableOp",
      "ReadVariableXlaSplitND",
      "ResizeBilinear",
      "ResizeBilinearGrad",
      "ResizeNearestNeighbor",
      "ResourceApplyAdaMax",
      "ResourceApplyAdadelta",
      "ResourceApplyAdagrad",
      "ResourceApplyAdagradDA",
      "ResourceApplyAdagradV2",
      "ResourceApplyAdam",
      "ResourceApplyAddSign",
      "ResourceApplyCenteredRMSProp",
      "ResourceApplyFtrl",
      "ResourceApplyFtrlV2",
      "ResourceApplyGradientDescent",
      "ResourceApplyKerasMomentum",
      "ResourceApplyMomentum",
      "ResourceApplyPowerSign",
      "ResourceApplyProximalAdagrad",
      "ResourceApplyProximalGradientDescent",
      "ResourceApplyRMSProp",
      "ResourceGather",
      "ResourceScatterAdd",
      "ResourceScatterDiv",
      "ResourceScatterMax",
      "ResourceScatterMin",
      "ResourceScatterMul",
      "ResourceScatterNdAdd",
      "ResourceScatterNdSub",
      "ResourceScatterNdUpdate",
      "ResourceScatterSub",
      "ResourceScatterUpdate",
      "RngReadAndSkip",
      "RngSkip",
      "Roll",
      "ScatterNd",
      "SegmentSumV2",
      "SegmentProdV2",
      "SegmentMinV2",
      "SegmentMaxV2",
      "SelfAdjointEigV2",
      "SoftmaxCrossEntropyWithLogits",
      "SpaceToBatch",
      "SpaceToBatchND",
      "SpaceToDepth",
      "SparseMatMul",
      "SparseToDense",
      "StackCloseV2",
      "StackPopV2",
      "StackPushV2",
      "StackV2",
      "StatefulPartitionedCall",
      "StatefulStandardNormalV2",
      "StatefulTruncatedNormal",
      "StatefulUniform",
      "StatefulUniformFullInt",
      "StatefulUniformInt",
      "StatelessCase",
      "StatelessIf",
      "StatelessMultinomial",
      "StatelessParameterizedTruncatedNormal",
      "StatelessRandomGetAlg",
      "StatelessRandomGetKeyCounter",
      "StatelessRandomGetKeyCounterAlg",
      "StatelessRandomNormal",
      "StatelessRandomNormalV2",
      "StatelessRandomUniform",
      "StatelessRandomUniformV2",
      "StatelessRandomUniformInt",
      "StatelessRandomUniformIntV2",
      "StatelessRandomUniformFullInt",
      "StatelessRandomUniformFullIntV2",
      "StatelessTruncatedNormal",
      "StatelessTruncatedNormalV2",
      "StatelessWhile",
      "StochasticCastToInt",
      "Svd",
      "SymbolicGradient",
      "TensorArrayCloseV3",
      "TensorArrayConcatV3",
      "TensorArrayGatherV3",
      "TensorArrayGradV3",
      "TensorArrayReadV3",
      "TensorArrayScatterV3",
      "TensorArraySizeV3",
      "TensorArraySplitV3",
      "TensorArrayV3",
      "TensorArrayWriteV3",
      "TensorListConcatV2",
      "TensorListElementShape",
      "TensorListFromTensor",
      "TensorListGather",
      "TensorListGetItem",
      "TensorListLength",
      "TensorListPopBack",
      "TensorListPushBack",
      "TensorListReserve",
      "TensorListSetItem",
      "TensorListSplit",
      "TensorListStack",
      "TensorScatterAdd",
      "TensorScatterMax",
      "TensorScatterMin",
      "TensorScatterSub",
      "TensorScatterUpdate",
      "ToBool",
      "TridiagonalSolve",
      "TridiagonalMatMul",
      "TruncatedNormal",
      "UniformDequantize",
      "UniformQuantize",
      "UniformQuantizedAdd",
      "UniformQuantizedClipByValue",
      "UniformQuantizedConvolution",
      "UniformQuantizedDot",
      "UniformRequantize",
      "Unique",
      "UniqueV2",
      "UpperBound",
      "UnsortedSegmentMax",
      "UnsortedSegmentMin",
      "UnsortedSegmentProd",
      "UnsortedSegmentSum",
      "VarIsInitializedOp",
      "VariableShape",
      "Where",
      "While",
      "XlaAllReduce",
      "XlaBroadcastHelper",
      "XlaCallModule",
      "XlaConcatND",
      "XlaConv",
      "XlaConvV2",
      "XlaCustomCall",
      "XlaCustomCallV2",
      "XlaDequantize",
      "XlaDot",
      "XlaDotV2",
      "XlaDynamicSlice",
      "XlaDynamicUpdateSlice",
      "XlaEinsum",
      "XlaGather",
      "XlaIf",
      "XlaKeyValueSort",
      "XlaOptimizationBarrier",
      "XlaPad",
      "XlaRecv",
      "XlaReduce",
      "XlaReducePrecision",
      "XlaReduceScatter",
      "XlaReduceWindow",
      "XlaRemoveDynamicDimensionSize",
      "XlaReplicaId",
      "XlaRngBitGenerator",
      "XlaScatter",
      "XlaSelectAndScatter",
      "XlaSelfAdjointEig",
      "XlaSend",
      "XlaSetBound",
      "XlaSetDynamicDimensionSize",
      "XlaSharding",
      "XlaSort",
      "XlaSplitND",
      "XlaSpmdFullToShardShape",
      "XlaSpmdShardToFullShape",
      "XlaSvd",
      "XlaVariadicReduce",
      "XlaVariadicReduceV2",
      "XlaVariadicSort",
      "XlaWhile",
      "Zeta",
      "_Arg",
      "_ArrayToList",
      "_ListToArray",
      "_Retval"};
  return result;
}
}  
}  