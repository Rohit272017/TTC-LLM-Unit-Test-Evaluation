#include "tensorflow/compiler/mlir/tensorflow/utils/cluster_util.h"
#include <algorithm>
#include <functional>
#include <string>
#include <vector>
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Block.h"  
#include "mlir/IR/BuiltinTypeInterfaces.h"  
#include "mlir/IR/Matchers.h"  
#include "mlir/Support/LLVM.h"  
#include "mlir/Transforms/RegionUtils.h"  
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
namespace mlir::TF {
namespace {
llvm::SetVector<Operation*> GetAllOpsDependOnCluster(
    const Cluster& c,
    const llvm::DenseMap<Operation*, Cluster*>& op_to_cluster_map) {
  llvm::SetVector<Operation*> ops_depend_on_cluster;
  for (Operation& op : *c.ops.front()->getBlock()) {
    if (op.isBeforeInBlock(c.ops.front()) || c.ops.contains(&op)) {
      continue;
    }
    llvm::SetVector<Value> live_ins(op.operand_begin(), op.operand_end());
    getUsedValuesDefinedAbove(op.getRegions(), live_ins);
    if (llvm::any_of(live_ins, [&](Value value) {
          Operation* defining_op = value.getDefiningOp();
          if (!defining_op) {
            return false;
          }
          return c.ops.contains(defining_op) ||
                 ops_depend_on_cluster.contains(defining_op);
        })) {
      ops_depend_on_cluster.insert(&op);
    }
  }
  llvm::SetVector<Operation*> same_cluster_ops_with_dependency(
      ops_depend_on_cluster.begin(), ops_depend_on_cluster.end());
  for (Operation* op : ops_depend_on_cluster) {
    Cluster* cluster = op_to_cluster_map.lookup(op);
    if (cluster == nullptr) {
      continue;
    }
    for (Operation* ops_in_same_cluster : cluster->ops) {
      same_cluster_ops_with_dependency.insert(ops_in_same_cluster);
    }
  }
  return same_cluster_ops_with_dependency;
}
bool CanMergeIntoCluster(
    const Cluster& c, Operation* to_merge,
    const TF::SideEffectAnalysis::Info& side_effect_analysis,
    std::function<std::string(Operation*)> get_target,
    const llvm::DenseMap<Operation*, Cluster*>& op_to_cluster_map) {
  const bool has_control_predecessors_after_cluster =
      !side_effect_analysis
           .DirectControlPredecessors(
               to_merge,
               [&c](Operation* pred) {
                 Operation* const last_c_op = c.ops.back();
                 return last_c_op->getBlock() == pred->getBlock() &&
                        last_c_op->isBeforeInBlock(pred);
               })
           .empty();
  if (has_control_predecessors_after_cluster) {
    return false;
  }
  llvm::SetVector<Operation*> ops_depend_on_cluster =
      GetAllOpsDependOnCluster(c, op_to_cluster_map);
  return llvm::none_of(to_merge->getOperands(), [&](Value value) {
    Operation* defining_op = value.getDefiningOp();
    return defining_op && ops_depend_on_cluster.contains(defining_op);
  });
}
}  
llvm::StringMap<SmallVector<Cluster>> BuildAllClusters(
    Block& block, const TF::SideEffectAnalysis::Info& side_effect_analysis,
    std::function<std::string(Operation*)> get_target,
    std::function<bool(Operation*)> is_ignored_op) {
  llvm::StringMap<SmallVector<Cluster>> all_clusters;
  llvm::DenseMap<Operation*, Cluster*> op_to_cluster_map;
  llvm::StringMap<Cluster> nearest_clusters;
  for (Operation& op : llvm::make_early_inc_range(block)) {
    if (is_ignored_op(&op)) {
      continue;
    }
    std::string target_name = get_target(&op);
    auto it = nearest_clusters.find(target_name);
    if (it == nearest_clusters.end()) {
      SetVector<Operation*> new_cluster_op_set;
      new_cluster_op_set.insert(&op);
      nearest_clusters[target_name] = Cluster{new_cluster_op_set, target_name};
      op_to_cluster_map[&op] = &nearest_clusters[target_name];
      continue;
    }
    Cluster& nearest_cluster = it->second;
    if (CanMergeIntoCluster(nearest_cluster, &op, side_effect_analysis,
                            get_target, op_to_cluster_map)) {
      nearest_cluster.ops.insert(&op);
      op_to_cluster_map[&op] = &nearest_cluster;
      continue;
    }
    all_clusters[target_name].push_back(nearest_cluster);
    SetVector<Operation*> new_cluster_op_set;
    new_cluster_op_set.insert(&op);
    nearest_clusters[target_name] = Cluster{new_cluster_op_set, target_name};
    op_to_cluster_map[&op] = &nearest_clusters[target_name];
  }
  for (auto& target_cluster : nearest_clusters) {
    all_clusters[target_cluster.first()].push_back(target_cluster.second);
  }
  return all_clusters;
}
void ReorderOpResultUses(mlir::Operation* cluster) {
  mlir::Block* const cluster_block = cluster->getBlock();
  llvm::SetVector<mlir::Operation*> ops_to_reorder;
  llvm::SmallVector<mlir::Value> worklist;
  llvm::append_range(worklist, cluster->getResults());
  while (!worklist.empty()) {
    mlir::Value value = worklist.back();
    worklist.pop_back();
    for (mlir::Operation* const user : value.getUsers()) {
      mlir::Operation* const op = cluster_block->findAncestorOpInBlock(*user);
      if (op == nullptr || !op->isBeforeInBlock(cluster)) {
        continue;
      }
      if (ops_to_reorder.insert(op)) {
        llvm::append_range(worklist, op->getResults());
      }
    }
  }
  llvm::SmallVector<mlir::Operation*, 0> sorted = ops_to_reorder.takeVector();
  llvm::sort(sorted, [](mlir::Operation* lhs, mlir::Operation* rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  for (mlir::Operation* const op : llvm::reverse(sorted)) {
    op->moveAfter(cluster);
  }
}
}  