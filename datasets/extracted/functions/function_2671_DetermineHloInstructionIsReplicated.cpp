#include "xla/service/hlo_replication_analysis.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/map_util.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
namespace xla {
HloReplicationAnalysis::HloReplication
HloReplicationAnalysis::DetermineHloInstructionIsReplicated(
    const HloInstruction* hlo, const ShapeIndex& index,
    bool cross_partition_spmd,
    const absl::flat_hash_map<const HloInstruction*, ShapeTree<HloReplication>>&
        hlo_replication,
    bool support_partial_replication) {
  const auto merge_operand_replication = [&hlo_replication](
                                             const HloInstruction* inst) {
    HloReplication replication = HloReplication::ReplicatedOnAllDevices();
    for (auto operand : inst->operands()) {
      auto operand_it = hlo_replication.find(operand);
      if (operand_it == hlo_replication.end()) {
        replication = replication.Merge(HloReplication::UniqueOnAllDevices());
      } else {
        replication = replication.Merge(operand_it->second.element({}));
      }
    }
    return replication;
  };
  if (hlo->opcode() == HloOpcode::kAllReduce ||
      hlo->opcode() == HloOpcode::kAllGather) {
    HloReplication replication = merge_operand_replication(hlo);
    if (replication.IsReplicatedOnAllDevices()) {
      return replication;
    }
    if (!hlo->channel_id().has_value()) {
      if (cross_partition_spmd) {
        return replication;
      }
      if (hlo->replica_groups().empty() || hlo->replica_groups().size() == 1) {
        return HloReplication::ReplicatedOnAllDevices();
      }
      if (support_partial_replication) {
        std::vector<absl::Span<const int64_t>> device_sets;
        for (const ReplicaGroup& replica_group : hlo->replica_groups()) {
          device_sets.push_back(replica_group.replica_ids());
        }
        return HloReplication::PartiallyReplicated(device_sets);
      } else {
        return HloReplication::UniqueOnAllDevices();
      }
    } else {
      bool global_id;
      if (hlo->opcode() == HloOpcode::kAllReduce) {
        global_id = Cast<HloAllReduceInstruction>(hlo)->use_global_device_ids();
      } else {
        global_id = Cast<HloAllGatherInstruction>(hlo)->use_global_device_ids();
      }
      if (global_id) {
        bool replicated_across_partitions = true;
        bool replicated_across_replicas = true;
        const int64_t num_partitions =
            hlo->GetModule()->config().num_partitions();
        absl::flat_hash_set<int64_t> visited_partitions;
        absl::flat_hash_set<int64_t> visited_replicas;
        for (const auto& group : hlo->replica_groups()) {
          visited_partitions.clear();
          visited_replicas.clear();
          visited_replicas.reserve(group.replica_ids().size());
          visited_partitions.reserve(group.replica_ids().size());
          for (int64_t id : group.replica_ids()) {
            int64_t rid = id / num_partitions;
            int64_t pid = id % num_partitions;
            visited_partitions.insert(pid);
            visited_replicas.insert(rid);
          }
          replicated_across_partitions &=
              visited_partitions.size() == num_partitions;
          replicated_across_replicas &=
              visited_replicas.size() ==
              hlo->GetModule()->config().replica_count();
        }
        if ((cross_partition_spmd && replicated_across_partitions) ||
            (!cross_partition_spmd && replicated_across_replicas)) {
          return HloReplication::ReplicatedOnAllDevices();
        } else {
          return HloReplication::UniqueOnAllDevices();
        }
      }
      if (cross_partition_spmd) {
        return HloReplication::ReplicatedOnAllDevices();
      }
      if (hlo->replica_groups().empty() || hlo->replica_groups().size() == 1) {
        return HloReplication::ReplicatedOnAllDevices();
      } else {
        return HloReplication::UniqueOnAllDevices();
      }
    }
  }
  if (hlo->HasSideEffectNoRecurse()) {
    return HloReplication::UniqueOnAllDevices();
  }
  if (hlo->opcode() == HloOpcode::kReplicaId) {
    return cross_partition_spmd ? HloReplication::ReplicatedOnAllDevices()
                                : HloReplication::UniqueOnAllDevices();
  }
  if (hlo->opcode() == HloOpcode::kPartitionId) {
    return cross_partition_spmd ? HloReplication::UniqueOnAllDevices()
                                : HloReplication::ReplicatedOnAllDevices();
  }
  auto it = hlo_replication.find(hlo);
  if (hlo->opcode() == HloOpcode::kParameter) {
    CHECK(it != hlo_replication.end());
    return it->second.element(index);
  }
  if (it != hlo_replication.end() &&
      it->second.element(index).IsUniqueOnAllDevices()) {
    return it->second.element(index);
  }
  if (hlo->opcode() == HloOpcode::kConstant) {
    return HloReplication::ReplicatedOnAllDevices();
  }
  if (hlo->opcode() == HloOpcode::kCustomCall &&
      (hlo->custom_call_target() == "X64SplitLow" ||
       hlo->custom_call_target() == "X64SplitHigh" ||
       hlo->custom_call_target() == "X64Combine")) {
    return merge_operand_replication(hlo);
  }
  if (support_partial_replication) {
    if (hlo->opcode() == HloOpcode::kDynamicSlice) {
      const HloInstruction* ds_buffer = hlo->operand(0);
      if (hlo->dynamic_slice_sizes().size() == 1 &&
          hlo->dynamic_slice_sizes()[0] == 1 &&
          ds_buffer->opcode() == HloOpcode::kConstant &&
          ds_buffer->shape().rank() == 1 &&
          ds_buffer->shape().element_type() == PrimitiveType::S32 &&
          ((cross_partition_spmd &&
            hlo->operand(1)->opcode() == HloOpcode::kPartitionId) ||
           (!cross_partition_spmd &&
            hlo->operand(1)->opcode() == HloOpcode::kReplicaId))) {
        const HloModule* hlo_module = hlo->GetModule();
        int64_t num_devices = cross_partition_spmd
                                  ? hlo_module->config().num_partitions()
                                  : hlo_module->config().replica_count();
        absl::flat_hash_map<int64_t, std::vector<int64_t>> value_to_device_set;
        for (int64_t device_id = 0; device_id < num_devices; ++device_id) {
          std::optional<int64_t> value =
              ds_buffer->literal().GetIntegralAsS64({device_id});
          value_to_device_set[*value].push_back(device_id);
        }
        std::vector<absl::Span<const int64_t>> device_sets;
        for (const auto& value_and_device_set : value_to_device_set) {
          device_sets.push_back(
              absl::Span<const int64_t>(value_and_device_set.second));
        }
        return HloReplication::PartiallyReplicated(device_sets);
      }
    }
  }
  if (hlo->IsElementwise() ||                             
      hlo->opcode() == HloOpcode::kConcatenate ||         
      hlo->opcode() == HloOpcode::kConvolution ||         
      hlo->opcode() == HloOpcode::kDot ||                 
      hlo->opcode() == HloOpcode::kReduce ||              
      hlo->opcode() == HloOpcode::kBroadcast ||           
      hlo->opcode() == HloOpcode::kTranspose ||           
      hlo->opcode() == HloOpcode::kReshape ||             
      hlo->opcode() == HloOpcode::kBitcast ||             
      hlo->opcode() == HloOpcode::kReverse ||             
      hlo->opcode() == HloOpcode::kGather ||              
      hlo->opcode() == HloOpcode::kScatter ||             
      hlo->opcode() == HloOpcode::kIota ||                
      hlo->opcode() == HloOpcode::kPad ||                 
      hlo->opcode() == HloOpcode::kSlice ||               
      hlo->opcode() == HloOpcode::kDynamicSlice ||        
      hlo->opcode() == HloOpcode::kDynamicUpdateSlice ||  
      hlo->opcode() == HloOpcode::kReduceWindow ||        
      hlo->opcode() == HloOpcode::kCopy) {
    return merge_operand_replication(hlo);
  }
  return HloReplication::UniqueOnAllDevices();
}
bool HloReplicationAnalysis::ComputeHloReplicationOnComputation(
    const HloComputation* computation, bool mark_everything_not_replicated) {
  bool changed = false;
  for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
    auto assign_or_combine_shapetree =
        [&](ShapeTree<HloReplication>&& to_combine,
            const HloInstruction* dest) {
          auto it = hlo_replication_.find(dest);
          if (it == hlo_replication_.end()) {
            hlo_replication_[dest] = std::move(to_combine);
            return true;
          }
          bool updated = false;
          it->second.ForEachMutableElement(
              [&](const ShapeIndex& index, HloReplication* element) {
                HloReplication new_replication =
                    element->Merge(to_combine.element(index));
                if (!element->Equal(new_replication)) {
                  *element = std::move(new_replication);
                  updated = true;
                }
              });
          return updated;
        };
    auto propagate_shapetree = [&](const HloInstruction* source,
                                   const HloInstruction* dest) {
      auto source_it = hlo_replication_.find(source);
      if (source_it == hlo_replication_.end()) {
        return false;
      }
      return assign_or_combine_shapetree(
          ShapeTree<HloReplication>(source_it->second), dest);
    };
    if (inst->opcode() == HloOpcode::kWhile) {
      while (true) {
        bool updated = propagate_shapetree(
            inst->operand(0),
            inst->while_condition()->parameter_instruction(0));
        updated |= propagate_shapetree(
            inst->while_body()->root_instruction(),
            inst->while_condition()->parameter_instruction(0));
        updated |= propagate_shapetree(
            inst->operand(0), inst->while_body()->parameter_instruction(0));
        updated |=
            propagate_shapetree(inst->while_body()->root_instruction(),
                                inst->while_body()->parameter_instruction(0));
        updated |= ComputeHloReplicationOnComputation(
            inst->while_condition(), mark_everything_not_replicated);
        if (!ContainsKey(loops_known_with_same_iterations_, inst) &&
            !hlo_replication_[inst->while_condition()->root_instruction()]
                 .element({})
                 .IsReplicatedOnAllDevices()) {
          updated |= ComputeHloReplicationOnComputation(
              inst->while_body(), true);
        } else {
          updated |= ComputeHloReplicationOnComputation(
              inst->while_body(), mark_everything_not_replicated);
        }
        if (!updated) {
          break;
        }
        changed = true;
      }
      changed |= propagate_shapetree(inst->operand(0), inst);
      changed |=
          propagate_shapetree(inst->while_body()->root_instruction(), inst);
    } else if (inst->opcode() == HloOpcode::kCall ||
               inst->opcode() == HloOpcode::kFusion) {
      auto called = inst->called_computations().front();
      for (int64_t i = 0; i < inst->operand_count(); ++i) {
        changed |= propagate_shapetree(inst->operand(i),
                                       called->parameter_instruction(i));
      }
      changed |= ComputeHloReplicationOnComputation(
          called, mark_everything_not_replicated);
      changed |= propagate_shapetree(called->root_instruction(), inst);
    } else if (inst->opcode() == HloOpcode::kConditional) {
      for (int64_t i = 0; i < inst->called_computations().size(); ++i) {
        changed |= propagate_shapetree(
            inst->operand(i + 1),
            inst->called_computations()[i]->parameter_instruction(0));
      }
      if (!hlo_replication_[inst->operand(0)]
               .element({})
               .IsReplicatedOnAllDevices()) {
        for (auto called : inst->called_computations()) {
          changed |= ComputeHloReplicationOnComputation(
              called,
              true);
        }
        changed |= assign_or_combine_shapetree(
            ShapeTree<HloReplication>(inst->shape(),
                                      HloReplication::UniqueOnAllDevices()),
            inst);
      } else {
        for (auto called : inst->called_computations()) {
          changed |= ComputeHloReplicationOnComputation(
              called, mark_everything_not_replicated);
          changed |= propagate_shapetree(called->root_instruction(), inst);
        }
      }
    } else if (inst->opcode() == HloOpcode::kTuple) {
      ShapeTree<HloReplication> shape_tree(
          inst->shape(), HloReplication::ReplicatedOnAllDevices());
      for (int64_t i = 0; i < inst->operand_count(); ++i) {
        shape_tree.CopySubtreeFrom(hlo_replication_[inst->operand(i)], {}, {i});
      }
      changed |= assign_or_combine_shapetree(std::move(shape_tree), inst);
    } else if (inst->opcode() == HloOpcode::kOptimizationBarrier) {
      ShapeTree<HloReplication> shape_tree = hlo_replication_[inst->operand(0)];
      changed |= assign_or_combine_shapetree(std::move(shape_tree), inst);
    } else if (inst->opcode() == HloOpcode::kGetTupleElement) {
      ShapeTree<HloReplication> shape_tree(
          inst->shape(), HloReplication::ReplicatedOnAllDevices());
      shape_tree.CopySubtreeFrom(hlo_replication_[inst->operand(0)],
                                 {inst->tuple_index()}, {});
      changed |= assign_or_combine_shapetree(std::move(shape_tree), inst);
    } else if (inst->opcode() == HloOpcode::kInfeed && cross_partition_spmd_) {
      ShapeTree<HloReplication> shape_tree(
          inst->shape(), HloReplication::UniqueOnAllDevices());
      if (inst->has_sharding()) {
        auto sharding = inst->sharding().GetAsShapeTree(inst->shape());
        shape_tree.ForEachMutableElement(
            [&sharding](const ShapeIndex& index, HloReplication* data) {
              *data = sharding.element(index).IsReplicated()
                          ? HloReplication::ReplicatedOnAllDevices()
                          : HloReplication::UniqueOnAllDevices();
            });
      }
      changed |= assign_or_combine_shapetree(std::move(shape_tree), inst);
    } else {
      if (mark_everything_not_replicated) {
        changed |= assign_or_combine_shapetree(
            ShapeTree<HloReplication>(inst->shape(),
                                      HloReplication::UniqueOnAllDevices()),
            inst);
      } else {
        ShapeTree<HloReplication> shape_tree(
            inst->shape(), HloReplication::ReplicatedOnAllDevices());
        ShapeUtil::ForEachSubshape(
            inst->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
              *shape_tree.mutable_element(index) =
                  DetermineHloInstructionIsReplicated(
                      inst, index, cross_partition_spmd_, hlo_replication_,
                      support_partial_replication_);
            });
        changed |= assign_or_combine_shapetree(std::move(shape_tree), inst);
      }
    }
  }
  return changed;
}
absl::Status HloReplicationAnalysis::ComputeHloReplication() {
  auto entry = module_->entry_computation();
  for (int i = 0; i < entry->num_parameters(); ++i) {
    auto param = entry->parameter_instruction(i);
    ShapeTree<HloReplication> shape_tree(param->shape(),
                                         HloReplication::UniqueOnAllDevices());
    const auto& replication = param->parameter_replicated_at_leaf_buffers();
    int leaf_index = 0;
    absl::Status status = ShapeUtil::ForEachSubshapeWithStatus(
        param->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
          if (!ShapeUtil::IsLeafIndex(param->shape(), index)) {
            return absl::OkStatus();
          }
          if (cross_partition_spmd_ && param->has_sharding()) {
            TF_ASSIGN_OR_RETURN(auto sharding_tree,
                                param->sharding().AsShapeTree(param->shape()));
            *shape_tree.mutable_element(index) =
                sharding_tree.element(index).IsReplicated()
                    ? HloReplication::ReplicatedOnAllDevices()
                    : HloReplication::UniqueOnAllDevices();
          }
          if (replication) {
            if (!cross_partition_spmd_ && (*replication)[leaf_index]) {
              *shape_tree.mutable_element(index) =
                  HloReplication::ReplicatedOnAllDevices();
            }
            if (cross_partition_spmd_ && !(*replication)[leaf_index]) {
              *shape_tree.mutable_element(index) =
                  HloReplication::UniqueOnAllDevices();
            }
            ++leaf_index;
          }
          return absl::OkStatus();
        });
    TF_RETURN_IF_ERROR(status);
    hlo_replication_[param] = std::move(shape_tree);
  }
  ComputeHloReplicationOnComputation(entry,
                                     false);
  return absl::OkStatus();
}
bool HloReplicationAnalysis::HloInstructionIsReplicatedAt(
    const HloInstruction* inst, const ShapeIndex& index) const {
  auto it = hlo_replication_.find(inst);
  if (it == hlo_replication_.end()) {
    return false;
  }
  return it->second.element(index).IsReplicatedOnAllDevices();
}
bool HloReplicationAnalysis::HloInstructionIsReplicatedAt(
    const HloInstruction* inst, const ShapeIndex& index,
    absl::Span<const ReplicaGroup> replica_groups) const {
  auto it = hlo_replication_.find(inst);
  if (it == hlo_replication_.end()) {
    return false;
  }
  VLOG(5) << "HloInstructionIsReplicatedAt is called on " << inst->name()
          << ", index: " << index.ToString()
          << ", replication: " << it->second.element(index).ToString();
  if (replica_groups.empty()) {
    return it->second.element(index).IsReplicatedOnAllDevices();
  }
  if (it->second.element(index).IsReplicatedOnAllDevices()) {
    return true;
  }
  if (it->second.element(index).IsUniqueOnAllDevices()) {
    return false;
  }
  for (const ReplicaGroup& replica_group : replica_groups) {
    if (!it->second.element(index).IsReplicatedWithinSubgroup(
            replica_group.replica_ids())) {
      return false;
    }
  }
  return true;
}
 absl::StatusOr<std::unique_ptr<HloReplicationAnalysis>>
HloReplicationAnalysis::Run(const HloModule* module,
                            bool cross_partition_spmd) {
  const absl::flat_hash_set<const HloInstruction*> empty;
  return Run(module, cross_partition_spmd, &empty);
}
 absl::StatusOr<std::unique_ptr<HloReplicationAnalysis>>
HloReplicationAnalysis::Run(const HloModule* module, bool cross_partition_spmd,
                            const absl::flat_hash_set<const HloInstruction*>*
                                loops_known_with_same_iterations) {
  auto analysis = absl::WrapUnique(new HloReplicationAnalysis(
      module, cross_partition_spmd, loops_known_with_same_iterations,
      false));
  TF_RETURN_IF_ERROR(analysis->ComputeHloReplication());
  return analysis;
}
 absl::StatusOr<std::unique_ptr<HloReplicationAnalysis>>
HloReplicationAnalysis::RunWithPartialReplication(const HloModule* module,
                                                  bool cross_partition_spmd) {
  const absl::flat_hash_set<const HloInstruction*> empty;
  auto analysis = absl::WrapUnique(
      new HloReplicationAnalysis(module, cross_partition_spmd, &empty,
                                 true));
  TF_RETURN_IF_ERROR(analysis->ComputeHloReplication());
  return analysis;
}
HloReplicationAnalysis::HloReplication::HloReplication()
    : state_(State::kReplicatedOnAllDevices) {}
HloReplicationAnalysis::HloReplication::HloReplication(
    HloReplicationAnalysis::HloReplication::State state,
    absl::Span<const int64_t> device_set_root)
    : state_(state),
      device_set_root_(device_set_root.begin(), device_set_root.end()) {
  CHECK(state == State::kPartiallyReplicated || device_set_root_.empty());
}
HloReplicationAnalysis::HloReplication
HloReplicationAnalysis::HloReplication::ReplicatedOnAllDevices() {
  return HloReplication(State::kReplicatedOnAllDevices, {});
}
HloReplicationAnalysis::HloReplication
HloReplicationAnalysis::HloReplication::UniqueOnAllDevices() {
  return HloReplication(State::kUniqueOnAllDevices, {});
}
HloReplicationAnalysis::HloReplication
HloReplicationAnalysis::HloReplication::PartiallyReplicated(
    absl::Span<const absl::Span<const int64_t>> device_sets) {
  int64_t max_device_id = 0;
  for (const absl::Span<const int64_t>& device_set : device_sets) {
    for (int64_t device_id : device_set) {
      max_device_id = std::max(max_device_id, device_id);
    }
  }
  std::vector<int64_t> device_set_root;
  device_set_root.resize(max_device_id + 1);
  for (const absl::Span<const int64_t>& device_set : device_sets) {
    int64_t min_device_id = *absl::c_min_element(device_set);
    for (int64_t device_id : device_set) {
      device_set_root[device_id] = min_device_id;
    }
  }
  return HloReplication(State::kPartiallyReplicated, device_set_root);
}
HloReplicationAnalysis::HloReplication
HloReplicationAnalysis::HloReplication::Merge(
    const HloReplication& other) const {
  switch (state_) {
    case State::kReplicatedOnAllDevices:
      return other;
    case State::kUniqueOnAllDevices:
      return *this;
    case State::kPartiallyReplicated: {
      switch (other.state_) {
        case State::kReplicatedOnAllDevices:
          return *this;
        case State::kUniqueOnAllDevices:
          return other;
        case State::kPartiallyReplicated: {
          absl::flat_hash_map<int64_t, std::vector<int64_t>>
              value_to_device_set;
          size_t num_devices = device_set_root_.size();
          for (int64_t device_id = 0; device_id < num_devices; ++device_id) {
            int64_t new_value = device_set_root_[device_id] * num_devices +
                                other.device_set_root_[device_id];
            value_to_device_set[new_value].push_back(device_id);
          }
          CHECK_LE(value_to_device_set.size(), num_devices);
          if (value_to_device_set.size() == 1) {
            return ReplicatedOnAllDevices();
          } else if (value_to_device_set.size() < num_devices) {
            std::vector<absl::Span<const int64_t>> device_sets;
            for (const auto& value_and_device_set : value_to_device_set) {
              device_sets.push_back(
                  absl::Span<const int64_t>(value_and_device_set.second));
            }
            return PartiallyReplicated(device_sets);
          } else {
            return UniqueOnAllDevices();
          }
        }
      }
    }
  }
}
bool HloReplicationAnalysis::HloReplication::Equal(
    const HloReplication& other) const {
  if (state_ != other.state_) {
    return false;
  }
  return absl::c_equal(device_set_root_, other.device_set_root_);
}
bool HloReplicationAnalysis::HloReplication::IsReplicatedOnAllDevices() const {
  return state_ == State::kReplicatedOnAllDevices;
}
bool HloReplicationAnalysis::HloReplication::IsUniqueOnAllDevices() const {
  return state_ == State::kUniqueOnAllDevices;
}
bool HloReplicationAnalysis::HloReplication::IsReplicatedWithinSubgroup(
    absl::Span<const int64_t> device_ids) const {
  if (device_ids.empty()) return true;
  return absl::c_all_of(device_ids, [this, &device_ids](int device_id) {
    return device_set_root_[device_id] == device_set_root_[device_ids.front()];
  });
}
std::string HloReplicationAnalysis::HloReplication::ToString() const {
  switch (state_) {
    case State::kReplicatedOnAllDevices:
      return "ReplicatedOnAllDevices";
    case State::kUniqueOnAllDevices:
      return "UniqueOnAllDevices";
    case State::kPartiallyReplicated:
      return absl::StrCat("PartiallyReplicated{",
                          absl::StrJoin(device_set_root_, ","), "}");
  }
}
}  