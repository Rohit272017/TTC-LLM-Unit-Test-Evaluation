#include "xla/service/gpu/transforms/all_reduce_splitter.h"
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/ir/collective_device_list.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/collective_opt_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace {
struct ARReplicaGroups {
  std::vector<ReplicaGroup> first_ar_replica_groups;
  std::vector<ReplicaGroup> second_ar_replica_groups;
};
struct AllReduceRewriteSpec {
  int split_dim;
  int group_size;
  HloAllReduceInstruction* all_reduce;
  HloDynamicSliceInstruction* dynamic_slice;
  ARReplicaGroups replica_groups;
  std::string ToString() {
    return absl::Substitute(
        "{\n split_dim=$0\n group_size=$1\n all_reduce=$2\n "
        "dynamic_slice=$3\n}\n",
        split_dim, group_size, all_reduce->ToString(),
        dynamic_slice->ToString());
  }
};
struct RewriteInfeasibleReason {
  const HloInstruction* ar;
  std::string message;
};
struct ReplicaGroups {
  std::vector<ReplicaGroup> replica_groups;
  template <typename H>
  friend H AbslHashValue(H h, const ReplicaGroups& rg) {
    return H::combine(std::move(h), rg.replica_groups.size());
  }
  friend bool operator==(const ReplicaGroups& item,
                         const ReplicaGroups& other) {
    if (item.replica_groups.size() != other.replica_groups.size()) {
      return false;
    }
    for (int i = 0; i < item.replica_groups.size(); i++) {
      const ReplicaGroup& item_replica_group = item.replica_groups[i];
      const ReplicaGroup& other_replica_group = other.replica_groups[i];
      for (int i = 0; i < item_replica_group.replica_ids_size(); i++) {
        if (item_replica_group.replica_ids(i) !=
            other_replica_group.replica_ids(i)) {
          return false;
        }
      }
    }
    return true;
  }
};
using ARReplicaGroupMap =
    absl::flat_hash_map<ReplicaGroups,
                        std::vector<const HloAllReduceInstruction*>>;
using RewriteDecision =
    std::variant<AllReduceRewriteSpec, RewriteInfeasibleReason>;
std::optional<int> GetSplitDim(const HloAllReduceInstruction& ar,
                               const HloDynamicSliceInstruction& ds) {
  int split_dim = -1;
  int num_dims = 0;
  for (int64_t dim = 0; dim < ar.shape().rank(); ++dim) {
    if (ar.shape().dimensions(dim) != ds.shape().dimensions(dim)) {
      num_dims++;
      split_dim = dim;
    }
  }
  if (num_dims != 1) {
    VLOG(2) << "No support for multiple nor 0 split dims.";
    return std::nullopt;
  }
  return split_dim;
}
std::optional<int> GetProcessGroupSize(const HloAllReduceInstruction& ar,
                                       const HloDynamicSliceInstruction& ds) {
  CHECK(ds.operand(0) == &ar) << "Irrelevant AR + DS pair.";
  std::optional<int> split_dim = GetSplitDim(ar, ds);
  if (!split_dim.has_value()) {
    return std::nullopt;
  }
  return ar.shape().dimensions(*split_dim) /
         ds.dynamic_slice_sizes()[*split_dim];
}
ARReplicaGroupMap GetReplicaGroupsMap(HloComputation& computation) {
  ARReplicaGroupMap map;
  hlo_query::ForEachInstructionWithOpcode(
      computation, HloOpcode::kAllReduce,
      [&map](const HloInstruction* instruction) {
        const HloAllReduceInstruction* ar =
            Cast<HloAllReduceInstruction>(instruction);
        auto rgs = ReplicaGroups{ar->replica_groups()};
        map[rgs].push_back(ar);
      });
  return map;
}
ARReplicaGroups GetNewReplicaGroups(int group_size, int num_partitions) {
  CHECK_EQ(num_partitions % group_size, 0);
  std::vector<ReplicaGroup> first_ar_rgs, second_ar_rgs;
  int num_units = num_partitions / group_size;
  first_ar_rgs.reserve(num_units);
  second_ar_rgs.reserve(group_size);
  for (int u = 0; u < group_size * num_units; u += group_size) {
    ReplicaGroup& group = first_ar_rgs.emplace_back();
    for (int r = u; r < u + group_size; r++) {
      group.add_replica_ids(r);
    }
  }
  for (int g = 0; g < group_size; g++) {
    ReplicaGroup& group = second_ar_rgs.emplace_back();
    for (int r = g; r < group_size * num_units; r += group_size) {
      group.add_replica_ids(r);
    }
  }
  return {
      first_ar_rgs,
      second_ar_rgs,
  };
}
bool IsLogicalReduceScatter(const HloModule& module,
                            const AllReduceRewriteSpec& spec,
                            HloComputation& computation) {
  HloAllReduceInstruction& ar = *spec.all_reduce;
  CHECK_EQ(ar.user_count(), 1);
  CHECK_EQ(module.config().replica_count(), 1);
  HloInstruction* first_ar =
      computation.AddInstruction(HloInstruction::CreateAllReduce(
          ar.shape(), ar.operands(), ar.to_apply(),
          CollectiveDeviceList(spec.replica_groups.first_ar_replica_groups),
          ar.constrain_layout(), hlo_query::NextChannelId(module),
          ar.use_global_device_ids()));
  HloInstruction* ds = ar.users()[0];
  auto* old_operand = ds->mutable_operand(0);
  if (!ds->ReplaceOperandWith(0, first_ar).ok()) {
    return false;
  }
  absl::Cleanup _ = [&] {
    CHECK_OK(ds->ReplaceOperandWith(0, old_operand));
    CHECK_OK(computation.RemoveInstruction(first_ar));
  };
  return MatchReduceScatter(Cast<HloAllReduceInstruction>(first_ar),
                            module.config().num_partitions(),
                            module.config().replica_count(),
                            false,
                            true)
      .has_value();
}
bool IsProfitableToSplit(const ARReplicaGroupMap& replica_map,
                         const AllReduceRewriteSpec& spec) {
  auto new_rgs = spec.replica_groups;
  bool first_replica_exists =
      replica_map.contains(ReplicaGroups{new_rgs.first_ar_replica_groups});
  bool second_replica_exists =
      replica_map.contains(ReplicaGroups{new_rgs.second_ar_replica_groups});
  return first_replica_exists || second_replica_exists;
}
RewriteDecision CanRewrite(const HloModule& module,
                           const ARReplicaGroupMap& replica_map,
                           HloComputation& computation,
                           HloInstruction& instruction) {
  const HloModuleConfig& config = module.config();
  if (config.use_auto_spmd_partitioning() || !config.use_spmd_partitioning() ||
      config.replica_count() != 1) {
    return RewriteInfeasibleReason{
        &instruction,
        "Supporting only SPMD partitioning scheme.",
    };
  }
  if (instruction.opcode() != HloOpcode::kAllReduce) {
    return RewriteInfeasibleReason{
        &instruction,
        "Cannot rewrite an AllReduce, since it's not AllReduce.",
    };
  }
  auto* ar = Cast<HloAllReduceInstruction>(&instruction);
  if (!ar->use_global_device_ids()) {
    return RewriteInfeasibleReason{
        &instruction,
        "Only global ids are supported currently.",
    };
  }
  if (ar->user_count() != 1 ||
      ar->users().front()->opcode() != HloOpcode::kDynamicSlice) {
    return RewriteInfeasibleReason{
        &instruction,
        "Cannot rewrite AllReduce if it is not a logical reduce scatter.",
    };
  }
  auto* ds = Cast<HloDynamicSliceInstruction>(ar->users().front());
  if (ds->user_count() > 1) {
    return RewriteInfeasibleReason{
        &instruction,
        "Exactly one user of dynamic slice is required for a rewrite.",
    };
  }
  int num_partitions = config.num_partitions();
  std::vector<ReplicaGroup> rgs = ar->replica_groups();
  if (rgs.size() != 1 || rgs.front().replica_ids_size() != num_partitions) {
    return RewriteInfeasibleReason{
        &instruction,
        absl::StrCat("Cannot determine a valid split with num_partitions: ",
                     num_partitions),
    };
  }
  std::optional<int> split_dim = GetSplitDim(*ar, *ds);
  if (!split_dim.has_value()) {
    return RewriteInfeasibleReason{
        &instruction,
        "Cannot get a split dim.",
    };
  }
  std::optional<int> group_size = GetProcessGroupSize(*ar, *ds);
  if (!group_size.has_value()) {
    return RewriteInfeasibleReason{
        &instruction,
        "Cannot determine a group size.",
    };
  }
  if (num_partitions == group_size) {
    return RewriteInfeasibleReason{
        &instruction,
        "Nothing to rewrite",
    };
  }
  if (num_partitions % *group_size != 0) {
    return RewriteInfeasibleReason{
        &instruction,
        "Group size does not evenly divide the number of partitions",
    };
  }
  auto spec = AllReduceRewriteSpec{
      *split_dim,
      *group_size,
      ar,
      ds,
      GetNewReplicaGroups(*group_size, num_partitions),
  };
  if (!IsLogicalReduceScatter(module, spec, computation)) {
    return RewriteInfeasibleReason{
        &instruction,
        "Not a logical reduce scatter.",
    };
  }
  if (!IsProfitableToSplit(replica_map, spec)) {
    return RewriteInfeasibleReason{
        &instruction,
        "Splitting is not profitable.",
    };
  }
  return spec;
}
absl::StatusOr<bool> SplitAllReduce(const HloModuleConfig& config,
                                    AllReduceRewriteSpec spec,
                                    HloComputation& computation) {
  int64_t next_channel_id =
      hlo_query::NextChannelId(*spec.all_reduce->GetModule());
  VLOG(1) << "AR splitting spec: " << spec.ToString();
  int num_partitions = config.num_partitions();
  int group_size = spec.group_size;
  CHECK_EQ(num_partitions % group_size, 0);
  HloAllReduceInstruction& ar = *spec.all_reduce;
  HloDynamicSliceInstruction& ds = *spec.dynamic_slice;
  const auto& [first_ar_replica_groups, second_ar_replica_groups] =
      spec.replica_groups;
  int channel_id = next_channel_id++;
  HloInstruction* first_ar =
      computation.AddInstruction(HloInstruction::CreateAllReduce(
          ar.shape(), ar.operands(), ar.to_apply(),
          CollectiveDeviceList(first_ar_replica_groups), ar.constrain_layout(),
          channel_id, ar.use_global_device_ids()));
  channel_id = next_channel_id++;
  HloInstruction* second_ar =
      computation.AddInstruction(HloInstruction::CreateAllReduce(
          ds.shape(), {&ds}, ar.to_apply(),
          CollectiveDeviceList(second_ar_replica_groups), ar.constrain_layout(),
          channel_id, ar.use_global_device_ids()));
  TF_RETURN_IF_ERROR(computation.ReplaceInstruction(&ar, first_ar));
  if (ds.IsRoot()) {
    computation.set_root_instruction(second_ar);
  }
  TF_RETURN_IF_ERROR(ds.ReplaceAllUsesWith(second_ar));
  return true;  
}
absl::StatusOr<bool> SplitAllReduce(const HloModule& module,
                                    const ARReplicaGroupMap& replica_map,
                                    HloComputation& computation,
                                    HloInstruction& instruction) {
  RewriteDecision spec =
      CanRewrite(module, replica_map, computation, instruction);
  if (std::holds_alternative<RewriteInfeasibleReason>(spec)) {
    auto reason = std::get<RewriteInfeasibleReason>(spec);
    VLOG(1) << "Cannot process {" << reason.ar->ToString()
            << "} due to : " << reason.message;
    return false;  
  }
  return SplitAllReduce(module.config(), std::get<AllReduceRewriteSpec>(spec),
                        computation);  
}
}  
absl::StatusOr<bool> AllReduceSplitter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (auto* computation : module->computations(execution_threads)) {
    ARReplicaGroupMap replica_map = GetReplicaGroupsMap(*computation);
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      TF_ASSIGN_OR_RETURN(bool rewritten, SplitAllReduce(*module, replica_map,
                                                         *computation, *instr));
      changed |= rewritten;
    }
  }
  return changed;
}
}  