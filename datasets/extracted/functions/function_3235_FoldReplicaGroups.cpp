#include "xla/service/all_reduce_folder.h"
#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/collective_device_list.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/all_reduce_key.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
namespace xla {
namespace {
std::optional<std::vector<ReplicaGroup>> FoldReplicaGroups(
    absl::Span<const ReplicaGroup> replica_groups0,
    absl::Span<const ReplicaGroup> replica_groups1) {
  int64_t num_replicas = 0;
  for (const ReplicaGroup &rg : replica_groups0) {
    for (int64_t id : rg.replica_ids()) {
      num_replicas = std::max(num_replicas, id);
    }
  }
  num_replicas++;
  std::vector<int> replica_group_no(num_replicas, -1);
  for (int group_no = 0; group_no < replica_groups0.size(); ++group_no) {
    for (int64_t id : replica_groups0[group_no].replica_ids()) {
      replica_group_no[id] = group_no;
    }
  }
  absl::flat_hash_map<std::vector<bool>, int64_t> contributor_set_id;
  std::vector<int64_t> contributing_replicas_set_id(num_replicas, 0);
  int64_t next_id = 1;
  for (const ReplicaGroup &rg : replica_groups1) {
    std::vector<bool> contributors(num_replicas, false);
    for (int64_t id : rg.replica_ids()) {
      int64_t group_no = replica_group_no[id];
      for (int64_t contrib : replica_groups0[group_no].replica_ids()) {
        if (contributors[contrib]) {
          return std::nullopt;
        }
        contributors[contrib] = true;
      }
    }
    int64_t set_id;
    auto it = contributor_set_id.find(contributors);
    if (it != contributor_set_id.end()) {
      set_id = it->second;
    } else {
      set_id = next_id++;
      contributor_set_id[contributors] = set_id;
    }
    for (int64_t id : rg.replica_ids()) {
      contributing_replicas_set_id[id] = set_id;
    }
  }
  std::vector<ReplicaGroup> new_replica_groups;
  new_replica_groups.reserve(contributor_set_id.size());
  for (const auto &it : contributor_set_id) {
    const std::vector<bool> &contributors = it.first;
    const int64_t set_id = it.second;
    new_replica_groups.emplace_back();
    ReplicaGroup &group = new_replica_groups.back();
    for (int64_t replica = 0; replica < num_replicas; ++replica) {
      if (contributors[replica]) {
        if (contributing_replicas_set_id[replica] != set_id) {
          return std::nullopt;
        }
        group.add_replica_ids(replica);
      }
    }
  }
  absl::c_sort(new_replica_groups,
               [](const ReplicaGroup &a, const ReplicaGroup &b) {
                 return a.replica_ids(0) < b.replica_ids(0);
               });
  return new_replica_groups;
}
}  
absl::StatusOr<bool> AllReduceFolder::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  if (hlo_query::ContainsLayoutConstrainedAllReduce(*module)) {
    VLOG(1) << "Skip AllReduceFolder because the module contains all-reduce "
               "with constrained layouts";
    return false;
  }
  int64_t next_channel_id = hlo_query::NextChannelId(*module);
  bool changed = false;
  for (auto computation : module->computations(execution_threads)) {
    for (HloInstruction *inst : computation->MakeInstructionPostOrder()) {
      if (inst->opcode() != HloOpcode::kAllReduce ||
          inst->operand(0)->opcode() != HloOpcode::kAllReduce) {
        continue;
      }
      auto *ar0 = Cast<HloAllReduceInstruction>(inst->mutable_operand(0));
      auto *ar1 = Cast<HloAllReduceInstruction>(inst);
      if (ar0->user_count() != 1) {
        continue;
      }
      std::optional<AllReduceKey> key0 = GetAllReduceKey(
          ar0, nullptr, true);
      std::optional<AllReduceKey> key1 = GetAllReduceKey(
          ar1, nullptr, true);
      if (!key0 || !key1 || *key0 != *key1 || ar0->replica_groups().empty() ||
          ar1->replica_groups().empty()) {
        continue;
      }
      std::optional<std::vector<ReplicaGroup>> new_replica_groups =
          FoldReplicaGroups(ar0->replica_groups(), ar1->replica_groups());
      if (!new_replica_groups) {
        continue;
      }
      std::optional<int64_t> channel_id;
      if (ar0->channel_id()) {
        channel_id = next_channel_id++;
      }
      HloInstruction *new_ar =
          computation->AddInstruction(HloInstruction::CreateAllReduce(
              ar0->shape(), ar0->operands(), ar0->to_apply(),
              CollectiveDeviceList(*new_replica_groups),
              false, channel_id,
              ar0->use_global_device_ids()));
      TF_RETURN_IF_ERROR(ar1->ReplaceAllUsesWith(new_ar));
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(ar1));
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(ar0));
      changed = true;
    }
  }
  return changed;
}
}  