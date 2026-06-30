#include "xla/service/gpu/transforms/reduce_scatter_creator.h"
#include <cstdint>
#include <optional>
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/collective_opt_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "tsl/platform/errors.h"
namespace xla {
namespace gpu {
absl::StatusOr<bool> ReduceScatterCreator::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  const HloModuleConfig &config = module->config();
  int64_t next_channel_id = hlo_query::NextChannelId(*module);
  bool changed = false;
  for (HloComputation *computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction *instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() != HloOpcode::kAllReduce) {
        continue;
      }
      auto *ar = Cast<HloAllReduceInstruction>(instruction);
      auto ar_spec = MatchReduceScatter(ar, config.num_partitions(),
                                        config.replica_count(),
                                        false,
                                        true);
      if (!ar_spec) {
        VLOG(2) << "Cannot match reduce-scatter " << ar->ToString();
        continue;
      }
      HloInstruction *ds = ar_spec->dynamic_slice;
      const int64_t split_dim = ar_spec->split_dim;
      Shape scatter_shape = ar->shape();
      const int64_t split_dim_size = scatter_shape.dimensions(split_dim);
      HloInstruction *rs_input = ar->mutable_operand(0);
      const int64_t scatter_dim_size = split_dim_size / ar_spec->group_size;
      TF_RET_CHECK(scatter_dim_size * ar_spec->group_size <= split_dim_size);
      if (split_dim_size % ar_spec->group_size != 0) {
        scatter_shape.set_dimensions(split_dim,
                                     scatter_dim_size * ar_spec->group_size);
        rs_input = computation->AddInstruction(HloInstruction::CreateSlice(
            scatter_shape, rs_input,
            std::vector<int64_t>(scatter_shape.rank(), 0),
            scatter_shape.dimensions(),
            std::vector<int64_t>(scatter_shape.rank(), 1)));
      }
      scatter_shape.set_dimensions(split_dim, scatter_dim_size);
      std::optional<int64_t> channel_id;
      if (ar->channel_id()) {
        channel_id = next_channel_id++;
      }
      HloInstruction *ars =
          computation->AddInstruction(HloInstruction::CreateReduceScatter(
              scatter_shape, {rs_input}, ar->to_apply(), ar->device_list(),
              ar->constrain_layout(), channel_id, ar->use_global_device_ids(),
              ar_spec->split_dim));
      HloInstruction *result = ars;
      HloInstruction *reshape = nullptr;
      if (ds->operand(0) != ar) {
        reshape = ds->mutable_operand(0);
        result = computation->AddInstruction(
            HloInstruction::CreateReshape(ds->shape(), result));
      }
      TF_RETURN_IF_ERROR(ds->ReplaceAllUsesWith(result));
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(ds));
      if (reshape) {
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(reshape));
      }
      TF_RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(ar));
      changed = true;
    }
  }
  return changed;
}
}  
}  