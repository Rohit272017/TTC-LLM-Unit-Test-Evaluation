#include "xla/service/gpu/transforms/convert_async_collectives_to_sync.h"
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace gpu {
absl::Status GpuConvertAsyncCollectivesToSync::ConvertAsyncInstructionsToSync(
    HloComputation* computation,
    absl::Span<const std::pair<HloInstruction*, HloInstruction*>> async_pairs)
    const {
  absl::flat_hash_map<HloInstruction*, HloInstruction*> replaced_ops;
  CollectiveBackendConfig sync_config;
  sync_config.set_is_sync(true);
  for (auto& [async_start, async_done] : async_pairs) {
    TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                        async_start->backend_config<GpuBackendConfig>());
    *gpu_config.mutable_collective_backend_config() = sync_config;
    TF_RETURN_IF_ERROR(async_start->set_backend_config(gpu_config));
    replaced_ops[async_start] = nullptr;
    replaced_ops[async_done] = async_start;
  }
  HloModule* module = computation->parent();
  const HloInstructionSequence& sequence =
      module->schedule().sequence(computation);
  std::vector<HloInstruction*> new_sequence;
  new_sequence.reserve(sequence.size());
  for (HloInstruction* instr : sequence.instructions()) {
    auto it = replaced_ops.find(instr);
    if (it == replaced_ops.end()) {
      new_sequence.push_back(instr);
      continue;
    }
    if (it->second == nullptr) {
      continue;
    }
    new_sequence.push_back(it->second);
    new_sequence.push_back(instr);
  }
  module->schedule().set_sequence(computation, new_sequence);
  return absl::OkStatus();
}
}  
}  