#include "xla/service/gpu/transforms/async_collective_annotator.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace gpu {
absl::StatusOr<bool> AsyncCollectiveAnnotator::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (!hlo_query::IsAsyncCollectiveStartOp(instruction)) {
        continue;
      }
      CollectiveBackendConfig config;
      config.set_is_sync(!is_collective_async_(instruction));
      TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                          instruction->backend_config<GpuBackendConfig>());
      *gpu_config.mutable_collective_backend_config() = config;
      TF_RETURN_IF_ERROR(instruction->set_backend_config(gpu_config));
      changed = true;
    }
  }
  return changed;
}
}  
}  