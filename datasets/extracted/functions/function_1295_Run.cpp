#include "xla/service/gpu/model/gpu_cost_model_stats_collection.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "tsl/platform/status.h"
namespace xla {
namespace gpu {
absl::StatusOr<bool> GpuCostModelStatsCollection::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  for (auto* computation : module->MakeComputationPostOrder()) {
    TF_CHECK_OK(computation->Accept(&cost_analysis_));
    for (auto* fusion_instr : computation->instructions()) {
      if (fusion_instr->opcode() != HloOpcode::kFusion) continue;
      GpuPerformanceModel::RecordEstimatedRunTime(
          fusion_instr, device_info_, &cost_analysis_,
          GpuPerformanceModelOptions::ForModule(module));
    }
  }
  return false;
}
}  
}  