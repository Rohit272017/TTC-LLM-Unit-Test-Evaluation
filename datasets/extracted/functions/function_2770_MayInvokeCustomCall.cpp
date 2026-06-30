#include "xla/service/gpu/transforms/schedule_postprocessing.h"
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace gpu {
namespace {
using CustomCallInComputation =
    absl::flat_hash_map<const HloComputation*, bool>;
bool MayInvokeCustomCall(
    const HloInstruction* hlo,
    const CustomCallInComputation& custom_call_in_computation) {
  if (hlo->opcode() == HloOpcode::kCustomCall) {
    return true;
  }
  return absl::c_any_of(
      hlo->called_computations(), [&](const HloComputation* callee) {
        return custom_call_in_computation.find(callee)->second;
      });
}
absl::StatusOr<bool> IsRelevantAsynchronousStart(const HloInstruction* hlo) {
  if (!hlo_query::IsAsyncCollectiveStartOp(hlo,
                                           false)) {
    return false;
  }
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                      hlo->backend_config<GpuBackendConfig>());
  const CollectiveBackendConfig& collective_backend_config =
      gpu_config.collective_backend_config();
  return !collective_backend_config.is_sync();
}
absl::StatusOr<bool> IsRelevantAsynchronousDone(const HloInstruction* hlo) {
  return hlo_query::IsAsyncCollectiveDoneOp(hlo,
                                            false);
}
absl::StatusOr<bool> ProcessComputation(
    const HloSchedule& schedule, HloComputation* computation,
    CustomCallInComputation& custom_call_in_computation) {
  bool changed = false;
  bool has_custom_call = false;
  absl::flat_hash_set<HloInstruction*> async_starts;
  const HloInstructionSequence& sequence = schedule.sequence(computation);
  const std::vector<HloInstruction*>& all_instructions =
      sequence.instructions();
  for (HloInstruction* hlo : all_instructions) {
    if (MayInvokeCustomCall(hlo, custom_call_in_computation)) {
      async_starts.clear();
      has_custom_call = true;
      continue;
    }
    TF_ASSIGN_OR_RETURN(bool is_async_start, IsRelevantAsynchronousStart(hlo));
    if (is_async_start) {
      async_starts.insert(hlo);
      continue;
    }
    TF_ASSIGN_OR_RETURN(bool is_async_done, IsRelevantAsynchronousDone(hlo));
    if (is_async_done) {
      HloInstruction* async_start = hlo->mutable_operand(0);
      if (async_starts.contains(async_start)) {
        changed = true;
        TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                            async_start->backend_config<GpuBackendConfig>());
        CollectiveBackendConfig& collective_backend_config =
            *gpu_config.mutable_collective_backend_config();
        collective_backend_config.set_no_parallel_custom_call(true);
        TF_RETURN_IF_ERROR(async_start->set_backend_config(gpu_config));
        async_starts.erase(async_start);
      }
    }
  }
  custom_call_in_computation[computation] = has_custom_call;
  return changed;
}
}  
absl::StatusOr<bool> SchedulePostprocessing::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!module->has_schedule()) return false;
  HloSchedule& schedule = module->schedule();
  bool changed = false;
  CustomCallInComputation custom_call_in_computation;
  std::vector<HloComputation*> all_computations =
      module->MakeComputationPostOrder(execution_threads);
  for (auto iter = all_computations.begin(); iter != all_computations.end();
       ++iter) {
    HloComputation* computation = *iter;
    if (computation->IsFusionComputation()) {
      custom_call_in_computation[computation] = false;
      continue;
    }
    TF_ASSIGN_OR_RETURN(
        bool result,
        ProcessComputation(schedule, computation, custom_call_in_computation));
    changed |= result;
  }
  return changed;
}
}  
}  