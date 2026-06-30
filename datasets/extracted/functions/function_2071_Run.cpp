#include "xla/service/while_loop_trip_count_annotator.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/while_loop_analysis.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
namespace xla {
absl::StatusOr<bool> WhileLoopTripCountAnnotator::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (const HloComputation* comp : module->computations(execution_threads)) {
    for (HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() != HloOpcode::kWhile) {
        continue;
      }
      if (auto trip_count = ComputeWhileLoopTripCount(instr)) {
        WhileLoopBackendConfig config;
        config.mutable_known_trip_count()->set_n(*trip_count);
        TF_RETURN_IF_ERROR(instr->set_backend_config(config));
        changed = true;
      }
    }
  }
  return changed;
}
}  