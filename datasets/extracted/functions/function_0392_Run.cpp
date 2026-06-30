#include "xla/service/collectives_schedule_linearizer.h"
#include <algorithm>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_reachability.h"
#include "tsl/platform/errors.h"
namespace xla {
absl::StatusOr<bool> CollectivesScheduleLinearizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (is_enabled_ && !is_enabled_(module)) {
    return false;
  }
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    std::unique_ptr<HloReachabilityMap> reachability;
    HloInstruction* prev_done = nullptr;
    for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
      auto* next = DynCast<HloCollectiveInstruction>(inst);
      if (!next) {
        continue;
      }
      if (!reachability) {
        reachability = HloReachabilityMap::Build(computation);
      }
      HloInstruction* start = next;
      HloInstruction* done = next;
      switch (next->opcode()) {
        case HloOpcode::kAllReduceStart:
        case HloOpcode::kAllGatherStart:
        case HloOpcode::kCollectivePermuteStart:
        case HloOpcode::kAsyncStart:
          CHECK_EQ(start->user_count(), 1);
          done = start->users()[0];
          break;
        default:
          break;
      }
      if (prev_done && !reachability->IsConnected(start, prev_done)) {
        TF_RETURN_IF_ERROR(prev_done->AddControlDependencyTo(next));
        VLOG(1) << "Adding control dependency from " << prev_done->ToString()
                << " to " << start->ToString();
        changed = true;
      }
      prev_done = done;
    }
  }
  return changed;
}
}  