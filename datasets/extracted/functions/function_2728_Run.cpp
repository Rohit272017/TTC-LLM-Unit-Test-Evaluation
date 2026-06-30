#include "xla/service/defuser.h"
#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/call_graph.h"
#include "xla/status_macros.h"
#include "xla/types.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
namespace xla {
absl::StatusOr<bool> Defuser::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(1) << "Defusing module " << module->name();
  XLA_VLOG_LINES(2, "Before defusion:\n" + module->ToString());
  bool changed = false;
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  TF_RETURN_IF_ERROR(call_graph->VisitNodes(
      [&](const CallGraphNode& call_graph_node) -> absl::Status {
        if (call_graph_node.computation()->IsFusionComputation()) {
          TF_RET_CHECK(call_graph_node.caller_callsites().size() == 1);
          HloInstruction* fusion_instruction =
              call_graph_node.caller_callsites()[0].instruction();
          TF_RETURN_IF_ERROR(fusion_instruction->Defuse());
          changed = true;
        }
        return absl::OkStatus();
      },
      true));
  XLA_VLOG_LINES(2, "After defusion:\n" + module->ToString());
  return changed;
}
}  