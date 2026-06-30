#include "xla/service/gpu/transforms/async_wrapper.h"
#include <algorithm>
#include <deque>
#include <iterator>
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
namespace xla::gpu {
absl::StatusOr<bool> AsyncWrapper::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  XLA_VLOG_LINES(
      1, absl::StrCat("AsyncWrapper will process the following module:\n",
                      module->ToString()));
  std::deque<HloComputation*> computations;
  computations.push_back(module->entry_computation());
  while (!computations.empty()) {
    HloComputation* computation = computations.front();
    computations.pop_front();
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (predicate_(instruction)) {
        XLA_VLOG_LINES(
            1, absl::StrCat(
                   "AsyncWrapper will make the following instruction async:\n",
                   instruction->ToString()));
        TF_RETURN_IF_ERROR(
            computation
                ->CreateAsyncInstructions(instruction,
                                          {ShapeUtil::MakeScalarShape(U32)})
                .status());
        changed = true;
        continue;
      }
      if (instruction->opcode() == HloOpcode::kCall) {
        std::copy(instruction->called_computations().begin(),
                  instruction->called_computations().end(),
                  std::back_inserter(computations));
      }
    }
  }
  XLA_VLOG_LINES(
      1,
      absl::StrCat("AsyncWrapper finished processing the following module:\n",
                   module->ToString()));
  return changed;
}
}  