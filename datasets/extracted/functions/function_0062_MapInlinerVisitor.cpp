#include "xla/service/map_inliner.h"
#include <memory>
#include <string>
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
namespace xla {
class MapInlinerVisitor : public DfsHloVisitorWithDefault {
 public:
  explicit MapInlinerVisitor(HloComputation* computation)
      : computation_(computation) {}
  absl::Status DefaultAction(HloInstruction* ) override {
    return absl::OkStatus();
  }
  absl::Status HandleMap(HloInstruction* map) override;
  absl::StatusOr<bool> Run(HloComputation* computation);
 private:
  HloComputation* computation_;
  bool changed_ = false;
};
absl::StatusOr<bool> MapInlinerVisitor::Run(HloComputation* computation) {
  changed_ = false;
  computation_ = computation;
  TF_RETURN_IF_ERROR(computation->root_instruction()->Accept(this));
  return changed_;
}
absl::Status MapInlinerVisitor::HandleMap(HloInstruction* map) {
  HloComputation* function = map->to_apply();
  HloInstruction& root = *function->root_instruction();
  if (hlo_query::AllOperandsAreParameters(root)) {
    if (root.opcode() == HloOpcode::kFusion) {
      return absl::OkStatus();
    }
    VLOG(10) << "inlining map({X ... Y}, op) => : op(X ... Y) with function "
             << root.ToShortString();
    if (root.opcode() == HloOpcode::kParameter) {
      TF_RETURN_IF_ERROR(
          map->ReplaceAllUsesWith(map->operands()[root.parameter_number()]));
      TF_RETURN_IF_ERROR(computation_->RemoveInstruction(map));
    } else if (root.opcode() == HloOpcode::kConstant) {
      HloInstruction* constant = computation_->AddInstruction(root.Clone());
      HloInstruction* placed_instruction = computation_->AddInstruction(
          HloInstruction::CreateBroadcast(map->shape(), constant, {}));
      TF_RETURN_IF_ERROR(
          computation_->ReplaceInstruction(map, placed_instruction));
    } else {
      std::vector<HloInstruction*> params;
      for (int64_t o = 0; o < root.operands().size(); o++) {
        params.push_back(map->operands()[root.operand(o)->parameter_number()]);
      }
      HloInstruction* placed_instruction = computation_->AddInstruction(
          root.CloneWithNewOperands(map->shape(), params));
      TF_RETURN_IF_ERROR(
          computation_->ReplaceInstruction(map, placed_instruction));
    }
    changed_ = true;
    return absl::OkStatus();
  }
  return absl::OkStatus();
}
absl::StatusOr<bool> MapInliner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  MapInlinerVisitor visitor(nullptr);
  bool changed = false;
  for (HloComputation* computation : module->computations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool computation_changed, visitor.Run(computation));
    changed |= computation_changed;
  }
  return changed;
}
}  