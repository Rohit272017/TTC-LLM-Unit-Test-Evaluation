#include "xla/service/gpu/transforms/horizontal_input_fusion.h"
#include <algorithm>
#include <cstddef>
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace gpu {
namespace {
Shape GetInputShapeForMultiOutputFusion(const HloInstruction& instr) {
  const HloInstruction* real_hero = GetRealHeroForMultiOutputFusion(instr);
  if (real_hero->operands().empty()) {
    return Shape();
  } else {
    return real_hero->operand(0)->shape();
  }
}
class HorizontalInputFusionImpl {
 public:
  explicit HorizontalInputFusionImpl(HloComputation* computation,
                                     const se::DeviceDescription& d)
      : computation_(computation), device_info_(d) {}
  ~HorizontalInputFusionImpl() = default;
  absl::StatusOr<bool> Run();
 private:
  HloComputation* computation_;
  const se::DeviceDescription& device_info_;
};  
bool CompareShapeDimsFromLeftToRight(const Shape& shape_a,
                                     const Shape& shape_b) {
  if (shape_a.rank() != shape_b.rank()) {
    return shape_a.rank() < shape_b.rank();
  }
  auto dims_a = shape_a.dimensions();
  auto dims_b = shape_b.dimensions();
  for (size_t i = 0; i < dims_a.size(); ++i) {
    if (dims_a[i] != dims_b[i]) {
      return dims_a[i] < dims_b[i];
    }
  }
  return true;
}
std::vector<HloInstruction*> FindAndSortFusionCandidates(
    HloInstruction* consumer) {
  absl::flat_hash_set<HloInstruction*> fusion_instr_set;
  std::vector<HloInstruction*> fusion_instrs;
  for (HloInstruction* opnd : consumer->operands()) {
    HloInstruction* predecessor = opnd->LatestNonGteAncestor();
    if (!predecessor->IsCustomFusion() &&
        IsInputFusibleReduction(*predecessor) &&
        IsConsumerTheOnlyNonRootUser(*predecessor, *consumer)) {
      if (fusion_instr_set.insert(predecessor).second) {
        fusion_instrs.push_back(predecessor);
      }
    }
  }
  std::sort(fusion_instrs.begin(), fusion_instrs.end(),
            [&](const HloInstruction* a, const HloInstruction* b) {
              Shape shape_a = GetInputShapeForMultiOutputFusion(*a);
              Shape shape_b = GetInputShapeForMultiOutputFusion(*b);
              if (!ShapeUtil::EqualIgnoringElementType(shape_a, shape_b)) {
                return CompareShapeDimsFromLeftToRight(shape_a, shape_b);
              }
              return GetInstrCountOfFusible(*a) < GetInstrCountOfFusible(*b);
            });
  return fusion_instrs;
}
absl::StatusOr<bool> HorizontalInputFusionImpl::Run() {
  bool changed = false;
  XLA_VLOG_LINES(3, computation_->ToString());
  std::vector<HloInstruction*> def_to_use_order =
      computation_->MakeInstructionPostOrder();
  for (HloInstruction* consumer : def_to_use_order) {
    auto candidates = FindAndSortFusionCandidates(consumer);
    if (candidates.size() <= 1) {
      continue;
    }
    for (size_t j = 0; j < candidates.size(); ++j) {
      if (candidates[j]->opcode() != HloOpcode::kFusion) {
        TF_ASSIGN_OR_RETURN(
            HloInstruction * fusion_instr,
            MakeFusionInstruction(candidates[j],
                                  HloInstruction::FusionKind::kInput));
        candidates[j] = fusion_instr;
        changed = true;
      }
    }
    size_t fusion_anchor_id = 0;
    for (size_t j = 1; j < candidates.size(); ++j) {
      HloInstruction* fusion_anchor = candidates[fusion_anchor_id];
      HloInstruction* fused = candidates[j];
      if (ShapesCompatibleForMultiOutputFusion(*fusion_anchor, *fused) &&
          FusionFitsInBudget(*fusion_anchor, *fused, device_info_)) {
        VLOG(3) << "Fuse " << fused->ToString() << " into "
                << fusion_anchor->ToString();
        fusion_anchor->MergeFusionInstructionIntoMultiOutput(fused);
        changed = true;
      } else {
        VLOG(3) << j - fusion_anchor_id - 1 << " instructions are fused.";
        fusion_anchor_id = j;
      }
    }
  }
  return changed;
}
}  
absl::StatusOr<bool> HorizontalInputFusion::RunOnComputation(
    HloComputation* computation) {
  HorizontalInputFusionImpl horizontal_fusion_impl(computation, device_info_);
  return horizontal_fusion_impl.Run();
}
absl::StatusOr<bool> HorizontalInputFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  VLOG(2) << "Run horizontal input fusion.";
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(changed, RunOnComputation(comp));
  }
  return changed;
}
}  
}  