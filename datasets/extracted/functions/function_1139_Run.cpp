#include "xla/service/memory_space_propagation.h"
#include <cstdint>
#include "xla/shape.h"
#include "xla/shape_util.h"
namespace xla {
absl::StatusOr<bool> MemorySpacePropagation::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool modified = false;
  TF_ASSIGN_OR_RETURN(auto dataflow_analysis,
                      HloDataflowAnalysis::Run(*module, false,
                                               true));
  dataflow_analysis_ = std::move(dataflow_analysis);
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kFusion) {
        for (int operand_idx = 0;
             operand_idx < instruction->fused_parameters().size();
             ++operand_idx) {
          ShapeUtil::ForEachLeafShape(
              instruction->operand(operand_idx)->shape(),
              [&](const Shape& sub_shape, const ShapeIndex& index) {
                int64_t memory_space = sub_shape.layout().memory_space();
                modified |=
                    Propagate(index, instruction->fused_parameter(operand_idx),
                              memory_space);
              });
        }
        ShapeUtil::ForEachLeafShape(
            instruction->shape(),
            [&](const Shape& sub_shape, const ShapeIndex& index) {
              int64_t memory_space = sub_shape.layout().memory_space();
              modified |= Propagate(index, instruction->fused_expression_root(),
                                    memory_space);
            });
      }
    }
  }
  return modified;
}
bool MemorySpacePropagation::Propagate(ShapeIndexView index,
                                       const HloInstruction* callee_instruction,
                                       int64_t memory_space) const {
  bool modified = false;
  const HloValue& value = dataflow_analysis_->GetUniqueValueAt(
      callee_instruction, ShapeIndex(index));
  for (const HloPosition& position : value.positions()) {
    HloInstruction* instruction = position.instruction;
    Shape* shape = ShapeUtil::GetMutableSubshape(instruction->mutable_shape(),
                                                 position.index);
    if (shape->layout().memory_space() == memory_space) {
      continue;
    }
    shape->mutable_layout()->set_memory_space(memory_space);
    modified = true;
    if (instruction->opcode() == HloOpcode::kFusion) {
      Propagate(position.index, instruction->fused_expression_root(),
                memory_space);
    }
    const HloInstruction* parent_fusion =
        instruction->parent()->FusionInstruction();
    if (instruction == instruction->parent()->root_instruction() &&
        parent_fusion->parent()->IsFusionComputation()) {
      Propagate(position.index, parent_fusion, memory_space);
    }
    if (instruction->opcode() == HloOpcode::kParameter &&
        parent_fusion->parent()->IsFusionComputation()) {
      const HloInstruction* fusion_operand =
          parent_fusion->operand(instruction->parameter_number());
      Propagate(position.index, fusion_operand, memory_space);
    }
  }
  for (const HloUse& use : value.GetUses()) {
    if (use.instruction->opcode() == HloOpcode::kFusion) {
      modified |= Propagate(
          use.operand_index,
          use.instruction->fused_parameter(use.operand_number), memory_space);
    }
  }
  return modified;
}
}  