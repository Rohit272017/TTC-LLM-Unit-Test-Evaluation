#include "xla/service/cpu/cpu_instruction_fusion.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/fusion_node_indexing_evaluation.h"
#include "xla/service/instruction_fusion.h"
#include "xla/service/llvm_ir/fused_ir_emitter.h"
namespace xla {
namespace cpu {
namespace {
bool CanBeLoopFused(const HloInstruction& hlo) {
  return hlo.IsElementwise() ||  
         hlo.opcode() == HloOpcode::kBitcast ||
         hlo.opcode() == HloOpcode::kBroadcast ||
         hlo.opcode() == HloOpcode::kConcatenate ||
         hlo.opcode() == HloOpcode::kDynamicSlice ||
         hlo.opcode() == HloOpcode::kDynamicUpdateSlice ||
         hlo.opcode() == HloOpcode::kGather ||
         hlo.opcode() == HloOpcode::kIota || hlo.opcode() == HloOpcode::kPad ||
         hlo.opcode() == HloOpcode::kReduce ||
         hlo.opcode() == HloOpcode::kReshape ||
         hlo.opcode() == HloOpcode::kReverse ||
         hlo.opcode() == HloOpcode::kSlice ||
         hlo.opcode() == HloOpcode::kTranspose;
}
bool IsNonComplexNonBatchedMatrixVectorDot(const HloInstruction* hlo) {
  const Shape& hlo_shape = hlo->shape();
  return !ShapeUtil::ElementIsComplex(hlo_shape) &&
         hlo->opcode() == HloOpcode::kDot && hlo_shape.dimensions_size() <= 1 &&
         hlo->dot_dimension_numbers().lhs_batch_dimensions_size() == 0;
}
bool HasExactlyOneUse(const HloInstruction& hlo_instr) {
  return hlo_instr.user_count() == 1 &&
         absl::c_count(hlo_instr.users().front()->operands(), &hlo_instr) == 1;
}
bool CanBeOutputFused(const HloInstruction* producer,
                      const HloInstruction* consumer) {
  return consumer->opcode() == HloOpcode::kAdd &&
         IsNonComplexNonBatchedMatrixVectorDot(producer) &&
         HasExactlyOneUse(*producer) == 1;
}
bool CanBeOutputFusedIntoSomeOperand(const HloInstruction* consumer) {
  return consumer->opcode() == HloOpcode::kAdd &&
         (CanBeOutputFused(consumer->operand(0), consumer) ||
          CanBeOutputFused(consumer->operand(1), consumer));
}
}  
FusionDecision CpuInstructionFusion::ShouldFuse(HloInstruction* consumer,
                                                int64_t operand_index) {
  HloInstruction* producer = consumer->mutable_operand(operand_index);
  VLOG(2) << "Considering for fusion: operand " << operand_index << " of "
          << consumer->ToString();
  constexpr int kFusionThresholdBytes = 16 * 1024;
  if (CanBeOutputFused(producer, consumer)) {
    VLOG(2) << "Fusion OK: Can create output fusion.";
    return FusionDecision::Allow();
  }
  if (CanBeOutputFusedIntoSomeOperand(producer)) {
    return FusionDecision::Forbid(
        "Bailing because producer can be output-fused into some operand.");
  }
  if (!CanBeLoopFused(*producer)) {
    return FusionDecision::Forbid("Producer is not loop-fusible.");
  }
  if (producer->opcode() != HloOpcode::kFusion && is_expensive(*producer) &&
      ReusesOperandElements(consumer, operand_index)) {
    return FusionDecision::Forbid("Fusion is not profitable.");
  }
  RETURN_IF_NOT_FUSIBLE(InstructionFusion::ShouldFuse(consumer, operand_index));
  if (producer->opcode() == HloOpcode::kConstant &&
      consumer->opcode() != HloOpcode::kFusion) {
    return FusionDecision::Forbid(
        "Not fusing: insufficient non-constant nodes.");
  }
  if (producer->opcode() == HloOpcode::kFusion) {
    return FusionDecision::Forbid(
        "Not fusing: producer is itself a fusion node.");
  }
  if (consumer->opcode() == HloOpcode::kFusion) {
    if (fusion_node_evaluations_.find(consumer) ==
        fusion_node_evaluations_.end()) {
      fusion_node_evaluations_.emplace(consumer,
                                       FusionNodeIndexingEvaluation(consumer));
    }
    if (fusion_node_evaluations_.at(consumer).CodeDuplicationTooHigh(
            producer)) {
      return FusionDecision::Forbid("Code duplication too high");
    }
  }
  if (consumer->opcode() == HloOpcode::kDot) {
    const Shape& output_shape = consumer->shape();
    if (output_shape.dimensions_size() <= 1) {
      if (consumer->operand(0)->shape().rank() == 1 && operand_index == 1 &&
          ShapeUtil::ByteSizeOfElements(consumer->operand(0)->shape()) <
              kFusionThresholdBytes) {
        VLOG(2) << "Fusing small matrix-vector product.";
        return FusionDecision::Allow();
      } else if (consumer->operand(1)->shape().rank() == 1 &&
                 operand_index == 0 &&
                 ShapeUtil::ByteSizeOfElements(consumer->operand(1)->shape()) <
                     kFusionThresholdBytes) {
        VLOG(2) << "Fusing small matrix-vector product.";
        return FusionDecision::Allow();
      }
    }
  }
  if (consumer->opcode() == HloOpcode::kReduce &&
      !absl::c_linear_search(
          consumer->dimensions(),
          LayoutUtil::Minor(consumer->operand(0)->shape().layout(), 0))) {
    return FusionDecision::Forbid(
        "Not fusing reductions over major dimensions");
  }
  if (producer->opcode() == HloOpcode::kReduce &&
      !absl::c_linear_search(
          producer->dimensions(),
          LayoutUtil::Minor(producer->operand(0)->shape().layout(), 0))) {
    return FusionDecision::Forbid(
        "Not fusing reductions over major dimensions");
  }
  if (consumer->IsLoopFusion()) {
    VLOG(2) << "Fusing: consumer is a fusion node.";
    return FusionDecision::Allow();
  }
  if (CanBeLoopFused(*consumer)) {
    VLOG(2) << "Fusing: consumer is elementwise or fusible.";
    return FusionDecision::Allow();
  }
  return FusionDecision::Forbid("Not fusing: not found a fusible case");
}
HloInstruction::FusionKind CpuInstructionFusion::ChooseKind(
    const HloInstruction* producer, const HloInstruction* consumer) {
  return CanBeOutputFused(producer, consumer)
             ? HloInstruction::FusionKind::kOutput
             : HloInstruction::FusionKind::kLoop;
}
HloInstruction* CpuInstructionFusion::FuseInstruction(
    HloInstruction* fusion_instruction, HloInstruction* producer) {
  auto evaluation = fusion_node_evaluations_.find(fusion_instruction);
  if (evaluation == fusion_node_evaluations_.end()) {
    evaluation = fusion_node_evaluations_
                     .emplace(fusion_instruction,
                              FusionNodeIndexingEvaluation(fusion_instruction))
                     .first;
  }
  auto indexing_users = evaluation->second.RemoveFusionOperand(producer);
  HloInstruction* new_producer =
      InstructionFusion::FuseInstruction(fusion_instruction, producer);
  evaluation->second.UpdateEvaluationCache(new_producer, indexing_users);
  return new_producer;
}
}  
}  