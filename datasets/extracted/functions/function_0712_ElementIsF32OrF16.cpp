#include "xla/service/gpu/transforms/instruction_fusion.h"
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/meta/type_traits.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/fusion_node_indexing_evaluation.h"
#include "xla/service/fusion_queue.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"
namespace xla {
namespace gpu {
namespace {
bool ElementIsF32OrF16(const Shape& shape) {
  PrimitiveType type = shape.element_type();
  return type == F32 || type == F16;
}
class EmptyFusionQueue : public FusionQueue {
 public:
  std::pair<HloInstruction*, std::vector<int64_t>>
  DequeueNextInstructionAndOperandsToFuseInOrder() override {
    return {nullptr, {}};
  }
  void RemoveInstruction(HloInstruction* instruction) override {};
  const std::vector<bool>* FusionConfiguration() override { return nullptr; };
};
}  
absl::StatusOr<bool> GpuInstructionFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  fusion_node_evaluations_.clear();
  auto fusible_computations =
      GetFusibleComputations(*module, execution_threads);
  fusible_computations_ = {fusible_computations.begin(),
                           fusible_computations.end()};
  return InstructionFusion::Run(module, execution_threads);
}
 bool GpuInstructionFusion::IsExpensive(
    const HloInstruction& instruction) {
  switch (instruction.opcode()) {
    case HloOpcode::kDivide:
    case HloOpcode::kSqrt:
    case HloOpcode::kRsqrt:
    case HloOpcode::kExp:
      if (ElementIsF32OrF16(instruction.shape())) {
        return false;
      }
      break;
    default:
      break;
  }
  return InstructionFusion::IsExpensive(instruction);
}
FusionDecision GpuInstructionFusion::ShouldFuseInexpensiveChecks(
    HloInstruction* consumer, int64_t operand_index) {
  HloInstruction* producer = consumer->mutable_operand(operand_index);
  if (producer->opcode() == HloOpcode::kFusion) {
    return FusionDecision::Forbid("the producer is a fusion");
  }
  if (consumer->IsCustomFusion()) {
    return FusionDecision::Forbid("the consumer is a custom fusion");
  }
  if (is_expensive(*producer) &&
      ReusesOperandElements(consumer, operand_index)) {
    return FusionDecision::Forbid(
        "the producer is expensive, and the consumer reuses inputs");
  }
  if (IsInputFusibleReduction(*consumer) &&
      IsPhysicallyTransposing(*producer)) {
    return FusionDecision::Forbid(
        "fusing the producer would break read coalescing");
  }
  RETURN_IF_NOT_FUSIBLE(IsProducerConsumerFusible(*producer, *consumer));
  if (CreatesHeavyComputation(*producer, *consumer)) {
    return FusionDecision::Forbid(
        "the fusion would create a heavy computation");
  }
  return InstructionFusion::ShouldFuse(consumer, operand_index);
}
FusionDecision GpuInstructionFusion::ShouldFuse(HloInstruction* consumer,
                                                int64_t operand_index) {
  RETURN_IF_NOT_FUSIBLE(ShouldFuseInexpensiveChecks(consumer, operand_index));
  auto producer = consumer->operand(operand_index);
  RETURN_IF_NOT_FUSIBLE(
      FusionFitsInBudget(*consumer, *producer, device_info_,
                         true));
  if (consumer->opcode() != HloOpcode::kFusion) {
    return FusionDecision::Allow();
  }
  if (fusion_node_evaluations_.find(consumer) ==
      fusion_node_evaluations_.end()) {
    fusion_node_evaluations_.emplace(consumer,
                                     FusionNodeIndexingEvaluation(consumer));
  }
  if (fusion_node_evaluations_.at(consumer).CodeDuplicationTooHigh(producer)) {
    return FusionDecision::Forbid(
        "the fusion would result in an overly large code duplication");
  }
  return FusionDecision::Allow();
}
HloInstruction::FusionKind GpuInstructionFusion::ChooseKind(
    const HloInstruction* producer, const HloInstruction* consumer) {
  return ChooseFusionKind(*producer, *consumer);
}
HloInstruction* GpuInstructionFusion::FuseInstruction(
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
std::unique_ptr<FusionQueue> GpuInstructionFusion::GetFusionQueue(
    HloComputation* computation) {
  if (fusible_computations_.contains(computation)) {
    return InstructionFusion::GetFusionQueue(computation);
  }
  return std::make_unique<EmptyFusionQueue>();
}
}  
}  