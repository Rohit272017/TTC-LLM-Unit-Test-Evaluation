#include "xla/service/gpu/transforms/all_gather_dynamic_slice_simplifier.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/service/collective_opt_utils.h"
namespace xla {
bool AllGatherDynamicSliceSimplifier::InstructionMatchesPattern(
    HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kDynamicSlice) {
    return false;
  }
  HloDynamicSliceInstruction* dynamic_slice =
      Cast<HloDynamicSliceInstruction>(instruction);
  HloInstruction* operand = dynamic_slice->mutable_operand(0);
  bool is_reshape = operand->opcode() == HloOpcode::kReshape;
  bool is_all_gather = operand->opcode() == HloOpcode::kAllGather;
  if (!is_reshape && !is_all_gather) {
    return false;
  }
  if (is_reshape && operand->operand(0)->opcode() != HloOpcode::kAllGather) {
    return false;
  }
  const HloModuleConfig& config = instruction->GetModule()->config();
  HloAllGatherInstruction* all_gather =
      is_reshape ? Cast<HloAllGatherInstruction>(operand->mutable_operand(0))
                 : Cast<HloAllGatherInstruction>(operand);
  bool match = AllGatherDynamicSliceCancellation(
      all_gather, config.num_partitions(), config.replica_count(),
      true,
      true, 1,
      HloPredicateIsOp<HloOpcode::kPartitionId>,
      HloPredicateIsOp<HloOpcode::kReplicaId>,
      false,
      true);
  return match;
}
absl::StatusOr<HloInstruction*>
AllGatherDynamicSliceSimplifier::ExpandInstruction(
    HloInstruction* instruction) {
  HloDynamicSliceInstruction* dynamic_slice =
      Cast<HloDynamicSliceInstruction>(instruction);
  HloInstruction* operand = dynamic_slice->mutable_operand(0);
  if (operand->opcode() != HloOpcode::kReshape) {
    return operand->mutable_operand(0);
  }
  HloReshapeInstruction* reshape = Cast<HloReshapeInstruction>(operand);
  HloAllGatherInstruction* all_gather =
      Cast<HloAllGatherInstruction>(reshape->mutable_operand(0));
  HloInstruction* all_gather_input = all_gather->mutable_operand(0);
  auto* new_reshape = instruction->parent()->AddInstruction(
      HloInstruction::CreateReshape(dynamic_slice->shape(), all_gather_input));
  return new_reshape;
}
}  