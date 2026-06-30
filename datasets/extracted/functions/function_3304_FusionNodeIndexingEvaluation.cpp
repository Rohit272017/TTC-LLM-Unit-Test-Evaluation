#include "xla/service/fusion_node_indexing_evaluation.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/elemental_ir_emitter.h"
#include "xla/types.h"
#include "tsl/platform/logging.h"
namespace xla {
FusionNodeIndexingEvaluation::FusionNodeIndexingEvaluation(
    const HloInstruction* fusion, int64_t root_usage_count)
    : fusion_(fusion) {
  HloInstruction* root = fusion->fused_expression_root();
  indexing_users_[root].insert(fusion);
  index_usage_count_[fusion] = root_usage_count;
  RecomputeCache();
}
const int64_t FusionNodeIndexingEvaluation::kAllowedCodeDuplication = 15;
namespace {
int64_t UserCount(const HloInstruction* hlo) {
  int64_t cnt = 0;
  for (HloInstruction* user : hlo->users()) {
    if (user->opcode() == HloOpcode::kFusion) {
      int64_t operand_index = user->operand_index(hlo);
      cnt += user->fused_parameter(operand_index)->user_count();
    } else {
      ++cnt;
    }
  }
  return cnt;
}
}  
bool FusionNodeIndexingEvaluation::CodeDuplicationTooHigh(
    const HloInstruction* producer) const {
  if (producer->opcode() == HloOpcode::kBroadcast) {
    return false;
  }
  int64_t emitted_instructions = EvaluateEmittedInstructions(producer);
  return emitted_instructions > kAllowedCodeDuplication ||
         (ElementalIrEmitter::OpInvalidatesCache(producer) &&
          (emitted_instructions > 1 || UserCount(producer) > 1));
}
bool FusionNodeIndexingEvaluation::MaxCodeDuplicationTooHigh() const {
  for (const auto& entry : index_usage_count_) {
    if (entry.second > kAllowedCodeDuplication ||
        (ElementalIrEmitter::OpInvalidatesCache(entry.first) &&
         (entry.second > 1 || UserCount(entry.first) > 1))) {
      return true;
    }
  }
  return false;
}
int64_t FusionNodeIndexingEvaluation::EvaluateEmittedInstructions(
    const HloInstruction* producer) const {
  int64_t total = 0;
  for (const auto* user : indexing_users_.at(producer)) {
    total += index_usage_count_.at(user);
  }
  return total;
}
void FusionNodeIndexingEvaluation::UpdateEvaluationCache(
    const HloInstruction* producer,
    absl::flat_hash_set<const HloInstruction*> indexing_users_of_producer) {
  CHECK(!indexing_users_.contains(producer));
  indexing_users_[producer] = std::move(indexing_users_of_producer);
  UpdateIndexUsageCount(producer);
  UpdateIndexingUsersOfOperands(producer);
}
absl::flat_hash_set<const HloInstruction*>
FusionNodeIndexingEvaluation::RemoveFusionOperand(
    HloInstruction* fusion_operand) {
  auto indexing_users_of_operand =
      std::move(indexing_users_.at(fusion_operand));
  indexing_users_.erase(fusion_operand);
  CHECK(!index_usage_count_.contains(fusion_operand));
  return indexing_users_of_operand;
}
void FusionNodeIndexingEvaluation::RecomputeCache() {
  auto postorder =
      fusion_->fused_instructions_computation()->MakeInstructionPostOrder();
  std::reverse(postorder.begin(), postorder.end());
  for (const auto* instruction : postorder) {
    if (instruction->opcode() == HloOpcode::kParameter) {
      continue;
    }
    UpdateIndexUsageCount(instruction);
    UpdateIndexingUsersOfOperands(instruction);
  }
}
void FusionNodeIndexingEvaluation::UpdateIndexUsageCount(
    const HloInstruction* instruction) {
  int64_t total = 0;
  for (const auto* user : indexing_users_[instruction]) {
    total += index_usage_count_.at(user);
  }
  CHECK(index_usage_count_.emplace(instruction, total).second);
}
void FusionNodeIndexingEvaluation::UpdateIndexingUsersOfOperands(
    const HloInstruction* instruction) {
  for (const auto* operand : instruction->operands()) {
    if (operand->opcode() == HloOpcode::kParameter) {
      operand = fusion_->operand(operand->parameter_number());
    }
    if (instruction->opcode() == HloOpcode::kTranspose ||
        Shape::Equal().IgnoreElementType()(operand->shape(),
                                           instruction->shape())) {
      indexing_users_[operand].insert(indexing_users_[instruction].begin(),
                                      indexing_users_[instruction].end());
    } else {
      indexing_users_[operand].insert(instruction);
    }
  }
}
}  