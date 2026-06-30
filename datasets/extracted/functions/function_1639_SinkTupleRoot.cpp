#include "xla/service/root_instruction_sinker.h"
#include "xla/service/tuple_util.h"
namespace xla {
namespace {
void SinkTupleRoot(HloComputation* computation) {
  HloInstruction* root = computation->root_instruction();
  CHECK(root->shape().IsTuple());
  HloInstruction* new_root = TupleUtil::Duplicate(root);
  HloInstructionSequence& sequence =
      computation->parent()->schedule().GetOrCreateSequence(computation);
  for (HloInstruction* operand : new_root->operands()) {
    sequence.push_back(operand);
  }
  sequence.push_back(new_root);
  computation->set_root_instruction(new_root);
}
void SinkNontupleRoot(HloComputation* computation) {
  HloInstruction* root = computation->root_instruction();
  CHECK(!root->shape().IsTuple());
  HloInstruction* new_root = computation->AddInstruction(
      HloInstruction::CreateBitcast(root->shape(), root));
  HloInstructionSequence& sequence =
      computation->parent()->schedule().GetOrCreateSequence(computation);
  sequence.push_back(new_root);
  computation->set_root_instruction(new_root);
}
}  
absl::StatusOr<bool> RootInstructionSinker::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_RET_CHECK(module->has_schedule());
  bool modified = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    HloInstructionSequence& sequence =
        module->schedule().GetOrCreateSequence(computation);
    if (computation->root_instruction() ==
        sequence.instructions().at(sequence.size() - 1)) {
      continue;
    }
    if (computation->root_instruction()->shape().IsTuple()) {
      SinkTupleRoot(computation);
    } else {
      SinkNontupleRoot(computation);
    }
    modified = true;
  }
  return modified;
}
}  