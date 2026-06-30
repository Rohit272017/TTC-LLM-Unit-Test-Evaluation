#include "xla/service/gpu/transforms/collective_permute_valid_iteration_annotator.h"
#include "xla/literal_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/while_loop_analysis.h"
namespace xla {
static const HloInstruction* NonConstantOperand(const HloInstruction* instr) {
  const HloInstruction* result = nullptr;
  for (const HloInstruction* operand : instr->operands()) {
    if (!operand->IsConstant()) {
      if (result != nullptr) {
        CHECK_EQ(result, operand);
      }
      result = operand;
    }
  }
  CHECK_NE(result, nullptr);
  return result;
}
std::optional<int64_t> GetStep(HloInstruction* while_inst) {
  std::optional<int64_t> indvar_tuple_idx =
      GetLoopInductionVarTupleIdx(while_inst);
  if (!indvar_tuple_idx) {
    return std::nullopt;
  };
  auto* while_body_indvar_update =
      while_inst->while_body()->root_instruction()->mutable_operand(
          *indvar_tuple_idx);
  auto* while_body_indvar = NonConstantOperand(while_body_indvar_update);
  HloInstruction* trip_count_increase_step_instr = nullptr;
  if (!Match(while_body_indvar_update,
             match::AddAnyOrder(match::Op().Is(while_body_indvar),
                                match::Op(&trip_count_increase_step_instr)))) {
    return std::nullopt;
  }
  return LiteralUtil::LiteralAsScalarInt64(
      trip_count_increase_step_instr->literal());
}
absl::StatusOr<bool> CollectivePermuteValidIterationAnnotator::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp : module->computations(execution_threads)) {
    for (HloInstruction* inst : comp->instructions()) {
      if (inst->opcode() != HloOpcode::kCollectivePermute) {
        continue;
      }
      if (inst->frontend_attributes().map().find(kSendRecvValidationAttr) !=
          inst->frontend_attributes().map().end()) {
        continue;
      }
      auto sourceTargetPairs = inst->source_target_pairs();
      if (!IsForwardCycle(sourceTargetPairs) &&
          !IsBackwardCycle(sourceTargetPairs)) {
        continue;
      }
      VLOG(2) << "Collective permute with cycle: " << inst->ToString();
      int64_t max_device_num = -1;
      for (auto [source, target] : sourceTargetPairs) {
        max_device_num = std::max(std::max(source, target), max_device_num);
      }
      int64_t num_devices = max_device_num + 1;
      HloInstruction* whileOp = inst->parent()->WhileCallInstruction();
      if (whileOp == nullptr) {
        VLOG(2) << "No surrounding while op found. Ignoring " << inst->name();
        continue;
      }
      if (!whileOp->frontend_attributes().map().contains(
              "is_pipelined_while_loop"))
        continue;
      TF_ASSIGN_OR_RETURN(WhileLoopBackendConfig config,
                          whileOp->backend_config<WhileLoopBackendConfig>());
      if (!config.has_known_trip_count()) {
        VLOG(2) << "Trip count for while loop (" << whileOp->name()
                << "): unknown";
        continue;
      }
      int64_t trip_count = config.known_trip_count().n();
      std::optional<int64_t> step = GetStep(whileOp);
      VLOG(2) << "Trip count for while loop (" << whileOp->name()
              << "): " << trip_count;
      if (!step) {
        VLOG(2) << "Could not find step for while operation";
        continue;
      }
      VLOG(2) << "Step for while loop (" << whileOp->name() << "): " << *step;
      if (*step != 1) {
        VLOG(2) << "Step is not 1. Skipping...";
        continue;
      }
      int64_t offset = trip_count - num_devices;
      std::vector<std::pair<int64_t, int64_t>> sendRecvValidation(
          sourceTargetPairs.size());
      for (size_t currIdx = 0; currIdx < sourceTargetPairs.size(); currIdx++) {
        sendRecvValidation[currIdx] = {currIdx, currIdx + offset};
      }
      if (IsBackwardCycle(sourceTargetPairs)) {
        std::reverse(sendRecvValidation.begin(), sendRecvValidation.end());
      }
      xla::FrontendAttributes attributes;
      std::string iteration_instances =
          "{" +
          absl::StrJoin(sendRecvValidation, ",",
                        [](std::string* out, std::pair<int64_t, int64_t> item) {
                          absl::StrAppend(out, "{", item.first, ",",
                                          item.second, "}");
                        }) +
          "}";
      (*attributes.mutable_map())[kSendRecvValidationAttr] =
          iteration_instances;
      inst->add_frontend_attributes(attributes);
      VLOG(1) << "Adding " << kSendRecvValidationAttr << " to " << inst->name()
              << ": " << iteration_instances;
      changed = true;
    }
  }
  return changed;
}
}  