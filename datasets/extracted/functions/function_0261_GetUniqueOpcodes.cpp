#include "xla/service/gpu/hlo_fusion_stats.h"
#include <set>
#include <string>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "tsl/platform/errors.h"
namespace xla {
namespace gpu {
namespace {
class OpcodeCollector : public ConstDfsHloVisitorWithDefault {
 public:
  std::set<std::string> GetUniqueOpcodes() { return opcodes_; }
 protected:
  absl::Status DefaultAction(const xla::HloInstruction* instr) final {
    switch (instr->opcode()) {
      case HloOpcode::kConstant:
        break;
      case HloOpcode::kParameter:
        break;
      case HloOpcode::kAbs:
      case HloOpcode::kCbrt:
      case HloOpcode::kCeil:
      case HloOpcode::kCos:
      case HloOpcode::kErf:
      case HloOpcode::kExp:
      case HloOpcode::kExpm1:
      case HloOpcode::kFloor:
      case HloOpcode::kLog:
      case HloOpcode::kLog1p:
      case HloOpcode::kLogistic:
      case HloOpcode::kNegate:
      case HloOpcode::kRoundNearestAfz:
      case HloOpcode::kRoundNearestEven:
      case HloOpcode::kRsqrt:
      case HloOpcode::kSign:
      case HloOpcode::kSin:
      case HloOpcode::kSqrt:
      case HloOpcode::kTan:
      case HloOpcode::kTanh:
      case HloOpcode::kAdd:
      case HloOpcode::kAtan2:
      case HloOpcode::kDivide:
      case HloOpcode::kMultiply:
      case HloOpcode::kSubtract:
        opcodes_.insert("cwise");
        break;
      default:
        opcodes_.insert(std::string(HloOpcodeString(instr->opcode())));
    }
    return absl::OkStatus();
  }
 private:
  std::set<std::string> opcodes_;
};
std::set<std::string> GetUniqueOpcodes(HloComputation* computation) {
  OpcodeCollector collector;
  if (!computation->Accept(&collector).ok()) {
    return {};
  }
  return collector.GetUniqueOpcodes();
}
}  
std::string HloOpcodeHistogram::ToString() {
  std::string result;
  for (const auto& entry : *this) {
    absl::StrAppend(&result, "{", absl::StrJoin(entry.first, ", "),
                    "}: ", entry.second, "\n");
  }
  return result;
}
absl::Status HloFusionStatsVisitor::RunOnModule(HloModule* module) {
  TF_RETURN_IF_ERROR(module->entry_computation()->Accept(this));
  return absl::OkStatus();
}
std::string HloFusionStatsVisitor::ToString() {
  return absl::StrCat("HLO Fusion Stats:\n",
                      "Number of fusion ops: ", num_fusions_, "\n",
                      "Number of kLoop fusions: ", num_loop_fusions_, "\n",
                      loop_fusion_opcode_histogram_.ToString(), "\n",
                      "Number of kInput fusions: ", num_input_fusions_, "\n",
                      input_fusion_opcode_histogram_.ToString());
}
absl::Status HloFusionStatsVisitor::DefaultAction(
    const xla::HloInstruction* instr) {
  return absl::OkStatus();
}
absl::Status HloFusionStatsVisitor::HandleFusion(const HloInstruction* fusion) {
  num_fusions_++;
  std::set<std::string> opcodes =
      GetUniqueOpcodes(fusion->fused_instructions_computation());
  if (fusion->fusion_kind() == HloInstruction::FusionKind::kLoop) {
    num_loop_fusions_++;
    loop_fusion_opcode_histogram_[opcodes]++;
  } else if (fusion->fusion_kind() == HloInstruction::FusionKind::kInput) {
    num_input_fusions_++;
    input_fusion_opcode_histogram_[opcodes]++;
  }
  return absl::OkStatus();
}
}  
}  