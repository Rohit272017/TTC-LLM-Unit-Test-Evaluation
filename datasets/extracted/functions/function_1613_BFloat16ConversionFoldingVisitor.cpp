#include "xla/service/bfloat16_conversion_folding.h"
#include <cstdint>
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/float_support.h"
#include "xla/service/hlo_dataflow_analysis.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
namespace xla {
class BFloat16ConversionFoldingVisitor : public DfsHloVisitorWithDefault {
 public:
  explicit BFloat16ConversionFoldingVisitor(
      HloComputation* computation, const FloatSupport* bfloat16_support,
      BFloat16ConversionFolding* bfloat16_conversion_folding)
      : computation_(computation),
        bfloat16_support_(bfloat16_support),
        bfloat16_conversion_folding_(bfloat16_conversion_folding) {}
  absl::Status DefaultAction(HloInstruction* hlo) override;
  absl::Status HandleAllReduce(HloInstruction* crs) override;
  static bool Run(HloComputation* computation,
                  const FloatSupport* bfloat16_support,
                  BFloat16ConversionFolding* bfloat16_conversion_folding) {
    BFloat16ConversionFoldingVisitor visitor(computation, bfloat16_support,
                                             bfloat16_conversion_folding);
    TF_CHECK_OK(computation->Accept(&visitor));
    return visitor.changed_;
  }
 private:
  absl::Status TryFoldBF16Conversions(HloInstruction* hlo);
  absl::Status FoldOutputConversions(HloInstruction* hlo);
  absl::Status FoldOperandConversion(HloInstruction* hlo,
                                     int64_t operand_index);
  HloComputation* computation_;
  const FloatSupport* bfloat16_support_;
  BFloat16ConversionFolding* bfloat16_conversion_folding_;
  bool changed_ = false;
};
absl::Status BFloat16ConversionFoldingVisitor::FoldOutputConversions(
    HloInstruction* hlo) {
  std::vector<HloInstruction*> materialized_users = hlo->users();
  hlo->mutable_shape()->set_element_type(BF16);
  bfloat16_conversion_folding_->UpdateLayout(hlo->mutable_shape());
  for (auto user : materialized_users) {
    CHECK_EQ(user->opcode(), HloOpcode::kConvert);
    TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(hlo));
    changed_ = true;
  }
  return absl::OkStatus();
}
absl::Status BFloat16ConversionFoldingVisitor::FoldOperandConversion(
    HloInstruction* hlo, int64_t operand_index) {
  auto operand = hlo->mutable_operand(operand_index);
  CHECK_EQ(operand->opcode(), HloOpcode::kConvert);
  TF_RETURN_IF_ERROR(
      hlo->ReplaceOperandWith(operand_index, operand->mutable_operand(0)));
  changed_ = true;
  return absl::OkStatus();
}
namespace {
bool AllUsersAreF32ToBF16Converts(const HloInstruction* hlo) {
  if (hlo->user_count() == 0 || hlo->shape().element_type() != F32) {
    return false;
  }
  for (const auto user : hlo->users()) {
    if (user->opcode() == HloOpcode::kConvert &&
        user->shape().element_type() == BF16) {
      continue;
    }
    return false;
  }
  return true;
}
}  
absl::Status BFloat16ConversionFoldingVisitor::TryFoldBF16Conversions(
    HloInstruction* hlo) {
  std::vector<int64_t> bf16_to_f32_operands;
  bool has_other_f32_operands = false;
  for (int64_t i = 0; i < hlo->operands().size(); ++i) {
    auto operand = hlo->operand(i);
    if (operand->shape().element_type() == F32) {
      if (operand->opcode() == HloOpcode::kConvert &&
          operand->operand(0)->shape().element_type() == BF16 &&
          bfloat16_support_->SupportsLowPrecisionOperand(*hlo, i)) {
        bf16_to_f32_operands.push_back(i);
      } else {
        has_other_f32_operands = true;
      }
      continue;
    }
  }
  const bool fold_output_conversion =
      AllUsersAreF32ToBF16Converts(hlo) &&
      bfloat16_support_->SupportsLowPrecisionOutput(*hlo);
  if (!bfloat16_support_->SupportsMixedPrecisions(*hlo)) {
    if (has_other_f32_operands ||
        (!fold_output_conversion && hlo->shape().element_type() == F32)) {
      return absl::OkStatus();
    }
  }
  if (fold_output_conversion) {
    TF_RETURN_IF_ERROR(FoldOutputConversions(hlo));
  }
  for (int64_t i : bf16_to_f32_operands) {
    TF_RETURN_IF_ERROR(FoldOperandConversion(hlo, i));
  }
  return absl::OkStatus();
}
absl::Status BFloat16ConversionFoldingVisitor::DefaultAction(
    HloInstruction* hlo) {
  if (hlo->opcode() == HloOpcode::kTuple ||                      
      hlo->opcode() == HloOpcode::kGetTupleElement ||            
      hlo->opcode() == HloOpcode::kConstant ||                   
      hlo->opcode() == HloOpcode::kParameter ||                  
      hlo->opcode() == HloOpcode::kFusion ||                     
      hlo->opcode() == HloOpcode::kBitcastConvert ||             
      hlo->opcode() == HloOpcode::kConvert ||                    
      hlo->opcode() == HloOpcode::kCall ||                       
      hlo->opcode() == HloOpcode::kCustomCall ||                 
      hlo->opcode() == HloOpcode::kWhile ||                      
      hlo->opcode() == HloOpcode::kConditional ||                
      hlo->opcode() == HloOpcode::kAsyncStart ||                 
      hlo->opcode() == HloOpcode::kAsyncDone ||                  
      HloDataflowAnalysis::IsInPlaceOperation(hlo->opcode()) ||  
      hlo->HasSideEffectNoRecurse()) {
    return absl::OkStatus();
  }
  if (hlo == computation_->root_instruction() &&
      !bfloat16_support_->SupportsMixedPrecisions(*hlo)) {
    return absl::OkStatus();
  }
  return TryFoldBF16Conversions(hlo);
}
absl::Status BFloat16ConversionFoldingVisitor::HandleAllReduce(
    HloInstruction* crs) {
  if (crs->HasSideEffectNoRecurse()) {
    return absl::OkStatus();
  }
  TF_RETURN_IF_ERROR(DefaultAction(crs));
  if (!bfloat16_support_->SupportsMixedPrecisions(*crs)) {
    return absl::OkStatus();
  }
  if (!crs->shape().IsTuple()) {
    return absl::OkStatus();
  }
  if (crs == computation_->root_instruction()) {
    return absl::OkStatus();
  }
  std::vector<std::vector<HloInstruction*>> per_tuple_element_gtes(
      crs->operand_count());
  for (auto user : crs->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      return absl::OkStatus();
    }
    per_tuple_element_gtes[user->tuple_index()].push_back(user);
  }
  for (int64_t i = 0; i < crs->operand_count(); ++i) {
    auto all_gte_users_are_bf16_convert = [&per_tuple_element_gtes, i]() {
      if (per_tuple_element_gtes[i].empty()) {
        return false;
      }
      for (auto gte : per_tuple_element_gtes[i]) {
        if (!AllUsersAreF32ToBF16Converts(gte)) {
          return false;
        }
      }
      return true;
    };
    if (!all_gte_users_are_bf16_convert()) {
      continue;
    }
    ShapeUtil::GetMutableSubshape(crs->mutable_shape(), {i})
        ->set_element_type(BF16);
    bfloat16_conversion_folding_->UpdateLayout(
        ShapeUtil::GetMutableSubshape(crs->mutable_shape(), {i}));
    for (auto gte : per_tuple_element_gtes[i]) {
      TF_RETURN_IF_ERROR(FoldOutputConversions(gte));
    }
  }
  return absl::OkStatus();
}
absl::StatusOr<bool> BFloat16ConversionFolding::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      2, "BFloat16ConversionFolding::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    if (BFloat16ConversionFoldingVisitor::Run(comp, bfloat16_support_, this)) {
      changed = true;
    }
  }
  XLA_VLOG_LINES(
      2, "BFloat16ConversionFolding::Run(), after:\n" + module->ToString());
  return changed;
}
}  