#include "xla/service/all_reduce_reassociate.h"
#include <cstdint>
#include <optional>
#include <string>
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/all_reduce_key.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
namespace xla {
namespace {
namespace m = match;
bool AreAllreduceKeysEqual(AllReduceKey& key0, AllReduceKey& key1,
                           bool ignore_element_type) {
  if (ignore_element_type) {
    return std::get<0>(key0) == std::get<0>(key1) &&
           std::get<2>(key0) == std::get<2>(key1) &&
           std::get<3>(key0) == std::get<3>(key1) &&
           std::get<4>(key0) == std::get<4>(key1) &&
           std::get<5>(key0) == std::get<5>(key1);
  } else {
    return key0 == key1;
  }
}
bool AreCompatible(const HloAllReduceInstruction* ar0,
                   const HloAllReduceInstruction* ar1, ReductionKind op_kind,
                   bool ignore_element_type) {
  std::optional<AllReduceKey> key0 = GetAllReduceKey(ar0);
  std::optional<AllReduceKey> key1 = GetAllReduceKey(ar1);
  auto kind0 = MatchReductionComputation(ar0->to_apply());
  return key0 && key1 && kind0 &&
         AreAllreduceKeysEqual(*key0, *key1, ignore_element_type) &&
         kind0 == op_kind;
}
HloInstruction* LookThroughForAllReduce(HloInstruction* instr,
                                        const Literal& reduction_identity) {
  if (instr->opcode() == HloOpcode::kDynamicSlice) {
    if (instr->operand(0)->opcode() != HloOpcode::kAllReduce ||
        instr->operand(0)->user_count() != 1 || instr->user_count() != 1) {
      return nullptr;
    }
    return instr;
  }
  while (instr->opcode() != HloOpcode::kAllReduce) {
    if (instr->user_count() != 1) {
      return nullptr;
    }
    if (instr->opcode() != HloOpcode::kReshape &&
        instr->opcode() != HloOpcode::kPad &&
        instr->opcode() != HloOpcode::kSlice &&
        instr->opcode() != HloOpcode::kConvert) {
      return nullptr;
    }
    if (instr->opcode() == HloOpcode::kPad) {
      if (!instr->operand(1)->IsConstant()) {
        return nullptr;
      }
      if (instr->operand(1)->literal() != reduction_identity) {
        return nullptr;
      }
    }
    instr = instr->mutable_operand(0);
  }
  if (instr->user_count() != 1) {
    return nullptr;
  }
  return instr;
}
bool ReassociateAllReduceIsProfitable(HloInstruction* ar0, HloInstruction* ar1,
                                      HloInstruction* reassociated_inst) {
  int64_t pre_reassociated_size = ShapeUtil::ElementsIn(ar0->shape());
  if (ar0 != ar1) {
    pre_reassociated_size += ShapeUtil::ElementsIn(ar1->shape());
  }
  return pre_reassociated_size >=
         ShapeUtil::ElementsIn(reassociated_inst->shape());
}
bool AreCompatibleConverts(const HloInstruction* convert0,
                           const HloInstruction* convert1) {
  bool is_compatible = true;
  if (convert0) {
    is_compatible &= primitive_util::CastPreservesValues(
        convert0->operand(0)->shape().element_type(),
        convert0->shape().element_type());
  }
  if (convert1) {
    is_compatible &= primitive_util::CastPreservesValues(
        convert1->operand(0)->shape().element_type(),
        convert1->shape().element_type());
  }
  if (convert0 && convert1) {
    CHECK(convert0->shape().element_type() == convert1->shape().element_type());
    is_compatible &= convert0->operand(0)->shape().element_type() ==
                     convert1->operand(0)->shape().element_type();
  }
  return is_compatible;
}
template <typename Pattern>
auto OptionalConvertWithOneUser(HloInstruction** optional_convert,
                                Pattern pattern) {
  return m::AnyOf<HloInstruction>(
      m::Convert(optional_convert, pattern).WithOneUser(), std::move(pattern));
}
bool MatchOperandsToAllReduceWithOptionalConvert(HloInstruction* inst,
                                                 HloInstruction** convert0,
                                                 HloInstruction** convert1) {
  auto ar_op_optional_convert_pattern =
      m::Op()
          .WithOperand(0, OptionalConvertWithOneUser(convert0, m::AllReduce()))
          .WithOperand(1, OptionalConvertWithOneUser(convert1, m::AllReduce()))
          .WithPredicate([](const HloInstruction* inst) {
            return inst->shape().IsArray();
          });
  return Match(inst, ar_op_optional_convert_pattern);
}
}  
absl::StatusOr<bool> AllReduceReassociate::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (hlo_query::ContainsLayoutConstrainedAllReduce(*module)) {
    VLOG(1)
        << "Skip AllReduceReassociate because the module contains all-reduce "
           "with constrained layouts";
    return false;
  }
  int64_t next_channel_id = hlo_query::NextChannelId(*module);
  bool changed = false;
  for (auto computation : module->computations(execution_threads)) {
    for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
      std::optional<ReductionKind> kind = MatchReductionInstruction(inst);
      if (!kind) {
        continue;
      }
      std::optional<Literal> reduction_identity =
          GetReductionIdentity(*kind, inst->shape().element_type());
      if (!reduction_identity) {
        continue;
      }
      HloInstruction* lhs = LookThroughForAllReduce(inst->mutable_operand(0),
                                                    *reduction_identity);
      if (lhs == nullptr) {
        continue;
      }
      HloInstruction* rhs = LookThroughForAllReduce(inst->mutable_operand(1),
                                                    *reduction_identity);
      if (rhs == nullptr) {
        continue;
      }
      if (!inst->shape().IsArray()) {
        continue;
      }
      if (lhs->opcode() != rhs->opcode() ||
          (lhs->opcode() == HloOpcode::kDynamicSlice &&
           !ShapeUtil::Compatible(lhs->operand(0)->shape(),
                                  rhs->operand(0)->shape()))) {
        continue;
      }
      HloAllReduceInstruction* ar0 = nullptr;
      HloAllReduceInstruction* ar1 = nullptr;
      bool reduce_scatter_pattern_match = false;
      if (lhs->opcode() == HloOpcode::kDynamicSlice) {
        HloInstruction* original_rhs_operand = rhs->mutable_operand(0);
        TF_RETURN_IF_ERROR(rhs->ReplaceOperandWith(0, lhs->mutable_operand(0)));
        if (!lhs->Identical(*rhs)) {
          TF_RETURN_IF_ERROR(rhs->ReplaceOperandWith(0, original_rhs_operand));
          continue;
        }
        TF_RETURN_IF_ERROR(rhs->ReplaceOperandWith(0, original_rhs_operand));
        ar0 = Cast<HloAllReduceInstruction>(lhs->mutable_operand(0));
        ar1 = Cast<HloAllReduceInstruction>(rhs->mutable_operand(0));
        reduce_scatter_pattern_match = true;
      } else {
        ar0 = Cast<HloAllReduceInstruction>(lhs);
        ar1 = Cast<HloAllReduceInstruction>(rhs);
      }
      if (!ReassociateAllReduceIsProfitable(lhs, rhs, inst)) {
        continue;
      }
      HloInstruction* convert0 = nullptr;
      HloInstruction* convert1 = nullptr;
      if (!MatchOperandsToAllReduceWithOptionalConvert(inst, &convert0,
                                                       &convert1)) {
        VLOG(2) << "One or both inputs are type-converted.";
      }
      bool should_promote_ar = convert0 || convert1;
      if (should_promote_ar) {
        if (!reassociate_converted_ar_) {
          VLOG(2) << "Promotions of all_reduces for reassociation will be "
                     "disabled.";
          continue;
        }
        if (!AreCompatibleConverts(convert0, convert1)) {
          VLOG(2) << "Inputs' Converts are not preserving "
                     "value, skipping";
          continue;
        }
      }
      HloInstruction* op_operand0 = inst->mutable_operand(0);
      HloInstruction* op_operand1 = inst->mutable_operand(1);
      if (convert0) {
        op_operand0 = convert0->mutable_operand(0);
      }
      if (convert1) {
        op_operand1 = convert1->mutable_operand(0);
      }
      if (!AreCompatible(ar0, ar1, *kind,
                         should_promote_ar)) {
        VLOG(2) << "All-Reduce operations are not compatible, skipping";
        continue;
      }
      VLOG(2) << "Reassociated:";
      VLOG(2) << "\tAR0: " << ar0->ToString();
      VLOG(2) << "\tAR1: " << ar1->ToString();
      auto op_users = inst->users();
      HloInstruction* new_op_operand0 = ar0->mutable_operand(0);
      HloInstruction* new_op_operand1 = ar1->mutable_operand(0);
      if (convert0) {
        HloInstruction* ar0_operand = ar0->mutable_operand(0);
        TF_RETURN_IF_ERROR(convert0->ReplaceOperandWith(0, ar0_operand));
        new_op_operand0 = convert0;
      }
      if (convert1) {
        HloInstruction* ar1_operand = ar1->mutable_operand(0);
        TF_RETURN_IF_ERROR(convert1->ReplaceOperandWith(0, ar1_operand));
        new_op_operand1 = convert1;
      }
      HloInstruction* new_op = inst;
      if (should_promote_ar) {
        new_op = computation->AddInstruction(inst->CloneWithNewOperands(
            inst->shape(), {new_op_operand0, new_op_operand1}));
      } else if (reduce_scatter_pattern_match) {
        new_op = computation->AddInstruction(inst->CloneWithNewOperands(
            ar0->shape(), {new_op_operand0, new_op_operand1}));
      }
      Shape new_ar_out_shape = inst->shape();
      CHECK(!should_promote_ar || !reduce_scatter_pattern_match);
      if (should_promote_ar) {
        new_ar_out_shape.set_element_type(
            new_op_operand0->shape().element_type());
      } else if (reduce_scatter_pattern_match) {
        new_ar_out_shape = ar0->shape();
      } else {
        TF_RETURN_IF_ERROR(ar0->ReplaceAllUsesWith(ar0->mutable_operand(0)));
        TF_RETURN_IF_ERROR(ar1->ReplaceAllUsesWith(ar1->mutable_operand(0)));
      }
      HloInstruction* new_ar = computation->AddInstruction(
          ar0->CloneWithNewOperands(new_ar_out_shape, {new_op}));
      if (new_ar->channel_id()) {
        new_ar->set_channel_id(next_channel_id++);
      }
      if (should_promote_ar) {
        HloComputation* to_apply = new_ar->to_apply();
        PrimitiveType type = new_ar->shape().element_type();
        std::string name = absl::StrCat(to_apply->name(), "_reassoc_promoted");
        HloComputation::Builder promoted(name);
        auto x = promoted.AddInstruction(HloInstruction::CreateParameter(
            0, ShapeUtil::MakeShape(type, {}), "x"));
        auto y = promoted.AddInstruction(HloInstruction::CreateParameter(
            1, ShapeUtil::MakeShape(type, {}), "y"));
        promoted.AddInstruction(HloInstruction::CreateBinary(
            ShapeUtil::MakeShape(type, {}),
            to_apply->root_instruction()->opcode(), x, y));
        HloComputation* to_apply_promoted =
            inst->GetModule()->AddEmbeddedComputation(promoted.Build());
        new_ar->set_to_apply(to_apply_promoted);
        TF_RETURN_IF_ERROR(inst->ReplaceAllUsesWith(new_ar));
      } else if (reduce_scatter_pattern_match) {
        auto dyn_slice_operands = lhs->mutable_operands();
        dyn_slice_operands[0] = new_ar;
        HloInstruction* new_dyn_slice = inst->parent()->AddInstruction(
            lhs->CloneWithNewOperands(inst->shape(), dyn_slice_operands));
        TF_RETURN_IF_ERROR(inst->ReplaceUsesWith(op_users, new_dyn_slice));
      } else {
        TF_RETURN_IF_ERROR(inst->ReplaceUsesWith(op_users, new_ar));
      }
      if (should_promote_ar || reduce_scatter_pattern_match) {
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(inst));
      }
      if (reduce_scatter_pattern_match) {
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(lhs));
        if (lhs != rhs) {
          TF_RETURN_IF_ERROR(computation->RemoveInstruction(rhs));
        }
      }
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(ar0));
      if (ar0 != ar1) {
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(ar1));
      }
      changed = true;
    }
  }
  return changed;
}
}  