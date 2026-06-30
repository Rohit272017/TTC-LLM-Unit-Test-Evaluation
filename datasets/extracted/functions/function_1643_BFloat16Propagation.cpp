#include "xla/service/bfloat16_propagation.h"
#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/map_util.h"
#include "xla/service/float_support.h"
#include "xla/service/hlo_dataflow_analysis.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_value.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
namespace xla {
BFloat16Propagation::BFloat16Propagation(const FloatSupport* bfloat16_support)
    : bfloat16_support_(bfloat16_support) {
  DCHECK_EQ(bfloat16_support->LowPrecisionType(), BF16);
}
void BFloat16Propagation::DetermineFusionComputationPrecision(
    HloInstruction* fusion) {
  CHECK_EQ(fusion->opcode(), HloOpcode::kFusion);
  if (!bfloat16_support_->SupportsMixedPrecisions(*fusion)) {
    return;
  }
  auto root = fusion->fused_instructions_computation()->root_instruction();
  ShapeUtil::ForEachSubshape(
      root->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (subshape.element_type() != F32) {
          return;
        }
        if (OutputTypeAfterChange(fusion, index) == BF16) {
          AddToOrRemoveFromBF16ChangeSet(root, index, BF16);
          VLOG(2) << "Fused root " << root->ToString() << " at shape index "
                  << index << " changed to BF16 precision for fusion "
                  << fusion->ToString();
        }
      });
  auto insts =
      fusion->fused_instructions_computation()->MakeInstructionPostOrder();
  for (auto inst_it = insts.rbegin(); inst_it != insts.rend(); ++inst_it) {
    DetermineInstructionPrecision(*inst_it, false);
  }
  computations_visited_in_backward_pass_.insert(
      fusion->fused_instructions_computation());
  RevertIfFusionInternalBF16Changes(fusion);
}
void BFloat16Propagation::RevertIfFusionInternalBF16Changes(
    HloInstruction* fusion) {
  auto has_changes = [this](HloInstruction* inst) {
    auto it = changes_to_bf16_.find(inst);
    return it != changes_to_bf16_.end() && !it->second.empty();
  };
  auto root = fusion->fused_instructions_computation()->root_instruction();
  absl::flat_hash_set<const HloValue*> changed_root_buffers;
  auto root_changes_it = changes_to_bf16_.find(root);
  if (root_changes_it != changes_to_bf16_.end()) {
    for (const auto& entry : root_changes_it->second) {
      for (const HloValue* value :
           dataflow_->GetValueSet(root, entry.second).values()) {
        changed_root_buffers.insert(value);
      }
    }
  }
  auto aliases_changed_root_buffer = [this, &changed_root_buffers](
                                         const HloInstruction* inst) {
    bool aliasing = false;
    ShapeUtil::ForEachSubshape(inst->shape(), [&](const Shape& subshape,
                                                  const ShapeIndex& index) {
      if (aliasing) {
        return;
      }
      if (subshape.element_type() != F32) {
        return;
      }
      aliasing = absl::c_any_of(dataflow_->GetValueSet(inst, index).values(),
                                IsValueIn(changed_root_buffers));
    });
    return aliasing;
  };
  for (auto inst :
       fusion->fused_instructions_computation()->MakeInstructionPostOrder()) {
    if (inst->opcode() == HloOpcode::kParameter) {
      continue;
    }
    if (aliases_changed_root_buffer(inst)) {
      continue;
    }
    if (inst->opcode() == HloOpcode::kFusion) {
      bool parameter_reverted = false;
      for (int64_t i = 0; i < inst->operand_count(); ++i) {
        if (has_changes(inst->mutable_operand(i))) {
          continue;
        }
        auto* fused_parameter = inst->fused_parameter(i);
        if (has_changes(fused_parameter)) {
          changes_to_bf16_.erase(fused_parameter);
          parameter_reverted = true;
        }
      }
      if (parameter_reverted) {
        RevertIfFusionInternalBF16Changes(inst);
      }
    }
    if (!has_changes(inst)) {
      continue;
    }
    bool revert_changes = true;
    for (auto operand : inst->operands()) {
      if (has_changes(operand)) {
        revert_changes = false;
        break;
      }
    }
    if (revert_changes) {
      changes_to_bf16_.erase(inst);
    }
  }
}
void BFloat16Propagation::DetermineWhileComputationsPrecision(
    HloInstruction* while_hlo) {
  CHECK_EQ(while_hlo->opcode(), HloOpcode::kWhile);
  HloComputation* body = while_hlo->while_body();
  auto body_root = body->root_instruction();
  HloComputation* condition = while_hlo->while_condition();
  ShapeUtil::ForEachSubshape(
      body_root->shape(), [this, while_hlo, body_root](
                              const Shape& subshape, const ShapeIndex& index) {
        if (subshape.element_type() != F32) {
          return;
        }
        if (OutputTypeAfterChange(while_hlo, index) == BF16) {
          AddToOrRemoveFromBF16ChangeSet(body_root, index, BF16);
          VLOG(2) << "While body root " << body_root->ToString()
                  << " at shape index " << index
                  << " changed to BF16 precision for while "
                  << while_hlo->ToString();
        }
      });
  auto body_insts = body->MakeInstructionPostOrder();
  for (auto inst_it = body_insts.rbegin(); inst_it != body_insts.rend();
       ++inst_it) {
    DetermineInstructionPrecision(*inst_it, false);
  }
  computations_visited_in_backward_pass_.insert(body);
  auto condition_insts = condition->MakeInstructionPostOrder();
  for (auto inst_it = condition_insts.rbegin();
       inst_it != condition_insts.rend(); ++inst_it) {
    DetermineInstructionPrecision(*inst_it, false);
  }
  computations_visited_in_backward_pass_.insert(condition);
}
void BFloat16Propagation::DetermineConditionalComputationsPrecision(
    HloInstruction* cond) {
  CHECK_EQ(cond->opcode(), HloOpcode::kConditional);
  for (int64_t i = 0; i < cond->branch_count(); ++i) {
    auto branch = cond->branch_computation(i);
    auto root = branch->root_instruction();
    ShapeUtil::ForEachSubshape(
        root->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
          if (subshape.element_type() != F32) {
            return;
          }
          if (OutputTypeAfterChange(cond, index) == BF16) {
            AddToOrRemoveFromBF16ChangeSet(root, index, BF16);
            VLOG(2) << "Conditional branch " << i << " root "
                    << root->ToString() << " at shape index " << index
                    << " changed to BF16 precision for conditional "
                    << cond->ToString();
          }
        });
    auto insts = branch->MakeInstructionPostOrder();
    for (auto inst_it = insts.rbegin(); inst_it != insts.rend(); ++inst_it) {
      DetermineInstructionPrecision(*inst_it, false);
    }
    computations_visited_in_backward_pass_.insert(branch);
  }
}
bool BFloat16Propagation::AllUsersConsumeBF16(const HloInstruction& hlo,
                                              const ShapeIndex& index) const {
  const Shape& subshape = ShapeUtil::GetSubshape(hlo.shape(), index);
  if (subshape.element_type() != BF16 && subshape.element_type() != F32) {
    return false;
  }
  auto& value_set = dataflow_->GetValueSet(&hlo, index);
  for (const HloValue* value : value_set.values()) {
    if (ContainsKey(values_that_must_be_kept_as_f32_, value)) {
      return false;
    }
    if (value->shape().element_type() == BF16) {
      continue;
    }
    for (const HloUse& use : value->GetUses()) {
      if (!ContainsKey(instructions_visited_in_backward_pass_,
                       use.instruction)) {
        continue;
      }
      if (use.instruction->HasSideEffectNoRecurse()) {
        return false;
      }
      if (use.instruction->opcode() == HloOpcode::kFusion) {
        auto* fused_parameter =
            use.instruction->fused_parameter(use.operand_number);
        if (OutputTypeAfterChange(fused_parameter, use.operand_index) != BF16) {
          return false;
        }
        continue;
      } else if (use.instruction->opcode() == HloOpcode::kWhile) {
        auto* cond_parameter =
            use.instruction->while_condition()->parameter_instruction(
                use.operand_number);
        if (OutputTypeAfterChange(cond_parameter, use.operand_index) != BF16) {
          return false;
        }
        auto* body_parameter =
            use.instruction->while_body()->parameter_instruction(
                use.operand_number);
        if (OutputTypeAfterChange(body_parameter, use.operand_index) != BF16) {
          return false;
        }
        continue;
      } else if (use.instruction->opcode() == HloOpcode::kConditional) {
        auto* cond_parameter =
            use.instruction->branch_computation(use.operand_number - 1)
                ->parameter_instruction(0);
        if (OutputTypeAfterChange(cond_parameter, use.operand_index) != BF16) {
          return false;
        }
        continue;
      }
      if (bfloat16_support_->EffectiveOperandPrecisionIsLowPrecision(
              *use.instruction, use.operand_number)) {
        continue;
      }
      if (bfloat16_support_->EffectiveOperandPrecisionIsOutputPrecision(
              *use.instruction, use.operand_number)) {
        if (use.instruction->opcode() == HloOpcode::kTuple ||
            (use.instruction->opcode() == HloOpcode::kAllReduce &&
             use.instruction->shape().IsTuple())) {
          ShapeIndex use_output_index{use.operand_number};
          for (int64_t i : use.operand_index) {
            use_output_index.push_back(i);
          }
          if (OutputTypeAfterChange(use.instruction, use_output_index) ==
              BF16) {
            continue;
          }
        } else if (use.instruction->opcode() == HloOpcode::kGetTupleElement) {
          ShapeIndex use_output_index;
          for (int64_t i = 1; i < use.operand_index.size(); ++i) {
            use_output_index.push_back(use.operand_index[i]);
          }
          if (OutputTypeAfterChange(use.instruction, use_output_index) ==
              BF16) {
            continue;
          }
        } else {
          if (OutputTypeAfterChange(use.instruction, use.operand_index) ==
              BF16) {
            continue;
          }
        }
      }
      return false;
    }
  }
  return true;
}
bool BFloat16Propagation::ShouldKeepPrecisionUnchanged(
    const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kFusion &&
      inst->fusion_kind() == HloInstruction::FusionKind::kCustom) {
    return ShouldKeepPrecisionUnchanged(
        inst->fused_instructions_computation()->root_instruction());
  }
  return (inst->opcode() == HloOpcode::kCustomCall &&
          !inst->IsCustomCall("AllocateBuffer")) ||
         inst->opcode() == HloOpcode::kCall ||
         inst->opcode() == HloOpcode::kBitcastConvert ||
         inst->HasSideEffectNoRecurse();
}
void BFloat16Propagation::DetermineInstructionPrecision(HloInstruction* hlo,
                                                        bool skip_parameters) {
  bool postpone_processing_called_computations = false;
  absl::Cleanup cleaner = [this, hlo,
                           &postpone_processing_called_computations] {
    if (!postpone_processing_called_computations) {
      if (hlo->opcode() == HloOpcode::kFusion) {
        DetermineFusionComputationPrecision(hlo);
      } else if (hlo->opcode() == HloOpcode::kWhile) {
        DetermineWhileComputationsPrecision(hlo);
      } else if (hlo->opcode() == HloOpcode::kConditional) {
        DetermineConditionalComputationsPrecision(hlo);
      }
    }
    instructions_visited_in_backward_pass_.insert(hlo);
  };
  if (hlo->opcode() == HloOpcode::kWhile &&
      (caller_counts_[hlo->while_condition()] > 1 ||
       caller_counts_[hlo->while_body()] > 1)) {
    postpone_processing_called_computations = true;
    return;
  }
  if (hlo->opcode() == HloOpcode::kConditional &&
      absl::c_any_of(hlo->branch_computations(), [&](const HloComputation* c) {
        return caller_counts_[c] > 1;
      })) {
    postpone_processing_called_computations = true;
    return;
  }
  CHECK(hlo->parent() != nullptr);
  if (hlo == hlo->parent()->root_instruction()) {
    if (!hlo->parent()->IsFusionComputation()) {
      ShapeUtil::ForEachSubshape(hlo->shape(), [&](const Shape& ,
                                                   const ShapeIndex& index) {
        if (OutputTypeAfterChange(hlo, index) != F32) {
          return;
        }
        for (const auto* value : dataflow_->GetValueSet(hlo, index).values()) {
          values_that_must_be_kept_as_f32_.insert(value);
        }
      });
    }
    return;
  }
  if (ShouldKeepPrecisionUnchanged(hlo) ||
      (hlo->opcode() == HloOpcode::kParameter && skip_parameters)) {
    return;
  }
  if (!ContainsKey(consider_using_bfloat16_, hlo)) {
    return;
  }
  if (!bfloat16_support_->SupportsLowPrecisionOutput(*hlo)) {
    return;
  }
  ShapeUtil::ForEachSubshape(
      hlo->shape(),
      [hlo, this](const Shape& , const ShapeIndex& index) {
        if (OutputTypeAfterChange(hlo, index) == F32 &&
            AllUsersConsumeBF16(*hlo, index)) {
          AddToOrRemoveFromBF16ChangeSet(hlo, index, BF16);
          VLOG(2) << "HloInstruction output at shape index " << index
                  << " changed to BF16 precision: " << hlo->ToString();
        }
      });
}
bool BFloat16Propagation::InstructionIsCandidateForBF16Output(
    HloInstruction* hlo) {
  if (!bfloat16_support_->SupportsMixedPrecisions(*hlo) &&
      hlo->opcode() != HloOpcode::kTuple &&
      hlo->opcode() != HloOpcode::kGetTupleElement &&
      hlo->opcode() != HloOpcode::kDomain &&
      hlo->shape().element_type() != BF16) {
    for (int64_t i = 0; i < hlo->operand_count(); ++i) {
      if (!bfloat16_support_->EffectiveOperandPrecisionIsOutputPrecision(*hlo,
                                                                         i) ||
          !ContainsKey(consider_using_bfloat16_, hlo->operand(i))) {
        return false;
      }
    }
  }
  return true;
}
void BFloat16Propagation::AdjustCalledComputationParameters(
    HloInstruction* hlo) {
  auto adjust_computation = [this, hlo](
                                HloComputation* computation,
                                absl::Span<HloInstruction* const> operands) {
    CHECK_EQ(operands.size(), computation->num_parameters());
    for (int64_t i = 0; i < operands.size(); ++i) {
      auto parameter = computation->parameter_instruction(i);
      ShapeUtil::ForEachSubshape(
          parameter->shape(),
          [this, i, hlo, &operands, parameter](const Shape& ,
                                               const ShapeIndex& index) {
            if (!ShapeUtil::IsLeafIndex(parameter->shape(), index)) {
              return;
            }
            PrimitiveType operand_type =
                OutputTypeAfterChange(operands[i], index);
            if (OutputTypeAfterChange(parameter, index) == operand_type) {
              return;
            }
            AddToOrRemoveFromBF16ChangeSet(parameter, index, operand_type);
            VLOG(2) << "Called computation parameter " << parameter->ToString()
                    << " at shape index " << index << " adjusted to "
                    << (operand_type == BF16 ? "BF16" : "F32")
                    << " to match operand in HLO " << hlo->ToString();
          });
    }
  };
  switch (hlo->opcode()) {
    case HloOpcode::kFusion:
      adjust_computation(hlo->fused_instructions_computation(),
                         hlo->operands());
      break;
    case HloOpcode::kWhile:
      adjust_computation(hlo->while_condition(), hlo->operands());
      adjust_computation(hlo->while_body(), hlo->operands());
      break;
    case HloOpcode::kConditional:
      for (int64_t i = 0; i < hlo->branch_count(); ++i) {
        adjust_computation(hlo->branch_computation(i),
                           {hlo->mutable_operand(i + 1)});
      }
      break;
    default:
      break;
  }
}
void BFloat16Propagation::AdjustCalledComputationRoot(HloInstruction* hlo) {
  auto adjust_computation = [this, hlo](HloComputation* computation,
                                        HloInstruction* output) {
    HloInstruction* root = computation->root_instruction();
    ShapeUtil::ForEachSubshape(root->shape(), [this, hlo, root, output](
                                                  const Shape& ,
                                                  const ShapeIndex& index) {
      if (!ShapeUtil::IsLeafIndex(hlo->shape(), index)) {
        return;
      }
      const PrimitiveType output_type = OutputTypeAfterChange(output, index);
      if (OutputTypeAfterChange(root, index) == output_type) {
        return;
      }
      AddToOrRemoveFromBF16ChangeSet(root, index, output_type);
      if (output_type == F32) {
        for (const auto* value : dataflow_->GetValueSet(root, index).values()) {
          values_that_must_be_kept_as_f32_.insert(value);
        }
      }
      VLOG(2) << "Called computation root " << root->ToString()
              << " at shape index " << index << " adjusted to "
              << (output_type == BF16 ? "BF16" : "F32")
              << " to match output shape of " << hlo->ToString();
    });
  };
  switch (hlo->opcode()) {
    case HloOpcode::kFusion:
      adjust_computation(hlo->fused_instructions_computation(), hlo);
      break;
    case HloOpcode::kWhile:
      adjust_computation(hlo->while_body(), hlo);
      break;
    case HloOpcode::kConditional:
      for (auto* branch : hlo->branch_computations()) {
        adjust_computation(branch, hlo);
      }
      break;
    default:
      break;
  }
}
bool BFloat16Propagation::ResolveInconsistencyOfAliasingBuffersHelper(
    HloComputation* computation,
    absl::flat_hash_set<const HloComputation*>* visited_computations) {
  bool parameter_changed = false;
  auto insts = computation->MakeInstructionPostOrder();
  while (true) {
    bool any_change = false;
    for (auto inst_it = insts.rbegin(); inst_it != insts.rend(); ++inst_it) {
      auto hlo = *inst_it;
      auto adjust_hlo_output = [&](const Shape& ,
                                   const ShapeIndex& index) {
        const PrimitiveType output_type = OutputTypeAfterChange(hlo, index);
        VLOG(2) << "output_type is " << ((output_type == BF16) ? "BF16" : "F32")
                << " for :" << hlo->ToString() << "\n";
        if (output_type != F32 && output_type != BF16) {
          return;
        }
        PrimitiveType type = BF16;
        for (const auto* value : dataflow_->GetValueSet(hlo, index).values()) {
          auto value_type = ValueTypeAfterChange(value);
          if (value_type == BF16) {
            continue;
          }
          VLOG(2) << "Adjust to F32 due to aliased dataflow value: "
                  << value->ToString() << "\n";
          CHECK_EQ(value_type, F32);
          type = F32;
          break;
        }
        for (const auto& operand_and_output_index :
             HloDataflowAnalysis::GetInPlaceInputOutputPairs(hlo)) {
          if (operand_and_output_index.second == index) {
            const HloOperandIndex& operand_index =
                operand_and_output_index.first;
            for (const auto* value :
                 dataflow_
                     ->GetValueSet(hlo->operand(operand_index.operand_number),
                                   operand_index.operand_index)
                     .values()) {
              auto value_type = ValueTypeAfterChange(value);
              if (value_type == BF16) {
                continue;
              }
              VLOG(2) << "Adjust to F32 due to InputOutPair: "
                      << value->ToString() << "\n";
              CHECK_EQ(value_type, F32);
              type = F32;
              break;
            }
          }
        }
        if (type == BF16 && !AllUsersConsumeBF16(*hlo, index)) {
          VLOG(2) << "Adjust to F32 due to All user consumeBF16 fail\n";
          type = F32;
        }
        if (type == F32) {
          for (const auto* value :
               dataflow_->GetValueSet(hlo, index).values()) {
            values_that_must_be_kept_as_f32_.insert(value);
          }
        }
        if (type != output_type) {
          any_change = true;
          AddToOrRemoveFromBF16ChangeSet(hlo, index, type);
          VLOG(2) << "HloInstruction output at shape index " << index
                  << " adjusted to " << (type == BF16 ? "BF16" : "F32") << ": "
                  << hlo->ToString();
          if (hlo->opcode() == HloOpcode::kParameter) {
            parameter_changed = true;
          }
        }
      };
      ShapeUtil::ForEachSubshape(hlo->shape(), adjust_hlo_output);
      AdjustCalledComputationRoot(hlo);
      if (hlo->opcode() == HloOpcode::kWhile) {
        absl::flat_hash_set<const HloComputation*> visited_in_while;
        while (ResolveInconsistencyOfAliasingBuffersHelper(
                   hlo->while_condition(), &visited_in_while) ||
               ResolveInconsistencyOfAliasingBuffersHelper(hlo->while_body(),
                                                           &visited_in_while)) {
          visited_in_while.clear();
          ShapeUtil::ForEachSubshape(hlo->shape(), adjust_hlo_output);
          AdjustCalledComputationRoot(hlo);
        }
        visited_computations->insert(visited_in_while.begin(),
                                     visited_in_while.end());
      } else if (hlo->opcode() == HloOpcode::kFusion) {
        ResolveInconsistencyOfAliasingBuffersHelper(
            hlo->fused_instructions_computation(), visited_computations);
      } else if (hlo->opcode() == HloOpcode::kConditional) {
        for (auto* branch : hlo->branch_computations()) {
          ResolveInconsistencyOfAliasingBuffersHelper(branch,
                                                      visited_computations);
        }
      }
    }
    if (!any_change) {
      break;
    }
  }
  for (auto inst_it = insts.rbegin(); inst_it != insts.rend(); ++inst_it) {
    AdjustCalledComputationParameters(*inst_it);
  }
  return parameter_changed;
}
void BFloat16Propagation::ResolveInconsistencyOfAliasingBuffers(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  const auto& computations_topological_order =
      module->MakeComputationPostOrder(execution_threads);
  absl::flat_hash_set<const HloComputation*> resolved;
  for (auto comp_it = computations_topological_order.rbegin();
       comp_it != computations_topological_order.rend(); ++comp_it) {
    if (ContainsKey(resolved, *comp_it)) {
      continue;
    }
    ResolveInconsistencyOfAliasingBuffersHelper(*comp_it, &resolved);
  }
}
absl::Status BFloat16Propagation::ResolveInconsistentFusions(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  for (auto computation : module->MakeComputationPostOrder(execution_threads)) {
    auto insts = computation->MakeInstructionPostOrder();
    for (auto inst_it = insts.rbegin(); inst_it != insts.rend(); ++inst_it) {
      auto hlo = *inst_it;
      if (hlo->opcode() != HloOpcode::kFusion) {
        continue;
      }
      auto fusion_computation = hlo->fused_instructions_computation();
      auto fusion_root = fusion_computation->root_instruction();
      if (ShapeUtil::Compatible(fusion_root->shape(), hlo->shape())) {
        continue;
      }
      ShapeTree<HloInstruction*> converted_outputs(hlo->shape());
      TF_ASSIGN_OR_RETURN(
          HloInstruction * copy,
          fusion_computation->DeepCopyInstructionWithCustomCopier(
              fusion_root,
              [hlo](HloInstruction* leaf, const ShapeIndex& leaf_index,
                    HloComputation* comp) {
                const Shape& hlo_subshape =
                    ShapeUtil::GetSubshape(hlo->shape(), leaf_index);
                if (ShapeUtil::Compatible(leaf->shape(), hlo_subshape)) {
                  return leaf;
                }
                return comp->AddInstruction(
                    HloInstruction::CreateConvert(hlo_subshape, leaf));
              }));
      fusion_computation->set_root_instruction(copy);
    }
  }
  return absl::OkStatus();
}
absl::Status BFloat16Propagation::ResolveConvertedConstants(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  for (auto computation : module->MakeComputationPostOrder(execution_threads)) {
    for (auto hlo : computation->MakeInstructionPostOrder()) {
      if (hlo->opcode() != HloOpcode::kConstant) {
        continue;
      }
      if (!Shape::Equal().MinorToMajorOnlyInLayout()(hlo->literal().shape(),
                                                     hlo->shape())) {
        TF_ASSIGN_OR_RETURN(auto converted_literal,
                            hlo->literal().ConvertToShape(hlo->shape()));
        auto new_constant = computation->AddInstruction(
            HloInstruction::CreateConstant(std::move(converted_literal)));
        UpdateLayout(new_constant->mutable_shape());
        TF_RETURN_IF_ERROR(hlo->ReplaceAllUsesWith(new_constant));
      }
    }
  }
  return absl::OkStatus();
}
absl::Status BFloat16Propagation::SkipNoopConversions(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  for (auto computation : module->computations(execution_threads)) {
    for (auto hlo : computation->MakeInstructionPostOrder()) {
      if (hlo->opcode() != HloOpcode::kConvert) {
        continue;
      }
      auto source = hlo->mutable_operand(0);
      if (!ShapeUtil::Equal(source->shape(), hlo->shape())) {
        continue;
      }
      const bool is_root = hlo == computation->root_instruction();
      TF_RETURN_IF_ERROR(hlo->ReplaceAllUsesWith(source));
      if (is_root) {
        computation->set_root_instruction(source);
      }
    }
  }
  return absl::OkStatus();
}
absl::StatusOr<bool> BFloat16Propagation::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  consider_using_bfloat16_.clear();
  instructions_visited_in_backward_pass_.clear();
  computations_visited_in_backward_pass_.clear();
  values_that_must_be_kept_as_f32_.clear();
  caller_counts_.clear();
  changes_to_bf16_.clear();
  changed_ = false;
  auto computations_topological_order =
      module->MakeComputationPostOrder(execution_threads);
  for (auto computation : computations_topological_order) {
    for (auto inst : computation->MakeInstructionPostOrder()) {
      if (inst->opcode() != HloOpcode::kWhile) {
        continue;
      }
      auto operand = inst->mutable_operand(0);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * copy,
          computation->DeepCopyInstructionWithCustomCopier(
              operand, [](HloInstruction* leaf, const ShapeIndex& leaf_index,
                          HloComputation* comp) {
                if (leaf->shape().element_type() != F32) {
                  return leaf;
                }
                return comp->AddInstruction(
                    HloInstruction::CreateConvert(leaf->shape(), leaf));
              }));
      TF_RETURN_IF_ERROR(operand->ReplaceUseWith(inst, copy));
    }
  }
  TF_ASSIGN_OR_RETURN(dataflow_, HloDataflowAnalysis::Run(*module));
  for (auto computation : computations_topological_order) {
    for (auto inst : computation->MakeInstructionPostOrder()) {
      if (InstructionIsCandidateForBF16Output(inst)) {
        consider_using_bfloat16_.insert(inst);
      }
    }
  }
  for (auto comp_it = computations_topological_order.rbegin();
       comp_it != computations_topological_order.rend(); ++comp_it) {
    if (ContainsKey(computations_visited_in_backward_pass_, *comp_it)) {
      continue;
    }
    auto insts = (*comp_it)->MakeInstructionPostOrder();
    for (auto inst_it = insts.rbegin(); inst_it != insts.rend(); ++inst_it) {
      DetermineInstructionPrecision(*inst_it,
                                    true);
    }
    computations_visited_in_backward_pass_.insert(*comp_it);
  }
  ResolveInconsistencyOfAliasingBuffers(module, execution_threads);
  for (auto& change : changes_to_bf16_) {
    auto inst = change.first;
    if (ShouldKeepPrecisionUnchanged(inst)) {
      auto users = inst->users();
      bool is_root = inst == inst->parent()->root_instruction();
      TF_ASSIGN_OR_RETURN(
          HloInstruction * copy,
          inst->parent()->DeepCopyInstructionWithCustomCopier(
              inst, [&](HloInstruction* leaf, const ShapeIndex& leaf_index,
                        HloComputation* comp) {
                if (!ContainsKey(change.second,
                                 ShapeUtil::GetMutableSubshape(
                                     inst->mutable_shape(), leaf_index))) {
                  return leaf;
                }
                auto converted_shape =
                    ShapeUtil::ChangeElementType(leaf->shape(), BF16);
                UpdateLayout(&converted_shape);
                return comp->AddInstruction(
                    HloInstruction::CreateConvert(converted_shape, leaf));
              }));
      for (auto user : users) {
        TF_RETURN_IF_ERROR(inst->ReplaceUseWithDifferentShape(user, copy));
      }
      if (is_root) {
        inst->parent()->set_root_instruction(copy,
                                             true);
      }
      continue;
    }
    for (const auto& entry : change.second) {
      auto subshape = entry.first;
      CHECK_EQ(subshape->element_type(), F32);
      subshape->set_element_type(BF16);
      UpdateLayout(subshape);
      changed_ = true;
    }
  }
  auto clean_up = [this, module, &execution_threads]() {
    TF_RETURN_IF_ERROR(SkipNoopConversions(module, execution_threads));
    TupleSimplifier tuple_simplifier;
    TF_RETURN_IF_ERROR(
        tuple_simplifier.Run(module, execution_threads).status());
    HloDCE dce;
    TF_RETURN_IF_ERROR(dce.Run(module, execution_threads).status());
    return absl::OkStatus();
  };
  if (!changed_) {
    TF_RETURN_IF_ERROR(clean_up());
    return false;
  }
  TF_RETURN_IF_ERROR(ResolveInconsistentFusions(module, execution_threads));
  TF_RETURN_IF_ERROR(ResolveConvertedConstants(module, execution_threads));
  TF_RETURN_IF_ERROR(clean_up());
  return true;
}
PrimitiveType BFloat16Propagation::OutputTypeAfterChange(
    HloInstruction* hlo, const ShapeIndex& index) const {
  Shape* subshape = ShapeUtil::GetMutableSubshape(hlo->mutable_shape(), index);
  const PrimitiveType type_on_hlo = subshape->element_type();
  if (type_on_hlo != F32) {
    return type_on_hlo;
  }
  auto it = changes_to_bf16_.find(hlo);
  if (it == changes_to_bf16_.end()) {
    return type_on_hlo;
  }
  return ContainsKey(it->second, subshape) ? BF16 : F32;
}
PrimitiveType BFloat16Propagation::ValueTypeAfterChange(
    const HloValue* value) const {
  auto hlo = value->defining_instruction();
  const auto& position = value->defining_position();
  return OutputTypeAfterChange(hlo, position.index);
}
void BFloat16Propagation::AddToOrRemoveFromBF16ChangeSet(
    HloInstruction* hlo, const ShapeIndex& index, PrimitiveType target_type) {
  if (target_type == BF16) {
    auto& entry = changes_to_bf16_[hlo];
    entry.emplace(ShapeUtil::GetMutableSubshape(hlo->mutable_shape(), index),
                  index);
  } else {
    CHECK_EQ(target_type, F32);
    auto it = changes_to_bf16_.find(hlo);
    if (it == changes_to_bf16_.end()) {
      return;
    }
    it->second.erase(
        ShapeUtil::GetMutableSubshape(hlo->mutable_shape(), index));
    if (it->second.empty()) {
      changes_to_bf16_.erase(it);
    }
  }
}
}  