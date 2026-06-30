#include "xla/service/reshape_mover.h"
#include <algorithm>
#include <memory>
#include <vector>
#include "absl/algorithm/container.h"
#include "xla/permutation_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
namespace xla {
namespace {
bool IsRearrange(const HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kReshape ||
         instruction->opcode() == HloOpcode::kTranspose;
}
bool AreEquivalentRearranges(const HloInstruction* a, const HloInstruction* b) {
  if (a->opcode() != b->opcode() ||
      !ShapeUtil::SameDimensions(a->shape(), b->shape())) {
    return false;
  }
  switch (a->opcode()) {
    case HloOpcode::kTranspose:
      return a->dimensions() == b->dimensions();
    case HloOpcode::kReshape:
      return ShapeUtil::SameDimensions(a->operand(0)->shape(),
                                       b->operand(0)->shape());
    default:
      return false;
  }
}
absl::InlinedVector<int64_t, 4> TransposedBcastDims(
    absl::Span<const int64_t> bcast_dims,
    absl::Span<const int64_t> transpose_dims) {
  auto inv_perm = InversePermutation(transpose_dims);
  absl::InlinedVector<int64_t, 4> new_bcast_dims;
  for (int64_t dim : bcast_dims) {
    new_bcast_dims.push_back(inv_perm[dim]);
  }
  return new_bcast_dims;
}
}  
bool ReshapeMover::CanTriviallyRearrange(const HloInstruction* instr,
                                         const HloInstruction* rearrange) {
  CHECK(IsRearrange(rearrange)) << rearrange->ToString();
  if (rearrange->opcode() == HloOpcode::kReshape &&
      ShapeUtil::Equal(rearrange->shape(), rearrange->operand(0)->shape())) {
    return true;
  }
  if (rearrange->opcode() == HloOpcode::kTranspose &&
      IsIdentityPermutation(rearrange->dimensions())) {
    return true;
  }
  if (instr->opcode() == HloOpcode::kConstant) {
    return true;
  }
  if (instr->opcode() == HloOpcode::kRng && instr->user_count() == 1) {
    return true;
  }
  if (instr->opcode() == HloOpcode::kBroadcast) {
    if (!absl::c_is_sorted(instr->dimensions())) {
      return false;
    }
    if (rearrange->opcode() == HloOpcode::kReshape) {
      return ShapeUtil::IsScalar(instr->operand(0)->shape()) ||
             (options_.reshape_of_1d_broadcast_is_cheap &&
              ShapeUtil::TrueRank(instr->operand(0)->shape()) <= 1) ||
             (options_.reshape_of_1d_broadcast_is_cheap &&
              ShapeUtil::ReshapeLeavesDimensionsUnmodified(
                  rearrange->shape(),
                  rearrange->operand(0)->shape(),
                  instr->dimensions())
                  .has_value());
    }
    if (rearrange->opcode() == HloOpcode::kTranspose) {
      return absl::c_is_sorted(TransposedBcastDims(
          instr->dimensions(), InversePermutation(rearrange->dimensions())));
    }
  }
  return false;
}
const HloInstruction* ReshapeMover::FirstNontrivialRearrange(
    absl::Span<const HloInstruction* const> instrs) {
  auto rearrange_it = absl::c_find_if(instrs, [&](const HloInstruction* instr) {
    return IsRearrange(instr) &&
           !CanTriviallyRearrange(instr->operand(0), instr);
  });
  if (rearrange_it == instrs.end()) {
    return nullptr;
  }
  return *rearrange_it;
}
bool ReshapeMover::IsReshapeMoveCandidate(HloInstruction* instruction) {
  auto print_no_metadata = HloPrintOptions().set_print_metadata(false);
  VLOG(5) << "** Checking instruction: "
          << instruction->ToString(print_no_metadata);
  if (!instruction->IsElementwise()) {
    return false;
  }
  const HloInstruction* rearrange =
      FirstNontrivialRearrange(instruction->operands());
  if (rearrange == nullptr) {
    return false;
  }
  return absl::c_all_of(
      instruction->operands(), [&](const HloInstruction* operand) {
        return (IsRearrange(operand) &&
                AreEquivalentRearranges(operand, rearrange)) ||
               (!IsRearrange(operand) &&
                CanTriviallyRearrange(operand, rearrange));
      });
}
absl::StatusOr<HloInstruction*> ReshapeMover::ApplyInverseRearrange(
    const HloInstruction* rearrange, HloInstruction* operand) {
  switch (rearrange->opcode()) {
    case HloOpcode::kReshape: {
      Shape new_shape = ShapeUtil::ChangeElementType(
          rearrange->operand(0)->shape(), operand->shape().element_type());
      if (operand->shape() != new_shape) {
        return MakeReshapeHlo(new_shape, operand);
      } else {
        return operand;
      }
    }
    case HloOpcode::kTranspose: {
      if (!IsIdentityPermutation(rearrange->dimensions())) {
        return MakeTransposeHlo(operand,
                                InversePermutation(rearrange->dimensions()));
      } else {
        return operand;
      }
    }
    default:
      LOG(FATAL) << "Invalid rearrange op: " << rearrange->ToString();
  }
}
absl::StatusOr<bool> ReshapeMover::SinkRearrangeOperands(
    HloInstruction* instruction) {
  auto print_no_metadata = HloPrintOptions().set_print_metadata(false);
  HloComputation* computation = instruction->parent();
  const HloInstruction* rearrange =
      FirstNontrivialRearrange(instruction->operands());
  CHECK(rearrange != nullptr);
  const Shape& new_operand_shape = rearrange->operand(0)->shape();
  VLOG(3) << "** Sinking reshape or transpose: "
          << instruction->ToString(print_no_metadata)
          << "\n\tfirst rearrange operand: "
          << rearrange->ToString(print_no_metadata)  
          << "\n\tnew operand shape: "
          << ShapeUtil::HumanString(new_operand_shape);
  auto operands = instruction->operands();
  for (size_t i = 0; i < operands.size(); ++i) {
    VLOG(3) << "Updating operand #" << i << ": "
            << operands[i]->ToString(print_no_metadata);
    TF_ASSIGN_OR_RETURN(operands[i],
                        ApplyInverseRearrange(rearrange, operands[i]));
    VLOG(3) << "Updated operand #" << i
            << " to: " << operands[i]->ToString(print_no_metadata);
  }
  HloInstruction* new_elementwise =
      computation->AddInstruction(instruction->CloneWithNewOperands(
          ShapeUtil::ChangeElementType(new_operand_shape,
                                       instruction->shape().element_type()),
          operands));
  std::unique_ptr<HloInstruction> new_rearrange;
  switch (rearrange->opcode()) {
    case HloOpcode::kReshape:
      VLOG(3) << "Creating new reshape for new elementwise op: "
              << new_elementwise->ToString(print_no_metadata);
      new_rearrange =
          HloInstruction::CreateReshape(instruction->shape(), new_elementwise);
      break;
    case HloOpcode::kTranspose:
      new_rearrange = HloInstruction::CreateTranspose(
          instruction->shape(), new_elementwise, rearrange->dimensions());
      break;
    default:
      LOG(FATAL) << "Bad opcode";
  }
  if (instruction->has_sharding()) {
    new_elementwise->clear_sharding();
  }
  TF_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
      instruction, std::move(new_rearrange)));
  return true;
}
absl::StatusOr<bool> ReshapeMover::TryReshapeMoveOnCandidates(
    HloInstructionSet* candidates) {
  bool removed = true;
  while (!candidates->empty() && removed) {
    if (VLOG_IS_ON(5)) {
      for (const HloInstruction* instruction : *candidates) {
        VLOG(5) << "candidate " << instruction->ToString();
      }
    }
    ConstHloInstructionSet rearrange_operands;
    for (const HloInstruction* instruction : *candidates) {
      for (const auto* operand : instruction->operands()) {
        if (IsRearrange(operand)) {
          rearrange_operands.insert(operand);
        }
      }
    }
    removed = false;
    for (auto operand : rearrange_operands) {
      if (absl::c_any_of(operand->users(), [&](HloInstruction* user) {
            return !candidates->count(user);
          })) {
        for (auto* user : operand->users()) {
          removed |= candidates->erase(user) > 0;
        }
      }
    }
  }
  if (candidates->empty()) {
    return false;
  }
  for (HloInstruction* instruction : *candidates) {
    if (!ConsumeFuel("reshape-mover", [&] {
          return absl::StrCat("instruction: ", instruction->ToString(),
                              "\nFull module:\n",
                              instruction->GetModule()->ToString());
        })) {
      break;
    }
    TF_ASSIGN_OR_RETURN(bool did_change, SinkRearrangeOperands(instruction));
    CHECK(did_change);
  }
  return true;
}
absl::StatusOr<bool> ReshapeMover::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    HloInstructionSet candidates;
    for (HloInstruction* instruction : comp->instructions()) {
      if (IsReshapeMoveCandidate(instruction)) {
        candidates.insert(instruction);
      }
    }
    TF_ASSIGN_OR_RETURN(bool did_change,
                        TryReshapeMoveOnCandidates(&candidates));
    changed |= did_change;
  }
  return changed;
}
}  