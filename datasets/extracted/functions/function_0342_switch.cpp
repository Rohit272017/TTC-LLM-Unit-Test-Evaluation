#include "xla/service/hlo_ordering.h"
#include <memory>
#include <utility>
#include <vector>
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/types.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
namespace xla {
bool HloOrdering::ExecutesBefore(const HloInstruction* a,
                                 const HloInstruction* b) const {
  switch (GetExecutionConstraint(a, b)) {
    case ExecutionConstraint::kIsSame:  
      return false;
    case ExecutionConstraint::kRunBeforeStart:
    case ExecutionConstraint::kRunBeforeEnd:
    case ExecutionConstraint::kRunExclusiveBefore:
      return true;
    case ExecutionConstraint::kRunExclusiveAfter:
    case ExecutionConstraint::kRunAfter:
    case ExecutionConstraint::kUnordered:
      return false;
  }
}
HloOrdering::ExecutionConstraint HloOrdering::GetExecutionConstraint(
    const HloInstruction* a, const HloInstruction* b) const {
  auto is_async_wrapped = [](const HloInstruction* a, const HloInstruction* b) {
    return a->IsAsynchronous() && a->async_wrapped_instruction() == b;
  };
  if (a == b || is_async_wrapped(a, b) || is_async_wrapped(b, a)) {
    return ExecutionConstraint::kIsSame;
  }
  const HloInstruction* a_ancestor;
  const HloInstruction* b_ancestor;
  std::tie(a_ancestor, b_ancestor) =
      call_graph_->NearestAncestorsInSameComputation(
          const_cast<HloInstruction*>(a), const_cast<HloInstruction*>(b));
  if (a_ancestor == nullptr) {
    VLOG(4) << "Ancestors in a common computation could not be found between"
            << a->ToString() << "\n and \n"
            << b->ToString() << "\n so consider them to be unordered.\n";
    return ExecutionConstraint::kUnordered;
  }
  CHECK_NE(b_ancestor, nullptr);
  CHECK_EQ(a_ancestor->parent(), b_ancestor->parent());
  if (a_ancestor == b_ancestor && a_ancestor->opcode() == HloOpcode::kWhile) {
    const HloComputation* body = a_ancestor->while_body();
    const HloComputation* condition = a_ancestor->while_condition();
    if (call_graph_->InstructionIsNestedIn(a, condition) &&
        call_graph_->InstructionIsNestedIn(b, body)) {
      return ExecutionConstraint::kRunBeforeEnd;
    }
  }
  if (a_ancestor == b_ancestor &&
      (a_ancestor->opcode() == HloOpcode::kConditional)) {
    int a_branch = -1;
    int b_branch = -1;
    for (int j = 0; j < a_ancestor->branch_count(); ++j) {
      if (call_graph_->InstructionIsNestedIn(
              a, a_ancestor->branch_computation(j))) {
        a_branch = j;
      }
      if (call_graph_->InstructionIsNestedIn(
              b, a_ancestor->branch_computation(j))) {
        b_branch = j;
      }
    }
    if (a_branch == -1 && b_branch == -1) {
      CHECK_EQ(a, a_ancestor);
      CHECK_EQ(b, b_ancestor);
      CHECK_EQ(a, b);
      return ExecutionConstraint::kIsSame;
    }
    if (b_branch == -1) {
      CHECK_EQ(b, a_ancestor);
      return ExecutionConstraint::kRunBeforeEnd;
    }
    if (a_branch == -1) {
      CHECK_EQ(a, a_ancestor);
      return ExecutionConstraint::kRunAfter;
    }
    if (a_branch < b_branch) {
      return ExecutionConstraint::kRunExclusiveBefore;
    }
    if (b_branch < a_branch) {
      return ExecutionConstraint::kRunExclusiveAfter;
    }
  }
  if (ExecutesBeforeInSameComputation(a_ancestor, b_ancestor)) {
    return ExecutionConstraint::kRunBeforeStart;
  }
  if (ExecutesBeforeInSameComputation(b_ancestor, a_ancestor)) {
    return ExecutionConstraint::kRunAfter;
  }
  VLOG(1) << "Cannot determine order between:" << a->ToString() << "\n"
          << "and " << b->ToString() << " which are in the same computation\n";
  return ExecutionConstraint::kUnordered;
}
bool HloOrdering::IsDefinedBefore(const HloValue& a, const HloValue& b) const {
  const HloModule* module = b.defining_instruction()->GetModule();
  if (b.defining_instruction()->parent() == module->entry_computation() &&
      b.defining_instruction()->opcode() == HloOpcode::kParameter) {
    return false;
  }
  if (a.defining_instruction()->parent() == module->entry_computation() &&
      a.defining_instruction()->opcode() == HloOpcode::kParameter) {
    return true;
  }
  auto is_body_or_condition_phi = [](const HloValue& v) {
    return v.is_phi() &&
           v.defining_instruction()->opcode() == HloOpcode::kParameter;
  };
  if (is_body_or_condition_phi(a) && !is_body_or_condition_phi(b) &&
      call_graph_->InstructionIsNestedIn(b.defining_instruction(),
                                         a.defining_instruction()->parent())) {
    return true;
  }
  if (is_body_or_condition_phi(b) &&
      call_graph_->InstructionIsNestedIn(a.defining_instruction(),
                                         b.defining_instruction()->parent())) {
    return false;
  }
  if (b.is_phi() && b.defining_instruction()->opcode() == HloOpcode::kWhile &&
      (call_graph_->InstructionIsNestedIn(
           a.defining_instruction(), b.defining_instruction()->while_body()) ||
       call_graph_->InstructionIsNestedIn(
           a.defining_instruction(),
           b.defining_instruction()->while_condition()))) {
    return true;
  }
  if (b.is_phi() &&
      b.defining_instruction()->opcode() == HloOpcode::kConditional) {
    for (int j = 0; j < b.defining_instruction()->branch_count(); ++j) {
      if (call_graph_->InstructionIsNestedIn(
              a.defining_instruction(),
              b.defining_instruction()->branch_computation(j))) {
        return true;
      }
    }
  }
  return ExecutesBefore(a.defining_instruction(), b.defining_instruction());
}
bool HloOrdering::UsesBeforeValueDefinition(
    absl::Span<const HloUse* const> uses, const HloValue& value,
    const HloDataflowAnalysis& dataflow,
    bool use_is_always_before_def_in_same_instr) const {
  bool has_use_in_exclusive_branches = false;
  bool has_escaped_use_in_conditional = false;
  auto UseIsBeforeValueDefinition = [&](const HloUse& use) {
    VLOG(4) << "UseIsBeforeValueDefinition(use=" << use
            << ", value=" << value.ToShortString() << ")";
    switch (
        GetExecutionConstraint(use.instruction, value.defining_instruction())) {
      case HloOrdering::ExecutionConstraint::kIsSame: {
        if (use_is_always_before_def_in_same_instr) {
          return true;
        }
        HloInstruction* operand =
            use.instruction->mutable_operand(use.operand_number);
        HloInstruction* user = value.defining_instruction();
        auto operand_index_ptr =
            std::make_unique<ShapeIndex>(use.operand_index);
        if (use.instruction->IsAsynchronous()) {
          if (value.defining_instruction()->parent() ==
              use.instruction->async_wrapped_computation()) {
            if (use.instruction->opcode() == HloOpcode::kAsyncStart) {
              operand = use.instruction->async_wrapped_computation()
                            ->parameter_instruction(use.operand_number);
            } else {
              CHECK_GT(use.operand_index.size(), 1);
              operand = use.instruction->async_wrapped_computation()
                            ->parameter_instruction(use.operand_index.at(1));
              operand_index_ptr = std::make_unique<ShapeIndex>(
                  absl::MakeSpan(use.operand_index)
                      .subspan(2, use.operand_index.size() - 2));
            }
          }
        }
        if (dataflow.CanShareOperandBufferWithUser(
                operand,
                *operand_index_ptr,
                user,
                value.defining_index())) {
          VLOG(4)
              << "  use is value def, and instruction can share use buffer.";
          return true;
        }
        break;
      }
      case HloOrdering::ExecutionConstraint::kRunExclusiveAfter:
        VLOG(4) << " use and value def are in exclusive branches.";
        if (!has_escaped_use_in_conditional) {
          has_use_in_exclusive_branches = true;
          VLOG(4) << "Allowing them to share buffer.\n";
          return true;
        }
        VLOG(4) << "value def has escaped use in conditional. \n";
        break;
      case HloOrdering::ExecutionConstraint::kRunExclusiveBefore:
      case HloOrdering::ExecutionConstraint::kRunBeforeStart:
      case HloOrdering::ExecutionConstraint::kRunBeforeEnd:
        VLOG(4)
            << "  use instruction executes before value-defining instruction";
        return true;
      case HloOrdering::ExecutionConstraint::kRunAfter:
        if (use_is_always_before_def_in_same_instr &&
            use.instruction->opcode() == HloOpcode::kCollectivePermuteDone &&
            use.instruction->operand(0) == value.instruction()) {
          return true;
        }
        break;
      case HloOrdering::ExecutionConstraint::kUnordered:
        break;
    }
    if (use.instruction->opcode() == HloOpcode::kWhile) {
      const HloInstruction* xla_while = use.instruction;
      if (call_graph_->InstructionIsNestedIn(value.defining_instruction(),
                                             xla_while->while_body())) {
        VLOG(4) << "  use is while " << use.instruction->name()
                << " and def is in body";
        return true;
      }
      if (call_graph_->InstructionIsNestedIn(value.defining_instruction(),
                                             xla_while->while_condition())) {
        if (value.defining_instruction() !=
            xla_while->while_condition()->parameter_instruction(0)) {
          VLOG(4) << "  use is while " << use.instruction->name()
                  << " and def is in condition and is not the parameter";
          return false;
        } else {
          VLOG(4) << "  use is while " << use.instruction->name()
                  << " and def is in condition and is the parameter";
          return true;
        }
      }
    }
    if (value.defining_instruction()->opcode() == HloOpcode::kWhile) {
      CHECK(value.is_phi());
      const HloInstruction* xla_while = value.defining_instruction();
      if (call_graph_->InstructionIsNestedIn(use.instruction,
                                             xla_while->while_body()) ||
          call_graph_->InstructionIsNestedIn(use.instruction,
                                             xla_while->while_condition())) {
        VLOG(4) << "  value is while " << value.defining_instruction()->name()
                << " and use is in condition or body";
        return true;
      }
    }
    if (use.instruction->opcode() == HloOpcode::kCall) {
      const HloInstruction* call = use.instruction;
      if (call_graph_->InstructionIsNestedIn(value.defining_instruction(),
                                             call->to_apply())) {
        VLOG(4) << "  use is call " << use.instruction->name()
                << " and def is in called computation";
        return true;
      }
    }
    if (use.instruction->IsAsynchronous()) {
      const HloInstruction* async = use.instruction;
      if (call_graph_->InstructionIsNestedIn(
              value.defining_instruction(),
              async->async_wrapped_computation())) {
        VLOG(4) << "  use is async " << use.instruction->name()
                << " and def is in called computation";
        return true;
      }
    }
    if (use.instruction->opcode() == HloOpcode::kConditional) {
      const HloInstruction* conditional = use.instruction;
      for (int j = 0; j < conditional->branch_count(); ++j) {
        if (call_graph_->InstructionIsNestedIn(
                value.defining_instruction(),
                conditional->branch_computation(j))) {
          if (!dataflow.ValueIsDefinedAt(
                  use.instruction->operand(use.operand_number), {})) {
            for (auto value_use : value.GetUses()) {
              VLOG(4) << "def have use:" << value_use << "\n";
              if (value_use.instruction ==
                  value_use.instruction->parent()->root_instruction()) {
                VLOG(4) << "def use is conditional root \n";
                has_escaped_use_in_conditional = true;
                break;
              }
            }
          }
          if (!has_use_in_exclusive_branches) {
            VLOG(4) << "  use is conditional " << use.instruction->name()
                    << " and def is in " << j << "th branch computation";
            return true;
          }
        }
      }
      if (value.defining_instruction() == use.instruction) {
        VLOG(4) << "  use is conditional " << use << " and def is "
                << value.ToShortString();
        return true;
      }
    }
    VLOG(4) << "  use is not before value definition";
    return false;
  };
  for (auto* use : uses) {
    if (!UseIsBeforeValueDefinition(*use)) {
      return false;
    }
  }
  return true;
}
bool HloOrdering::LiveRangeStrictlyBefore(
    const HloValue& a, const HloValue& b, const HloDataflowAnalysis& dataflow,
    bool use_is_always_before_def_in_same_instr) const {
  VLOG(4) << "LiveRangeStrictlyBefore(a = " << a.ToShortString()
          << ", b = " << b.ToShortString() << ")";
  VLOG(4) << "Parent:" << a.instruction()->parent()->ToString() << "\n";
  if (!IsDefinedBefore(a, b)) {
    VLOG(4) << a << " not defined before " << b;
    return false;
  }
  if (a.live_out_of_module()) {
    VLOG(4) << a << " is live out of module and not defined before " << b;
    return false;
  }
  for (const HloPosition& pos : a.positions()) {
    if (pos.instruction->parent()->root_instruction() == pos.instruction &&
        call_graph().InstructionIsNestedIn(b.instruction(),
                                           pos.instruction->parent())) {
      return false;
    }
  }
  std::vector<const HloUse*> uses;
  for (const HloUse& use : a.GetUses()) {
    if (dataflow.DoesNotUseOperandBuffer(a.instruction(), a.index(),
                                         use.instruction)) {
      continue;
    }
    uses.push_back(&use);
  }
  if (!UsesBeforeValueDefinition(uses, b, dataflow,
                                 use_is_always_before_def_in_same_instr)) {
    VLOG(4) << "uses of " << a << "not before " << b << " is defined";
    return false;
  }
  if (a.IsRootOf(b.instruction()->parent())) {
    VLOG(4) << a << " is live out of computation and defined before " << b
            << " which is in same computation";
    return false;
  }
  return true;
}
bool HloOrdering::MayInterfere(const HloValue& a, const HloValue& b,
                               const HloDataflowAnalysis& dataflow) const {
  return !LiveRangeStrictlyBefore(a, b, dataflow) &&
         !LiveRangeStrictlyBefore(b, a, dataflow);
}
PredecessorHloOrdering::PredecessorHloOrdering(const HloModule* module)
    : HloOrdering(module) {}
bool PredecessorHloOrdering::ExecutesBeforeInSameComputation(
    const HloInstruction* a, const HloInstruction* b) const {
  CHECK_EQ(a->parent(), b->parent());
  return a != b && predecessors_.at(a->parent())->IsReachable(a, b);
}
std::string PredecessorHloOrdering::ToStringHelper(
    const std::string& name) const {
  std::vector<std::string> pieces;
  pieces.push_back(name);
  for (auto* computation : module_->MakeNonfusionComputations()) {
    pieces.push_back(absl::StrFormat("computation %s:", computation->name()));
    const auto all = computation->MakeInstructionPostOrder();
    for (auto instruction : all) {
      pieces.push_back(
          absl::StrFormat("  %s predecessors:", instruction->name()));
      for (auto predecessor : all) {
        if (predecessors_.at(computation)
                ->IsReachable(predecessor, instruction)) {
          pieces.push_back(absl::StrFormat("    %s", predecessor->name()));
        }
      }
    }
  }
  return absl::StrJoin(pieces, "\n");
}
DependencyHloOrdering::DependencyHloOrdering(const HloModule* module)
    : PredecessorHloOrdering(module) {
  for (auto* computation : module->MakeNonfusionComputations()) {
    predecessors_.emplace(computation, HloReachabilityMap::Build(computation));
  }
}
std::string DependencyHloOrdering::ToString() const {
  return ToStringHelper("DependencyHloOrdering");
}
SequentialHloOrdering::SequentialHloOrdering(const HloSchedule& schedule)
    : HloOrdering(schedule.module()), schedule_(schedule) {
  Initialize();
}
SequentialHloOrdering::SequentialHloOrdering(HloSchedule&& schedule)
    : HloOrdering(schedule.module()), schedule_(std::move(schedule)) {
  Initialize();
}
void SequentialHloOrdering::Initialize() {
  TF_DCHECK_OK(schedule_.Verify());
  for (const auto& computation_sequence : schedule_.sequences()) {
    const auto& order = computation_sequence.second.instructions();
    for (int i = 0; i < order.size(); ++i) {
      InsertOrDie(&order_position_, order[i], i);
    }
  }
}
bool SequentialHloOrdering::ExecutesBeforeInSameComputation(
    const HloInstruction* a, const HloInstruction* b) const {
  CHECK_EQ(a->parent(), b->parent());
  if (!order_position_.contains(a) || !order_position_.contains(b)) {
    return false;
  }
  if (a->parent()->root_instruction() == a) {
    return false;
  }
  return order_position_.at(a) < order_position_.at(b);
}
const HloInstructionSequence* SequentialHloOrdering::SequentialOrder(
    const HloComputation& computation) const {
  return schedule_.is_computation_scheduled(&computation)
             ? &schedule_.sequence(&computation)
             : nullptr;
}
std::string SequentialHloOrdering::ToString() const {
  return absl::StrCat("SequentialHloOrdering\n", schedule_.ToString());
}
}  