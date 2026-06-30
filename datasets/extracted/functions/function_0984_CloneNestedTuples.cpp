#include "xla/service/conditional_code_motion.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <stack>
#include <string>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/literal.h"
#include "xla/map_util.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/types.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
namespace xla {
namespace conditional_opt {
HloInstruction* CloneNestedTuples(HloInstruction* tuple) {
  if (!tuple->shape().IsTuple()) {
    return tuple;
  }
  std::vector<HloInstruction*> tuple_users, gte_users;
  for (int i = 0; i < tuple->shape().tuple_shapes_size(); ++i) {
    gte_users.push_back(nullptr);
  }
  for (auto* tuple_user : tuple->users()) {
    VLOG(2) << "tuple_user: " << tuple_user->ToString() << "\n";
    if (tuple_user->opcode() != HloOpcode::kGetTupleElement ||
        tuple_user == tuple->parent()->root_instruction()) {
      tuple_users.push_back(tuple_user);
    } else {
      gte_users[tuple_user->tuple_index()] = tuple_user;
    }
  }
  if (!tuple_users.empty() || tuple->user_count() == 0 ||
      tuple == tuple->parent()->root_instruction()) {
    VLOG(5) << "CLONING: " << tuple->ToString() << "\n";
    int64_t tuple_size = tuple->shape().tuple_shapes_size();
    std::vector<HloInstruction*> operands;
    operands.reserve(tuple_size);
    for (int64_t j = 0; j < tuple_size; ++j) {
      HloInstruction* gte =
          (gte_users[j] == nullptr)
              ? tuple->parent()->AddInstruction(
                    HloInstruction::CreateGetTupleElement(
                        tuple->shape().tuple_shapes(j), tuple, j))
              : gte_users[j];
      CHECK_NE(gte, nullptr);
      operands.push_back(CloneNestedTuples(gte));
    }
    HloInstruction* new_tuple =
        tuple->parent()->AddInstruction(HloInstruction::CreateTuple(operands));
    VLOG(2) << "new_tuple: " << new_tuple->ToString() << "\n";
    if (tuple == tuple->parent()->root_instruction()) {
      tuple->parent()->set_root_instruction(new_tuple,
                                            true);
    } else {
      for (auto tuple_user : tuple_users) {
        TF_CHECK_OK(tuple->ReplaceUseWithDifferentShape(tuple_user, new_tuple));
      }
    }
    return new_tuple;
  }
  for (auto gte_user : gte_users) {
    if (gte_user != nullptr) {
      auto gte = CloneNestedTuples(gte_user);
      CHECK_NE(gte, nullptr);
    }
  }
  return tuple;
}
class BoundaryVisitor {
 public:
  explicit BoundaryVisitor(HloInstruction* conditional) {
    Boundary b(Boundary::Position::kInsideBranch);
    b.mutable_operands().push_back(conditional);
    worklist_.push_back(b);
  }
  BoundaryVisitor() {}
  Boundary PopNextBoundary() {
    CHECK(!worklist_.empty());
    Boundary b = worklist_.front();
    worklist_.pop_front();
    while (!worklist_.empty() && ContainsKey(visited_, b)) {
      b = worklist_.front();
      worklist_.pop_front();
    }
    visited_.insert(b);
    return b;
  }
  void AddToWorkList(const Boundary& b) {
    CHECK(!b.operands().empty());
    worklist_.push_back(b);
  }
  bool HasNextBoundary() {
    while (!worklist_.empty()) {
      Boundary b = worklist_.front();
      if (!ContainsKey(visited_, b)) {
        break;
      }
      worklist_.pop_front();
    }
    return !worklist_.empty();
  }
 private:
  std::deque<Boundary> worklist_;
  absl::flat_hash_set<Boundary> visited_;
};
template <class OpCollection>
int64_t CountNonLeafOps(const OpCollection& ops) {
  absl::flat_hash_set<HloInstruction*> op_set;
  for (auto op : ops) {
    if (!op_set.contains(op) && op->opcode() != HloOpcode::kConstant) {
      op_set.insert(op);
    }
  }
  return op_set.size();
}
int64_t ReusesCarriedBy(HloOpcode op, HloOpcode user) {
  switch (user) {
    case HloOpcode::kGetTupleElement:
      return 0;
    case HloOpcode::kConvert:
      switch (op) {
        case HloOpcode::kConvolution:
        case HloOpcode::kDot:
          return 0;
        default:
          break;
      }
      break;
    default:
      break;
  }
  switch (op) {
    case HloOpcode::kParameter:
    case HloOpcode::kConstant:
    case HloOpcode::kGetTupleElement:
      return 0;
    case HloOpcode::kConditional:
      return 10;
    default:
      return -10;
  }
}
bool WorthHoisting(HloOpcode op, HloOpcode child_op) {
  switch (op) {
    case HloOpcode::kConvert:
      switch (child_op) {
        case HloOpcode::kAllReduce:
        case HloOpcode::kReshape:
        case HloOpcode::kGetTupleElement:
          return true;
        default:
          return false;
      }
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kTuple:
      switch (child_op) {
        case HloOpcode::kParameter:
          return false;
        default:
          return true;
      }
    case HloOpcode::kAllReduce:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kReduce:
    case HloOpcode::kConstant:
    case HloOpcode::kReshape:
    case HloOpcode::kBroadcast:
      return true;
    default:
      if (HloInstruction::IsOpElementwise(op)) {
        return true;
      }
      return false;
  }
}
bool InstructionWithinBranchIdentical(
    const std::vector<HloInstruction*>& instructions,
    bool is_layout_sensitive) {
  auto eq_operand = [&](const HloInstruction* a, const HloInstruction* b) {
    bool eq_operands = is_layout_sensitive
                           ? ShapeUtil::Equal(a->shape(), b->shape())
                           : ShapeUtil::Compatible(a->shape(), b->shape());
    return eq_operands;
  };
  auto eq_computations = [](const HloComputation* a, const HloComputation* b) {
    return *a == *b;
  };
  if (instructions.empty()) {
    return false;
  }
  if (instructions[0]->IsCrossModuleAllReduce()) {
    return std::all_of(
        instructions.begin(), instructions.end(),
        [&](HloInstruction* instruction) {
          if (!instruction->IsCrossModuleAllReduce()) {
            return false;
          }
          auto old_channel_id = instruction->channel_id();
          instruction->set_channel_id(instructions[0]->channel_id());
          bool eq_instructions = instructions[0]->Identical(
              *instruction, eq_operand, eq_computations, is_layout_sensitive);
          instruction->set_channel_id(old_channel_id);
          return eq_instructions;
        });
  }
  return std::all_of(instructions.begin(), instructions.end(),
                     [&](HloInstruction* instruction) {
                       return instructions[0]->Identical(
                           *instruction, eq_operand, eq_computations,
                           is_layout_sensitive);
                     });
}
void CopyOutOfConditional(
    Boundary& boundary, HloInstruction* conditional,
    absl::flat_hash_map<Boundary, Boundary>& hoisted_boundaries) {
  CHECK(boundary.IsInsideBranch());
  absl::InlinedVector<HloInstruction*, 4> new_operands;
  const HloInstruction* branch0_inst = boundary.operands()[0];
  for (int i = 0; i < branch0_inst->operands().size(); ++i) {
    Boundary operand_boundary(boundary.GetPosition());
    for (HloInstruction* operand : boundary.operands()) {
      operand_boundary.mutable_operands().push_back(operand->operands()[i]);
    }
    VLOG(2) << "Looking for: " << operand_boundary.ToString();
    auto hoisted_boundaries_it = hoisted_boundaries.find(operand_boundary);
    CHECK(hoisted_boundaries_it != hoisted_boundaries.end());
    Boundary hoisted_boundary = hoisted_boundaries_it->second;
    CHECK(hoisted_boundary.IsOutsideBranchUser());
    CHECK_EQ(hoisted_boundary.operands().size(), 1);
    new_operands.push_back(hoisted_boundary.operands()[0]);
  }
  HloInstruction* new_instruction = conditional->parent()->AddInstruction(
      branch0_inst->CloneWithNewOperands(branch0_inst->shape(), new_operands));
  VLOG(2) << "new instruction:" << new_instruction->ToString();
  Boundary hoisted_boundary(Boundary::Position::kOutsideBranchUser);
  hoisted_boundary.mutable_operands().push_back(new_instruction);
  hoisted_boundaries[boundary] = hoisted_boundary;
}
void CopyIntoConditional(
    Boundary& boundary, HloInstruction* conditional,
    absl::flat_hash_map<Boundary, Boundary>& hoisted_boundaries) {
  CHECK(boundary.IsOutsideBranchUser() || boundary.IsOutsideBranchOperand());
  CHECK_EQ(boundary.operands().size(), 1);
  int num_branches = conditional->branch_count();
  std::vector<absl::InlinedVector<HloInstruction*, 4>> new_operands(
      num_branches);
  HloInstruction* op = boundary.operands()[0];
  for (HloInstruction* operand : op->operands()) {
    Boundary operand_boundary(boundary.GetPosition());
    operand_boundary.mutable_operands().push_back(operand);
    VLOG(2) << "Looking for: " << operand_boundary.ToString();
    auto hoisted_boundaries_it = hoisted_boundaries.find(operand_boundary);
    if (hoisted_boundaries_it != hoisted_boundaries.end()) {
      Boundary hoisted_boundary = hoisted_boundaries_it->second;
      CHECK(hoisted_boundary.IsInsideBranch());
      CHECK_EQ(hoisted_boundary.operands().size(), num_branches);
      for (int j = 0; j < num_branches; ++j) {
        new_operands[j].push_back(hoisted_boundary.operands()[j]);
      }
    } else {
      for (int j = 0; j < num_branches; ++j) {
        switch (operand->opcode()) {
          case HloOpcode::kConstant: {
            auto new_operand =
                conditional->branch_computation(j)->AddInstruction(
                    operand->Clone());
            VLOG(2) << "new instruction:" << new_operand->ToString();
            new_operands[j].push_back(new_operand);
            break;
          }
          case HloOpcode::kGetTupleElement: {
            auto gte = Cast<HloGetTupleElementInstruction>(operand);
            int64_t index = gte->tuple_index();
            HloInstruction* root =
                conditional->branch_computation(j)->root_instruction();
            CHECK(root->opcode() == HloOpcode::kTuple &&
                  index < root->operand_count())
                << root->ToString() << " " << gte->ToString();
            auto new_operand = root->mutable_operand(index);
            VLOG(2) << "new instruction:" << new_operand->ToString();
            new_operands[j].push_back(new_operand);
            break;
          }
          default:
            LOG(FATAL) << "Unexpected out-of-boundary instruction:"
                       << operand->ToString() << "\n";
        }
      }
    }
  }
  Boundary hoisted_boundary(Boundary::Position::kInsideBranch);
  for (int j = 0; j < num_branches; ++j) {
    HloInstruction* new_instruction =
        conditional->branch_computation(j)->AddInstruction(
            op->CloneWithNewOperands(op->shape(), new_operands[j]));
    VLOG(2) << "new instruction:" << new_instruction->ToString();
    hoisted_boundary.mutable_operands().push_back(new_instruction);
  }
  hoisted_boundaries[boundary] = hoisted_boundary;
}
absl::flat_hash_set<int64_t> FindSpecialConverts(HloInstruction* old_root,
                                                 int branch_count,
                                                 HloInstruction* conditional,
                                                 bool is_layout_sensitive) {
  absl::flat_hash_set<int64_t> special_convert;
  auto convert_invalid =
      [](const HloInstruction* convert_set_candidate) -> bool {
    bool invalid_user = absl::c_any_of(
        convert_set_candidate->users(), [](const HloInstruction* user) -> bool {
          return (user->opcode() == HloOpcode::kConvert);
        });
    bool invalid_producer =
        absl::c_any_of(convert_set_candidate->operands(),
                       [](const HloInstruction* operand) -> bool {
                         return (operand->opcode() == HloOpcode::kConvert);
                       });
    return (invalid_user || invalid_producer);
  };
  for (int64_t operand_num = 0; operand_num < old_root->operand_count();
       ++operand_num) {
    if (old_root->operand(operand_num)->opcode() != HloOpcode::kConvert) {
      continue;
    }
    bool replica = true;
    HloInstruction* special_convert_candidate =
        old_root->mutable_operand(operand_num);
    auto repeated =
        absl::c_count_if(old_root->operands(),
                         [&](const HloInstruction* operand) -> bool {
                           return (special_convert_candidate == operand);
                         }) > 1;
    if (convert_invalid(special_convert_candidate) || repeated) {
      continue;
    }
    for (int others = 1; others < branch_count; ++others) {
      HloInstruction* others_root =
          conditional->branch_computation(others)->root_instruction();
      const HloInstruction* other_convert = others_root->operand(operand_num);
      if (other_convert->opcode() != HloOpcode::kConvert ||
          convert_invalid(other_convert)) {
        replica = false;
        break;
      }
      bool eq_shape =
          is_layout_sensitive
              ? ShapeUtil::Equal(other_convert->shape(),
                                 special_convert_candidate->shape()) &&
                    ShapeUtil::Equal(
                        other_convert->operand(0)->shape(),
                        special_convert_candidate->operand(0)->shape())
              : ShapeUtil::Compatible(other_convert->shape(),
                                      special_convert_candidate->shape()) &&
                    ShapeUtil::Compatible(
                        other_convert->operand(0)->shape(),
                        special_convert_candidate->operand(0)->shape());
      if (!eq_shape) {
        replica = false;
        break;
      }
      auto repeated =
          absl::c_count_if(others_root->operands(),
                           [&](const HloInstruction* operand) -> bool {
                             return (special_convert_candidate == operand);
                           }) > 1;
      if (repeated) {
        replica = false;
        break;
      }
    }
    if (replica) {
      special_convert.insert(operand_num);
    }
  }
  return special_convert;
}
absl::Status RestructureConditionalInstruction(HloComputation* computation,
                                               HloInstruction* conditional) {
  HloInstruction* old_root = computation->root_instruction();
  std::vector<HloInstruction*> new_operands;
  int cur_index = 0;
  for (; cur_index < ShapeUtil::TupleElementCount(conditional->shape());
       ++cur_index) {
    new_operands.push_back(
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            ShapeUtil::GetTupleElementShape(conditional->shape(), cur_index),
            conditional, cur_index)));
  }
  HloInstruction* new_tuple =
      computation->AddInstruction(HloInstruction::CreateTuple(new_operands));
  if (old_root == conditional) {
    computation->set_root_instruction(new_tuple);
  } else {
    std::vector<HloInstruction*> new_tuple_users;
    for (auto conditional_user : conditional->users()) {
      auto is_new_gte = absl::c_find_if(
          new_operands,
          [&](HloInstruction* instr) { return instr == conditional_user; });
      if (is_new_gte == new_operands.end()) {
        new_tuple_users.push_back(conditional_user);
      }
    }
    for (auto new_tuple_user : new_tuple_users) {
      TF_RETURN_IF_ERROR(
          conditional->ReplaceUseWith(new_tuple_user, new_tuple));
    }
  }
  VLOG(2) << "computation after root restructure:\n" << computation->ToString();
  return absl::OkStatus();
}
absl::StatusOr<bool> ConvertSpecialMove(HloInstruction* conditional,
                                        bool is_layout_sensitive) {
  int branch_count = conditional->branch_count();
  if (branch_count <= 0) {
    return false;
  }
  for (int branch_num = 0; branch_num < branch_count; ++branch_num) {
    HloInstruction* branch_root =
        conditional->branch_computation(branch_num)->root_instruction();
    if (branch_root->opcode() != HloOpcode::kTuple) {
      return false;
    }
  }
  HloInstruction* old_root =
      conditional->branch_computation(0)->root_instruction();
  VLOG(2) << "BEFORE :" << conditional->GetModule()->ToString();
  auto find_gte = [](const HloInstruction* conditional_result,
                     int64_t index) -> HloInstruction* {
    for (HloInstruction* instr : conditional_result->users()) {
      if (instr->opcode() != HloOpcode::kGetTupleElement) {
        return nullptr;
      }
      if (instr->tuple_index() == index) {
        return instr;
      }
    }
    return nullptr;
  };
  absl::flat_hash_set<int64_t> special_convert = FindSpecialConverts(
      old_root, branch_count, conditional, is_layout_sensitive);
  if (special_convert.empty()) {
    return false;
  }
  TF_RETURN_IF_ERROR(
      RestructureConditionalInstruction(conditional->parent(), conditional));
  for (int branch = 0; branch < branch_count; branch++) {
    old_root = conditional->branch_computation(branch)->root_instruction();
    absl::flat_hash_map<HloInstruction*, int64_t> map_inst_to_tuple_index;
    std::vector<HloInstruction*> new_operands(old_root->operand_count());
    absl::flat_hash_set<HloInstruction*> to_hoist_set;
    for (int64_t operand_num = 0; operand_num < old_root->operand_count();
         ++operand_num) {
      map_inst_to_tuple_index[old_root->mutable_operand(operand_num)] =
          operand_num;
    }
    for (int64_t operand_num = 0; operand_num < old_root->operand_count();
         ++operand_num) {
      HloInstruction* hoist = old_root->mutable_operand(operand_num);
      if (!special_convert.contains(operand_num)) {
        new_operands[operand_num] = old_root->mutable_operand(operand_num);
        continue;
      }
      to_hoist_set.insert(hoist);
      int64_t new_tuple_count = old_root->operand_count();
      bool inplace = true;
      CHECK(!hoist->operands().empty());
      for (HloInstruction* prod : hoist->operands()) {
        if (inplace) {
          map_inst_to_tuple_index[prod] = map_inst_to_tuple_index[hoist];
          new_operands[map_inst_to_tuple_index[hoist]] = prod;
          inplace = false;
        } else {
          map_inst_to_tuple_index[prod] = new_tuple_count++;
          new_operands.push_back(prod);
        }
      }
    }
    HloComputation* cur_branch = conditional->branch_computation(branch);
    HloInstruction* new_branch_root =
        cur_branch->AddInstruction(HloInstruction::CreateTuple(new_operands));
    cur_branch->set_root_instruction(new_branch_root, true );
    TF_CHECK_OK(cur_branch->RemoveInstruction(old_root));
    if (branch != 0) {
      continue;
    }
    HloComputation* conditional_parent = conditional->parent();
    HloInstruction* newconditional =
        conditional_parent->AddInstruction(HloInstruction::CreateConditional(
            cur_branch->root_instruction()->shape(),
            conditional->mutable_operand(0),
            absl::MakeSpan(conditional->branch_computations()),
            absl::MakeSpan(conditional->operands()).subspan(1)));
    TF_RETURN_IF_ERROR(
        conditional->ReplaceAllUsesWithDifferentShape(newconditional));
    TF_CHECK_OK(conditional_parent->RemoveInstruction(conditional));
    conditional = newconditional;
    for (HloInstruction* hoist : to_hoist_set) {
      VLOG(2) << "Hoisting instruction:" << hoist->ToString();
      int64_t hoist_index = map_inst_to_tuple_index[hoist];
      HloInstruction* gte_hoist = find_gte(conditional, hoist_index);
      CHECK(gte_hoist != nullptr);
      std::vector<HloInstruction*> new_operands;
      for (HloInstruction* op : hoist->operands()) {
        HloInstruction* gte = conditional_parent->AddInstruction(
            HloInstruction::CreateGetTupleElement(op->shape(), conditional,
                                                  map_inst_to_tuple_index[op]));
        new_operands.push_back(gte);
      }
      HloInstruction* hoisted = conditional_parent->AddInstruction(
          hoist->CloneWithNewOperands(hoist->shape(), new_operands));
      VLOG(2) << "Hoisted instruction in parent:" << hoisted->ToString();
      TF_RETURN_IF_ERROR(gte_hoist->ReplaceAllUsesWith(hoisted));
      TF_CHECK_OK(conditional_parent->RemoveInstruction(gte_hoist));
    }
  }
  VLOG(2) << "AFTER :" << conditional->GetModule()->ToString();
  return true;
}
absl::StatusOr<bool> ConditionalCodeMotion::MoveInstructionOut(
    HloInstruction* conditional, std::vector<Boundary>& to_move_out,
    std::vector<Boundary>& new_boundaries) {
  if (to_move_out.empty()) {
    return false;
  }
  VLOG(1) << "Modifying code--number of boundaries to move out of conditional:"
          << to_move_out.size() << "\n";
  HloComputation* conditional_parent = conditional->parent();
  std::vector<HloInstruction*> old_conditional_users = conditional->users();
  absl::flat_hash_map<Boundary, Boundary> hoisted_boundaries;
  VLOG(2) << "before opt:"
          << conditional_parent->ToString(HloPrintOptions::Fingerprint())
          << "\n";
  int64_t op_index = 0;
  for (const Boundary& b : new_boundaries) {
    HloInstruction* op = b.operands()[0];
    CHECK(op != nullptr);
    VLOG(2) << "Mapping new boundary instr: " << op->ToString() << "\n";
    HloInstruction* gtr = conditional_parent->AddInstruction(
        HloInstruction::CreateGetTupleElement(op->shape(), conditional,
                                              op_index++));
    Boundary b2(Boundary::Position::kOutsideBranchUser);
    b2.mutable_operands().push_back(gtr);
    hoisted_boundaries[b] = b2;
  }
  for (int64_t i = to_move_out.size() - 1; i >= 0; i--) {
    CopyOutOfConditional(to_move_out[i], conditional, hoisted_boundaries);
  }
  VLOG(2) << "Done copy branch instructions out\n"
          << conditional_parent->ToString(HloPrintOptions::Fingerprint())
          << "\n";
  for (auto user_instr : old_conditional_users) {
    VLOG(2) << "Checking conditional user: " << user_instr->ToString() << "\n";
    CHECK(user_instr->opcode() == HloOpcode::kGetTupleElement);
    auto tuple_opd = static_cast<HloGetTupleElementInstruction*>(user_instr);
    int64_t index = tuple_opd->tuple_index();
    Boundary old_user_boundary(Boundary::Position::kInsideBranch);
    for (const HloComputation* called_computation :
         conditional->called_computations()) {
      HloInstruction* root = called_computation->root_instruction();
      CHECK(root->operands().size() > index);
      old_user_boundary.mutable_operands().push_back(root->operands()[index]);
    }
    CHECK(ContainsKey(hoisted_boundaries, old_user_boundary));
    HloInstruction* new_opd =
        hoisted_boundaries[old_user_boundary].operands()[0];
    CHECK(new_opd != nullptr);
    VLOG(2) << "Try replace all uses of :" << old_user_boundary.ToString()
            << "\n";
    TF_RETURN_IF_ERROR(user_instr->ReplaceAllUsesWith(new_opd));
    TF_RETURN_IF_ERROR(conditional_parent->RemoveInstruction(user_instr));
  }
  VLOG(2) << "Done changing conditional users\n"
          << conditional_parent->ToString() << "\n";
  int64_t branch_count = conditional->branch_count();
  for (int i = 0; i < branch_count; i++) {
    auto computation = conditional->branch_computation(i);
    std::vector<HloInstruction*> elements;
    for (const auto& b1 : new_boundaries) {
      HloInstruction* op = b1.operands()[i];
      CHECK(op != nullptr);
      VLOG(2) << "Adding to root " << i << " with " << op->ToString() << "\n";
      elements.push_back(op);
    }
    HloInstruction* tuple =
        computation->AddInstruction(HloInstruction::CreateTuple(elements));
    computation->set_root_instruction(tuple, true);
    VLOG(2) << "computation is :" << computation->ToString() << "\n";
    for (const auto& b2 : to_move_out) {
      auto instr_to_remove = b2.operands()[i];
      if (!computation->IsMarkedAsDead(instr_to_remove) &&
          instr_to_remove->IsDead()) {
        VLOG(2) << "Removing boundary:" << b2.ToString() << "\n";
        VLOG(2) << "computation: " << computation->ToString() << "\n";
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(instr_to_remove));
      }
    }
  }
  HloInstruction* new_root =
      conditional->branch_computation(0)->root_instruction();
  *conditional->mutable_shape() = new_root->shape();
  conditional->copy_sharding(new_root);
  VLOG(2) << "done moving instructions out of branches\n"
          << conditional_parent->ToString(HloPrintOptions::Fingerprint());
  return true;
}
absl::StatusOr<bool> ConditionalCodeMotion::MoveUserInstructionsIn(
    HloInstruction* conditional, std::vector<Boundary>& to_move_in) {
  if (to_move_in.empty()) {
    return false;
  }
  absl::flat_hash_map<Boundary, Boundary> hoisted_boundaries;
  int64_t to_move_in_size = to_move_in.size();
  int64_t branch_count = conditional->branch_count();
  HloGetTupleElementInstruction* tuple_use =
      DynCast<HloGetTupleElementInstruction>(to_move_in[0].operands()[0]);
  int64_t use_index = (tuple_use != nullptr && tuple_use->user_count() == 1)
                          ? tuple_use->tuple_index()
                          : -1;
  VLOG(2) << "Tuple use index = " << use_index << "\n";
  int64_t op_index =
      conditional->shape().IsTuple()
          ? ((use_index >= 0) ? conditional->shape().tuple_shapes_size() - 1
                              : conditional->shape().tuple_shapes_size())
          : 0;
  Boundary b_opd_use(Boundary::Position::kInsideBranch);
  Boundary b_old_root(Boundary::Position::kInsideBranch);
  for (int i = 0; i < branch_count; i++) {
    auto computation = conditional->branch_computation(i);
    auto old_root = computation->root_instruction();
    b_old_root.mutable_operands().push_back(old_root);
    std::vector<HloInstruction*> operands;
    if (old_root->opcode() == HloOpcode::kTuple) {
      for (int i = 0; i < old_root->operand_count(); ++i) {
        if (i != use_index) {
          operands.push_back(old_root->operands()[i]);
        } else {  
          b_opd_use.mutable_operands().push_back(old_root->operands()[i]);
        }
      }
    } else if (old_root->shape().IsTuple()) {
      const Shape& old_shape = old_root->shape();
      for (int i = 0; i < old_shape.tuple_shapes_size(); ++i) {
        auto element =
            computation->AddInstruction(HloInstruction::CreateGetTupleElement(
                old_shape.tuple_shapes(i), old_root, i));
        if (i != use_index) {
          operands.push_back(element);
        } else {
          b_opd_use.mutable_operands().push_back(element);
        }
      }
    } else {
      b_opd_use.mutable_operands().push_back(conditional);
    }
    HloInstruction* new_root =
        computation->AddInstruction(HloInstruction::CreateTuple(operands));
    VLOG(2) << "setting new root: " << new_root->ToString() << "\n";
    computation->set_root_instruction(new_root,
                                       true);
    if (old_root->opcode() == HloOpcode::kTuple) {
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(old_root));
    }
    VLOG(2) << "new branch computation: " << computation->ToString() << "\n";
  }
  if (use_index != -1) {
    for (auto* user : conditional->users()) {
      if (user->opcode() == HloOpcode::kGetTupleElement &&
          user->tuple_index() > use_index) {
        user->set_tuple_index(user->tuple_index() - 1);
      }
    }
  }
  Boundary conditional_boundary(Boundary::Position::kOutsideBranchUser);
  conditional_boundary.mutable_operands().push_back(conditional);
  hoisted_boundaries[conditional_boundary] = b_old_root;
  if (use_index >= 0) {
    VLOG(2) << "Mapping GTE: " << tuple_use->ToString() << "\n";
    Boundary tuple_use_boundary(Boundary::Position::kOutsideBranchUser);
    tuple_use_boundary.mutable_operands().push_back(tuple_use);
    hoisted_boundaries[tuple_use_boundary] = b_opd_use;
  }
  int64_t cp_start = (tuple_use != nullptr) ? 1 : 0;
  for (int64_t to_move_index = cp_start; to_move_index < to_move_in_size;
       to_move_index++) {
    Boundary b_to_move = to_move_in[to_move_index];
    HloInstruction* op = b_to_move.operands()[0];
    CHECK(op != nullptr);
    bool to_be_used_outside = true;
    VLOG(2) << "Mapping new boundary instr: " << op->ToString() << "\n";
    if (to_move_index < to_move_in_size - 1 && op->user_count() == 1 &&
        op->users()[0] == to_move_in[to_move_index + 1].operands()[0]) {
      to_be_used_outside = false;
      VLOG(2) << "Instruction is not to be used outside the branch\n";
    }
    Boundary b(Boundary::Position::kInsideBranch);
    CopyIntoConditional(b_to_move, conditional, hoisted_boundaries);
    if (to_be_used_outside) {
      for (int i = 0; i < branch_count; ++i) {
        auto computation = conditional->branch_computation(i);
        auto new_op = hoisted_boundaries[b_to_move].operands()[i];
        auto new_root = computation->root_instruction();
        new_root->AppendOperand(new_op);
        *new_root->mutable_shape()->add_tuple_shapes() = new_op->shape();
        VLOG(2) << "Extending conditional root " << i << " : "
                << new_root->ToString() << "\n";
      }
      HloInstruction* gtr = conditional->parent()->AddInstruction(
          HloInstruction::CreateGetTupleElement(op->shape(), conditional,
                                                op_index++));
      TF_RETURN_IF_ERROR(op->ReplaceAllUsesWith(gtr));
      if (conditional->parent()->root_instruction() == op) {
        conditional->parent()->set_root_instruction(gtr);
      }
    }
  }
  VLOG(2) << "Done copying instructions inside branch: "
          << conditional->ToString(HloPrintOptions::Fingerprint()) << "\n";
  HloInstruction* new_root =
      conditional->branch_computation(0)->root_instruction();
  *conditional->mutable_shape() = new_root->shape();
  conditional->copy_sharding(new_root);
  if (use_index != -1) {
    for (auto* user : conditional->users()) {
      if (user->opcode() == HloOpcode::kGetTupleElement) {
        VLOG(2) << "Resetting shape of user: " << user->ToString() << "\n";
        *user->mutable_shape() =
            conditional->shape().tuple_shapes(user->tuple_index());
      }
    }
  }
  VLOG(2) << "Done moving user instructions inside branches\n"
          << conditional->parent()->ToString(HloPrintOptions::Fingerprint());
  return true;
}
class MoveOperandIntoBranch {
 public:
  MoveOperandIntoBranch() = default;
  absl::Status operator()(HloInstruction* inst, HloInstruction*& user) {
    VLOG(1) << "operand to move into branch: " << inst->ToString();
    VLOG(2) << "MoveIntoBranches user =" << user->ToString() << "\n";
    CHECK(inst->user_count() == 1 || inst->opcode() == HloOpcode::kBroadcast);
    absl::InlinedVector<HloInstruction*, 4> new_operands;
    std::vector<std::vector<int64_t>> matching_tuple_indices;
    TF_RETURN_IF_ERROR(
        ReplaceInputInUser(inst, user, new_operands, matching_tuple_indices));
    TF_RETURN_IF_ERROR(
        MoveInputIntoBranch(inst, user, new_operands, matching_tuple_indices));
    if (inst->user_count() == 0) {
      TF_RETURN_IF_ERROR(inst->parent()->RemoveInstruction(inst));
    }
    return absl::OkStatus();
  }
 private:
  HloInstruction* InsertIntoBranch(HloInstruction* inst,
                                   HloInstruction* branch_input) {
    VLOG(2) << "Branch input=" << branch_input->ToString() << "\n";
    auto branch_comp = branch_input->parent();
    std::vector<HloInstruction*> operands(inst->operand_count());
    for (int64_t i = 0; i < inst->operand_count(); ++i) {
      VLOG(2) << "processing operand =" << i << "\n";
      if (branch_input->shape().IsTuple()) {
        int64_t j = std::find(inst->operands().begin(), inst->operands().end(),
                              inst->operands()[i]) -
                    inst->operands().begin();
        VLOG(2) << "operand index = " << j << "\n";
        CHECK(j < branch_input->shape().tuple_shapes_size());
        if (j < i) {
          operands[i] = operands[j];
        } else {
          CHECK(op_map_.contains(inst->operands()[i]));
          int64_t index = op_map_[inst->operands()[i]];
          operands[i] =
              branch_comp->AddInstruction(HloInstruction::CreateGetTupleElement(
                  branch_input->shape().tuple_shapes(index), branch_input,
                  index));
        }
      } else {
        CHECK(inst->operands()[i] == inst->operands()[0]);
        operands[i] = branch_input;
      }
    }
    return branch_comp->AddInstruction(
        inst->CloneWithNewOperands(inst->shape(), operands));
  }
  bool UpdateParamShape(
      std::vector<std::vector<int64_t>>& matching_tuple_indices,
      const Shape* param_shape, HloInstruction*& branch_param,
      HloInstruction*& param_tuple) {
    bool used = false;
    for (int64_t matching_index = matching_tuple_indices.size() - 1;
         matching_index >= 0; --matching_index) {
      auto* new_tuple = CloneNestedTuples(branch_param);
      CHECK_NE(new_tuple, nullptr);
      VLOG(5) << "Cloned new tuple:" << new_tuple->parent()->ToString() << "\n";
      std::vector<std::vector<HloInstruction*>> gte_users;
      gte_users.reserve(branch_param->shape().tuple_shapes_size());
      for (int64_t j = 0; j < branch_param->shape().tuple_shapes_size(); ++j) {
        gte_users.push_back(std::vector<HloInstruction*>());
      }
      for (auto* param_user : branch_param->users()) {
        if (param_user->opcode() == HloOpcode::kGetTupleElement) {
          CHECK_LT(param_user->tuple_index(), gte_users.size());
          gte_users[param_user->tuple_index()].push_back(param_user);
        }
      }
      used = false;
      *branch_param->mutable_shape() = *param_shape;
      const Shape* new_param_shape = nullptr;
      for (auto param_users : gte_users) {
        if (param_users.empty()) continue;
        CHECK_EQ(param_users[0]->opcode(), HloOpcode::kGetTupleElement);
        auto tuple_index = param_users[0]->tuple_index();
        VLOG(1) << "Processing gte users: " << param_users.size() << "\n";
        VLOG(1) << "tuple_index: " << tuple_index << "\n";
        VLOG(1) << "matching_tuple_indices: "
                << matching_tuple_indices[matching_index][0] << "\n";
        if (matching_tuple_indices[matching_index].end() ==
            std::find(matching_tuple_indices[matching_index].begin(),
                      matching_tuple_indices[matching_index].end(),
                      tuple_index)) {
          continue;
        }
        for (HloInstruction* param_user : param_users) {
          VLOG(1) << "param_user: " << param_user->ToString() << "\n";
          if (new_param_shape == nullptr) {
            branch_param = param_user;
            if (matching_index > 0) {
              param_tuple = branch_param;
            }
            CHECK_GT(param_shape->tuple_shapes_size(), tuple_index);
            new_param_shape = &param_shape->tuple_shapes(tuple_index);
            param_shape = new_param_shape;
            VLOG(1) << "new_param_shape: " << param_shape->ToString();
            *param_user->mutable_shape() = *new_param_shape;
            VLOG(1) << "branch parameter: " << param_user->ToString();
            used = true;
          } else {
            VLOG(1) << "new_param_shape=" << new_param_shape->ToString();
            *param_user->mutable_shape() = *new_param_shape;
            TF_CHECK_OK(param_user->ReplaceAllUsesWith(branch_param));
          }
        }
      }
      if (!used) {
        break;
      }
    }
    return used;
  }
  absl::Status ReplaceInputInUser(
      HloInstruction* input, HloInstruction*& user,
      absl::InlinedVector<HloInstruction*, 4>& new_operands,
      std::vector<std::vector<int64_t>>& matching_tuple_indices) {
    for (int64_t j = 0; j < input->operand_count(); ++j) {
      VLOG(2) << "Push back input operand index: " << j;
      auto operand = input->mutable_operand(j);
      if (std::find(new_operands.begin(), new_operands.end(), operand) ==
          new_operands.end()) {
        new_operands.push_back(operand);
      }
    }
    if (user->opcode() == HloOpcode::kTuple) {
      for (HloInstruction *input_now = input, *user_now = user;
           user_now->opcode() != HloOpcode::kConditional;
           input_now = user_now, user_now = user_now->users()[0]) {
        std::vector<int64_t> matching_tuple_index;
        for (int64_t i = 0; i < user_now->operand_count(); ++i) {
          if (user_now->operand(i) != input_now) {
            continue;
          }
          matching_tuple_index.push_back(i);
        }
        CHECK(!matching_tuple_index.empty());
        matching_tuple_indices.push_back(matching_tuple_index);
        CHECK_EQ(user_now->user_count(), 1);
      }
      CHECK(!matching_tuple_indices.empty());
      int64_t repl_count = 0;
      for (auto opd_index : matching_tuple_indices[0]) {
        HloInstruction* new_input =
            (repl_count < new_operands.size())
                ? new_operands[repl_count++]
                : input->AddInstruction(HloInstruction::CreateTuple({}));
        op_map_[new_input] = opd_index;
        VLOG(2) << "Mapping operand " << repl_count << " = "
                << new_input->ToString() << " to " << opd_index;
        TF_RETURN_IF_ERROR(
            user->ReplaceOperandWithDifferentShape(opd_index, new_input));
        *user->mutable_shape()->mutable_tuple_shapes(opd_index) =
            new_input->shape();
      }
      while (repl_count < new_operands.size()) {
        HloInstruction* new_input = new_operands[repl_count++];
        auto new_input_in_user = std::find(user->operands().begin(),
                                           user->operands().end(), new_input);
        int64_t opd_index = (new_input_in_user == user->operands().end())
                                ? user->operand_count()
                                : new_input_in_user - user->operands().begin();
        op_map_[new_input] = opd_index;
        CHECK(op_map_.contains(new_input));
        VLOG(2) << "Mapping operand " << new_input->ToString() << " to "
                << opd_index;
        user->AppendOperand(new_input);
        user->mutable_shape()->mutable_tuple_shapes()->push_back(
            new_input->shape());
      }
      int64_t nesting_index = 1;
      for (auto user_now = user->users()[0];
           nesting_index < matching_tuple_indices.size() &&
           user_now->opcode() != HloOpcode::kConditional;
           user = user_now, user_now = user_now->users()[0], nesting_index++) {
        VLOG(2) << "Replacing tuple: " << user->ToString();
        CHECK(user_now->shape().IsTuple());
        for (auto opd_index : matching_tuple_indices[nesting_index]) {
          *user_now->mutable_shape()->mutable_tuple_shapes(opd_index) =
              user->shape();
        }
        VLOG(2) << "Done replacing tuple:" << user->ToString();
        CHECK_EQ(user_now->user_count(), 1);
      }
      VLOG(2) << "User: " << user->ToString() << "\n";
    }
    return absl::OkStatus();
  }
  absl::Status MoveInputIntoBranch(
      HloInstruction* input, HloInstruction*& user,
      absl::InlinedVector<HloInstruction*, 4>& new_operands,
      std::vector<std::vector<int64_t>>& matching_tuple_indices) {
    HloInstruction* cond =
        (user->opcode() != HloOpcode::kConditional && user->user_count() == 1)
            ? user->users()[0]
            : user;
    if (user == cond) {
      auto new_input =
          input->AddInstruction(HloInstruction::CreateTuple(new_operands));
      for (int64_t i = 0; i < new_operands.size(); ++i) {
        op_map_[new_operands[i]] = i;
      }
      user = new_input;
      TF_RETURN_IF_ERROR(input->ReplaceUseWithDifferentShape(cond, new_input));
    }
    TF_RET_CHECK(cond->opcode() == HloOpcode::kConditional)
        << "User has non-conditional users";
    for (int64_t branch = 0; branch < cond->branch_count(); ++branch) {
      if (cond->operand(branch + 1) != user) {
        continue;
      }
      VLOG(2) << "Modifying conditional branch: " << branch << "\n";
      auto branch_comp = cond->branch_computation(branch);
      auto branch_param = branch_comp->parameter_instruction(0);
      auto* param_shape = &user->shape();
      VLOG(2) << "param_shape: " << param_shape->ToString() << "\n";
      VLOG(2) << "branch parameter: " << branch_param->ToString() << "\n";
      HloInstruction* param_tuple = branch_param;
      if (matching_tuple_indices.empty()) {
        VLOG(2) << "The original input is passed in as conditional parameter "
                   "directly.";
        VLOG(5) << branch_comp->ToString() << "\n";
        *branch_param->mutable_shape() = *param_shape;
        if (branch_param == branch_comp->root_instruction()) {
          VLOG(2) << "Cloning root user";
          auto new_user =
              branch_comp->AddInstruction(HloInstruction::CreateGetTupleElement(
                  branch_param->shape().tuple_shapes(0), branch_param, 0));
          VLOG(2) << "new_user: " << new_user->ToString() << "\n";
          branch_comp->set_root_instruction(new_user,
                                            true);
        }
      } else {
        if (!UpdateParamShape(matching_tuple_indices, param_shape, branch_param,
                              param_tuple)) {
          VLOG(2) << "instruction is not used in this branch.";
          continue;
        }
      }
      auto inserted = InsertIntoBranch(input, param_tuple);
      VLOG(2) << "Inserted operands:" << inserted->ToString() << "\n";
      std::vector<HloInstruction*> tuple_users = branch_param->users();
      for (auto param_user : tuple_users) {
        if (param_user == inserted ||
            (param_user->opcode() == HloOpcode::kGetTupleElement &&
             param_user != branch_comp->root_instruction())) {
          continue;
        }
        TF_RETURN_IF_ERROR(
            branch_param->ReplaceUseWithDifferentShape(param_user, inserted));
        if (branch_comp->root_instruction()->opcode() ==
                HloOpcode::kGetTupleElement &&
            !branch_comp->root_instruction()->operand(0)->shape().IsTuple()) {
          branch_comp->set_root_instruction(
              branch_comp->root_instruction()->mutable_operands()[0]);
        }
        UpdateTupleUsers(inserted);
      }
    }
    return absl::OkStatus();
  }
  void UpdateTupleUsers(HloInstruction* param_user) {
    for (auto new_user : param_user->users()) {
      if (new_user->opcode() == HloOpcode::kTuple) {
        for (int64_t opd_index = 0; opd_index < new_user->operand_count();
             ++opd_index) {
          if (new_user->operands()[opd_index] != param_user) {
            continue;
          }
          *new_user->mutable_shape()->mutable_tuple_shapes(opd_index) =
              param_user->shape();
          UpdateTupleUsers(new_user);
          VLOG(2) << "Updated tuple user: " << new_user->ToString() << "\n";
        }
      }
    }
  }
  absl::flat_hash_map<const HloInstruction*, int64_t> op_map_;
};
absl::StatusOr<bool> ConditionalCodeMotion::MoveOperandInstructionsIn(
    HloInstruction* conditional, std::vector<Boundary>& to_move_in) {
  int64_t to_move_in_size = to_move_in.size();
  CHECK_GT(to_move_in_size, 0);
  VLOG(2) << "Before moving operand instructions inside branch: "
          << conditional->ToString(HloPrintOptions::Fingerprint()) << "\n";
  HloInstruction* user = conditional;
  int64_t user_index = 0;
  MoveOperandIntoBranch move_into_branch;
  for (int64_t to_move_index = 0; to_move_index < to_move_in_size;
       to_move_index++) {
    Boundary b_to_move = to_move_in[to_move_index];
    HloInstruction* op = b_to_move.operands()[0];
    CHECK_NE(op, nullptr);
    if (op->user_count() == 1) {
      user = op->users()[0];
      user_index = user->operand_index(op);
    }
    if (op->opcode() == HloOpcode::kTuple) {
      continue;
    }
    VLOG(1) << "Mapping new boundary instr: " << op->ToString() << "\n";
    VLOG(1) << "current user = " << user->ToString();
    std::vector<std::pair<HloInstruction*, int64_t>> users;
    for (auto* user_now = user; user_now != conditional;
         user_now = user_now->users()[0]) {
      CHECK_EQ(user_now->user_count(), 1);
      VLOG(1) << "Saving user: " << user_now->ToString() << "\n";
      users.push_back(std::make_pair(
          user_now->users()[0], user_now->users()[0]->operand_index(user_now)));
    }
    TF_RETURN_IF_ERROR(move_into_branch(op, user));
    for (int64_t i = users.size() - 1; i > 0; --i) {
      CHECK_NE(users[i].first, nullptr);
      CHECK_NE(users[i - 1].first, nullptr);
      users[i - 1].first = users[i].first->mutable_operand(users[i].second);
    }
    if (!users.empty()) {
      user = users.front().first->mutable_operand(users.front().second);
      VLOG(1) << "Updated user: " << user->ToString() << "\n";
    }
  }
  VLOG(2) << "Done moving operand instructions inside branch: "
          << conditional->ToString(HloPrintOptions::Fingerprint()) << "\n";
  return true;
}
class GroupConnectedBoundaries {
 private:
  std::vector<Boundary> connected_boundaries_, new_boundaries_;
  int64_t connected_boundaries_memory_increase_ = 0;
  HloInstruction* conditional_;
  HloComputation* conditional_parent_;
  bool is_layout_sensitive_;
  absl::flat_hash_map<HloInstruction*, int>& visited_count_;
  std::vector<std::vector<int64_t>>& move_config_;
  std::vector<std::vector<int64_t>>& reuse_config_;
  absl::Span<int64_t> search_config_vec_;
  int64_t& search_config_;
  int64_t search_subscript_;
  absl::flat_hash_map<const int64_t*, int64_t> flipped_;
  int64_t FlipMutation(int64_t* loc, const int64_t non_zero,
                       const std::string& msg) {
    if (search_config_ == 0 || ContainsKey(flipped_, loc)) {
      VLOG(2) << "Configured not to search or loc is already flipped.";
      return *loc;
    }
    int c = ConditionalCodeMotion::flip_start(search_config_);
    VLOG(2) << "flip start index = " << c << "\n";
    if (c > 0) {
      search_config_--;
      return *loc;
    }
    auto flip_count = ConditionalCodeMotion::DecrementMaxFlip(&search_config_);
    VLOG(2) << "max flip count = " << flip_count << "\n";
    VLOG(2) << "Updating max Flipping configuration = " << search_config_
            << "\n";
    if (flip_count == 0) {
      VLOG(2) << "Maximum flip count has reached. ";
      if (search_subscript_ + 1 < search_config_vec_.size()) {
        VLOG(2) << "search_subscript_ = " << search_subscript_;
        VLOG(2) << "search config vec size = " << search_config_vec_.size();
        search_config_ = search_config_vec_[++search_subscript_];
      } else {
        return *loc;
      }
    }
    auto flip_stride = ConditionalCodeMotion::flip_stride(search_config_);
    search_config_ += flip_stride;
    VLOG(2) << "flip stride = " << flip_stride << "\n";
    VLOG(2) << "Updating Flipping Stride = " << search_config_ << "\n";
    flipped_[loc] = *loc;
    switch (*loc) {
      case 0:
        *loc = non_zero;
        break;
      default:
        *loc = 0;
        break;
    }
    VLOG(2) << "Flipping decision for: " << msg << ": from " << flipped_[loc]
            << " to " << *loc << "\n";
    return *loc;
  }
  static std::vector<int64_t>& EnsureSearchConfig(
      std::vector<int64_t>& search_config) {
    if (search_config.empty()) {
      search_config.push_back(0);
    }
    return search_config;
  }
 public:
  explicit GroupConnectedBoundaries(
      HloInstruction* conditional, bool is_layout_sensitive,
      absl::flat_hash_map<HloInstruction*, int>& visited_count,
      std::vector<std::vector<int64_t>>* move_config,
      std::vector<std::vector<int64_t>>* reuse_config,
      std::vector<int64_t>& search_config)
      : conditional_(conditional),
        conditional_parent_(conditional->parent()),
        is_layout_sensitive_(is_layout_sensitive),
        visited_count_(visited_count),
        move_config_(*move_config),
        reuse_config_(*reuse_config),
        search_config_vec_(EnsureSearchConfig(search_config)),
        search_config_(search_config_vec_.front()),
        search_subscript_(0) {
    VLOG(2) << "Initializing Group Connected Boundaries\n";
  }
  int64_t ReusesCarriedBy(HloInstruction* op, HloInstruction* user) {
    std::vector<int64_t>& curconfig =
        reuse_config_[static_cast<uint32_t>(op->opcode())];
    int64_t config =
        (search_config_ < 0)
            ? FlipMutation(&curconfig[static_cast<uint32_t>(user->opcode())],
                           -10,
                           absl::StrCat(HloOpcodeString(op->opcode()), "->",
                                        HloOpcodeString(user->opcode())))
            : curconfig[static_cast<uint32_t>(user->opcode())];
    VLOG(2) << "ConditionalCodeMotion: Add reuses carried by instr: "
            << op->ToString() << "=>" << user->ToString() << " : " << config
            << "\n";
    if (config < 0) {
      int count1 = CountNonLeafOps(op->users());
      int count2 = CountNonLeafOps(user->operands());
      return (-config) / count1 / count2;
    }
    return config;
  }
  void clear_recently_visited() {
    for (const auto& boundary : new_boundaries_) {
      visited_count_.erase(boundary.operands()[0]);
    }
  }
  bool WorthHoisting(HloInstruction* instruction, Boundary::Position pos,
                     int64_t index) {
    VLOG(1) << "Check Worth hoisting\n";
    HloOpcode opcode = instruction->opcode();
    if (opcode == HloOpcode::kTuple &&
        instruction == conditional_parent_->root_instruction()) {
      VLOG(1) << "Do not move conditional parent.";
      return false;
    }
    if (pos == Boundary::Position::kOutsideBranchOperand) {
      if (opcode == HloOpcode::kTuple && instruction->has_sharding()) {
        VLOG(1) << "Not moving operand because of sharding annotations.";
        return false;
      }
      if (instruction->user_count() > 1) {
        VLOG(1) << "Not moving operand b/c it has >1 users.";
        return false;
      }
      if (instruction->HasSideEffect()) {
        VLOG(1) << "Not moving operand b/c it has side effects.";
        return false;
      }
      if (opcode == HloOpcode::kGetTupleElement) {
        VLOG(1) << "Do not move GetTupleElement.";
        return false;
      }
    }
    if (DynCast<HloChannelInstruction>(instruction) &&
        pos != Boundary::Position::kInsideBranch) {
      VLOG(1) << "It is not safe to move collectives inside branches.";
      return false;
    }
    if (opcode == HloOpcode::kParameter) {
      return false;
    }
    if (opcode == HloOpcode::kGetTupleElement &&
        pos == Boundary::Position::kOutsideBranchOperand) {
      return false;
    }
    std::vector<int64_t>& curconfig =
        move_config_[static_cast<uint32_t>(opcode)];
    auto col = (curconfig.size() == 1) ? 0
               : (instruction->operand_count() > 0)
                   ? static_cast<uint32_t>(instruction->operand(0)->opcode())
                   : 0;
    VLOG(2) << "column = " << col << "\n";
    VLOG(2) << "config size = " << curconfig.size() << "\n";
    VLOG(2) << "search_config = " << search_config_ << "\n";
    CHECK(col < curconfig.size());
    uint32_t config =
        (search_config_ > 0)
            ? FlipMutation(&curconfig[col], 1,
                           absl::StrCat("Move-", HloOpcodeString(opcode)))
            : curconfig[col];
    VLOG(2) << "Checking instruction is worth moving: " << config << "\n";
    VLOG(2) << "after checking search_config = " << search_config_ << "\n";
    return (config != 0);
  }
  int64_t ReusesBeforeBoundary(HloInstruction* user) {
    int64_t reuses = 0;
    for (auto op : user->operands()) {
      if (!ContainsKey(visited_count_, op) && op != conditional_) {
        continue;
      }
      if (auto tuple_gte = DynCast<HloGetTupleElementInstruction>(user)) {
        if (op->opcode() == HloOpcode::kConditional) {
          auto tuple = op->branch_computation(0)->root_instruction();
          if (tuple->opcode() == HloOpcode::kTuple) {
            auto index = tuple_gte->tuple_index();
            CHECK(index < tuple->operand_count());
            op = tuple->mutable_operand(index);
          }
        }
        reuses += ReusesCarriedBy(op, user->users()[0]);
      } else {
        reuses += ReusesCarriedBy(op, user);
      }
    }
    VLOG(2) << "Reuses before instruction " << user->ToString() << ":" << reuses
            << "\n";
    return reuses;
  }
  int64_t ReusesAfterBoundary(HloInstruction* user, int64_t tuple_idx = -1) {
    CHECK(user != nullptr);
    if (user->opcode() == HloOpcode::kConstant) {
      return 0;
    }
    auto all_users = user->users();
    if (tuple_idx < 0 && all_users.size() > 1) {
      VLOG(2) << "Having multiple users from: " << user->ToString() << "\n";
      return 0;
    }
    if (!all_users.empty()) {
      auto op = all_users[0];
      int64_t reuses = 0;
      if (tuple_idx >= 0) {
        VLOG(2) << "Reuse for conditional operands with tuple index = "
                << tuple_idx << "\n";
        VLOG(2) << "user op = " << op->ToString();
        if (op->opcode() == HloOpcode::kConditional) {
          int64_t reuse_count = 0;
          for (int64_t i = 0; i < conditional_->branch_count(); ++i) {
            VLOG(5) << "Counting use in branch " << i << "\n";
            if (conditional_->operand(i + 1) != user) {
              continue;
            }
            CHECK_EQ(conditional_->branch_computation(i)
                         ->parameter_instructions()
                         .size(),
                     1);
            auto param_i =
                conditional_->branch_computation(i)->parameter_instruction(0);
            if (param_i ==
                conditional_->branch_computation(i)->root_instruction()) {
              VLOG(5) << "parameter is root.\n";
              reuse_count++;
              continue;
            }
            if (!param_i->shape().IsTuple() && param_i->user_count() > 0) {
              VLOG(5) << "parameter is not tuple and is used. \n";
              reuse_count++;
              continue;
            }
            for (auto* param_i_user : param_i->users()) {
              if (param_i_user->opcode() == HloOpcode::kGetTupleElement &&
                  param_i_user->tuple_index() == tuple_idx) {
                reuse_count++;
                VLOG(5) << "Found user" << param_i_user->ToString() << "\n";
                break;
              }
            }
          }
          VLOG(2) << "Reuse count for conditional:" << reuse_count << "\n";
          if (reuse_count < conditional_->branch_count()) {
            reuses += 10;
          }
        } else if (op->opcode() == HloOpcode::kTuple) {
          VLOG(2) << "new tuple index = " << op->operand_index(user);
          return ReusesAfterBoundary(op, op->operand_index(user));
        } else {
          return ReusesAfterBoundary(op, tuple_idx);
        }
      } else if (op ==
                 conditional_->branch_computation(0)->root_instruction()) {
        int64_t index = op->operand_index(user);
        for (auto op2 : conditional_->users()) {
          if (op2->opcode() == HloOpcode::kGetTupleElement) {
            auto tuple_opd = static_cast<HloGetTupleElementInstruction*>(op2);
            if (index == tuple_opd->tuple_index()) {
              all_users = op2->users();
              if (!all_users.empty()) {
                reuses += ReusesCarriedBy(user, all_users[0]);
                break;
              }
            }
          }
        }
      } else if (ContainsKey(visited_count_, op)) {
        reuses += ReusesCarriedBy(user, op);
      }
      VLOG(2) << "reuses after instruction " << user->ToString() << ":"
              << reuses << "\n";
      return reuses;
    }
    return 0;
  }
  int64_t BenefitForMovingBoundaries(const std::vector<Boundary>& boundaries,
                                     bool perform_reuse_analysis = true) {
    int64_t reuses_before = 0, reuses_after = 0;
    if ((boundaries[0].IsInsideBranch() ||
         boundaries[0].IsOutsideBranchOperand()) &&
        absl::c_count_if(boundaries, [](const Boundary& b) {
          return b.operands()[0]->opcode() != HloOpcode::kTuple;
        }) == 0) {
      return -1;
    }
    if (boundaries.size() == 1) {
      if (boundaries[0].IsOutsideBranchUser() &&
          boundaries[0].operands()[0]->opcode() ==
              HloOpcode::kGetTupleElement) {
        return -1;
      }
    }
    if (!perform_reuse_analysis) {
      return 1;
    }
    auto get_copy_folding_benefit = [&](HloInstruction* hlo) -> int64_t {
      if (hlo->opcode() != HloOpcode::kCopy) {
        return 0;
      }
      const HloGetTupleElementInstruction* gte =
          DynCast<HloGetTupleElementInstruction>(hlo->operand(0));
      if (gte == nullptr) {
        return 0;
      }
      const HloInstruction* conditional = gte->operand(0);
      if (conditional != conditional_) {
        return 0;
      }
      int64_t benefit = 0;
      for (auto* branch : conditional->called_computations()) {
        HloInstruction* root = branch->root_instruction();
        if (root->opcode() == HloOpcode::kTuple) {
          const auto* tuple_operand = root->operand(gte->tuple_index());
          if (tuple_operand->opcode() == HloOpcode::kCopy) {
            if (Shape::Equal()(tuple_operand->operand(0)->shape(),
                               hlo->shape())) {
              benefit += 10;
            }
          }
        }
      }
      return benefit;
    };
    for (const Boundary& b : boundaries) {
      auto op = b.operands()[0];
      if (op == conditional_->branch_computation(0)->root_instruction()) {
        continue;
      }
      VLOG(2) << "Benefit for " << op->ToString();
      reuses_before += ReusesBeforeBoundary(op);
      VLOG(2) << "Reuses before boundary so far: " << reuses_before << "\n";
      reuses_after += ReusesAfterBoundary(
          op, boundaries[0].IsOutsideBranchOperand() ? 0 : -1);
      VLOG(2) << "Reuese after boundary so far : " << reuses_after << "\n";
    }
    int64_t copy_folding_benefit = 0;
    if (boundaries[0].IsOutsideBranchUser()) {
      for (const Boundary& b : boundaries) {
        auto op = b.operands()[0];
        copy_folding_benefit += get_copy_folding_benefit(op);
      }
    }
    VLOG(2) << "Copy folding benefit: " << copy_folding_benefit;
    if (reuses_after == 0 && reuses_before == 0 && copy_folding_benefit == 0) {
      return -1;
    } else if (boundaries[0].IsInsideBranch()) {
      return reuses_after - reuses_before;
    } else if (boundaries[0].IsOutsideBranchUser()) {
      return reuses_before - reuses_after - 1 + copy_folding_benefit;
    } else {
      CHECK(boundaries[0].IsOutsideBranchOperand());
      return reuses_after > 0;
    }
  }
  Boundary GetNextBoundary(const Boundary& b, int64_t op_index) {
    Boundary b2(b.GetPosition());
    for (int j = 0; j < b.operands().size(); ++j) {
      HloInstruction* inst = b.operands()[j];
      CHECK(inst != nullptr);
      HloInstruction* op = (b.IsInsideBranch() || b.IsOutsideBranchOperand())
                               ? inst->operands()[op_index]
                               : inst->users()[op_index];
      CHECK(op != nullptr);
      b2.mutable_operands().push_back(op);
    }
    return b2;
  }
  bool IsSafeToMoveBoundary(const Boundary& next_boundary) {
    VLOG(1) << "Check is safe to move boundary.\n";
    int64_t next_boundary_count =
        (next_boundary.IsInsideBranch() ||
         next_boundary.IsOutsideBranchOperand())
            ? next_boundary.operands()[0]->user_count()
            : CountNonLeafOps(next_boundary.operands()[0]->operands());
    if (next_boundary_count <= 1) {
      if (next_boundary.IsOutsideBranchOperand() &&
          next_boundary.operands()[0]->users()[0] == conditional_ &&
          next_boundary.operands()[0] == conditional_->operand(0)) {
        return false;
      }
      return true;
    } else {
      if (!ContainsKey(visited_count_, next_boundary.operands()[0])) {
        VLOG(1) << "Skip next boundary " << next_boundary.ToString() << "\n"
                << " because it has multiple dependents: "
                << next_boundary_count << "\n";
        visited_count_[next_boundary.operands()[0]] = 1;
        new_boundaries_.push_back(next_boundary);
      } else {
        auto pos = std::find(new_boundaries_.begin(), new_boundaries_.end(),
                             next_boundary);
        if (pos != new_boundaries_.end() ||
            next_boundary.operands().size() == 1) {
          int count = ++visited_count_[next_boundary.operands()[0]];
          if (count == next_boundary_count) {
            VLOG(2) << "Recovering next boundary " << next_boundary.ToString()
                    << "\n"
                    << " because all of its dependents have been visited: "
                    << next_boundary_count << "\n";
            visited_count_.erase(next_boundary.operands()[0]);
            if (pos != new_boundaries_.end()) {
              new_boundaries_.erase(pos);
            }
            return true;
          }
        } else {
          VLOG(1) << "Skip incompatible multi-dependent boundary: "
                  << next_boundary.ToString() << ":" << next_boundary_count
                  << "\n";
        }
      }
    }
    return false;
  }
  void AddBoundaries(const Boundary& boundary) {
    auto calc_memory_size = [](const HloInstruction* hlo) -> int64_t {
      if (hlo->shape().IsTuple()) {
        return 0;
      }
      return ShapeUtil::ByteSizeOf(hlo->shape(), 1) >> 9;
    };
    BoundaryVisitor visitor;
    visitor.AddToWorkList(boundary);
    int64_t boundary_index = 0;
    while (visitor.HasNextBoundary()) {
      Boundary b = visitor.PopNextBoundary();
      VLOG(1) << "visiting boundary " << b.ToString() << "\n";
      VLOG(1) << "boundary index=" << boundary_index << "\n";
      if ((b.IsOutsideBranchUser() || b.IsOutsideBranchOperand() ||
           InstructionWithinBranchIdentical(b.operands(),
                                            is_layout_sensitive_)) &&
          IsSafeToMoveBoundary(b) &&
          WorthHoisting(b.operands()[0], b.GetPosition(), boundary_index)) {
        connected_boundaries_.push_back(b);
        boundary_index++;
        auto output_size = calc_memory_size(b.operands()[0]);
        connected_boundaries_memory_increase_ -= output_size;
        VLOG(1) << "memory incr = " << connected_boundaries_memory_increase_
                << " after subtracting output size.\n";
        VLOG(1) << "boundary can be moved.";
        int64_t operand_count =
            (b.IsInsideBranch() || b.IsOutsideBranchOperand())
                ? b.operands()[0]->operand_count()
                : b.operands()[0]->users().size();
        for (int i = 0; i < operand_count; i++) {
          Boundary next_boundary = GetNextBoundary(b, i);
          VLOG(1) << "Add operand/user " << i << " to visit later\n";
          visitor.AddToWorkList(next_boundary);
          connected_boundaries_memory_increase_ +=
              calc_memory_size(next_boundary.operands()[0]);
          VLOG(1) << "memory incr = " << connected_boundaries_memory_increase_
                  << " after adding shape size of operand " << i << "\n";
        }
      } else if (b.IsOutsideBranchOperand() &&
                 b.operands()[0]->opcode() == HloOpcode::kBroadcast &&
                 connected_boundaries_.size() > 1 &&
                 absl::c_find(
                     b.operands()[0]->users(),
                     connected_boundaries_[connected_boundaries_.size() - 1]
                         .operands()[0]) != b.operands()[0]->users().end() &&
                 connected_boundaries_[connected_boundaries_.size() - 1]
                         .operands()[0]
                         ->opcode() != HloOpcode::kTuple) {
        VLOG(1) << "Replicating multi-use broadcasts:" << b.ToString() << "\n";
        connected_boundaries_.push_back(b);
        auto output_size = calc_memory_size(b.operands()[0]) -
                           calc_memory_size(b.operands()[0]->operand(0));
        connected_boundaries_memory_increase_ -= output_size;
        VLOG(1) << "memory incr = " << connected_boundaries_memory_increase_;
        VLOG(1) << "boundary can be moved.";
      } else {
        VLOG(1) << "boundary cannot be moved\n";
        visited_count_[b.operands()[0]] = 1;
        new_boundaries_.push_back(b);
      }
    }
  }
  std::pair<std::vector<Boundary>, int64_t> BoundariesToMoveInOrOut(
      HloInstruction* conditional, const Boundary& b) {
    HloInstruction* inst = b.operands()[0];
    if (inst == conditional) {
      int branch_count = inst->branch_count();
      Boundary boundary_in(Boundary::Position::kInsideBranch);
      for (int i = 0; i < branch_count; i++) {
        HloComputation* branch_computation = inst->branch_computation(i);
        HloInstruction* root_inst = branch_computation->root_instruction();
        CHECK(root_inst != nullptr);
        boundary_in.mutable_operands().push_back(root_inst);
      }
      new_boundaries_.push_back(boundary_in);
      for (auto u : inst->users()) {
        Boundary boundary_in(Boundary::Position::kOutsideBranchUser);
        boundary_in.mutable_operands().push_back(u);
        new_boundaries_.push_back(boundary_in);
      }
      for (int64_t opd_idx = 1; opd_idx < inst->operand_count(); opd_idx++) {
        HloInstruction* u = inst->mutable_operand(opd_idx);
        Boundary boundary_in(Boundary::Position::kOutsideBranchOperand);
        boundary_in.mutable_operands().push_back(u);
        new_boundaries_.push_back(boundary_in);
      }
    } else {
      AddBoundaries(b);
    }
    return std::pair<std::vector<Boundary>, int64_t>(
        connected_boundaries_, connected_boundaries_memory_increase_);
  }
  void AddNewBoundaries(std::vector<Boundary>& b) {
    b.insert(b.end(), new_boundaries_.begin(), new_boundaries_.end());
  }
};
ConditionalCodeMotion::Decision ConditionalCodeMotion::ConsiderCodeMotion(
    HloInstruction* conditional, const Boundary& cur_boundary,
    std::vector<Boundary>& to_move, std::vector<Boundary>& new_boundaries,
    absl::flat_hash_map<HloInstruction*, int>& visited_count) {
  GroupConnectedBoundaries connect(conditional, is_layout_sensitive_,
                                   visited_count, &move_config_, &reuse_config_,
                                   search_config_);
  auto move_in_or_out =
      connect.BoundariesToMoveInOrOut(conditional, cur_boundary);
  if (!move_in_or_out.first.empty()) {
    auto benefit = connect.BenefitForMovingBoundaries(
        move_in_or_out.first, search_config_map_.empty());
    VLOG(2) << "benefit of moving in or out "
            << cur_boundary.operands()[0]->ToString() << ":" << benefit << "\n";
    if (benefit >= 0) {
      if (move_in_or_out.second > 0 &&
          move_in_or_out.second / move_in_or_out.first.size() >
              memory_increase_allowance_) {
        VLOG(1) << "Stop moving operands because of memory pressure: "
                << move_in_or_out.second << " / " << move_in_or_out.first.size()
                << " > " << memory_increase_allowance_ << "\n";
        benefit = -1;
      } else {
        VLOG(1) << "Increase memory pressure by " << move_in_or_out.second
                << "\n";
        memory_increase_ += move_in_or_out.second;
      }
    }
    if (benefit >= 0) {
      new_boundaries.clear();
      connect.AddNewBoundaries(new_boundaries);
      to_move = move_in_or_out.first;
      return Decision(to_move[0].IsInsideBranch()
                          ? Decision::Direction::kMoveOutOfBranch
                          : Decision::Direction::kMoveIntoBranch,
                      benefit);
    } else {
      connect.clear_recently_visited();
    }
  } else {
    connect.AddNewBoundaries(new_boundaries);
  }
  return Decision(Decision::Direction::kNoChange, 0);
}
absl::StatusOr<bool> ConditionalCodeMotion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(2) << "Begin a new pass of conditional code motion optimization.\n";
  if (!ConsumeFuel("conditional_code_motion", [&] {
        return "Skipping conditional opt after allowed limit reaching 0.\n";
      })) {
    return false;
  }
  bool changed = false;
  bool cleanup_changed = false;
  {
    HloPassPipeline subpipeline("before_conditional_code_motion");
    subpipeline.AddPass<HloCSE>(is_layout_sensitive_);
    subpipeline.AddPass<HloDCE>();
    TF_ASSIGN_OR_RETURN(auto cleanup_changed_now,
                        subpipeline.Run(module, execution_threads));
    cleanup_changed |= cleanup_changed_now;
  }
  std::vector<HloInstruction*> conditional_ops;
  absl::flat_hash_map<HloComputation*, int> conditional_computations;
  for (auto* comp : module->MakeComputationPostOrder(execution_threads)) {
    for (auto* instr : comp->MakeInstructionPostOrder()) {
      if (instr->opcode() == HloOpcode::kConditional) {
        int branch_count = instr->branch_count();
        for (int i = 0; i < branch_count; ++i) {
          HloComputation* branch_i = instr->branch_computation(i);
          if (ContainsKey(conditional_computations, branch_i)) {
            conditional_computations[branch_i]++;
          } else {
            conditional_computations[branch_i] = 0;
          }
        }
        if (instr->shape().IsTuple()) {
          bool can_change_tuple_shape = true;
          for (auto user : instr->users()) {
            VLOG(2) << "user is : " << user->ToString() << "\n";
            if (user->opcode() != HloOpcode::kGetTupleElement) {
              can_change_tuple_shape = false;
            }
          }
          if (can_change_tuple_shape) {
            conditional_ops.push_back(instr);
          }
        } else {
          conditional_ops.push_back(instr);
        }
      }
    }
  }
  int64_t conditional_index = 0;
  HloCloneContext clone_context(module);
  for (HloInstruction* conditional : conditional_ops) {
    if (conditional_index == 0 || !search_config_map_.empty()) {
      auto config_entry = search_config_map_.find(conditional_index);
      if (config_entry != search_config_map_.end()) {
        search_config_ = (*config_entry).second;
        VLOG(2) << "config entry value extracted:" << search_config_.size();
        search_config_index_ = 0;
      }
      VLOG(2) << "Obtaining default configuration for conditional "
              << conditional_index << "\n";
      SetDefaultMoveConfig();
      VLOG(2) << "Done obtaining default configuration\n";
      conditional_index++;
    }
    int branch_count = conditional->branch_count();
    bool conditional_is_shared = false;
    for (int i = 0; i < branch_count; ++i) {
      HloComputation* branch_i = conditional->branch_computation(i);
      if (conditional_computations[branch_i] > 0) {
        conditional_is_shared = true;
        break;
      }
    }
    std::vector<std::vector<Boundary>> to_move_out, to_move_in;
    std::vector<std::vector<Boundary>> new_boundaries_for_moveout;
    std::vector<std::vector<Boundary>> new_boundaries_for_movein;
    absl::flat_hash_map<HloInstruction*, int> visited_count;
    int benefit_move_out = 0, benefit_move_in = 0;
    Decision::Direction final_d = Decision::Direction::kNoChange;
    BoundaryVisitor visitor(conditional);
    VLOG(2) << "Analyzing conditional:" << conditional->ToString() << "\n";
    while (visitor.HasNextBoundary()) {
      std::vector<Boundary> to_move, next_boundary;
      Boundary boundary = visitor.PopNextBoundary();
      VLOG(2) << "Analyzing boundary:" << boundary.ToString() << "\n";
      auto d = ConsiderCodeMotion(conditional, boundary, to_move, next_boundary,
                                  visited_count);
      switch (d.GetDirection()) {
        case Decision::Direction::kMoveOutOfBranch:
          VLOG(2) << "Local Decision is move out of branch\n";
          to_move_out.push_back(to_move);
          new_boundaries_for_moveout.push_back(next_boundary);
          benefit_move_out += d.GetBenefit();
          if (benefit_move_out >= benefit_move_in) {
            final_d = Decision::Direction::kMoveOutOfBranch;
            VLOG(2) << "Current Decision is move out of branch ("
                    << to_move_out.size() << ")\n";
          } else {
            VLOG(2) << "Current Decision remains move into branch\n";
          }
          break;
        case Decision::Direction::kMoveIntoBranch:
          VLOG(2) << "Decision is move into branch\n";
          to_move_in.push_back(to_move);
          new_boundaries_for_movein.push_back(next_boundary);
          benefit_move_in += d.GetBenefit();
          if (benefit_move_out >= benefit_move_in) {
            VLOG(2) << "Current Decision remains move out of branch\n";
          } else {
            final_d = Decision::Direction::kMoveIntoBranch;
            VLOG(2) << "Current Decision is move into branch ("
                    << to_move_in.size() << ")\n";
          }
          break;
        case Decision::Direction::kNoChange:
          VLOG(2) << "Decision is no change\n";
          for (const Boundary& b : next_boundary) {
            visitor.AddToWorkList(b);
            VLOG(2) << "Adding new boundary to worklist:" << b.ToString()
                    << "\n";
          }
          break;
      }
    }
    if (final_d != Decision::Direction::kNoChange && conditional_is_shared) {
      for (int i = 0; i < branch_count; ++i) {
        HloComputation* branch_i = conditional->branch_computation(i);
        if (conditional_computations[branch_i] > 0) {
          HloComputation* clone_i =
              conditional->GetModule()->AddEmbeddedComputation(
                  branch_i->Clone("clone", &clone_context));
          conditional->set_branch_computation(i, clone_i);
          conditional_computations[branch_i]--;
          auto update_boundary = [&](Boundary& boundary) {
            auto cloned_instr =
                clone_context.FindInstruction(boundary.operands()[i]);
            CHECK(cloned_instr != nullptr);
            VLOG(2) << "boundary before cloning:" << boundary.operands()[i]
                    << "\n";
            boundary.mutable_operands()[i] = cloned_instr;
            VLOG(2) << "boundary after cloning:" << boundary.operands()[i]
                    << "\n";
          };
          if (final_d == Decision::Direction::kMoveOutOfBranch) {
            for (int i = 0; i < to_move_out.size(); ++i) {
              std::vector<Boundary>& m = to_move_out[i];
              std::for_each(m.begin(), m.end(), update_boundary);
            }
            for (int i = 0; i < new_boundaries_for_moveout.size(); ++i) {
              std::vector<Boundary>& m = new_boundaries_for_moveout[i];
              std::for_each(m.begin(), m.end(), update_boundary);
            }
          }
        }
      }
      VLOG(2) << "Cloned branches as needed: " << conditional->ToString()
              << "\n";
    }
    if (final_d == Decision::Direction::kMoveOutOfBranch) {
      CHECK(to_move_out.size() == new_boundaries_for_moveout.size());
      for (int i = 0; i < to_move_out.size(); ++i) {
        TF_ASSIGN_OR_RETURN(bool result,
                            MoveInstructionOut(conditional, to_move_out[i],
                                               new_boundaries_for_moveout[i]));
        changed |= result;
      }
      VLOG(2) << "Done moving out of branches " << to_move_out.size()
              << " times. \n";
      if (!ConsumeFuel("conditional_code_motion", [&] {
            return "Skipping conditional opt after allowed limit reaching "
                   "0.\n";
          })) {
        break;
      }
    } else if (final_d == Decision::Direction::kMoveIntoBranch) {
      CHECK(to_move_in.size() == new_boundaries_for_movein.size());
      for (int i = 0; i < to_move_in.size(); ++i) {
        if (to_move_in[i].empty()) {
          continue;
        }
        VLOG(2) << "before opt:"
                << conditional->parent()->ToString(
                       HloPrintOptions::Fingerprint());
        if (to_move_in[i][0].IsOutsideBranchOperand()) {
          VLOG(1) << "Modifying code---number of operand boundaries to move in:"
                  << to_move_in[i].size() << "\n";
          TF_ASSIGN_OR_RETURN(bool result, MoveOperandInstructionsIn(
                                               conditional, to_move_in[i]));
          changed |= result;
        } else {
          VLOG(1) << "Modifying code---number of user boundaries to move in:"
                  << to_move_in[i].size() << "\n";
          CHECK(to_move_in[i][0].IsOutsideBranchUser());
          TF_ASSIGN_OR_RETURN(
              bool result, MoveUserInstructionsIn(conditional, to_move_in[i]));
          changed |= result;
        }
        VLOG(2) << "Before removing instructions:"
                << conditional->parent()->ToString() << "\n";
        for (int64_t j = to_move_in[i].size() - 1; j >= 0; j--) {
          Boundary boundary_to_move_in = to_move_in[i][j];
          HloInstruction* op = boundary_to_move_in.operands()[0];
          if (op->user_count() == 0 && op->parent() != nullptr) {
            VLOG(2) << "Removing boundary:" << boundary_to_move_in.ToString()
                    << "\n";
            TF_RETURN_IF_ERROR(conditional->parent()->RemoveInstruction(op));
            VLOG(2) << "Done removing boundary.\n";
          }
        }
        VLOG(2) << "Done moving instructions inside branches\n"
                << conditional->parent()->ToString(
                       HloPrintOptions::Fingerprint())
                << "\n";
        VLOG(2) << "Done moving into branches " << to_move_in.size()
                << " times. \n";
        if (!ConsumeFuel("conditional_code_motion", [&] {
              return "Skipping conditional opt after allowed limit reaching "
                     "0.\n";
            })) {
          break;
        }
      }
    } else if (pursue_full_conditional_code_motion_ && !conditional_is_shared) {
      TF_ASSIGN_OR_RETURN(
          bool convert_result,
          ConvertSpecialMove(conditional, is_layout_sensitive_));
      if (convert_result) {
        VLOG(2) << "Done special moving of convert\n";
        if (!ConsumeFuel("conditional_code_motion", [&] {
              return "Skipping conditional opt after allowed limit reaching "
                     "0.\n";
            })) {
          break;
        }
      }
      changed |= convert_result;
    }
  }
  if (changed) {
    HloPassPipeline subpipeline(
        "after_conditional_code_motion_after_convert_hoisting");
    VLOG(2) << "starting after motion passes: DCE\n";
    subpipeline.AddPass<HloDCE>();
    subpipeline.AddPass<TupleSimplifier>();
    subpipeline.AddPass<HloDCE>();
    TF_ASSIGN_OR_RETURN(auto cleanup_changed_now, subpipeline.Run(module));
    cleanup_changed |= cleanup_changed_now;
  }
  if (cleanup_changed) {
    VLOG(2) << "subpipeline cleanup have modified code\n";
  }
  return changed;
}
void ConditionalCodeMotion::SetDefaultMoveConfig() {
  VLOG(2) << "search_config_index = " << search_config_index_ << "\n";
  VLOG(2) << "search_config_ size = " << search_config_.size() << "\n";
  int64_t cur_search_config = (search_config_index_ < 0 ||
                               search_config_index_ >= search_config_.size())
                                  ? 0
                                  : search_config_[search_config_index_];
  enum class TuningOption {
    kDoNotTune = 0,
    kTuneTransformationDecision = 1,
    kTuneReuseModel = 2,
  };
  TuningOption tuning_option =
      (cur_search_config == 0)  ? TuningOption::kDoNotTune
      : (cur_search_config > 0) ? TuningOption::kTuneTransformationDecision
                                : TuningOption::kTuneReuseModel;
  auto row = HloOpcodeCount();
  auto col = row;
  VLOG(2) << "Start setting default configuration\n";
  reuse_config_.clear();
  move_config_.clear();
  reuse_config_.reserve(row);
  move_config_.reserve(row);
  for (int64_t opcode = 0; opcode < row; ++opcode) {
    std::vector<int64_t> reuse_vec(col, 0);
    for (uint32_t j = 0; j < col; ++j) {
      reuse_vec[j] = ReusesCarriedBy(static_cast<HloOpcode>(opcode),
                                     static_cast<HloOpcode>(j));
    }
    reuse_config_.push_back(reuse_vec);
    std::vector<int64_t> move_vec;
    switch (tuning_option) {
      case TuningOption::kTuneTransformationDecision:
        move_vec.push_back(1);
        break;
      case TuningOption::kTuneReuseModel:
      case TuningOption::kDoNotTune:
        move_vec.reserve(col);
        for (uint32_t j = 0; j < col; ++j) {
          move_vec.push_back(WorthHoisting(static_cast<HloOpcode>(opcode),
                                           static_cast<HloOpcode>(j)));
        }
        break;
    }
    move_config_.push_back(move_vec);
  }
}
void ConditionalCodeMotion::ParseSearchConfiguration(
    const std::string& search_config) {
  if (search_config.empty()) {
    return;
  }
  search_config_index_ = 0;
  std::vector<std::string> configs = absl::StrSplit(search_config, ';');
  for (const std::string& config : configs) {
    std::vector<std::string> specs = absl::StrSplit(config, ',');
    CHECK_EQ(specs.size(), 4);
    int64_t condition_index;
    CHECK(absl::SimpleAtoi(specs[0], &condition_index));
    auto& cur_config_entry = search_config_map_[condition_index];
    int64_t flip_start, max_flip, flip_stride;
    CHECK(absl::SimpleAtoi(specs[1], &flip_start));
    CHECK(absl::SimpleAtoi(specs[2], &max_flip));
    CHECK(absl::SimpleAtoi(specs[3], &flip_stride));
    int64_t cur_config = MakeSearchConfig(flip_start, max_flip, flip_stride);
    cur_config_entry.push_back(cur_config);
    VLOG(2) << "Setting search config " << condition_index << "->" << cur_config
            << "\n";
  }
}
}  
}  