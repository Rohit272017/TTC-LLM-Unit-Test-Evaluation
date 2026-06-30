#include "xla/service/while_loop_simplifier.h"
#include <cstdint>
#include <optional>
#include <utility>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/call_inliner.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/while_loop_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/union_find.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace m = match;
using hlo_query::ContainsInstrWithOpcode;
using std::optional;
static absl::StatusOr<bool> TryRemoveTrivialCompare(HloInstruction* while_op) {
  std::optional<int64_t> indvar_index = GetLoopInductionVarTupleIdx(while_op);
  if (indvar_index.has_value()) {
    if (while_op->operand(0)->operand(*indvar_index)->IsConstant()) {
      const HloConstantInstruction* init_value_hlo =
          Cast<HloConstantInstruction>(
              while_op->operand(0)->operand(*indvar_index));
      std::optional<int64_t> trip_count = MatchTrivialLoopTripCount(
          while_op, indvar_index.value(), init_value_hlo->literal());
      if (trip_count.has_value()) {
        std::optional<int64_t> init_value =
            LiteralUtil::LiteralAsScalarInt64(init_value_hlo->literal());
        for (HloInstruction* body_instr :
             while_op->while_body()->instructions()) {
          HloInstruction* constant;
          if (Match(body_instr,
                    m::Compare(m::GetTupleElement(m::Parameter(),
                                                  indvar_index.value()),
                               m::Constant(&constant).IsConstantScalar()))) {
            std::optional<int64_t> constant_value =
                LiteralUtil::LiteralAsScalarInt64(constant->literal());
            if (constant_value.has_value()) {
              if (constant_value.value() <= init_value.value()) {
                if (body_instr->comparison_direction() ==
                    ComparisonDirection::kLt) {
                  TF_RETURN_IF_ERROR(while_op->while_body()->ReplaceInstruction(
                      body_instr, MakeScalarLike(body_instr, false)));
                  return true;
                } else if (body_instr->comparison_direction() ==
                           ComparisonDirection::kGt) {
                  TF_RETURN_IF_ERROR(while_op->while_body()->ReplaceInstruction(
                      body_instr, MakeScalarLike(body_instr, true)));
                  return true;
                }
              }
              if (constant_value.value() >=
                  init_value.value() + trip_count.value()) {
                if (body_instr->comparison_direction() ==
                    ComparisonDirection::kLt) {
                  TF_RETURN_IF_ERROR(while_op->while_body()->ReplaceInstruction(
                      body_instr, MakeScalarLike(body_instr, true)));
                  return true;
                } else if (body_instr->comparison_direction() ==
                           ComparisonDirection::kGt) {
                  TF_RETURN_IF_ERROR(while_op->while_body()->ReplaceInstruction(
                      body_instr, MakeScalarLike(body_instr, false)));
                  return true;
                }
              }
            }
          }
        }
      }
    }
  }
  return false;
}
void CopyFrontendAttributes(HloInstruction* old_while_op,
                            HloInstruction* new_while_op) {
  new_while_op->add_frontend_attributes(old_while_op->frontend_attributes());
}
void CopyMetadata(HloInstruction* old_while_op, HloInstruction* new_while_op) {
  new_while_op->set_metadata(old_while_op->metadata());
}
static absl::StatusOr<HloInstruction*> RemoveDeadTupleIndices(
    HloInstruction* while_op, absl::flat_hash_set<int64_t>& used_tuple_indices,
    int64_t index_for_replaced = -1) {
  std::vector<int64_t> new_to_old_tuple_idx(used_tuple_indices.begin(),
                                            used_tuple_indices.end());
  absl::c_sort(new_to_old_tuple_idx);
  HloModule* module = while_op->GetModule();
  HloComputation* computation = while_op->parent();
  HloInstruction* while_init = while_op->mutable_operand(0);
  HloComputation* while_cond = while_op->while_condition();
  HloComputation* while_body = while_op->while_body();
  HloInstruction* while_body_root = while_body->root_instruction();
  auto print_no_metadata = HloPrintOptions().set_print_metadata(false);
  absl::flat_hash_map<int64_t, int64_t> old_to_new_tuple_idx;
  for (int64_t new_idx = 0; new_idx < new_to_old_tuple_idx.size(); ++new_idx) {
    int64_t old_idx = new_to_old_tuple_idx[new_idx];
    old_to_new_tuple_idx[old_idx] = new_idx;
    VLOG(2) << "Remapping tuple index " << old_idx << " to " << new_idx;
  }
  std::vector<const Shape*> new_while_tuple_elem_shapes;
  new_while_tuple_elem_shapes.reserve(new_to_old_tuple_idx.size());
  for (int64_t old_idx : new_to_old_tuple_idx) {
    new_while_tuple_elem_shapes.push_back(
        &while_init->shape().tuple_shapes(old_idx));
  }
  Shape new_while_shape =
      ShapeUtil::MakeTupleShapeWithPtrs(new_while_tuple_elem_shapes);
  auto make_while_computation_replacements = [&](const HloComputation* comp) {
    absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
        replacements;
    auto* param = comp->parameter_instruction(0);
    replacements.emplace(param, HloInstruction::CreateParameter(
                                    0, new_while_shape, param->name()));
    std::vector<HloInstruction*> materialized_users(param->users().begin(),
                                                    param->users().end());
    for (const auto* user : materialized_users) {
      if (user == while_body_root) {
        continue;
      }
      CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement)
          << user->ToString(print_no_metadata);
      int64_t old_idx = user->tuple_index();
      auto new_idx_iter = old_to_new_tuple_idx.find(old_idx);
      if (new_idx_iter != old_to_new_tuple_idx.end()) {
        replacements.emplace(
            user, HloInstruction::CreateGetTupleElement(user->shape(), param,
                                                        new_idx_iter->second));
      } else {
        replacements.emplace(user, nullptr);
      }
    }
    for (const auto* hlo : comp->MakeInstructionPostOrder()) {
      if (hlo == comp->root_instruction() || replacements.contains(hlo)) {
        continue;
      }
      for (const auto* operand : hlo->operands()) {
        auto op_it = replacements.find(operand);
        if (op_it != replacements.end() && op_it->second == nullptr) {
          replacements[hlo] = nullptr;
          break;
        }
      }
    }
    return replacements;
  };
  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      while_cond_replacements = make_while_computation_replacements(while_cond);
  std::unique_ptr<HloComputation> new_while_cond =
      while_cond->CloneWithReplacements(&while_cond_replacements);
  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      while_body_replacements = make_while_computation_replacements(while_body);
  std::vector<HloInstruction*> new_while_body_root_elems;
  new_while_body_root_elems.reserve(new_to_old_tuple_idx.size());
  for (int64_t old_idx : new_to_old_tuple_idx) {
    new_while_body_root_elems.push_back(
        while_body_root->mutable_operand(old_idx));
  }
  while_body_replacements.emplace(
      while_body_root, HloInstruction::CreateTuple(new_while_body_root_elems));
  std::unique_ptr<HloComputation> new_while_body =
      while_body->CloneWithReplacements(&while_body_replacements);
  std::vector<HloInstruction*> new_while_init_elems;
  new_while_init_elems.reserve(new_to_old_tuple_idx.size());
  for (int64_t old_idx : new_to_old_tuple_idx) {
    new_while_init_elems.push_back(
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            while_init->shape().tuple_shapes(old_idx), while_init, old_idx)));
  }
  auto* new_while_init = computation->AddInstruction(
      HloInstruction::CreateTuple(new_while_init_elems));
  auto* new_while_op = computation->AddInstruction(HloInstruction::CreateWhile(
      new_while_shape,
      module->AddEmbeddedComputation(std::move(new_while_cond)),
      module->AddEmbeddedComputation(std::move(new_while_body)),
      new_while_init));
  new_while_op->CopyBackendConfigFrom(while_op);
  CopyFrontendAttributes(while_op, new_while_op);
  CopyMetadata(while_op, new_while_op);
  std::vector<HloInstruction*> new_tuple_elems;
  const int64_t tuple_size = ShapeUtil::TupleElementCount(while_init->shape());
  for (int64_t old_idx = 0; old_idx < tuple_size; ++old_idx) {
    auto new_tuple_idx_it = old_to_new_tuple_idx.find(old_idx);
    if (new_tuple_idx_it != old_to_new_tuple_idx.end() ||
        index_for_replaced != -1) {
      int64_t gte_idx = new_tuple_idx_it != old_to_new_tuple_idx.end()
                            ? new_tuple_idx_it->second
                            : index_for_replaced;
      new_tuple_elems.push_back(
          computation->AddInstruction(HloInstruction::CreateGetTupleElement(
              new_while_op->shape().tuple_shapes(gte_idx), new_while_op,
              gte_idx)));
    } else {
      new_tuple_elems.push_back(
          computation->AddInstruction(HloInstruction::CreateGetTupleElement(
              while_init->shape().tuple_shapes(old_idx), while_init, old_idx)));
    }
  }
  HloInstruction* new_tuple =
      computation->AddInstruction(HloInstruction::CreateTuple(new_tuple_elems));
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(while_op, new_tuple));
  return new_while_op;
}
absl::StatusOr<bool> TryRemoveDeadWhileParams(HloInstruction* while_op) {
  CHECK_EQ(while_op->opcode(), HloOpcode::kWhile);
  if (!while_op->parent()->IsSafelyRemovable(while_op)) {
    VLOG(2) << "Can't remove dead parameters from non-removable while op.";
    return false;
  }
  HloInstruction* while_init = while_op->mutable_operand(0);
  HloComputation* while_cond = while_op->while_condition();
  HloComputation* while_body = while_op->while_body();
  HloInstruction* while_body_root = while_body->root_instruction();
  if (!while_init->shape().IsTuple()) {
    VLOG(2) << "While op's carried value isn't tuple shaped.";
    return false;
  }
  if (while_body_root->opcode() != HloOpcode::kTuple) {
    VLOG(2) << "While body's root is not a tuple(...) instruction.";
    return false;
  }
  const int64_t tuple_size = ShapeUtil::TupleElementCount(while_init->shape());
  auto print_no_metadata = HloPrintOptions().set_print_metadata(false);
  absl::flat_hash_set<int64_t> used_tuple_indices;
  for (int64_t i = 0; i < tuple_size; ++i) {
    used_tuple_indices.insert(i);
  }
  for (const HloInstruction* instr : {while_body->parameter_instruction(0),
                                      while_cond->parameter_instruction(0)}) {
    for (const HloInstruction* user : instr->users()) {
      if (user->opcode() != HloOpcode::kGetTupleElement) {
        VLOG(2) << "Cowardly refusing to analyze while loop with "
                << instr->ToString(print_no_metadata)
                << " used by non-GTE instruction "
                << user->ToString(print_no_metadata) << " in computation "
                << instr->parent()->name();
        return false;
      }
    }
  }
  if (tuple_size == 0) {
    VLOG(2) << "Can't remove elements from while loop's tuple -- it's already "
               "empty.";
    return false;
  }
  absl::flat_hash_set<int64_t> used_indices_after_loop;
  if (while_op == while_op->parent()->root_instruction()) {
    for (int64_t i = 0; i < while_body_root->operand_count(); ++i) {
      used_indices_after_loop.insert(i);
    }
  }
  for (auto user : while_op->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      for (int64_t i = 0; i < while_body_root->operand_count(); ++i) {
        used_indices_after_loop.insert(i);
      }
      break;
    }
    used_indices_after_loop.insert(user->tuple_index());
  }
  struct InputIndicesSet {
    void Merge(const InputIndicesSet& other) {
      if (all.size() + other.all.size() <= all.capacity() && owned == nullptr) {
        absl::c_copy(other.all, std::back_inserter(all));
        return;
      }
      if (owned == nullptr) {
        owned = std::make_unique<absl::flat_hash_set<int64_t>>();
        owned->reserve(other.all.front()->size() * 2);
      }
      for (auto* deps : all) {
        if (deps == owned.get()) {
          continue;
        }
        owned->insert(deps->begin(), deps->end());
      }
      for (auto* deps : other.all) {
        owned->insert(deps->begin(), deps->end());
      }
      all.clear();
      all.push_back(owned.get());
    }
    void Add(int64_t index) {
      if (owned == nullptr) {
        CHECK(all.empty());
        owned = std::make_unique<absl::flat_hash_set<int64_t>>();
        all.push_back(owned.get());
      }
      owned->insert(index);
    }
    std::unique_ptr<absl::flat_hash_set<int64_t>> owned;
    absl::InlinedVector<const absl::flat_hash_set<int64_t>*, 4> all;
  };
  absl::flat_hash_map<HloInstruction*, InputIndicesSet> inst_input_deps;
  absl::flat_hash_map<HloInstruction*, UnionFind<HloInstruction*>>
      disjoint_sets;
  for (HloComputation* comp : {while_body, while_cond}) {
    HloInstruction* while_input = comp->parameter_instruction(0);
    for (HloInstruction* inst : comp->instructions()) {
      if (inst == while_input || inst == while_body_root) {
        continue;
      }
      disjoint_sets[inst].Get() = inst;
    }
  }
  absl::flat_hash_set<int64_t> side_effecting_indices;
  for (HloComputation* comp : {while_body, while_cond}) {
    HloInstruction* while_input = comp->parameter_instruction(0);
    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (inst == while_input || inst == while_body_root) {
        continue;
      }
      auto& deps = inst_input_deps[inst];
      auto& my_set = disjoint_sets[inst];
      if (inst->opcode() == HloOpcode::kGetTupleElement &&
          inst->operand(0) == while_input) {
        deps.Add(inst->tuple_index());
        HloInstruction* output =
            while_body_root->mutable_operand(inst->tuple_index());
        if (output != inst) {
          disjoint_sets[output].Merge(&my_set);
        }
      } else {
        for (HloInstruction* operand : inst->operands()) {
          disjoint_sets[operand].Merge(&my_set);
          deps.Merge(inst_input_deps[operand]);
        }
      }
      if (inst->HasSideEffect() || inst == while_cond->root_instruction()) {
        for (auto* dep : deps.all) {
          side_effecting_indices.insert(dep->begin(), dep->end());
        }
      }
    }
  }
  absl::flat_hash_set<int64_t> indices_affecting_others;
  for (int64_t i = 0; i < tuple_size; ++i) {
    HloInstruction* output = while_body_root->mutable_operand(i);
    for (auto* deps : inst_input_deps[output].all) {
      for (int64_t index : *deps) {
        if (index != i) {
          indices_affecting_others.insert(index);
        }
      }
    }
  }
  for (int64_t i = 0; i < tuple_size; ++i) {
    if (!indices_affecting_others.contains(i) &&
        !used_indices_after_loop.contains(i) &&
        !side_effecting_indices.contains(i)) {
      VLOG(2) << "Remove with dependencies " << i;
      used_tuple_indices.erase(i);
    }
  }
  absl::flat_hash_map<HloInstruction*, absl::flat_hash_set<int64_t>> groups;
  for (int64_t i = 0; i < tuple_size; ++i) {
    HloInstruction* output = while_body_root->mutable_operand(i);
    groups[disjoint_sets[output].Get()].insert(i);
  }
  for (HloComputation* comp : {while_body, while_cond}) {
    HloInstruction* while_input = comp->parameter_instruction(0);
    for (HloInstruction* gte : while_input->users()) {
      groups[disjoint_sets[gte].Get()].insert(gte->tuple_index());
    }
  }
  for (const auto& group : groups) {
    if (absl::c_any_of(group.second, [&](int64_t index) {
          const HloInstruction* output = while_body_root->operand(index);
          return side_effecting_indices.contains(index) ||
                 (used_indices_after_loop.contains(index) &&
                  !(output->opcode() == HloOpcode::kGetTupleElement &&
                    output->operand(0) ==
                        while_body->parameter_instruction(0) &&
                    output->tuple_index() == index));
        })) {
      continue;
    }
    VLOG(2) << "Remove with groups:";
    for (int64_t index : group.second) {
      VLOG(2) << "    index " << index;
      used_tuple_indices.erase(index);
    }
  }
  if (used_tuple_indices.size() == tuple_size) {
    VLOG(2) << "Loop " << while_op->ToString(print_no_metadata)
            << " uses all of its inputs; no simplification possible.";
    return false;
  }
  CHECK_LT(used_tuple_indices.size(), tuple_size);
  VLOG(1) << "Eliminating " << tuple_size - used_tuple_indices.size()
          << " elements from tuple of "
          << while_op->ToString(print_no_metadata);
  TF_ASSIGN_OR_RETURN(while_op,
                      RemoveDeadTupleIndices(while_op, used_tuple_indices));
  return true;
}
static absl::StatusOr<HloInstruction*> TryRemoveRepeatedWhileTupleIndicesHelper(
    HloInstruction* while_op, const int64_t tuple_index, bool replace_with_init,
    absl::flat_hash_set<int64_t>& duplicates) {
  HloComputation* while_cond = while_op->while_condition();
  HloComputation* while_body = while_op->while_body();
  HloInstruction* while_init = while_op->mutable_operand(0);
  VLOG(2) << "while_init " << while_init->ToString() << " operands "
          << while_init->operand_count();
  VLOG(2) << "while_body_root " << while_body->root_instruction()->ToString()
          << " operands " << while_body->root_instruction()->operand_count();
  for (HloComputation* comp : {while_body, while_cond}) {
    auto new_get = comp->AddInstruction(HloInstruction::CreateGetTupleElement(
        comp->parameter_instruction(0)->shape().tuple_shapes(tuple_index),
        comp->parameter_instruction(0), tuple_index));
    std::vector<HloInstruction*> instrs_to_replace;
    for (auto* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kGetTupleElement &&
          duplicates.contains(instr->tuple_index()) &&
          instr->operand(0) == comp->parameter_instruction(0)) {
        instrs_to_replace.push_back(instr);
      }
    }
    for (auto instr : instrs_to_replace) {
      TF_RETURN_IF_ERROR(comp->ReplaceInstruction(instr, new_get));
    }
  }
  absl::flat_hash_set<int64_t> used_tuple_indices;
  for (int index = 0; index < while_init->shape().tuple_shapes_size();
       ++index) {
    if (!duplicates.count(index)) {
      used_tuple_indices.insert(index);
    }
  }
  TF_ASSIGN_OR_RETURN(
      while_op, RemoveDeadTupleIndices(while_op, used_tuple_indices,
                                       replace_with_init ? -1 : tuple_index));
  return while_op;
}
static bool IsDynamicUpdateSliceWhileInsertion(
    const HloInstruction* instr, const HloComputation* while_body) {
  return instr->opcode() == HloOpcode::kDynamicUpdateSlice &&
         instr->operand(0)->opcode() == HloOpcode::kGetTupleElement &&
         instr->operand(0)->operand(0) == while_body->parameter_instruction(0);
}
static absl::StatusOr<bool> TryRemoveRepeatedWhileTupleIndices(
    HloInstruction* while_op) {
  CHECK_EQ(while_op->opcode(), HloOpcode::kWhile);
  int index_to_investigate = 0;
  if (!while_op->parent()->IsSafelyRemovable(while_op)) {
    VLOG(2) << "Can't remove dead parameters from non-removable while op.";
    return false;
  }
  HloInstruction* while_init = while_op->mutable_operand(0);
  HloComputation* while_cond = while_op->while_condition();
  HloComputation* while_body = while_op->while_body();
  HloInstruction* while_body_root = while_body->root_instruction();
  if (!while_init->shape().IsTuple()) {
    VLOG(2) << "While op's carried value isn't tuple shaped.";
    return false;
  }
  bool changed = false;
  while (index_to_investigate < while_init->shape().tuple_shapes_size()) {
    if (!while_init->shape().IsTuple() ||
        while_init->opcode() != HloOpcode::kTuple) {
      VLOG(2) << "While op's carried value isn't tuple shaped.";
      return false;
    }
    if (while_body_root->opcode() != HloOpcode::kTuple) {
      VLOG(2) << "While body's root is not a tuple(...) instruction.";
      return false;
    }
    auto& while_shape = while_init->shape();
    VLOG(2) << "Iterating " << index_to_investigate;
    absl::flat_hash_set<int64_t> duplicates;
    auto* pivot_init_elem = while_init->operand(index_to_investigate);
    auto* pivot_body_elem = while_body_root->operand(index_to_investigate);
    bool replace_with_init = true;
    if (pivot_body_elem->opcode() == HloOpcode::kGetTupleElement &&
        pivot_body_elem->operand(0) == while_body->parameter_instruction(0)) {
      if (pivot_body_elem->tuple_index() != index_to_investigate) {
        VLOG(2) << "Mismatch between pivot_body_elem->tuple_index() "
                << pivot_body_elem->tuple_index() << " index_to_investigate "
                << index_to_investigate;
        index_to_investigate++;
        continue;
      }
    } else if (IsDynamicUpdateSliceWhileInsertion(pivot_body_elem,
                                                  while_body)) {
      if (pivot_body_elem->operand(0)->tuple_index() != index_to_investigate) {
        VLOG(2)
            << "Mismatch between pivot_body_elem->operand(0)->tuple_index() "
            << pivot_body_elem->operand(0)->tuple_index()
            << " index_to_investigate " << index_to_investigate;
        index_to_investigate++;
        continue;
      }
    } else {
      index_to_investigate++;
      continue;
    }
    for (int64_t i = index_to_investigate + 1;
         i < while_shape.tuple_shapes_size(); ++i) {
      auto* init_elem = while_init->operand(i);
      auto* body_elem = while_body_root->operand(i);
      if (pivot_body_elem->opcode() == HloOpcode::kGetTupleElement &&
          body_elem->opcode() == HloOpcode::kGetTupleElement &&
          body_elem->operand(0) == while_body->parameter_instruction(0)) {
        if (body_elem->tuple_index() != i) {
          VLOG(2) << "Mismatch between body_elem->tuple_index() "
                  << body_elem->tuple_index() << " i " << i;
          continue;
        }
      } else if (IsDynamicUpdateSliceWhileInsertion(pivot_body_elem,
                                                    while_body) &&
                 IsDynamicUpdateSliceWhileInsertion(body_elem, while_body)) {
        if (pivot_body_elem->operand_count() != body_elem->operand_count()) {
          VLOG(2) << "Mismatch in operand count of dynamic-update-slice "
                  << pivot_body_elem->operand_count() << " vs "
                  << body_elem->operand_count();
          continue;
        }
        if (body_elem->operand(0)->tuple_index() != i) {
          VLOG(2) << "Mismatch between body_elem->operand(0)->tuple_index() "
                  << body_elem->tuple_index() << " i " << i;
          continue;
        }
        if (pivot_body_elem->operand(0) == body_elem->operand(0)) {
          VLOG(2) << "Inserting in the same input index";
          continue;
        }
        bool mismatch = false;
        for (int64_t i = 1; i < body_elem->operand_count(); ++i) {
          if (body_elem->operand(i) != pivot_body_elem->operand(i)) {
            VLOG(2) << "Mismatch in insertion indices or values";
            mismatch = true;
            break;
          }
        }
        if (mismatch) {
          continue;
        }
        replace_with_init = false;
      } else {
        continue;
      }
      if (pivot_init_elem == init_elem) {
        VLOG(2) << "init_elem " << init_elem->ToString() << " pivot_init_elem "
                << pivot_init_elem->ToString();
        VLOG(2) << "body_elem " << body_elem->ToString() << " pivot_body_elem "
                << pivot_body_elem->ToString();
        duplicates.insert(i);
      }
    }
    if (!duplicates.empty()) {
      VLOG(2) << "Duplicate found " << duplicates.size() << " pivot_init "
              << pivot_init_elem->ToString();
      TF_ASSIGN_OR_RETURN(while_op, TryRemoveRepeatedWhileTupleIndicesHelper(
                                        while_op, index_to_investigate,
                                        replace_with_init, duplicates));
      changed = true;
      VLOG(2) << "Changed while_op " << while_op->ToString()
              << " while_op operand count " << while_op->operand_count();
      while_init = while_op->mutable_operand(0);
      while_cond = while_op->while_condition();
      while_body = while_op->while_body();
      while_body_root = while_body->root_instruction();
    }
    index_to_investigate++;
  }
  return changed;
}
static absl::StatusOr<bool> TryRemoveConstantParams(HloInstruction* while_op) {
  HloModule* module = while_op->GetModule();
  HloComputation* computation = while_op->parent();
  auto* while_init = while_op->mutable_operand(0);
  auto* while_body = while_op->while_body();
  auto* while_cond = while_op->while_condition();
  auto* while_body_root = while_body->root_instruction();
  if (while_init->opcode() != HloOpcode::kTuple ||
      while_body_root->opcode() != HloOpcode::kTuple) {
    return false;
  }
  TF_RET_CHECK(while_cond->num_parameters() == 1);
  TF_RET_CHECK(while_body->num_parameters() == 1);
  TF_RET_CHECK(
      ShapeUtil::Compatible(while_init->shape(), while_body_root->shape()));
  absl::flat_hash_set<int64_t> constant_tuple_indices;
  const auto& while_shape = while_init->shape();
  for (int i = 0; i < while_shape.tuple_shapes_size(); ++i) {
    auto* init_elem = while_init->operand(i);
    auto* body_elem = while_body_root->operand(i);
    if (init_elem->opcode() == HloOpcode::kConstant &&
        body_elem->opcode() == HloOpcode::kConstant &&
        init_elem->literal() == body_elem->literal()) {
      constant_tuple_indices.insert(i);
    }
  }
  if (constant_tuple_indices.empty()) {
    return false;
  }
  std::vector<const Shape*> new_while_shape_elems;
  for (int i = 0; i < while_shape.tuple_shapes_size(); ++i) {
    if (!constant_tuple_indices.count(i)) {
      new_while_shape_elems.push_back(&while_shape.tuple_shapes(i));
    }
  }
  Shape new_while_shape =
      ShapeUtil::MakeTupleShapeWithPtrs(new_while_shape_elems);
  std::vector<std::unique_ptr<HloInstruction>> new_instrs;
  auto add_new_instr = [&](std::unique_ptr<HloInstruction> instr) {
    new_instrs.push_back(std::move(instr));
    return new_instrs.back().get();
  };
  auto remove_constant_elems = [&](HloInstruction* instr) {
    CHECK(ShapeUtil::Compatible(instr->shape(), while_shape));
    std::vector<HloInstruction*> tuple_elems;
    for (int i = 0; i < while_shape.tuple_shapes_size(); ++i) {
      if (!constant_tuple_indices.count(i)) {
        tuple_elems.push_back(
            add_new_instr(HloInstruction::CreateGetTupleElement(
                while_shape.tuple_shapes(i), instr, i)));
      }
    }
    return HloInstruction::CreateTuple(tuple_elems);
  };
  auto add_constant_elems = [&](HloInstruction* instr) {
    CHECK(ShapeUtil::Compatible(instr->shape(), new_while_shape));
    std::vector<HloInstruction*> tuple_elems;
    int64_t j = 0;
    for (int i = 0; i < while_shape.tuple_shapes_size(); ++i) {
      if (constant_tuple_indices.count(i)) {
        tuple_elems.push_back(while_init->mutable_operand(i));
      } else {
        tuple_elems.push_back(
            add_new_instr(HloInstruction::CreateGetTupleElement(
                while_shape.tuple_shapes(i), instr, j)));
        ++j;
      }
    }
    return HloInstruction::CreateTuple(tuple_elems);
  };
  if (ShapeUtil::IsEmptyTuple(new_while_shape)) {
    TF_RETURN_IF_ERROR(computation->ReplaceInstruction(while_op, while_init));
    return true;
  }
  std::unique_ptr<HloComputation> new_while_cond =
      while_cond->CloneWithReplacementPairs({
          while_cond->parameter_instruction(0),
          add_constant_elems(add_new_instr(HloInstruction::CreateParameter(
              0, new_while_shape,
              while_cond->parameter_instruction(0)->name()))),
      });
  std::unique_ptr<HloComputation> new_while_body =
      while_body->CloneWithReplacementPairs(
          {
              while_body->parameter_instruction(0),
              add_constant_elems(add_new_instr(HloInstruction::CreateParameter(
                  0, new_while_shape,
                  while_cond->parameter_instruction(0)->name()))),
          },
          {
              while_body->root_instruction(),
              remove_constant_elems(
                  add_new_instr(while_body->root_instruction()->Clone())),
          });
  new_instrs.clear();
  auto* new_while_op = computation->AddInstruction(HloInstruction::CreateWhile(
      new_while_shape,
      module->AddEmbeddedComputation(std::move(new_while_cond)),
      module->AddEmbeddedComputation(std::move(new_while_body)),
      add_new_instr(remove_constant_elems(while_init))));
  new_while_op->CopyBackendConfigFrom(while_op);
  CopyFrontendAttributes(while_op, new_while_op);
  CopyMetadata(while_op, new_while_op);
  TF_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
      while_op, add_constant_elems(new_while_op)));
  for (auto& instr : new_instrs) {
    computation->AddInstruction(std::move(instr));
  }
  return true;
}
static absl::StatusOr<bool> TryRemoveWhileLoop(HloInstruction* while_op) {
  if (!while_op->parent()->IsSafelyRemovable(while_op)) {
    VLOG(2) << "Not attempting to remove while loop that is not removable: "
            << while_op->ToShortString();
    return false;
  }
  if (while_op->while_condition()->HasSideEffect()) {
    VLOG(2) << "Not attempting to remove while loop whose condition contains "
               "side-effecting instructions: "
            << while_op->ToShortString();
    return false;
  }
  optional<int64_t> trip_count =
      ComputeWhileLoopTripCount(while_op, 1);
  if (trip_count && *trip_count == 0) {
    auto computation = while_op->parent();
    TF_RETURN_IF_ERROR(computation->ReplaceInstruction(
        while_op, while_op->mutable_operand(0)));
    return true;
  }
  const auto& attrs = while_op->frontend_attributes().map();
  bool skip_trip_count_one_simplification =
      attrs.contains("skip-simplify-while-loops_trip-count-one") &&
      (attrs.at("skip-simplify-while-loops_trip-count-one") == "true");
  if (trip_count && *trip_count == 1 && !skip_trip_count_one_simplification) {
    bool has_side_effects = absl::c_any_of(
        while_op->called_computations(), [](const HloComputation* computation) {
          return computation->HasSideEffect();
        });
    if (!has_side_effects) {
      auto computation = while_op->parent();
      auto call_op = computation->AddInstruction(HloInstruction::CreateCall(
          while_op->shape(), while_op->operands(), while_op->while_body()));
      TF_RETURN_IF_ERROR(computation->ReplaceInstruction(while_op, call_op));
      TF_ASSIGN_OR_RETURN(auto inlined_instructions_map,
                          CallInliner::Inline(call_op));
      (void)inlined_instructions_map;
      return true;
    } else {
      VLOG(2) << "Not attempting to simplify while loop because it contains a "
                 "side-effecting node: "
              << while_op->ToShortString();
    }
  }
  return false;
}
static absl::StatusOr<bool> TryPropagateConstant(HloInstruction* while_op) {
  auto while_init = while_op->operand(0);
  if (while_init->opcode() != HloOpcode::kTuple) {
    return false;
  }
  auto while_body = while_op->while_body();
  auto while_body_root = while_body->root_instruction();
  if (while_body_root->opcode() != HloOpcode::kTuple) {
    return false;
  }
  auto while_body_param = while_body->parameter_instruction(0);
  const HloInstruction::InstructionVector& root_operands =
      while_body_root->operands();
  absl::flat_hash_map<int, const HloInstruction*> index_to_constant;
  for (int i = 0; i < root_operands.size(); i++) {
    const HloInstruction* init_tuple_elem = nullptr;
    if (Match(root_operands[i],
              m::GetTupleElement(m::Op().Is(while_body_param), i)
                  .WithShape(m::Shape().IsScalar())) &&
        Match(while_init->operand(i), m::Constant(&init_tuple_elem))) {
      VLOG(3) << "Found loop invariant tuple element " << i << " "
              << init_tuple_elem->ToString();
      index_to_constant[i] = init_tuple_elem;
    }
  }
  if (index_to_constant.empty()) {
    return false;
  }
  auto propagate_constant =
      [&](HloComputation* computation) -> absl::StatusOr<bool> {
    HloInstruction* param = computation->parameter_instruction(0);
    bool changed = false;
    for (auto instr : param->users()) {
      if (instr->opcode() == HloOpcode::kGetTupleElement) {
        VLOG(3) << "tuple index " << instr->tuple_index() << " "
                << instr->ToString();
        auto iter = index_to_constant.find(instr->tuple_index());
        if (iter != index_to_constant.end()) {
          const HloInstruction* hlo_constant = (*iter).second;
          VLOG(3) << "Replace use of " << instr->ToString() << " with "
                  << hlo_constant->ToString();
          TF_RETURN_IF_ERROR(instr->ReplaceAllUsesWith(
              computation->AddInstruction(hlo_constant->Clone())));
          changed = true;
        }
      }
    }
    return changed;
  };
  TF_ASSIGN_OR_RETURN(bool changed_cond,
                      propagate_constant(while_op->while_condition()));
  TF_ASSIGN_OR_RETURN(bool changed_body, propagate_constant(while_body));
  return changed_cond || changed_body;
}
static std::unique_ptr<HloInstruction> UnflattenTupleInstr(
    absl::Span<HloInstruction*> instrs, const Shape& desired_shape,
    std::vector<std::unique_ptr<HloInstruction>>* new_instrs) {
  CHECK(desired_shape.IsTuple()) << ShapeUtil::HumanString(desired_shape);
  std::vector<HloInstruction*> elems;
  for (int i = 0; i < desired_shape.tuple_shapes_size(); ++i) {
    const Shape& subshape = desired_shape.tuple_shapes(i);
    if (!subshape.IsTuple()) {
      elems.push_back(instrs[0]);
      instrs.remove_prefix(1);
      continue;
    }
    int64_t num_leaves = 0;
    ShapeUtil::ForEachSubshape(
        subshape, [&](const Shape& s, const ShapeIndex& ) {
          if (!s.IsTuple()) {
            ++num_leaves;
          }
        });
    std::unique_ptr<HloInstruction> subinstr =
        UnflattenTupleInstr(instrs.subspan(0, num_leaves),
                            desired_shape.tuple_shapes(i), new_instrs);
    elems.push_back(subinstr.get());
    new_instrs->push_back(std::move(subinstr));
    instrs.remove_prefix(num_leaves);
  }
  return HloInstruction::CreateTuple(elems);
}
static std::vector<HloInstruction*> GetFlatTupleElems(
    HloInstruction* instr,
    std::vector<std::unique_ptr<HloInstruction>>* new_instrs) {
  const auto& shape = instr->shape();
  if (!shape.IsTuple()) {
    return {instr};
  }
  std::vector<HloInstruction*> elems;
  for (int i = 0; i < shape.tuple_shapes_size(); ++i) {
    const Shape& subshape = shape.tuple_shapes(i);
    new_instrs->push_back(
        HloInstruction::CreateGetTupleElement(subshape, instr, i));
    auto* gte = new_instrs->back().get();
    auto flattened_subshape = GetFlatTupleElems(gte, new_instrs);
    elems.insert(elems.end(), flattened_subshape.begin(),
                 flattened_subshape.end());
  }
  return elems;
}
static absl::StatusOr<bool> TryFlattenNestedTuples(HloInstruction* while_op) {
  HloModule* module = while_op->GetModule();
  HloComputation* computation = while_op->parent();
  auto* while_init = while_op->mutable_operand(0);
  auto* while_body = while_op->while_body();
  auto* while_cond = while_op->while_condition();
  auto* while_body_root = while_body->root_instruction();
  if (while_init->opcode() != HloOpcode::kTuple ||
      while_body_root->opcode() != HloOpcode::kTuple) {
    return false;
  }
  TF_RET_CHECK(while_cond->num_parameters() == 1);
  TF_RET_CHECK(while_body->num_parameters() == 1);
  TF_RET_CHECK(
      ShapeUtil::Compatible(while_init->shape(), while_body_root->shape()));
  Shape while_shape = while_init->shape();
  if (!ShapeUtil::IsNestedTuple(while_shape)) {
    return false;
  }
  std::vector<const Shape*> flattened_shape_elems;
  ShapeUtil::ForEachSubshape(while_shape,
                             [&](const Shape& s, const ShapeIndex& ) {
                               if (!s.IsTuple()) {
                                 flattened_shape_elems.push_back(&s);
                               }
                             });
  Shape flattened_shape =
      ShapeUtil::MakeTupleShapeWithPtrs(flattened_shape_elems);
  std::vector<std::unique_ptr<HloInstruction>> new_instrs;
  auto add_new_instr = [&](std::unique_ptr<HloInstruction> instr) {
    new_instrs.push_back(std::move(instr));
    return new_instrs.back().get();
  };
  auto nested = [&](HloInstruction* instr) {
    std::vector<HloInstruction*> gtes;
    const Shape& flat_shape = instr->shape();
    gtes.reserve(flat_shape.tuple_shapes_size());
    for (int i = 0; i < flat_shape.tuple_shapes_size(); ++i) {
      gtes.push_back(add_new_instr(HloInstruction::CreateGetTupleElement(
          flat_shape.tuple_shapes(i), instr, i)));
    }
    auto nested_instr =
        UnflattenTupleInstr(absl::MakeSpan(gtes), while_shape, &new_instrs);
    CHECK(ShapeUtil::Compatible(nested_instr->shape(), while_shape))
        << ShapeUtil::HumanString(nested_instr->shape()) << " vs "
        << ShapeUtil::HumanString(while_shape);
    return nested_instr;
  };
  auto flattened = [&](HloInstruction* instr) {
    return HloInstruction::CreateTuple(GetFlatTupleElems(instr, &new_instrs));
  };
  std::unique_ptr<HloComputation> new_while_cond =
      while_cond->CloneWithReplacementPairs({
          while_cond->parameter_instruction(0),
          nested(add_new_instr(HloInstruction::CreateParameter(
              0, flattened_shape,
              while_cond->parameter_instruction(0)->name()))),
      });
  std::unique_ptr<HloComputation> new_while_body =
      while_body->CloneWithReplacementPairs(
          {
              while_body->parameter_instruction(0),
              nested(add_new_instr(HloInstruction::CreateParameter(
                  0, flattened_shape,
                  while_body->parameter_instruction(0)->name()))),
          },
          {
              while_body->root_instruction(),
              flattened(add_new_instr(while_body->root_instruction()->Clone())),
          });
  new_instrs.clear();
  auto* new_while_op = computation->AddInstruction(HloInstruction::CreateWhile(
      flattened_shape,
      module->AddEmbeddedComputation(std::move(new_while_cond)),
      module->AddEmbeddedComputation(std::move(new_while_body)),
      computation->AddInstruction(flattened(while_init))));
  new_while_op->CopyBackendConfigFrom(while_op);
  CopyFrontendAttributes(while_op, new_while_op);
  CopyMetadata(while_op, new_while_op);
  TF_RETURN_IF_ERROR(
      computation->ReplaceWithNewInstruction(while_op, nested(new_while_op)));
  for (auto& instr : new_instrs) {
    computation->AddInstruction(std::move(instr));
  }
  return true;
}
static absl::StatusOr<HloInstruction*> TryMergeInductionVariables(
    HloInstruction* while_op, PrimitiveType elem_ty) {
  CHECK(primitive_util::IsIntegralType(elem_ty)) << PrimitiveType_Name(elem_ty);
  HloModule* module = while_op->GetModule();
  HloComputation* computation = while_op->parent();
  auto* while_init = while_op->mutable_operand(0);
  auto* while_body = while_op->while_body();
  auto* while_cond = while_op->while_condition();
  auto* while_body_root = while_body->root_instruction();
  if (while_init->opcode() != HloOpcode::kTuple ||
      while_body_root->opcode() != HloOpcode::kTuple) {
    return nullptr;
  }
  TF_RET_CHECK(while_cond->num_parameters() == 1);
  TF_RET_CHECK(while_body->num_parameters() == 1);
  TF_RET_CHECK(
      ShapeUtil::Compatible(while_init->shape(), while_body_root->shape()));
  Shape while_shape = while_init->shape();
  std::optional<int64_t> trip_counter;
  absl::flat_hash_map<int64_t, const HloConstantInstruction*> induction_vars;
  for (int64_t i = 0; i < while_body_root->operand_count(); ++i) {
    HloInstruction* constant;
    if (!Match(while_body_root->mutable_operand(i),
               m::AddAnyOrder(m::GetTupleElement(m::Parameter(), i),
                              m::ConstantScalar(&constant))
                   .WithShape(m::Shape().WithElementType(elem_ty)))) {
      continue;
    }
    if (!trip_counter && constant->literal().IsAll(1) &&
        while_init->operand(i)->IsConstant() &&
        while_init->operand(i)->literal().IsAll(0)) {
      VLOG(10) << "Found existing trip counter at index " << i;
      trip_counter = i;
    } else {
      VLOG(10) << "Found induction variable at index " << i;
      induction_vars.emplace(i, Cast<HloConstantInstruction>(constant));
    }
  }
  if (induction_vars.size() + (trip_counter.has_value() ? 1 : 0) < 2) {
    return nullptr;
  }
  std::vector<std::unique_ptr<HloInstruction>> new_instrs;
  auto add_new_instr = [&](std::unique_ptr<HloInstruction> instr) {
    new_instrs.push_back(std::move(instr));
    return new_instrs.back().get();
  };
  auto add_binary_op = [&](const Shape& shape, HloOpcode opcode,
                           HloInstruction* lhs, HloInstruction* rhs) {
    if (!ShapeUtil::Compatible(shape, lhs->shape())) {
      lhs = add_new_instr(HloInstruction::CreateReshape(shape, lhs));
    }
    if (!ShapeUtil::Compatible(shape, rhs->shape())) {
      rhs = add_new_instr(HloInstruction::CreateReshape(shape, rhs));
    }
    return add_new_instr(HloInstruction::CreateBinary(shape, opcode, lhs, rhs));
  };
  auto add_gte = [&](HloInstruction* src, int64_t idx) {
    return add_new_instr(HloInstruction::CreateGetTupleElement(
        src->shape().tuple_shapes(idx), src, idx));
  };
  Shape new_while_shape = while_shape;
  bool added_trip_counter = false;
  if (!trip_counter) {
    VLOG(10) << "Adding new trip counter to end of loop's tuple.";
    trip_counter = new_while_shape.tuple_shapes_size();
    *new_while_shape.add_tuple_shapes() =
        ShapeUtil::MakeShape(elem_ty, {});
    added_trip_counter = true;
  }
  auto convert_to_old_form = [&](HloInstruction* instr) {
    CHECK(ShapeUtil::Compatible(instr->shape(), new_while_shape));
    std::vector<HloInstruction*> tuple_elems;
    for (int i = 0; i < while_shape.tuple_shapes_size(); ++i) {
      const auto& elem_shape = while_shape.tuple_shapes(i);
      if (!induction_vars.count(i)) {
        tuple_elems.push_back(add_gte(instr, i));
        continue;
      }
      tuple_elems.push_back(add_binary_op(
          elem_shape, HloOpcode::kAdd, add_gte(instr, i),
          add_binary_op(elem_shape, HloOpcode::kMultiply,
                        add_gte(instr, *trip_counter),
                        add_new_instr(induction_vars.at(i)->Clone()))));
    }
    return HloInstruction::CreateTuple(tuple_elems);
  };
  auto convert_to_new_form = [&](HloInstruction* old_root,
                                 HloParameterInstruction* loop_body_param) {
    CHECK(ShapeUtil::Compatible(old_root->shape(), while_shape));
    std::vector<HloInstruction*> tuple_elems;
    tuple_elems.reserve(while_shape.tuple_shapes_size());
    for (int i = 0; i < while_shape.tuple_shapes_size(); ++i) {
      tuple_elems.push_back(
          add_gte((induction_vars.count(i) ? loop_body_param : old_root), i));
    }
    if (added_trip_counter) {
      tuple_elems.push_back(add_binary_op(
          new_while_shape.tuple_shapes(*trip_counter), HloOpcode::kAdd,
          add_gte(loop_body_param, *trip_counter),
          add_new_instr(
              HloInstruction::CreateConstant(LiteralUtil::One(elem_ty)))));
    }
    return HloInstruction::CreateTuple(tuple_elems);
  };
  auto get_new_while_init = [&](HloInstruction* init) {
    CHECK(ShapeUtil::Compatible(init->shape(), while_shape));
    if (!added_trip_counter) {
      return init;
    }
    std::vector<HloInstruction*> tuple_elems;
    tuple_elems.reserve(while_shape.tuple_shapes_size());
    for (int i = 0; i < while_shape.tuple_shapes_size(); ++i) {
      tuple_elems.push_back(add_gte(init, i));
    }
    tuple_elems.push_back(add_new_instr(
        HloInstruction::CreateConstant(LiteralUtil::Zero(elem_ty))));
    return add_new_instr(HloInstruction::CreateTuple(tuple_elems));
  };
  std::unique_ptr<HloComputation> new_while_cond =
      while_cond->CloneWithReplacementPairs({
          while_cond->parameter_instruction(0),
          convert_to_old_form(add_new_instr(HloInstruction::CreateParameter(
              0, new_while_shape,
              while_cond->parameter_instruction(0)->name()))),
      });
  HloComputation* temp_new_while_body =
      module->AddEmbeddedComputation(while_body->CloneWithReplacementPairs({
          while_body->parameter_instruction(0),
          convert_to_old_form(add_new_instr(HloInstruction::CreateParameter(
              0, new_while_shape,
              while_body->parameter_instruction(0)->name()))),
      }));
  std::unique_ptr<HloComputation> new_while_body =
      temp_new_while_body->CloneWithReplacementPairs({
          temp_new_while_body->root_instruction(),
          convert_to_new_form(
              add_new_instr(temp_new_while_body->root_instruction()->Clone()),
              Cast<HloParameterInstruction>(
                  temp_new_while_body->parameter_instruction(0))),
      });
  TF_RETURN_IF_ERROR(module->RemoveEmbeddedComputation(temp_new_while_body));
  new_instrs.clear();
  auto* new_while = computation->AddInstruction(HloInstruction::CreateWhile(
      new_while_shape,
      module->AddEmbeddedComputation(std::move(new_while_cond)),
      module->AddEmbeddedComputation(std::move(new_while_body)),
      get_new_while_init(while_init)));
  new_while->CopyBackendConfigFrom(while_op);
  CopyFrontendAttributes(while_op, new_while);
  CopyMetadata(while_op, new_while);
  TF_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
      while_op, convert_to_old_form(new_while)));
  for (auto& instr : new_instrs) {
    computation->AddInstruction(std::move(instr));
  }
  return new_while;
}
absl::StatusOr<bool> WhileLoopSimplifier::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(3,
                 "WhileLoopSimplifier::Run(), before:\n" + module->ToString());
  bool changed = false;
  std::vector<HloInstruction*> while_ops;
  for (auto* comp : module->computations(execution_threads)) {
    for (auto* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kWhile) {
        while_ops.push_back(instr);
      }
    }
  }
  for (HloInstruction* while_op : while_ops) {
    TF_ASSIGN_OR_RETURN(bool result,
                        TryRemoveRepeatedWhileTupleIndices(while_op));
    changed |= result;
    if (result) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(result, TryFlattenNestedTuples(while_op));
    changed |= result;
    if (result) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(result, TryRemoveDeadWhileParams(while_op));
    changed |= result;
    if (result) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(result, TryRemoveConstantParams(while_op));
    changed |= result;
    if (result) {
      continue;
    }
    if (simplify_compare_instrs_) {
      TF_ASSIGN_OR_RETURN(result, TryRemoveTrivialCompare(while_op));
      changed |= result;
      if (result) {
        continue;
      }
    }
    if (ContainsInstrWithOpcode(while_op->while_body(),
                                {HloOpcode::kSend, HloOpcode::kSendDone,
                                 HloOpcode::kRecv, HloOpcode::kRecvDone}) ||
        ContainsInstrWithOpcode(while_op->while_condition(),
                                {HloOpcode::kSend, HloOpcode::kSendDone,
                                 HloOpcode::kRecv, HloOpcode::kRecvDone})) {
      VLOG(2) << "Not attempting to simplify while loop because it contains a "
                 "send/recv node: "
              << while_op->ToShortString();
      continue;
    }
    TF_ASSIGN_OR_RETURN(result, TryPropagateConstant(while_op));
    changed |= result;
    TF_ASSIGN_OR_RETURN(result, TryRemoveWhileLoop(while_op));
    changed |= result;
    if (result) {
      continue;
    }
    if (ContainsInstrWithOpcode(while_op->while_body(), {HloOpcode::kDomain}) ||
        ContainsInstrWithOpcode(while_op->while_condition(),
                                {HloOpcode::kDomain})) {
      continue;
    }
    bool merged_induction_vars = false;
    for (auto elem_ty : {S8, U8, S32, U32, S64, U64}) {
      TF_ASSIGN_OR_RETURN(auto* new_while_op,
                          TryMergeInductionVariables(while_op, elem_ty));
      if (new_while_op) {
        while_op = new_while_op;
        changed = true;
        merged_induction_vars = true;
      }
    }
    if (merged_induction_vars) {
      continue;
    }
  }
  HloDCE dce;
  TF_ASSIGN_OR_RETURN(bool dce_changed, dce.Run(module));
  changed |= dce_changed;
  XLA_VLOG_LINES(3,
                 "WhileLoopSimplifier::Run(), after:\n" + module->ToString());
  return changed;
}
}  