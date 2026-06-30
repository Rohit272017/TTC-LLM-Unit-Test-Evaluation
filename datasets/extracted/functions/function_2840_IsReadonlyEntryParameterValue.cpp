#include "xla/service/copy_insertion.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/frontend_attributes.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_reachability.h"
#include "xla/map_util.h"
#include "xla/service/call_graph.h"
#include "xla/service/compile_time_cap.h"
#include "xla/service/dump.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_dataflow_analysis.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_ordering.h"
#include "xla/service/hlo_value.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace {
using absl::StrAppend;
bool IsReadonlyEntryParameterValue(const HloValue& value) {
  const HloComputation* computation = value.defining_instruction()->parent();
  return value.defining_instruction()->opcode() == HloOpcode::kParameter &&
         computation == computation->parent()->entry_computation() &&
         !computation->parent()->input_output_alias_config().ParameterHasAlias(
             value.defining_instruction()->parameter_number(), value.index());
}
bool IsConstantValue(const HloValue& value) {
  return value.defining_instruction()->opcode() == HloOpcode::kConstant;
}
bool ValueIsReadOnly(const HloValue& value) {
  return IsConstantValue(value) || IsReadonlyEntryParameterValue(value);
}
struct SpecialCaseCopyPolicy {
  bool copy_root_replicated_buffers = false;
  bool copy_parameters_and_constants = false;
};
SpecialCaseCopyPolicy GetSpecialCaseCopyPolicy(const CallGraphNode& node,
                                               HloModule* module,
                                               HloComputation* computation) {
  SpecialCaseCopyPolicy policy;
  if (computation == module->entry_computation()) {
    policy.copy_parameters_and_constants = true;
    policy.copy_root_replicated_buffers = true;
  }
  return policy;
}
bool ShouldCopyRootValue(const HloValue& value,
                         const SpecialCaseCopyPolicy& policy) {
  if (policy.copy_parameters_and_constants) {
    return ValueIsReadOnly(value);
  }
  return false;
}
absl::StatusOr<std::pair<HloInstruction*, HloInstruction*>>
DeepCopyAndAddControlEdges(HloInstruction* from, HloInstruction* to,
                           const ShapeTree<bool>& indices_to_copy) {
  DCHECK(ShapeUtil::Compatible(from->shape(), to->shape()));
  ShapeTree<HloInstruction*> from_copy_tree(from->shape(),
                                            nullptr);
  TF_ASSIGN_OR_RETURN(HloInstruction * from_deep_copy,
                      from->parent()->DeepCopyInstruction(
                          from, &indices_to_copy, &from_copy_tree));
  ShapeTree<HloInstruction*> to_copy_tree(to->shape(), nullptr);
  TF_ASSIGN_OR_RETURN(
      HloInstruction * to_deep_copy,
      to->parent()->DeepCopyInstruction(to, &indices_to_copy, &to_copy_tree));
  for (const auto& pair : from_copy_tree) {
    const ShapeIndex& index = pair.first;
    HloInstruction* from_copy = pair.second;
    HloInstruction* to_copy = to_copy_tree.element(index);
    if (from_copy == nullptr) {
      TF_RET_CHECK(to_copy == nullptr);
      continue;
    }
    TF_RET_CHECK(to_copy != nullptr);
    TF_RETURN_IF_ERROR(from_copy->AddControlDependencyTo(to_copy));
  }
  return std::make_pair(from_deep_copy, to_deep_copy);
}
bool IndicesToCopyForWhile(const HloDataflowAnalysis& dataflow,
                           const HloInstruction* xla_while,
                           ShapeTree<bool>* indices_to_copy) {
  DCHECK(ShapeUtil::Compatible(indices_to_copy->shape(), xla_while->shape()));
  bool any_copies = false;
  const HloInstruction* init = xla_while->operand(0);
  for (auto& pair : *indices_to_copy) {
    const ShapeIndex& index = pair.first;
    bool& should_copy = pair.second;
    if (dataflow.GetValueSet(init, index).values().size() > 1 ||
        dataflow.GetValueSet(xla_while, index).values().size() > 1) {
      should_copy = true;
    } else {
      should_copy = dataflow.GetUniqueValueAt(xla_while, index) !=
                    dataflow.GetUniqueValueAt(init, index);
    }
    any_copies |= should_copy;
  }
  return any_copies;
}
bool IndicesToCopyForConditional(const HloDataflowAnalysis& dataflow,
                                 const HloInstruction* xla_conditional,
                                 ShapeTree<bool>* indices_to_copy) {
  DCHECK(ShapeUtil::Compatible(indices_to_copy->shape(),
                               xla_conditional->shape()));
  bool any_copies = false;
  for (auto& pair : *indices_to_copy) {
    const ShapeIndex& index = pair.first;
    bool& should_copy = pair.second;
    CHECK_EQ(dataflow.GetValueSet(xla_conditional, index).values().size(), 1);
    auto value = dataflow.GetValueSet(xla_conditional, index).values()[0];
    should_copy =
        value->is_phi() && value->defining_instruction() == xla_conditional;
    any_copies |= should_copy;
  }
  return any_copies;
}
absl::Status AddCopiesForWhile(const HloAliasAnalysis& alias_analysis,
                               HloInstruction* xla_while) {
  VLOG(2) << "Adding copies for kWhile instruction " << xla_while->name();
  TF_RET_CHECK(xla_while->opcode() == HloOpcode::kWhile);
  ShapeTree<bool> indices_to_copy(xla_while->shape());
  if (!IndicesToCopyForWhile(alias_analysis.dataflow_analysis(), xla_while,
                             &indices_to_copy)) {
    VLOG(2) << "No copies necessary for kWhile instruction "
            << xla_while->name();
    return absl::OkStatus();
  }
  VLOG(2) << "Adding copies for " << xla_while->name() << " at indices:";
  for (auto& pair : indices_to_copy) {
    if (pair.second) {
      VLOG(2) << "  " << pair.first;
    }
  }
  HloInstruction* while_init = xla_while->mutable_operand(0);
  TF_ASSIGN_OR_RETURN(
      HloInstruction * while_init_copy,
      xla_while->parent()->DeepCopyInstruction(while_init, &indices_to_copy));
  TF_RETURN_IF_ERROR(while_init->ReplaceUseWith(xla_while, while_init_copy));
  HloComputation* body = xla_while->while_body();
  HloInstruction* param = body->parameter_instruction(0);
  HloInstruction* root = body->root_instruction();
  TF_RET_CHECK(param != root);
  std::vector<HloInstruction*> param_users = param->users();
  TF_ASSIGN_OR_RETURN(auto pair,
                      DeepCopyAndAddControlEdges(param, root, indices_to_copy));
  HloInstruction* param_copy = pair.first;
  HloInstruction* root_copy = pair.second;
  for (HloInstruction* user : param_users) {
    TF_RETURN_IF_ERROR(param->ReplaceUseWith(user, param_copy));
  }
  body->set_root_instruction(root_copy);
  return absl::OkStatus();
}
absl::Status AddCopiesForInPlaceOperation(
    const HloAliasAnalysis& alias_analysis, HloInstruction* in_place_op,
    int64_t operand_number) {
  VLOG(2) << "Adding copies for in-place operation " << in_place_op->name();
  HloInstruction* operand = in_place_op->mutable_operand(operand_number);
  TF_ASSIGN_OR_RETURN(HloInstruction * deep_copy,
                      in_place_op->parent()->DeepCopyInstruction(operand));
  TF_RETURN_IF_ERROR(
      operand->ReplaceUseWith(in_place_op, operand_number, deep_copy));
  return absl::OkStatus();
}
absl::Status AddCopiesForAliasedInputOutputs(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  HloComputation* entry = module->entry_computation();
  if (!HloInstruction::IsThreadIncluded(entry->execution_thread(),
                                        execution_threads)) {
    return absl::OkStatus();
  }
  HloInstruction* root = entry->root_instruction();
  ShapeTree<bool> output_indices_to_copy(root->shape());
  std::vector<std::optional<ShapeTree<HloInstruction*>>> copied_parameters(
      entry->num_parameters());
  bool has_alias = false;
  for (auto* param : entry->parameter_instructions()) {
    bool param_has_alias = false;
    ShapeTree<bool> param_indices_to_copy(param->shape());
    module->input_output_alias_config().ForEachAlias(
        [&](const ShapeIndex& output_index,
            const HloInputOutputAliasConfig::Alias& alias) {
          if (alias.parameter_number == param->parameter_number()) {
            param_has_alias = true;
            *(param_indices_to_copy.mutable_element(alias.parameter_index)) =
                true;
            *(output_indices_to_copy.mutable_element(output_index)) = true;
          }
        });
    if (!param_has_alias) {
      continue;
    }
    TF_RET_CHECK(param->parameter_number() < entry->num_parameters());
    TF_RET_CHECK(!copied_parameters[param->parameter_number()]);
    has_alias = true;
    std::vector<HloInstruction*> users = param->users();
    ShapeTree<HloInstruction*> param_copy_tree(param->shape(),
                                               nullptr);
    TF_ASSIGN_OR_RETURN(HloInstruction * copied,
                        entry->DeepCopyInstruction(
                            param, &param_indices_to_copy, &param_copy_tree));
    if (param == root) {
      entry->set_root_instruction(copied);
      root = copied;
    }
    for (HloInstruction* user : users) {
      TF_RETURN_IF_ERROR(param->ReplaceUseWith(user, copied));
    }
    copied_parameters[param->parameter_number()] = param_copy_tree;
  }
  if (!has_alias) {
    return absl::OkStatus();
  }
  ShapeTree<HloInstruction*> output_copy_tree(root->shape(),
                                              nullptr);
  TF_ASSIGN_OR_RETURN(HloInstruction * root_copied,
                      root->parent()->DeepCopyInstruction(
                          root, &output_indices_to_copy, &output_copy_tree));
  TF_RETURN_IF_ERROR(module->input_output_alias_config().ForEachAliasWithStatus(
      [&](const ShapeIndex& output_index,
          const HloInputOutputAliasConfig::Alias& alias) -> absl::Status {
        if (!copied_parameters[alias.parameter_number]) {
          return absl::OkStatus();
        }
        HloInstruction* from =
            copied_parameters[alias.parameter_number]->element(
                alias.parameter_index);
        HloInstruction* to = output_copy_tree.element(output_index);
        TF_RET_CHECK(from != nullptr);
        TF_RET_CHECK(to != nullptr);
        TF_RETURN_IF_ERROR(from->AddControlDependencyTo(to));
        return absl::OkStatus();
      }));
  entry->set_root_instruction(root_copied);
  return absl::OkStatus();
}
absl::Status StripControlDependenciesFrom(HloInstruction* instruction) {
  while (!instruction->control_successors().empty()) {
    TF_RETURN_IF_ERROR(instruction->RemoveControlDependencyTo(
        instruction->control_successors().front()));
  }
  while (!instruction->control_predecessors().empty()) {
    TF_RETURN_IF_ERROR(
        instruction->control_predecessors().front()->RemoveControlDependencyTo(
            instruction));
  }
  return absl::OkStatus();
}
class LiveRangeRegions {
 public:
  struct InstructionInfo {
    InstructionInfo() : value_definition(nullptr), is_definition(false) {}
    HloInstruction* value_definition;
    bool is_definition;
    std::string ToString() const {
      return absl::StrCat(
          "is_definition: ", std::to_string(is_definition),
          ", value_definition: ",
          value_definition ? value_definition->name() : "nullptr");
    }
  };
  typedef HloInstructionMap<InstructionInfo> InstructionMap;
  typedef std::pair<HloInstruction*, InstructionInfo> InstructionEntry;
  typedef absl::flat_hash_map<const HloComputation*, InstructionMap>
      ComputationMap;
  InstructionMap& operator[](const HloComputation* computation) {
    if (computation_map_.find(computation) == computation_map_.end()) {
      computation_vector_.push_back(computation);
    }
    return computation_map_[computation];
  }
  const InstructionMap& operator[](const HloComputation* computation) const {
    ComputationMap::const_iterator p = computation_map_.find(computation);
    CHECK(p != computation_map_.end());
    return p->second;
  }
  absl::InlinedVector<const HloComputation*, 5>::const_iterator begin() const {
    return computation_vector_.begin();
  }
  absl::InlinedVector<const HloComputation*, 5>::const_iterator end() const {
    return computation_vector_.end();
  }
  int64_t size() const {
    CHECK_EQ(computation_vector_.size(), computation_map_.size());
    return computation_vector_.size();
  }
  bool empty() const { return size() == 0; }
  const HloComputation* Computation(int64_t index) const {
    return computation_vector_[index];
  }
  bool contains(HloInstruction* instr) const {
    CHECK_NE(instr, nullptr);
    auto* computation = instr->parent();
    auto p = computation_map_.find(computation);
    if (p == computation_map_.end()) {
      return false;
    }
    auto instr_map = (*p).second;
    return instr_map.find(instr) != instr_map.end();
  }
  std::string ToString() const {
    std::string result;
    for (const auto* computation : computation_vector_) {
      StrAppend(&result, "computation: ", computation->name(), "\n");
      for (const auto& entry : computation_map_.at(computation)) {
        StrAppend(&result, "  entry: ", entry.first->name(), ", ",
                  entry.second.ToString(), "\n");
      }
    }
    return result;
  }
 private:
  ComputationMap computation_map_;
  absl::InlinedVector<const HloComputation*, 5> computation_vector_;
};
namespace {
class Relation {
 public:
  enum RuntimeOrder {
    kNoOverlap = 0,
    kSameInstr = 1,
    kBeforeStart = 2,
    kBeforeStartOrSameInstr = kBeforeStart | kSameInstr,
    kAfterEnd = 4,
    kAfterEndOrSameInstr = kAfterEnd | kSameInstr,
    kBeforeStartOrAfterEnd = kBeforeStart | kAfterEnd,
    kBeforeOrAfterOrOverlap = kBeforeStart | kAfterEnd | kSameInstr,
  };
  Relation() : intercept_def_use_(false) {}
  explicit Relation(RuntimeOrder order, bool intercept_def_use = false)
      : intercept_def_use_(intercept_def_use) {
    orders_.push_back(order);
  }
  Relation(const Relation& that)
      : intercept_def_use_(that.intercept_def_use_), orders_(that.orders_) {}
  bool operator==(const Relation& that) const {
    return intercept_def_use_ == that.intercept_def_use_ &&
           absl::c_equal(orders_, that.orders_);
  }
  bool UseImpliesInterception() const {
    CHECK_EQ(orders_.size(), 1);
    return UseImpliesInterception(orders_[0]);
  }
  bool DefinitionImpliesInterception() const {
    CHECK_EQ(orders_.size(), 1);
    return DefinitionImpliesInterception(orders_[0]);
  }
  bool InterceptDefUse() const { return intercept_def_use_; }
  void UpdateInterception(bool value) {
    CHECK_EQ(orders_.size(), 1);
    intercept_def_use_ = value;
  }
  Relation::RuntimeOrder GetRuntimeOrder() const {
    if (orders_.empty()) {
      return Relation::kNoOverlap;
    }
    CHECK_EQ(orders_.size(), 1);
    return orders_[0];
  }
  bool RuntimeOrderOverlap() const {
    return absl::c_any_of(orders_, ImpliesOverlap);
  }
  bool RuntimeOrderIsUnordered() const {
    return orders_.size() == 1 && orders_[0] == kBeforeStartOrAfterEnd;
  }
  bool RuntimeOrderIsNoOverlap() const {
    return orders_.empty() || (orders_.size() == 1 && orders_[0] == kNoOverlap);
  }
  bool RuntimeOrderIsRunBefore() const {
    return orders_.size() == 1 && orders_[0] == kBeforeStart;
  }
  bool RuntimeOrderIsRunAfter() const {
    return orders_.size() == 1 && orders_[0] == kAfterEnd;
  }
  std::string ToString() const {
    return absl::StrCat("Interception = ", intercept_def_use_, ";",
                        absl::StrJoin(orders_, ","));
  }
  static bool DefinitionImpliesInterception(RuntimeOrder definition) {
    return (definition == kAfterEnd || definition == kBeforeStartOrAfterEnd);
  }
  static bool UseImpliesInterception(RuntimeOrder use) {
    return (use == kBeforeStart || use == kBeforeStartOrAfterEnd);
  }
  void UnionRelationFromSameSource(const Relation& rel) {
    CHECK_LE(orders_.size(), 1);
    CHECK_EQ(rel.orders_.size(), 1);
    if (orders_.empty()) {
      orders_.push_back(rel.orders_[0]);
    } else {
      orders_[0] = Union(orders_[0], rel.orders_[0]);
    }
    intercept_def_use_ = intercept_def_use_ || rel.intercept_def_use_;
  }
  void UnionRelationFromDifferentSource(const Relation& rel) {
    if (rel.orders_.empty()) {
      return;
    }
    CHECK_EQ(rel.orders_.size(), 1);
    intercept_def_use_ = intercept_def_use_ || rel.intercept_def_use_;
    for (auto& local_order : orders_) {
      if (OverwriteIfSubsume(rel.orders_[0], &local_order)) {
        return;
      }
    }
    orders_.push_back(rel.orders_[0]);
  }
  static Relation::RuntimeOrder ReverseRuntimeOrder(RuntimeOrder order) {
    switch (order) {
      case kNoOverlap:
      case kSameInstr:
      case kBeforeStartOrAfterEnd:
      case kBeforeOrAfterOrOverlap:
        return order;
      case kBeforeStart:
        return kAfterEnd;
      case kBeforeStartOrSameInstr:
        return kAfterEndOrSameInstr;
      case kAfterEnd:
        return kBeforeStart;
      case kAfterEndOrSameInstr:
        return kBeforeStartOrSameInstr;
    }
  }
 private:
  bool intercept_def_use_;
  absl::InlinedVector<RuntimeOrder, 4> orders_;
  static RuntimeOrder Union(RuntimeOrder o1, RuntimeOrder o2) {
    return static_cast<Relation::RuntimeOrder>(o1 | o2);
  }
  static bool ImpliesOverlap(RuntimeOrder o) {
    return o >= RuntimeOrder::kBeforeStartOrAfterEnd;
  }
  static bool Subsume(RuntimeOrder o1, RuntimeOrder o2) {
    return Union(o1, o2) == o1;
  }
  static bool OverwriteIfSubsume(RuntimeOrder o2, RuntimeOrder* o1) {
    if (*o1 == o2) {
      return true;
    }
    CHECK_NE(o1, nullptr);
    if (Subsume(o2, *o1)) {
      *o1 = o2;
      return true;
    } else if (Subsume(*o1, o2)) {
      return true;
    }
    return false;
  }
};
class ComputeRelativeLocation {
 public:
  typedef LiveRangeRegions::InstructionEntry InstructionEntry;
  explicit ComputeRelativeLocation(HloOrdering* ordering)
      : ordering_(ordering) {
    VLOG(3) << "New analysis";
  }
  Relation Compute(const InstructionEntry& entry1,
                   const InstructionEntry& entry2, bool instr2_can_modify) {
    auto def = entry1.second.value_definition;
    auto use = entry1.first;
    Relation::RuntimeOrder order =
        ComputeRuntimeOrdering(entry2.first, entry1.first);
    if (order == Relation::kSameInstr &&
        entry1.second.is_definition != entry2.second.is_definition) {
      if (entry1.second.is_definition) {
        order = Relation::kBeforeStart;
      } else {
        order = Relation::kAfterEnd;
      }
    }
    bool intercept = AlwaysForceInterception(entry2.first);
    if (def == nullptr || !instr2_can_modify) {
      return Relation(order, intercept);
    }
    if (def->opcode() == HloOpcode::kParameter &&
        use == use->parent()->root_instruction()) {
      VLOG(3) << "Setting interception due to parameter/root relation";
      return Relation(order, true);
    }
    if (use->parent() == def->parent() &&
        ComputeRuntimeOrdering(use, entry2.first) == Relation::kAfterEnd &&
        def->opcode() == HloOpcode::kWhile &&
        entry2.first->parent() == def->while_body()) {
      return Relation(order, false);
    }
    if (use->parent() == def->parent() &&
        ComputeRuntimeOrdering(def, entry2.first) == Relation::kBeforeStart &&
        use->opcode() == HloOpcode::kWhile &&
        entry2.first->parent() == use->while_body()) {
      return Relation(order, false);
    }
    if (use->parent() == def->parent() &&
        def->parent()->IsConditionalBranchComputation() &&
        def == entry2.first && def->shape().IsTuple()) {
      VLOG(3) << "Setting interception for multi-output instruction inside "
                 "conditional branch: "
              << def->name();
      return Relation(order, true);
    }
    if (Relation::UseImpliesInterception(order)) {
      auto order2 = ComputeRuntimeOrdering(entry2.first, def);
      if (Relation::DefinitionImpliesInterception(order2)) {
        VLOG(3) << "Setting interception for " << def->ToString()
                << " with use: " << entry1.first->ToString();
        intercept = true;
      }
    }
    return Relation(order, intercept);
  }
  Relation Compute(const LiveRangeRegions& range1,
                   const LiveRangeRegions& range2) {
    Relation dir_src_dest;
    for (const auto* computation1 : range1) {
      for (const auto* computation2 : range2) {
        for (auto instr_entry2 : range2[computation2]) {
          if (!ordering_->call_graph().Dominates(computation1, computation2)) {
            continue;
          }
          VLOG(3) << "Locationing " << instr_entry2.first->ToString();
          bool instr2_can_modify =
              InstructionCanIntercept(instr_entry2, range1);
          Relation instr2_relation;
          std::vector<InstructionEntry> unordered_ops;
          bool unordered_intercept = false;
          for (auto instr_entry1 : range1[computation1]) {
            auto rel = Compute(instr_entry1, instr_entry2, instr2_can_modify);
            VLOG(3) << "New relation with " << instr_entry1.first->name()
                    << ": " << rel.ToString();
            if (!rel.RuntimeOrderIsUnordered()) {
              instr2_relation.UnionRelationFromSameSource(rel);
            } else {
              unordered_ops.push_back(instr_entry1);
              unordered_intercept |= rel.InterceptDefUse();
            }
            VLOG(3) << "instr2 relation: " << instr2_relation.ToString();
          }
          if (!ForceRuntimeOrder(unordered_ops, instr_entry2,
                                 instr2_relation.GetRuntimeOrder())) {
            VLOG(3) << "Unable to force ordering of unordered ops";
            instr2_relation.UnionRelationFromSameSource(Relation(
                Relation::kBeforeStartOrAfterEnd, unordered_intercept));
          }
          dir_src_dest.UnionRelationFromDifferentSource(instr2_relation);
          VLOG(3) << "Resulting relation: " << dir_src_dest.ToString();
        }
      }
    }
    return dir_src_dest;
  }
  bool AddControlDependenceForUnorderedOps() {
    if (ctrl_deps_.empty()) {
      return true;
    }
    PredecessorHloOrdering* ordering =
        dynamic_cast<PredecessorHloOrdering*>(ordering_);
    if (ordering == nullptr) {
      return false;
    }
    for (const auto& comp_it : ctrl_deps_) {
      HloComputation* parent = comp_it.first;
      HloReachabilityMap& reachability_map = ordering->reachability_map(parent);
      for (const auto& instr_it : comp_it.second) {
        HloInstruction* entry1 = instr_it.first;
        for (HloInstruction* entry2 : instr_it.second) {
          VLOG(3) << "Add control dependence between " << entry2->name()
                  << " vs " << entry1->name();
          TF_CHECK_OK(entry2->AddControlDependencyTo(entry1));
        }
        reachability_map.UpdateReachabilityThroughInstruction(entry1);
        for (HloInstruction* entry2 : instr_it.second) {
          DCHECK(ordering_->GetExecutionConstraint(entry1, entry2) ==
                 HloOrdering::ExecutionConstraint::kRunAfter);
        }
      }
    }
    return true;
  }
 private:
  enum ComputeStatus {
    kFullyComputed,
    kPartiallyComputed,
    kNotComputed,
  };
  typedef std::pair<ComputeStatus, Relation::RuntimeOrder> SavedRelation;
  bool ForceRuntimeOrder(absl::Span<const InstructionEntry> unordered_ops,
                         const InstructionEntry entry2,
                         Relation::RuntimeOrder desired_relation) {
    if (unordered_ops.empty()) {
      return true;
    }
    if (desired_relation != Relation::kBeforeStart &&
        desired_relation != Relation::kAfterEnd) {
      return false;
    }
    auto ModifiesNonCopy = [](HloInstruction* instr, const HloInstruction* op) {
      auto in_place = HloDataflowAnalysis::GetInPlaceInputOutputPairs(instr);
      if (in_place.empty()) {
        return false;
      }
      return absl::c_any_of(
          in_place, [&](const std::pair<HloOperandIndex, ShapeIndex>&
                            operand_and_output_index) {
            auto* op2 =
                instr->operand(operand_and_output_index.first.operand_number);
            return (op == nullptr) ? (op2->opcode() == HloOpcode::kCopy)
                                   : (op2 == op);
          });
    };
    for (const InstructionEntry& entry1 : unordered_ops) {
      if (entry1.first->parent() != entry2.first->parent()) {
        return false;
      }
      HloInstruction* pred = (desired_relation == Relation::kBeforeStart)
                                 ? entry2.first
                                 : entry1.first;
      HloInstruction* succ = (desired_relation == Relation::kBeforeStart)
                                 ? entry1.first
                                 : entry2.first;
      if (pred == pred->parent()->root_instruction()) {
        return false;
      }
      if (succ->opcode() == HloOpcode::kCopy &&
          ModifiesNonCopy(pred, succ->operand(0))) {
        VLOG(3) << "Failed to force unordered op ordering due to copy ordering "
                << " between " << pred->name() << " vs " << succ->name();
        return false;
      }
    }
    for (const InstructionEntry& entry1 : unordered_ops) {
      Save(entry2.first, entry1.first, desired_relation,
           true);
    }
    return true;
  }
  static bool AlwaysForceInterception(HloInstruction* instr) {
    if (HloDataflowAnalysis::IsAsynchronousOperationStart(instr->opcode()) ||
        HloDataflowAnalysis::IsAsynchronousOperationDone(instr->opcode())) {
      return true;
    }
    switch (instr->opcode()) {
      case HloOpcode::kCollectivePermute:
        return true;
      default:
        return false;
    }
  }
  bool InstructionCanIntercept(const InstructionEntry& entry,
                               const LiveRangeRegions& region) {
    auto instr = entry.first;
    if (!entry.second.is_definition) {
      for (const auto& operand_and_output_index :
           HloDataflowAnalysis::GetInPlaceInputOutputPairs(instr)) {
        const HloOperandIndex& operand_index = operand_and_output_index.first;
        if (region.contains(
                instr->mutable_operand(operand_index.operand_number))) {
          return true;
        }
      }
      return false;
    }
    switch (instr->opcode()) {
      case HloOpcode::kCopy: {
        HloInstruction* operand = instr->mutable_operand(0);
        if (operand->opcode() == HloOpcode::kGetTupleElement) {
          operand = operand->mutable_operand(0);
        }
        if (region.contains(operand) &&
            ShapeUtil::Equal(instr->shape(), instr->operand(0)->shape())) {
          return false;  
        }
        return true;
      }
      case HloOpcode::kParameter:
      case HloOpcode::kTuple:
      case HloOpcode::kGetTupleElement:
      case HloOpcode::kWhile:
      case HloOpcode::kCall:
      case HloOpcode::kConditional:
        return false;
      default:
        return true;
    }
    return true;
  }
  SavedRelation AlreadyComputed(HloInstruction* op1, HloInstruction* op2) {
    auto p2 = saved_relations_.find(op2);
    if (p2 != saved_relations_.end()) {
      auto p1 = (*p2).second.find(op1);
      if (p1 != (*p2).second.end()) {
        return SavedRelation(kFullyComputed, (*p1).second);
      }
    }
    p2 = saved_relations_.find(op1);
    if (p2 != saved_relations_.end()) {
      auto p1 = (*p2).second.find(op2);
      if (p1 != (*p2).second.end()) {
        return SavedRelation(kPartiallyComputed,
                             Relation::ReverseRuntimeOrder((*p1).second));
      }
    }
    return SavedRelation(kNotComputed, Relation::kNoOverlap);
  }
  Relation::RuntimeOrder Save(HloInstruction* entry1, HloInstruction* entry2,
                              const Relation::RuntimeOrder relation,
                              bool is_unordered_originally = false) {
    CHECK_EQ(AlreadyComputed(entry1, entry2).first, kNotComputed);
    CHECK_NE(relation, Relation::kBeforeStartOrAfterEnd);
    saved_relations_[entry2][entry1] = relation;
    if (is_unordered_originally) {
      CHECK(relation == Relation::kBeforeStart ||
            relation == Relation::kAfterEnd)
          << relation;
      HloInstruction* pred =
          (relation == Relation::kBeforeStart) ? entry1 : entry2;
      HloInstruction* succ =
          (relation == Relation::kBeforeStart) ? entry2 : entry1;
      VLOG(3) << "Save unordered relation: " << pred->name() << " vs "
              << succ->name();
      CHECK_EQ(succ->parent(), pred->parent());
      auto& dep_vec = ctrl_deps_[succ->parent()][succ];
      for (HloInstruction*& op : dep_vec) {
        auto rel = AlreadyComputed(pred, op);
        if (rel.first != kNotComputed) {
          if (rel.second == Relation::kAfterEnd) {
            op = pred;
          } else {
            CHECK(rel.second == Relation::kBeforeStart);
          }
          return relation;
        }
      }
      VLOG(2) << "Forcing unordered: " << pred->name() << " vs "
              << succ->name();
      dep_vec.push_back(pred);
    }
    return relation;
  }
  Relation::RuntimeOrder ComputeRuntimeOrdering(HloInstruction* instr1,
                                                HloInstruction* instr2) {
    auto saved_relation = AlreadyComputed(instr1, instr2);
    if (saved_relation.first != kNotComputed) {
      VLOG(3) << "Already computed between " << instr1->name() << " vs "
              << instr2->name();
      return saved_relation.second;
    }
    auto constraint = ordering_->GetExecutionConstraint(instr1, instr2);
    switch (constraint) {
      case HloOrdering::ExecutionConstraint::kIsSame:
        return Save(instr1, instr2, Relation::kSameInstr);
      case HloOrdering::ExecutionConstraint::kRunBeforeEnd:
        return Save(instr1, instr2, Relation::kBeforeStartOrSameInstr);
      case HloOrdering::ExecutionConstraint::kRunBeforeStart:
        return Save(instr1, instr2, Relation::kBeforeStart);
      case HloOrdering::ExecutionConstraint::kRunAfter:
        return Save(instr1, instr2, Relation::kAfterEnd);
      case HloOrdering::ExecutionConstraint::kRunExclusiveBefore:
      case HloOrdering::ExecutionConstraint::kRunExclusiveAfter:
        return Save(instr1, instr2, Relation::kNoOverlap);
      case HloOrdering::ExecutionConstraint::kUnordered: {
        if (instr1->parent() != instr2->parent()) {
          return Relation::kBeforeStartOrAfterEnd;
        }
        auto ControlDependenceBefore = [&](HloInstruction* op1,
                                           HloInstruction* op2) {
          auto constraint = ComputeRuntimeOrdering(op1, op2);
          if (constraint == Relation::kBeforeStart ||
              constraint == Relation::kSameInstr ||
              constraint == Relation::kBeforeStartOrSameInstr) {
            return true;
          } else {
            return false;
          }
        };
        if (!ctrl_deps_.empty()) {
          auto ctrl_deps = ctrl_deps_[instr1->parent()];
          if (absl::c_any_of(ctrl_deps[instr2], [&](HloInstruction* pred2) {
                return ControlDependenceBefore(instr1, pred2);
              })) {
            VLOG(2) << "control-dependent: " << instr1->name() << " vs "
                    << instr2->name();
            return Save(instr1, instr2, Relation::kBeforeStart);
          } else if (absl::c_any_of(
                         ctrl_deps[instr1], [&](HloInstruction* pred1) {
                           return ControlDependenceBefore(instr2, pred1);
                         })) {
            VLOG(2) << "control-dependent: " << instr2->name() << " vs "
                    << instr1->name();
            return Save(instr1, instr2, Relation::kAfterEnd);
          }
        }
        return Relation::kBeforeStartOrAfterEnd;
      }
    }
  }
  HloOrdering* ordering_;
  absl::flat_hash_map<
      HloInstruction*,
      absl::flat_hash_map<HloInstruction*, Relation::RuntimeOrder>>
      saved_relations_;
  absl::flat_hash_map<
      HloComputation*,
      absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>>
      ctrl_deps_;
};
}  
class CopyRemover {
 public:
  struct ValueNode {
    explicit ValueNode(const HloValue* v) : value(v) {}
    const HloValue* value;
    std::vector<const HloUse*> uses;
    ValueNode* prev = nullptr;
    ValueNode* next = nullptr;
  };
  CopyRemover(const HloModule& module, const HloAliasAnalysis& alias_analysis,
              HloOrdering* ordering, bool check_live_range_ordering,
              const absl::flat_hash_set<absl::string_view>& execution_threads)
      : dataflow_(alias_analysis.dataflow_analysis()), ordering_(ordering) {
    absl::flat_hash_map<int, int64_t> instruction_ids;
    int64_t id = 0;
    for (HloComputation* computation : module.MakeComputationPostOrder()) {
      for (HloInstruction* instruction :
           computation->MakeInstructionPostOrder()) {
        instruction_ids[instruction->unique_id()] = id++;
      }
    }
    absl::flat_hash_map<const HloValue*, ValueNode*> value_to_node;
    for (const HloBuffer& buffer : alias_analysis.buffers()) {
      if (buffer.values().at(0)->defining_instruction()->IsFused()) {
        continue;
      }
      if (check_live_range_ordering) {
        auto should_skip_value = [&execution_threads](const HloValue* value) {
          return value->defining_instruction()->parent() != nullptr &&
                 !HloInstruction::IsThreadIncluded(value->defining_instruction()
                                                       ->parent()
                                                       ->execution_thread(),
                                                   execution_threads);
        };
        for (const HloValue* value_a : buffer.values()) {
          if (value_a->shape().IsToken()) {
            continue;
          }
          if (should_skip_value(value_a)) {
            continue;
          }
          for (const HloValue* value_b : buffer.values()) {
            if (!should_skip_value(value_b) && value_a != value_b) {
              DCHECK(ordering_->LiveRangeStrictlyBefore(
                         *value_a, *value_b, dataflow_,
                         true) ||
                     ordering_->LiveRangeStrictlyBefore(
                         *value_b, *value_a, dataflow_,
                         true))
                  << value_a->ToString() << " and " << value_b->ToString()
                  << " are not ordered";
            }
          }
        }
      }
      std::vector<const HloValue*> values = buffer.values();
      absl::c_sort(values, [this, &instruction_ids](const HloValue* a,
                                                    const HloValue* b) {
        if (a == b) {
          return false;
        }
        const bool a_has_smaller_id =
            instruction_ids.at(a->defining_instruction()->unique_id()) <
            instruction_ids.at(b->defining_instruction()->unique_id());
        if (a_has_smaller_id) {
          if (ordering_->IsDefinedBefore(*a, *b)) {
            return true;
          }
          if (ordering_->IsDefinedBefore(*b, *a)) {
            return false;
          }
        } else {
          if (ordering_->IsDefinedBefore(*b, *a)) {
            return false;
          }
          if (ordering_->IsDefinedBefore(*a, *b)) {
            return true;
          }
        }
        return a_has_smaller_id;
      });
      AddValueList(values, &value_to_node);
    }
    CreateCopyMap(module, value_to_node);
    XLA_VLOG_LINES(3, ToString());
    TF_DCHECK_OK(Verify());
  }
  void AddValueList(
      absl::Span<const HloValue* const> values,
      absl::flat_hash_map<const HloValue*, ValueNode*>* value_to_node) {
    ValueNode* tail = nullptr;
    ValueNode* head = nullptr;
    for (const HloValue* value : values) {
      auto new_node = new ValueNode(value);
      (*value_to_node)[value] = new_node;
      new_node->uses.reserve(value->GetUses().size());
      for (const HloUse& use : value->GetUses()) {
        new_node->uses.push_back(&use);
      }
      if (tail == nullptr) {
        head = new_node;
      } else {
        tail->next = new_node;
        new_node->prev = tail;
      }
      tail = new_node;
    }
    tail->next = head;
    head->prev = tail;
    value_lists_.insert(head);
  }
  void CreateCopyMap(
      const HloModule& module,
      const absl::flat_hash_map<const HloValue*, ValueNode*>& value_to_node) {
    for (HloComputation* computation : module.MakeNonfusionComputations()) {
      for (HloInstruction* instruction : computation->instructions()) {
        if (instruction->opcode() == HloOpcode::kCopy) {
          const HloValueSet& src_value_set =
              dataflow_.GetValueSet(instruction->operand(0));
          if (src_value_set.values().size() == 1) {
            CopyNodes& copy_node = copy_map_[instruction];
            copy_node.dest =
                value_to_node.at(&dataflow_.GetUniqueValueAt(instruction));
            copy_node.src = value_to_node.at(&src_value_set.GetUniqueValue());
          }
        }
      }
    }
  }
  ~CopyRemover() {
    for (const ValueNode* head : value_lists_) {
      const ValueNode* p = head;
      do {
        const ValueNode* tmp = p->next;
        delete p;
        p = tmp;
      } while (p != head);
    }
  }
  absl::Status Verify() const {
    for (const ValueNode* head : value_lists_) {
      const ValueNode* p = head;
      do {
        TF_RET_CHECK(p->prev->next == p);
        TF_RET_CHECK(p->next->prev == p);
        const HloInstruction* def = p->value->defining_instruction();
        if (def->opcode() == HloOpcode::kCopy && ContainsKey(copy_map_, def)) {
          TF_RET_CHECK(copy_map_.at(def).dest == p);
        }
        for (const HloUse* use : p->uses) {
          if (use->instruction->opcode() == HloOpcode::kCopy &&
              ContainsKey(copy_map_, use->instruction)) {
            TF_RET_CHECK(copy_map_.at(use->instruction).src == p);
          }
        }
        p = p->next;
      } while (p != head);
    }
    return absl::OkStatus();
  }
  LiveRangeRegions ComputeLiveRangeRegions(const ValueNode* head) {
    LiveRangeRegions live_range;
    auto VisitValueNode = [&](const ValueNode* node) {
      HloInstruction* def_op = node->value->instruction();
      HloComputation* def_parent = def_op->parent();
      live_range[def_parent][def_op].is_definition = true;
      for (const auto& use : node->uses) {
        auto* use_op = use->instruction;
        HloComputation* use_parent = use_op->parent();
        live_range[use_parent][use_op].value_definition = def_op;
      }
    };
    ForEachValueInRange(head, VisitValueNode);
    return live_range;
  }
  bool TryElideCopy(const HloInstruction* copy,
                    int64_t* region_analysis_limit) {
    VLOG(2) << "Trying to remove " << copy->name();
    CHECK_NE(region_analysis_limit, nullptr);
    if (!ContainsKey(copy_map_, copy)) {
      VLOG(2) << copy->name() << " is not removable";
      return false;
    }
    if (!ShapeUtil::Equal(copy->shape(), copy->operand(0)->shape())) {
      VLOG(2) << copy->name() << " is not removable (shape mismatch)";
      return false;
    }
    const CopyNodes& copy_node = copy_map_.at(copy);
    DCHECK(copy_node.src != nullptr);
    DCHECK(copy_node.dest != nullptr);
    int64_t live_range_size1 = 0, live_range_size2 = 0;
    ForEachValueInRange(copy_node.src, [&](const ValueNode* node) {
      live_range_size1 += 1 + node->uses.size();
    });
    ForEachValueInRange(copy_node.dest, [&](const ValueNode* node) {
      live_range_size2 += 1 + node->uses.size();
    });
    bool use_region_analysis =
        copy->operand(0)->opcode() != HloOpcode::kBroadcast &&
        (*region_analysis_limit < 0 ||
         live_range_size1 * live_range_size2 <= *region_analysis_limit);
    *region_analysis_limit = 0;
    VLOG(3) << copy->name() << " copies value "
            << copy_node.src->value->ToShortString();
    VLOG(3) << "Source buffer values: " << ValueListToString(copy_node.src);
    VLOG(3) << "Dest buffer values: " << ValueListToString(copy_node.dest);
    auto CheckLiveRangeBefore = [&](ValueNode* src, ValueNode* dest) {
      for (ValueNode* next_dest = dest; next_dest != nullptr;
           next_dest = Next(*next_dest)) {
        for (ValueNode* prev_src = src; prev_src != nullptr;
             prev_src = Prev(*prev_src)) {
          if (!LiveRangeBefore(*prev_src, *next_dest)) {
            VLOG(2) << "Live range of " << prev_src->value->ToShortString()
                    << " is not before " << next_dest->value->ToShortString();
            return false;
          }
        }
      }
      return true;
    };
    auto CheckLiveRangeInterference = [&](ValueNode* src, ValueNode* dest,
                                          const CombineLiveRangeOption option) {
      CHECK_NE(src, nullptr);
      CHECK_NE(dest, nullptr);
      if (!use_region_analysis) {
        VLOG(2) << "Configured to not use region-based analysis.";
        return true;
      }
      *region_analysis_limit += live_range_size1 * live_range_size2;
      if (ValuesInterfere(src, dest, option)) {
        VLOG(2) << "Region-based interference is true.";
        return true;
      }
      VLOG(2) << "Region-based interference is false.";
      return false;
    };
    if (copy_node.src->next == copy_node.dest) {
      VLOG(2) << copy->name() << " source and destination buffers are same.";
    } else if (IsHead(*copy_node.dest)) {
      VLOG(2) << copy->name() << " defines the first value in its buffer";
      bool live_range_before =
          CheckLiveRangeBefore(copy_node.src, Next(*copy_node.dest)) &&
          CheckLiveRangeBefore(copy_node.dest->prev, Next(*copy_node.src));
      VLOG(2) << "LiveRangeBefore result: " << live_range_before;
      if (!live_range_before &&
          CheckLiveRangeInterference(copy_node.src, copy_node.dest,
                                     kMergeFirstDestInSource)) {
        return false;
      }
      VLOG(2) << "Splice dest after source.";
      SpliceAfter(copy_node.dest, copy_node.src);
    } else if (IsTail(*copy_node.src)) {
      VLOG(2) << copy->name() << " copies the last value ("
              << copy_node.src->value->ToShortString() << ") in its buffer";
      bool live_range_before =
          CheckLiveRangeBefore(Prev(*copy_node.dest), copy_node.src->next) &&
          CheckLiveRangeBefore(copy_node.src, Next(*copy_node.dest));
      VLOG(2) << "LiveRangeBefore result: " << live_range_before;
      if (!live_range_before &&
          CheckLiveRangeInterference(copy_node.src, copy_node.dest,
                                     kMergeLastSourceInDest)) {
        VLOG(2) << "Region-based analysis concludes interference.";
        return false;
      }
      VLOG(2) << "Splice src after prev of dest.";
      SpliceAfter(copy_node.src->next, Prev(*copy_node.dest));
    } else {
      VLOG(2) << copy->name()
              << " copies value in middle of source buffer to value in middle "
                 "of destination buffer";
      return false;
    }
    RemoveCopyValue(copy_node.dest);
    XLA_VLOG_LINES(4, ToString());
    TF_DCHECK_OK(Verify());
    return true;
  }
  void RemoveCopyValue(ValueNode* copy_value_node) {
    CHECK_EQ(copy_value_node->value->defining_instruction()->opcode(),
             HloOpcode::kCopy);
    ValueNode* operand_node = copy_value_node->prev;
    CHECK(operand_node != copy_value_node);
    VLOG(2) << "Removing copy " << operand_node->value->ToShortString()
            << " => " << copy_value_node->value->ToShortString();
    operand_node->next = copy_value_node->next;
    copy_value_node->next->prev = operand_node;
    auto it = absl::c_find_if(operand_node->uses, [copy_value_node](
                                                      const HloUse* use) {
      return use->instruction == copy_value_node->value->defining_instruction();
    });
    CHECK(it != operand_node->uses.end());
    operand_node->uses.erase(it);
    for (const HloUse* copy_use : copy_value_node->uses) {
      operand_node->uses.push_back(copy_use);
      if (copy_use->instruction->opcode() == HloOpcode::kCopy &&
          ContainsKey(copy_map_, copy_use->instruction)) {
        copy_map_.at(copy_use->instruction).src = operand_node;
      }
    }
    copy_map_.erase(copy_value_node->value->defining_instruction());
    delete copy_value_node;
  }
  bool LiveRangeBefore(const ValueNode& a, const ValueNode& b) {
    if (a.uses.empty()) {
      VLOG(2) << "Empty uses for " << *a.value;
      return ordering_->IsDefinedBefore(*a.value, *b.value);
    }
    VLOG(3) << "Checking live ranges before: " << ValueListToString(&a)
            << " vs " << ValueListToString(&b);
    if (a.value->IsRootOf(b.value->defining_instruction()->parent())) {
      VLOG(3) << "Value is root of the same computation";
      return false;
    }
    return ordering_->UsesBeforeValueDefinition(
        a.uses, *b.value, dataflow_,
        false);
  }
  bool IsTail(const ValueNode& node) const {
    return ContainsKey(value_lists_, node.next);
  }
  bool IsHead(const ValueNode& node) const {
    return ContainsKey(value_lists_, &node);
  }
  ValueNode* Next(const ValueNode& node) const {
    if (IsTail(node)) {
      return nullptr;
    } else {
      return node.next;
    }
  }
  ValueNode* Prev(const ValueNode& node) const {
    if (IsHead(node)) {
      return nullptr;
    } else {
      return node.prev;
    }
  }
  void SpliceAfter(ValueNode* head, ValueNode* insert_after) {
    DCHECK(IsHead(*head));
    value_lists_.erase(head);
    ValueNode* tail = head->prev;
    tail->next = insert_after->next;
    insert_after->next->prev = tail;
    insert_after->next = head;
    head->prev = insert_after;
  }
  enum CombineLiveRangeOption {
    kMergeFirstDestInSource = 1,
    kMergeLastSourceInDest = 2
  };
  bool ValuesInterfere(const ValueNode* src, const ValueNode* dest,
                       CombineLiveRangeOption merge_location) {
    auto src_live_range = ComputeLiveRangeRegions(src);
    auto dest_live_range = ComputeLiveRangeRegions(dest);
    VLOG(5) << "src value: " << src->value->ToString();
    VLOG(5) << "src live range:\n" << src_live_range.ToString();
    VLOG(5) << "dest value: " << dest->value->ToString();
    VLOG(5) << "dest live range:\n" << dest_live_range.ToString();
    ComputeRelativeLocation relative_location_analysis(ordering_);
    auto rel1 =
        relative_location_analysis.Compute(src_live_range, dest_live_range);
    VLOG(3) << "Location of dest in relation to src: " << rel1.ToString()
            << " with interception set to " << rel1.InterceptDefUse();
    auto rel2 =
        relative_location_analysis.Compute(dest_live_range, src_live_range);
    VLOG(3) << "Location of src in relation to dest: " << rel2.ToString()
            << " with interception set to " << rel2.InterceptDefUse();
    if (rel1.RuntimeOrderOverlap() && rel2.RuntimeOrderOverlap()) {
      VLOG(3) << "Both relations are overlap.";
      return true;
    }
    if (rel1.RuntimeOrderOverlap() || rel2.RuntimeOrderOverlap()) {
      VLOG(3) << "At least one relation is overlap.";
      if (rel1.RuntimeOrderOverlap()) {
        VLOG(3) << "rel1 is overlap, with interception = "
                << rel1.InterceptDefUse();
        if (rel1.InterceptDefUse() ||
            (merge_location != kMergeFirstDestInSource &&
             rel2.InterceptDefUse())) {
          return true;
        }
      } else {
        VLOG(3) << "rel2 is overlap, with interception = "
                << rel2.InterceptDefUse();
        if (rel2.InterceptDefUse() ||
            (merge_location != kMergeLastSourceInDest &&
             rel1.InterceptDefUse())) {
          return true;
        }
      }
    }
    if (relative_location_analysis.AddControlDependenceForUnorderedOps()) {
      return false;
    } else {
      return true;
    }
  }
  void ForEachValueInRange(const ValueNode* element,
                           absl::FunctionRef<void(const ValueNode*)> visitor) {
    const ValueNode* head = element;
    for (const ValueNode* p = head; p != nullptr; p = Next(*p)) {
      visitor(p);
    }
    while (!IsHead(*head)) {
      head = Prev(*head);
    }
    for (const ValueNode* p = head; p != element; p = Next(*p)) {
      visitor(p);
    }
  }
  std::string ValueListToString(const ValueNode* element) {
    std::string result = "{";
    auto VisitValueNode = [&](const ValueNode* node) {
      if (result == "{") {
        StrAppend(&result, node->value->ToShortString());
      } else {
        StrAppend(&result, ", ", node->value->ToShortString());
      }
    };
    ForEachValueInRange(element, VisitValueNode);
    StrAppend(&result, "}");
    return result;
  }
  std::string ToString() const {
    std::string out = absl::StrCat("CopyRemover:\n");
    StrAppend(&out, "  Def-use chains in each buffer:\n");
    for (const ValueNode* head : value_lists_) {
      StrAppend(&out, "    Buffer defined by ", head->value->ToShortString(),
                ":\n");
      const ValueNode* p = head;
      do {
        StrAppend(&out, "      ", p->value->ToShortString(), ", uses: ",
                  absl::StrJoin(p->uses, "; ",
                                [](std::string* s, const HloUse* use) {
                                  StrAppend(s, use->ToString());
                                }),
                  "\n");
        p = p->next;
      } while (p != head);
    }
    StrAppend(&out, "  Potentially removable copies:\n");
    for (const auto& pair : copy_map_) {
      const HloInstruction* copy = pair.first;
      const CopyNodes& copy_info = pair.second;
      StrAppend(&out, "    ", copy->name(), " : ",
                copy_info.src->value->ToShortString(), " => ",
                copy_info.dest->value->ToShortString(), "\n");
    }
    return out;
  }
 private:
  const HloDataflowAnalysis& dataflow_;
  HloOrdering* ordering_;
  absl::flat_hash_set<const ValueNode*> value_lists_;
  struct CopyNodes {
    ValueNode* src = nullptr;
    ValueNode* dest = nullptr;
  };
  absl::flat_hash_map<const HloInstruction*, CopyNodes> copy_map_;
};
}  
absl::Status CopyInsertion::AddCopiesForConditional(
    const HloAliasAnalysis& alias_analysis, HloInstruction* conditional) {
  VLOG(2) << "Adding copies for kConditional instruction "
          << conditional->name();
  ShapeTree<bool> indices_to_copy(conditional->shape());
  TF_RET_CHECK(conditional->opcode() == HloOpcode::kConditional);
  if (!IndicesToCopyForConditional(alias_analysis.dataflow_analysis(),
                                   conditional, &indices_to_copy)) {
    VLOG(2) << "No copies necessary for kConditional instruction "
            << conditional->name();
    return absl::OkStatus();
  }
  for (HloComputation* computation : conditional->branch_computations()) {
    HloInstruction* root = computation->root_instruction();
    std::vector<HloInstruction*> users = root->users();
    TF_ASSIGN_OR_RETURN(
        HloInstruction * deep_copy,
        computation->DeepCopyInstruction(root, &indices_to_copy));
    for (HloInstruction* user : users) {
      TF_RETURN_IF_ERROR(root->ReplaceUseWith(user, deep_copy));
    }
    computation->set_root_instruction(deep_copy);
  }
  return absl::OkStatus();
}
absl::Status CopyInsertion::AddCopiesToResolveInterference(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(module, can_share_buffer_));
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    if (computation->IsAsyncComputation()) {
      continue;
    }
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kWhile) {
        TF_RETURN_IF_ERROR(AddCopiesForWhile(*alias_analysis, instruction));
      } else if (instruction->opcode() == HloOpcode::kConditional) {
        TF_RETURN_IF_ERROR(
            AddCopiesForConditional(*alias_analysis, instruction));
      } else {
        absl::flat_hash_set<int64_t> copied_operands;
        for (const auto& operand_and_output_index :
             HloDataflowAnalysis::GetInPlaceInputOutputPairs(
                 instruction->opcode() == HloOpcode::kAsyncStart
                     ? instruction->async_wrapped_instruction()
                     : instruction)) {
          const HloOperandIndex& operand_index = operand_and_output_index.first;
          if (copied_operands.contains(operand_index.operand_number)) {
            continue;
          }
          bool can_share_buffer = false;
          if (can_share_buffer_ != nullptr) {
            auto maybe_can_share_buffer = can_share_buffer_(
                instruction, instruction->operand(operand_index.operand_number),
                operand_index.operand_index);
            if (maybe_can_share_buffer.has_value()) {
              can_share_buffer = maybe_can_share_buffer.value();
            }
          }
          if (can_share_buffer &&
              HasDisjointReadWriteRegionsAttr(instruction) &&
              absl::c_all_of(
                  instruction->operand(operand_index.operand_number)->users(),
                  [&instruction](const HloInstruction* user) {
                    return user == instruction;
                  })) {
            continue;
          }
          copied_operands.insert(operand_index.operand_number);
          TF_RETURN_IF_ERROR(AddCopiesForInPlaceOperation(
              *alias_analysis, instruction, operand_index.operand_number));
        }
      }
    }
  }
  TF_RETURN_IF_ERROR(
      AddCopiesForAliasedInputOutputs(module, execution_threads));
  return absl::OkStatus();
}
absl::Status CopyInsertion::AddSpecialCaseCopies(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  return AddSpecialCaseCopies(*call_graph, execution_threads, module);
}
absl::Status CopyInsertion::AddSpecialCaseCopies(
    const CallGraph& call_graph,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    HloModule* module) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(module, can_share_buffer_));
  HloInstructionMap<ShapeTree<bool>> instructions_to_copy;
  auto add_index_to_copy = [&instructions_to_copy](HloInstruction* instruction,
                                                   const ShapeIndex& index) {
    auto it = instructions_to_copy.find(instruction);
    if (it == instructions_to_copy.end()) {
      auto it_added = instructions_to_copy.emplace(
          std::piecewise_construct, std::forward_as_tuple(instruction),
          std::forward_as_tuple(instruction->shape(), false));
      it = it_added.first;
    }
    *it->second.mutable_element(index) = true;
  };
  for (const HloValue* value : alias_analysis->dataflow_analysis().values()) {
    HloBuffer& buffer = alias_analysis->GetBufferContainingValue(*value);
    if (buffer.values().size() > 1 && ValueIsReadOnly(*value)) {
      VLOG(2) << "Value " << value->ToShortString()
              << " is read only, but its buffer contains more than one value. "
                 "Copying.";
      add_index_to_copy(value->defining_instruction(), value->defining_index());
    }
    for (const HloValue* value2 : buffer.values()) {
      if (value2 == value) {
        continue;
      }
      HloPosition position = value2->defining_position();
      for (const HloUse& use : value->GetUses()) {
        if (use.instruction == position.instruction) {
          VLOG(3) << "Same instruction: " << position.instruction->ToString();
          if (!alias_analysis->dataflow_analysis()
                   .CanShareOperandBufferWithUser(
                       use.instruction->mutable_operand(
                           use.operand_number),
                       use.operand_index,
                       position.instruction,
                       position.index)) {
            VLOG(2) << "Adding back copy: "
                    << use.instruction->operand(use.operand_number)->ToString()
                    << "@" << use.operand_index.ToString()
                    << " instr: " << position.instruction->ToString() << "@"
                    << position.index;
            add_index_to_copy(
                use.instruction->mutable_operand(use.operand_number),
                use.operand_index);
          }
        }
      }
    }
  }
  for (HloComputation* computation : module->computations(execution_threads)) {
    const CallGraphNode& node = call_graph.GetNode(computation);
    if (node.context() == CallContext::kEmbedded) {
      continue;
    }
    TF_RET_CHECK(node.context() == CallContext::kControlFlow);
    SpecialCaseCopyPolicy policy =
        GetSpecialCaseCopyPolicy(node, module, computation);
    HloInstruction* root = computation->root_instruction();
    absl::flat_hash_map<const HloBuffer*, ShapeIndex> seen;
    ShapeUtil::ForEachSubshape(
        root->shape(), [&](const Shape& , const ShapeIndex& index) {
          std::vector<const HloBuffer*> buffers_at_index =
              alias_analysis->ComputeBuffersAt(root, index);
          bool buffer_seen_before = false;
          for (const HloBuffer* buffer : buffers_at_index) {
            buffer_seen_before |= !seen.emplace(buffer, index).second;
          }
          if (buffer_seen_before && policy.copy_root_replicated_buffers &&
              computation == module->entry_computation() &&
              module->input_output_alias_config().OutputHasAlias(index) &&
              buffers_at_index.size() == 1) {
            std::optional<HloInputOutputAliasConfig::Alias> alias =
                module->input_output_alias_config().GetAliasedParameter(index);
            CHECK(alias) << "Alias does not exist";
            const ShapeIndex& other_index = seen[buffers_at_index[0]];
            VLOG(2) << "Output indices " << index.ToString() << " and "
                    << other_index.ToString() << " are both aliased to "
                    << alias->parameter_number << " copying " << other_index;
            add_index_to_copy(root, other_index);
            return;
          }
          if (buffers_at_index.size() > 1 ||
              (buffer_seen_before && policy.copy_root_replicated_buffers)) {
            VLOG(2) << "Index " << index << " of computation "
                    << computation->name() << " (" << root->name()
                    << ") has ambiguous or non-distinct buffer. Copying.";
            add_index_to_copy(root, index);
          }
        });
    for (const auto& pair :
         alias_analysis->dataflow_analysis().GetInstructionValueSet(root)) {
      const ShapeIndex& index = pair.first;
      const HloValueSet& value_set = pair.second;
      for (const HloValue* value : value_set.values()) {
        if (ShouldCopyRootValue(*value, policy)) {
          VLOG(2) << "Root of (" << root->name() << ") of computation("
                  << computation->name()
                  << ") has constant or parameter value at index " << index
                  << ". Copying.";
          add_index_to_copy(root, index);
        }
      }
    }
  }
  for (const auto& pair : instructions_to_copy) {
    HloInstruction* instruction = pair.first;
    const ShapeTree<bool>& indices_to_copy = pair.second;
    ShapeTree<HloInstruction*> copies_added(indices_to_copy.shape());
    std::vector<HloInstruction*> users = instruction->users();
    TF_ASSIGN_OR_RETURN(HloInstruction * deep_copy,
                        instruction->parent()->DeepCopyInstruction(
                            instruction, &indices_to_copy, &copies_added));
    for (HloInstruction* user : users) {
      TF_RETURN_IF_ERROR(instruction->ReplaceUseWith(user, deep_copy));
    }
    if (instruction == instruction->parent()->root_instruction()) {
      instruction->parent()->set_root_instruction(deep_copy);
    }
  }
  return absl::OkStatus();
}
static int64_t GetNumExistingCopies(
    const HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  int64_t num_existing_copies = 0;
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kCopy) {
        ++num_existing_copies;
      }
    }
  }
  return num_existing_copies;
}
absl::Status CopyInsertion::RemoveUnnecessaryCopies(
    HloModule* module, bool check_live_range_ordering,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      4, module->ToString(HloPrintOptions().set_syntax_sugar_async_ops(false)));
  std::unique_ptr<HloOrdering> ordering;
  if (module->has_schedule()) {
    ordering = std::make_unique<SequentialHloOrdering>(module->schedule());
  } else {
    ordering = std::make_unique<DependencyHloOrdering>(module);
  }
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(module, can_share_buffer_));
  CopyRemover copy_remover(*module, *alias_analysis, ordering.get(),
                           check_live_range_ordering, execution_threads);
  if (VLOG_IS_ON(3)) {
    LOG(INFO) << "Removing unnecessary copies in " << module->name();
    LOG(INFO) << "Buffer values, in dependency order: ";
    for (const HloBuffer& buffer : alias_analysis->buffers()) {
      LOG(INFO) << "    HloBuffer " << buffer.id();
    }
  }
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  int64_t num_existing_copies = GetNumExistingCopies(module, execution_threads);
  bool changed = true;
  int64_t num_iterations = -1;
  VLOG(6) << "Copy Insertion analyzing module with instruction count = "
          << module->instruction_count();
  BoundNonLinearCompilerAnalysis allowance(module, name(), 10);
  while (changed) {
    CHECK_LE(++num_iterations, num_existing_copies);
    changed = false;
    VLOG(2) << "Running fixpoint iteration " << num_iterations
            << " of copy elision";
    for (HloComputation* computation :
         module->computations(execution_threads)) {
      VLOG(2) << "computation:" << computation->name();
      for (HloInstruction* instruction : computation->instructions()) {
        if (instruction->opcode() != HloOpcode::kCopy) continue;
        int64_t region_analysis_cost_now =
            (use_region_based_live_range_analysis_ == 0)
                ? 0
                : std::min(allowance.analysis_allowance(),
                           use_region_based_live_range_analysis_);
        if (copy_remover.TryElideCopy(instruction, &region_analysis_cost_now)) {
          changed = true;
          TF_RETURN_IF_ERROR(StripControlDependenciesFrom(instruction));
          TF_RETURN_IF_ERROR(
              instruction->ReplaceAllUsesWith(instruction->mutable_operand(0)));
          VLOG(6) << "succeeded in eliminating copy.";
        }
        if (allowance.ContinueAnalysis() && region_analysis_cost_now > 0) {
          VLOG(6) << "Copy Insertion analyzing module cost: "
                  << region_analysis_cost_now;
          VLOG(6) << "instruction:" << instruction->ToString();
          allowance.DeductCost(region_analysis_cost_now);
          VLOG(6) << "allowance:" << allowance.analysis_allowance();
        }
      }
    }
  }
  return absl::OkStatus();
}
absl::StatusOr<bool> CopyInsertion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  if (!call_graph->IsFlattened()) {
    return FailedPrecondition(
        "Call graph must be flattened before copy insertion.");
  }
  int64_t num_copies_before = GetNumExistingCopies(module, execution_threads);
  TF_RETURN_IF_ERROR(AddCopiesToResolveInterference(module, execution_threads));
  TupleSimplifier tuple_simplifier;
  HloDCE dce;
  TF_RETURN_IF_ERROR(tuple_simplifier.Run(module, execution_threads).status());
  TF_RETURN_IF_ERROR(dce.Run(module, execution_threads).status());
  DumpHloModuleDuringPassIfEnabled(
      name(), "after adding copies to resolve interference", *module);
  TF_RETURN_IF_ERROR(RemoveUnnecessaryCopies(module,
                                             true,
                                             execution_threads));
  DumpHloModuleDuringPassIfEnabled(name(), "after removing unnecessary copies",
                                   *module);
  TF_RETURN_IF_ERROR(
      AddSpecialCaseCopies(*call_graph, execution_threads, module));
  DumpHloModuleDuringPassIfEnabled(name(), "after adding special-case copies",
                                   *module);
  TF_RETURN_IF_ERROR(tuple_simplifier.Run(module, execution_threads).status());
  TF_RETURN_IF_ERROR(dce.Run(module, execution_threads).status());
  VLOG(1) << "Num copies before copy-insertion: " << num_copies_before;
  VLOG(1) << "Num copies after copy-insertion: "
          << GetNumExistingCopies(module, execution_threads);
  return true;
}
}  