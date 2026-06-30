#include "xla/service/hlo_dataflow_analysis.h"
#include <algorithm>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/map_util.h"
#include "xla/shape_util.h"
#include "xla/types.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
namespace xla {
namespace {
int64_t CalculatePostOrderScheduleHelper(
    const HloComputation* comp, int64_t start_ordinal,
    absl::flat_hash_map<HloInstruction*, int64_t>* ordinal_map) {
  int64_t ordinal = start_ordinal;
  for (HloInstruction* instruction : comp->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kCall ||
        instruction->opcode() == HloOpcode::kConditional) {
      for (const HloComputation* called_computation :
           instruction->called_computations()) {
        ordinal = CalculatePostOrderScheduleHelper(called_computation, ordinal,
                                                   ordinal_map);
      }
    }
    if (instruction->opcode() == HloOpcode::kWhile) {
      ordinal = CalculatePostOrderScheduleHelper(instruction->while_condition(),
                                                 ordinal, ordinal_map);
      ordinal = CalculatePostOrderScheduleHelper(instruction->while_body(),
                                                 ordinal, ordinal_map);
    }
    ordinal_map->insert({instruction, ordinal++});
  }
  return ordinal;
}
absl::flat_hash_map<HloInstruction*, int64_t> CalculatePostOrderSchedule(
    const HloModule& module) {
  absl::flat_hash_map<HloInstruction*, int64_t> map;
  CalculatePostOrderScheduleHelper(module.entry_computation(), 0, &map);
  return map;
}
}  
using absl::StrAppend;
using absl::StrCat;
HloDataflowAnalysis::HloDataflowAnalysis(
    const HloModule& module, bool ssa_form, bool bitcast_defines_value,
    const CanShareBuffer& can_share_buffer, const ForwardsValue& forwards_value,
    absl::flat_hash_set<absl::string_view> execution_threads)
    : module_(module),
      execution_threads_(std::move(execution_threads)),
      ssa_form_(ssa_form),
      bitcast_defines_value_(bitcast_defines_value),
      call_graph_(CallGraph::Build(&module)),
      can_share_buffer_(can_share_buffer),
      forwards_value_(forwards_value) {}
bool HloDataflowAnalysis::AreTransitiveUsesElementwiseOrTuple(
    const HloInstruction* inst) {
  absl::flat_hash_set<const HloInstruction*> visited;
  absl::InlinedVector<const HloInstruction*, 4> stack;
  stack.push_back(inst);
  while (!stack.empty()) {
    const HloInstruction* current = stack.back();
    stack.pop_back();
    visited.insert(current);
    for (const HloInstruction* user : current->users()) {
      for (const int64_t use_index : user->OperandIndices(current)) {
        if (!user->IsElementwiseOnOperand(use_index) &&
            user->opcode() != HloOpcode::kTuple) {
          return false;
        }
      }
      if (!visited.contains(user)) {
        stack.push_back(user);
      }
    }
  }
  return true;
}
namespace {
bool Is1dSliceWithoutStrides(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kSlice &&
         1 == instr->slice_starts().size() &&
         1 == instr->slice_limits().size() &&
         1 == instr->slice_strides().size() &&
         1 == instr->slice_strides().at(0);
}
bool IsSliceInputFusion(const HloInstruction& unnested_hlo) {
  if (!unnested_hlo.IsInputFusion()) {
    return false;
  }
  const HloInstruction* root = unnested_hlo.fused_expression_root();
  if (root->opcode() != HloOpcode::kTuple) {
    return false;
  }
  return absl::c_all_of(root->operands(), [](const HloInstruction* instr) {
    return Is1dSliceWithoutStrides(instr);
  });
}
struct ConcatUsageInfo {
  const HloInstruction* prev_concat;
  int64_t concat_opnd_idx;
  const HloInstruction* slice_to_recover_opnd;
};
std::optional<ConcatUsageInfo> ConcatIsEffectivelyElementwise(
    const HloInstruction& concat, const HloInstruction& operand,
    const ConcatUsageInfo& info) {
  std::vector<HloInstruction*> users = concat.users();
  if (!absl::c_all_of(users, Is1dSliceWithoutStrides)) {
    return std::optional<ConcatUsageInfo>();
  }
  if (users.size() != concat.operand_count() ||
      concat.operand_count() != concat.unique_operands().size()) {
    return std::optional<ConcatUsageInfo>();
  }
  absl::c_sort(users, [](const HloInstruction* a, const HloInstruction* b) {
    return a->slice_starts().at(0) < b->slice_starts().at(0);
  });
  int64_t prev_limit = 0;
  for (int64_t i = 0; i < users.size(); ++i) {
    const HloInstruction* u = users[i];
    int64_t slice_size = u->slice_limits().at(0) - u->slice_starts().at(0);
    if (u->slice_starts().at(0) != prev_limit ||
        slice_size != ShapeUtil::ElementsIn(concat.operand(i)->shape())) {
      return std::optional<ConcatUsageInfo>();
    }
    prev_limit = u->slice_limits().at(0);
  }
  int64_t operand_idx = concat.operand_index(&operand);
  if (info.prev_concat != nullptr) {
    bool is_concat_identical = info.prev_concat->Identical(
        concat,
        [](const HloInstruction*, const HloInstruction*) {
          return true;
        });
    if (!is_concat_identical || info.concat_opnd_idx != operand_idx) {
      return std::optional<ConcatUsageInfo>();
    }
  }
  const HloInstruction* slice_to_recover_opnd = users.at(operand_idx);
  return std::optional<ConcatUsageInfo>(
      ConcatUsageInfo{&concat, operand_idx, slice_to_recover_opnd});
}
bool AreTransitiveUsesEffectivelyElementwise(const HloInstruction* param,
                                             const HloInstruction* root_tuple,
                                             const ShapeIndex& out_shape_idx) {
  CHECK_EQ(root_tuple->opcode(), HloOpcode::kTuple);
  CHECK_EQ(out_shape_idx.size(), 1);
  absl::flat_hash_set<const HloInstruction*> visited;
  absl::InlinedVector<const HloInstruction*, 4> stack;
  stack.push_back(param);
  ConcatUsageInfo concat_usage_info{nullptr, 0, nullptr};
  bool is_output_reachable = false;
  while (!stack.empty()) {
    const HloInstruction* current = stack.back();
    stack.pop_back();
    visited.insert(current);
    for (const HloInstruction* user : current->users()) {
      VLOG(3) << "Visiting: " << user->ToString();
      switch (user->opcode()) {
        case HloOpcode::kTuple:
          if (user == root_tuple &&
              current == root_tuple->operand(out_shape_idx.back())) {
            is_output_reachable = true;
          }
          break;
        case HloOpcode::kReshape:
          if (!ShapeUtil::ReshapeIsBitcast(current->shape(), user->shape())) {
            return false;
          }
          break;
        case HloOpcode::kConcatenate: {
          std::optional<ConcatUsageInfo> optional_concat_info =
              ConcatIsEffectivelyElementwise(*user, *current,
                                             concat_usage_info);
          if (!optional_concat_info) {
            return false;
          }
          concat_usage_info = *optional_concat_info;
          CHECK(!visited.contains(concat_usage_info.slice_to_recover_opnd));
          stack.push_back(concat_usage_info.slice_to_recover_opnd);
          continue;
        }
        default:
          for (const int64_t use_index : user->OperandIndices(current)) {
            if (!user->IsElementwiseOnOperand(use_index)) {
              return false;
            }
          }
          if (!LayoutUtil::Equal(current->shape().layout(),
                                 user->shape().layout())) {
            return false;
          }
          break;
      }  
      if (!visited.contains(user)) {
        stack.push_back(user);
      }
    }
  }
  return is_output_reachable;
}
}  
bool HloDataflowAnalysis::ValueIsDefinedAt(const HloInstruction* instruction,
                                           const ShapeIndex& index) const {
  const HloValueSet& value_set = GetValueSet(instruction, index);
  if (value_set.values().size() != 1) {
    return false;
  }
  return value_set.GetUniqueValue().defining_instruction() == instruction;
}
const HloValue& HloDataflowAnalysis::GetValueDefinedAt(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  CHECK(ValueIsDefinedAt(instruction, index)) << instruction->ToString();
  return GetUniqueValueAt(instruction, index);
}
HloValue& HloDataflowAnalysis::GetValueDefinedAt(
    const HloInstruction* instruction, const ShapeIndex& index) {
  CHECK(ValueIsDefinedAt(instruction, index));
  return GetUniqueValueAt(instruction, index);
}
HloValue* HloDataflowAnalysis::NewHloValue(HloInstruction* instruction,
                                           const ShapeIndex& index,
                                           bool is_phi) {
  const int64_t value_id = next_value_id_++;
  auto result =
      values_.insert({value_id, std::make_unique<HloValue>(
                                    value_id, instruction, index, is_phi)});
  CHECK(result.second);
  VLOG(4) << "NewHloValue = " << result.first->second->ToShortString();
  return result.first->second.get();
}
void HloDataflowAnalysis::MarkValueForDeletion(HloValue::Id value_id) {
  const HloValue& value = GetValue(value_id);
  VLOG(4) << "MarkValueForDeletion(" << value.ToShortString() << ")";
  value_ids_to_delete_.push_back(value_id);
}
void HloDataflowAnalysis::DeleteMarkedValues() {
  absl::flat_hash_set<HloValue::Id> id_set(value_ids_to_delete_.begin(),
                                           value_ids_to_delete_.end());
#ifndef NDEBUG
  for (const auto& pair : value_sets_) {
    const HloInstruction* instruction = pair.first;
    const InstructionValueSet& instruction_value_set = *pair.second;
    for (const auto& index_value_set : instruction_value_set) {
      const HloValueSet& value_set = index_value_set.second;
      for (const HloValue* value : value_set.values()) {
        DCHECK(!ContainsKey(id_set, value->id()))
            << "Value " << value->ToShortString()
            << " marked for deletion, but still exists in value set for "
               "instruction "
            << instruction->name();
      }
    }
  }
#endif
  for (HloValue::Id value_id : id_set) {
    values_.erase(value_id);
  }
  value_ids_to_delete_.clear();
}
std::string HloDataflowAnalysis::ToString() const {
  std::string out =
      StrCat("HloDataflowAnalysis, module ", module_.name(), "\n");
  StrAppend(&out, "  Instruction value sets:\n");
  for (const HloComputation* computation : module_.computations()) {
    if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                          execution_threads_)) {
      continue;
    }
    for (const HloInstruction* instruction : computation->instructions()) {
      StrAppend(&out, "Instruction: \n  ", instruction->name(), ":\n");
      if (instruction->shape().IsTuple()) {
        GetInstructionValueSet(instruction)
            .ForEachElement([this, &instruction, &out](
                                const ShapeIndex& index,
                                const HloValueSet& value_set) {
              StrAppend(&out, "      tuple index ", index.ToString(), ":\n");
              for (const HloValue* value : value_set.values()) {
                StrAppend(&out, "        ", value->ToShortString(),
                          ValueIsDefinedAt(instruction, index) ? " (def)" : "",
                          "\n");
              }
            });
      } else {
        const HloValueSet& top_level_value_set =
            GetValueSet(instruction, {});
        for (const HloValue* value : top_level_value_set.values()) {
          StrAppend(&out, "      ", value->ToShortString(),
                    ValueIsDefinedAt(instruction) ? " (def)" : "", "\n");
        }
      }
    }
  }
  StrAppend(&out, "  HloValues:\n");
  for (const HloValue* value : values()) {
    StrAppend(&out, value->ToString(4));
  }
  return out;
}
bool HloDataflowAnalysis::Phi(
    HloInstruction* instruction,
    absl::Span<const InstructionValueSet* const> inputs) {
  CHECK(ssa_form_);
  VLOG(4) << "Phi(" << instruction->name() << ")";
  VLOG(5) << "instruction value set = "
          << GetInstructionValueSet(instruction).ToString();
  for (const InstructionValueSet* input : inputs) {
    VLOG(5) << "input value set = " << input->ToString();
  }
  if (bitcast_defines_value_) {
    absl::c_for_each(inputs, [&](const InstructionValueSet* input) {
      DCHECK(ShapeUtil::Compatible(instruction->shape(), input->shape()));
    });
  } else {
    const Shape& shape = instruction->shape();
    PrimitiveType ty = shape.element_type();
    bool is_array = shape.IsArray();
    absl::c_for_each(inputs, [&](const InstructionValueSet* input) {
      DCHECK(
          ty == input->shape().element_type() &&
          (!is_array ||
           ShapeUtil::ElementsIn(shape) ==
               ShapeUtil::ElementsIn(input->shape()) ||
           ShapeUtil::ArraySize(shape) == ShapeUtil::ArraySize(input->shape())))
          << shape.ToString() << " vs." << input->shape().ToString();
    });
  }
  bool changed = false;
  for (auto& pair : GetInstructionValueSet(instruction)) {
    const ShapeIndex& index = pair.first;
    HloValueSet& value_set = pair.second;
    CHECK_LE(value_set.values().size(), 1);
    const HloValue* current_value =
        value_set.values().size() == 1 ? value_set.values()[0] : nullptr;
    std::vector<HloValue::Id> input_value_ids;
    for (const InstructionValueSet* input : inputs) {
      for (const HloValue* value : input->element(index).values()) {
        input_value_ids.push_back(value->id());
      }
    }
    bool current_value_defined_here =
        (current_value != nullptr &&
         current_value->defining_instruction() == instruction &&
         current_value->defining_index() == index);
    VLOG(5) << "after input_value_ids.size = " << input_value_ids.size();
    if (input_value_ids.empty()) {
      CHECK_EQ(value_set.values().size(), 0)
          << "Instruction " << instruction->name() << " at index " << index
          << " previously had non-empty value set. Value set: " << value_set;
    } else if (input_value_ids.size() == 1) {
      const HloValue& new_value = GetValue(input_value_ids[0]);
      if (current_value == nullptr) {
        value_set.Clear();
        value_set.AddValue(&new_value);
        changed = true;
      } else if (current_value != &new_value) {
        if (current_value_defined_here) {
          MarkValueForDeletion(current_value->id());
        }
        value_set.Clear();
        value_set.AddValue(&new_value);
        changed = true;
      }
    } else {
      CHECK_GT(input_value_ids.size(), 1);
      bool phi_defined_here =
          current_value_defined_here && current_value->is_phi();
      if (current_value == nullptr || !phi_defined_here) {
        value_set.Clear();
        value_set.AddValue(NewHloValue(instruction, index, true));
        std::vector<HloValue*> inputs;
        inputs.reserve(input_value_ids.size());
        for (HloValue::Id id : input_value_ids) {
          inputs.push_back(&GetValue(id));
        }
        phi_graph_.RegisterPhi(*value_set.values()[0], inputs);
        changed = true;
      } else if (phi_defined_here) {
        std::vector<HloValue*> new_inputs;
        new_inputs.reserve(input_value_ids.size());
        for (HloValue::Id id : input_value_ids) {
          new_inputs.push_back(&GetValue(id));
        }
        if (!phi_graph_.InputsEqualTo(*current_value, new_inputs)) {
          VLOG(1) << current_value->ToShortString() << " has new phi inputs: ";
          phi_graph_.RegisterPhi(*current_value, new_inputs);
          changed = true;
        }
      }
    }
  }
  return changed;
}
const HloValue& HloDataflowAnalysis::GetValue(HloValue::Id value_id) const {
  DCHECK(values_.contains(value_id)) << "Value not found: " << value_id;
  return *values_.find(value_id)->second;
}
HloValue& HloDataflowAnalysis::GetValue(HloValue::Id value_id) {
  DCHECK(values_.contains(value_id)) << "Value not found: " << value_id;
  return *values_.find(value_id)->second;
}
HloValueSet HloDataflowAnalysis::GetFlattenedValueSet(
    const HloInstruction* instruction) const {
  HloValueSet value_set;
  const InstructionValueSet& value_set_tree =
      GetInstructionValueSet(instruction);
  std::vector<const HloValueSet*> all_sets;
  for (auto& pair : value_set_tree) {
    const HloValueSet& value_set = pair.second;
    all_sets.push_back(&value_set);
  }
  value_set.AssignUnionOf(all_sets);
  return value_set;
}
const HloValueSet& HloDataflowAnalysis::GetValueSet(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  return GetInstructionValueSet(instruction).element(index);
}
HloValueSet& HloDataflowAnalysis::GetValueSet(const HloInstruction* instruction,
                                              const ShapeIndex& index) {
  return *GetInstructionValueSet(instruction).mutable_element(index);
}
const HloValueSet& HloDataflowAnalysis::GetValueSet(
    const HloPosition& position) const {
  return GetValueSet(position.instruction, position.index);
}
HloValueSet& HloDataflowAnalysis::GetValueSet(const HloPosition& position) {
  return GetValueSet(position.instruction, position.index);
}
bool HloDataflowAnalysis::UpdateBitcastValueSet(HloInstruction* bitcast) {
  CHECK_EQ(bitcast->opcode(), HloOpcode::kBitcast);
  const InstructionValueSet& operand_set =
      GetInstructionValueSet(bitcast->operand(0));
  InstructionValueSet& bitcast_set = GetInstructionValueSet(bitcast);
  if (!bitcast_defines_value_ && operand_set != bitcast_set) {
    bitcast_set = operand_set;
    return true;
  }
  return false;
}
bool HloDataflowAnalysis::UpdateSendValueSet(HloInstruction* send) {
  CHECK_EQ(send->opcode(), HloOpcode::kSend);
  bool changed = false;
  for (auto& pair : GetInstructionValueSet(send->operand(0))) {
    const ShapeIndex& operand_index = pair.first;
    const HloValueSet& operand_value_set = pair.second;
    ShapeIndex index = {0};
    for (int64_t i : operand_index) {
      index.push_back(i);
    }
    HloValueSet& value_set = GetValueSet(send, index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}
bool HloDataflowAnalysis::UpdateAsyncStartValueSet(
    HloInstruction* async_start) {
  CHECK_EQ(async_start->opcode(), HloOpcode::kAsyncStart);
  bool changed = false;
  for (int64_t i = 0; i < async_start->operand_count(); ++i) {
    const HloInstruction* operand = async_start->operand(i);
    ShapeUtil::ForEachSubshape(
        operand->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
          if (!subshape.IsArray()) {
            return;
          }
          const HloValueSet& operand_value_set = GetValueSet(operand, index);
          ShapeIndex output_index = {0, i};
          output_index.insert(output_index.end(), index.begin(), index.end());
          HloValueSet& value_set = GetValueSet(async_start, output_index);
          if (value_set != operand_value_set) {
            value_set = operand_value_set;
            changed = true;
          }
        });
  }
  if (!HloInstruction::IsThreadIncluded(async_start->async_execution_thread(),
                                        execution_threads_)) {
    return changed;
  }
  HloInstruction* root =
      async_start->async_wrapped_computation()->root_instruction();
  ShapeUtil::ForEachSubshape(
      root->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (!subshape.IsArray()) {
          return;
        }
        const HloValueSet& root_value_set = GetValueSet(root, index);
        ShapeIndex output_index = {1};
        output_index.insert(output_index.end(), index.begin(), index.end());
        HloValueSet& value_set = GetValueSet(async_start, output_index);
        if (value_set != root_value_set) {
          value_set = root_value_set;
          changed = true;
        }
      });
  return changed;
}
bool HloDataflowAnalysis::UpdateAsyncUpdateValueSet(
    HloInstruction* async_update) {
  CHECK_EQ(async_update->opcode(), HloOpcode::kAsyncUpdate);
  CHECK_EQ(async_update->shape(), async_update->operand(0)->shape());
  bool changed = false;
  HloInstruction* root =
      HloInstruction::IsThreadIncluded(async_update->async_execution_thread(),
                                       execution_threads_)
          ? async_update->async_wrapped_computation()->root_instruction()
          : nullptr;
  ShapeUtil::ForEachSubshape(
      async_update->operand(0)->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) {
        if (!subshape.IsArray()) {
          return;
        }
        const HloValueSet& operand_value_set =
            GetValueSet(async_update->operand(0), index);
        HloValueSet& value_set = GetValueSet(async_update, index);
        CHECK_GE(index.size(), 0);
        if (index[0] != 1) {
          if (value_set != operand_value_set) {
            value_set = operand_value_set;
            changed = true;
          }
        } else if (root != nullptr) {
          ShapeIndex root_index(index.begin() + 1, index.end());
          const HloValueSet& root_value_set = GetValueSet(root, root_index);
          changed |=
              value_set.AssignUnionOf({&operand_value_set, &root_value_set});
        } else if (value_set != operand_value_set) {
          value_set = operand_value_set;
          changed = true;
        }
      });
  return changed;
}
bool HloDataflowAnalysis::UpdateAsyncDoneValueSet(HloInstruction* async_done) {
  CHECK_EQ(async_done->opcode(), HloOpcode::kAsyncDone);
  bool changed = false;
  HloInstruction* root =
      HloInstruction::IsThreadIncluded(async_done->async_execution_thread(),
                                       execution_threads_)
          ? async_done->async_wrapped_computation()->root_instruction()
          : nullptr;
  ShapeUtil::ForEachSubshape(
      async_done->operand(0)->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) {
        if (!subshape.IsArray() || index.front() != 1) {
          return;
        }
        const HloValueSet& operand_value_set =
            GetValueSet(async_done->operand(0), index);
        ShapeIndex output_index(index.begin() + 1, index.end());
        HloValueSet& value_set = GetValueSet(async_done, output_index);
        if (root != nullptr) {
          const HloValueSet& root_value_set = GetValueSet(root, output_index);
          changed |=
              value_set.AssignUnionOf({&operand_value_set, &root_value_set});
        } else if (value_set != operand_value_set) {
          value_set = operand_value_set;
          changed = true;
        }
      });
  return changed;
}
bool HloDataflowAnalysis::UpdateCopyStartValueSet(HloInstruction* copy_start) {
  CHECK_EQ(copy_start->opcode(), HloOpcode::kCopyStart);
  bool changed = false;
  const HloValueSet& operand_value_set = GetValueSet(copy_start->operand(0));
  HloValueSet& value_set = GetValueSet(copy_start, {1});
  if (value_set != operand_value_set) {
    value_set = operand_value_set;
    changed = true;
  }
  return changed;
}
bool HloDataflowAnalysis::UpdateCopyDoneValueSet(HloInstruction* copy_done) {
  CHECK_EQ(copy_done->opcode(), HloOpcode::kCopyDone);
  bool changed = false;
  const HloValueSet& operand_value_set =
      GetValueSet(copy_done->operand(0), {0});
  HloValueSet& value_set = GetValueSet(copy_done);
  if (value_set != operand_value_set) {
    value_set = operand_value_set;
    changed = true;
  }
  return changed;
}
bool HloDataflowAnalysis::UpdateRecvDoneValueSet(HloInstruction* recv_done) {
  CHECK_EQ(recv_done->opcode(), HloOpcode::kRecvDone);
  bool changed = false;
  for (auto& pair : GetInstructionValueSet(recv_done)) {
    ShapeIndex& index = pair.first;
    HloValueSet& value_set = pair.second;
    if (index.empty() || index[0] != 0) {
      continue;
    }
    const HloValueSet& operand_value_set =
        GetValueSet(recv_done->operand(0), index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}
bool HloDataflowAnalysis::UpdateCallValueSet(HloInstruction* call) {
  CHECK_EQ(call->opcode(), HloOpcode::kCall);
  InstructionValueSet& value_set = GetInstructionValueSet(call);
  InstructionValueSet& root_value_set =
      GetInstructionValueSet(call->to_apply()->root_instruction());
  if (value_set != root_value_set) {
    value_set = root_value_set;
    return true;
  }
  return false;
}
bool HloDataflowAnalysis::UpdateConditionalValueSet(
    HloInstruction* conditional) {
  CHECK_EQ(conditional->opcode(), HloOpcode::kConditional);
  std::vector<const InstructionValueSet*> inputs(conditional->branch_count());
  for (int j = 0; j < conditional->branch_count(); ++j) {
    inputs[j] = &GetInstructionValueSet(
        conditional->branch_computation(j)->root_instruction());
  }
  if (ssa_form_) {
    return Phi(conditional, inputs);
  } else {
    return GetInstructionValueSet(conditional).AssignUnionOf(inputs);
  }
}
bool HloDataflowAnalysis::UpdateCopyValueSet(HloInstruction* copy) {
  CHECK_EQ(copy->opcode(), HloOpcode::kCopy);
  bool changed = false;
  for (auto& pair : GetInstructionValueSet(copy)) {
    const ShapeIndex& index = pair.first;
    if (index.empty()) {
      continue;
    }
    HloValueSet& value_set = pair.second;
    HloValueSet& operand_value_set = GetValueSet(copy->operand(0), index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}
bool HloDataflowAnalysis::UpdateOptimizationBarrierValueSet(
    HloInstruction* barrier) {
  CHECK_EQ(barrier->opcode(), HloOpcode::kOptimizationBarrier);
  bool changed = false;
  for (auto& pair : GetInstructionValueSet(barrier)) {
    const ShapeIndex& index = pair.first;
    HloValueSet& value_set = pair.second;
    HloValueSet& operand_value_set = GetValueSet(barrier->operand(0), index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}
bool HloDataflowAnalysis::UpdateDomainValueSet(HloInstruction* domain) {
  CHECK_EQ(domain->opcode(), HloOpcode::kDomain);
  bool changed = false;
  for (auto& pair : GetInstructionValueSet(domain)) {
    const ShapeIndex& index = pair.first;
    HloValueSet& value_set = pair.second;
    HloValueSet& operand_value_set = GetValueSet(domain->operand(0), index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}
bool HloDataflowAnalysis::UpdateAddDependencyValueSet(
    HloInstruction* add_dependency) {
  CHECK_EQ(add_dependency->opcode(), HloOpcode::kAddDependency);
  const InstructionValueSet& operand_set =
      GetInstructionValueSet(add_dependency->operand(0));
  InstructionValueSet& add_dependency_set =
      GetInstructionValueSet(add_dependency);
  if (operand_set != add_dependency_set) {
    add_dependency_set = operand_set;
    return true;
  }
  return false;
}
bool HloDataflowAnalysis::UpdateGetTupleElementValueSet(HloInstruction* gte) {
  CHECK_EQ(gte->opcode(), HloOpcode::kGetTupleElement);
  bool changed = false;
  for (auto& pair : GetInstructionValueSet(gte)) {
    const ShapeIndex& index = pair.first;
    HloValueSet& value_set = pair.second;
    ShapeIndex operand_index = {gte->tuple_index()};
    for (int64_t i : index) {
      operand_index.push_back(i);
    }
    HloValueSet& operand_value_set =
        GetValueSet(gte->operand(0), operand_index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}
bool HloDataflowAnalysis::UpdateParameterValueSet(HloInstruction* parameter) {
  CHECK_EQ(parameter->opcode(), HloOpcode::kParameter);
  const CallGraphNode& call_graph_node =
      call_graph_->GetNode(parameter->parent());
  if (call_graph_node.context() == CallContext::kEmbedded ||
      call_graph_node.caller_callsites().empty()) {
    return false;
  }
  CHECK_EQ(call_graph_node.context(), CallContext::kControlFlow);
  std::vector<const InstructionValueSet*> inputs;
  bool need_phi = false;
  for (const CallSite& callsite : call_graph_node.caller_callsites()) {
    const HloOpcode& opcode = callsite.instruction()->opcode();
    if (opcode == HloOpcode::kCall) {
      inputs.push_back(&GetInstructionValueSet(
          callsite.instruction()->operand(parameter->parameter_number())));
    } else if (opcode == HloOpcode::kWhile) {
      CHECK_EQ(parameter->parameter_number(), 0);
      inputs.push_back(
          &GetInstructionValueSet(callsite.instruction()->operand(0)));
      if (parameter !=
          callsite.instruction()->while_body()->root_instruction()) {
        inputs.push_back(&GetInstructionValueSet(
            callsite.instruction()->while_body()->root_instruction()));
      }
      need_phi = true;
    } else if (opcode == HloOpcode::kConditional) {
      CHECK_EQ(parameter->parameter_number(), 0);
      auto conditional = callsite.instruction();
      bool found_parent = false;
      for (int j = 0; j < conditional->branch_count(); ++j) {
        if (parameter->parent() == conditional->branch_computation(j)) {
          inputs.push_back(
              &GetInstructionValueSet(conditional->operand(j + 1)));
          found_parent = true;
          break;
        }
      }
      CHECK(found_parent);
      need_phi = true;
    } else if (opcode == HloOpcode::kAsyncStart) {
      inputs.push_back(&GetInstructionValueSet(
          callsite.instruction()->operand(parameter->parameter_number())));
    } else if (opcode == HloOpcode::kAsyncUpdate ||
               opcode == HloOpcode::kAsyncDone) {
      return GetInstructionValueSet(parameter).AssignUnionOf(
          GetInstructionValueSet(callsite.instruction()->operand(0)),
          {0, parameter->parameter_number()});
    } else {
      LOG(FATAL) << "CallContext::kSequential computations should only be "
                    "called from call, while, or conditional instructions";
    }
  }
  if (ssa_form_ && need_phi) {
    return Phi(parameter, inputs);
  } else {
    return GetInstructionValueSet(parameter).AssignUnionOf(inputs);
  }
}
bool HloDataflowAnalysis::UpdateTupleValueSet(HloInstruction* tuple) {
  CHECK_EQ(tuple->opcode(), HloOpcode::kTuple);
  bool changed = false;
  for (int64_t i = 0; i < tuple->operands().size(); ++i) {
    for (auto& pair : GetInstructionValueSet(tuple->operand(i))) {
      const ShapeIndex& operand_index = pair.first;
      HloValueSet& operand_value_set = pair.second;
      ShapeIndex index = {i};
      for (int64_t op_index : operand_index) {
        index.push_back(op_index);
      }
      HloValueSet& value_set = GetValueSet(tuple, index);
      if (value_set != operand_value_set) {
        value_set = operand_value_set;
        changed = true;
      }
    }
  }
  return changed;
}
bool HloDataflowAnalysis::UpdateWhileValueSet(HloInstruction* xla_while) {
  CHECK_EQ(xla_while->opcode(), HloOpcode::kWhile);
  const InstructionValueSet* const inputs[] = {
      &GetInstructionValueSet(xla_while->while_body()->root_instruction()),
      &GetInstructionValueSet(xla_while->operand(0))};
  if (ssa_form_) {
    return Phi(xla_while, inputs);
  } else {
    return GetInstructionValueSet(xla_while).AssignUnionOf(inputs);
  }
}
bool HloDataflowAnalysis::UpdateAllGatherStartValueSet(
    HloInstruction* all_gather_start) {
  CHECK_EQ(all_gather_start->opcode(), HloOpcode::kAllGatherStart);
  bool changed = false;
  for (int64_t i = 0; i < all_gather_start->operand_count(); ++i) {
    const HloValueSet& operand_value_set =
        GetValueSet(all_gather_start->operand(i));
    ShapeIndex output_index = {0};
    if (all_gather_start->operand_count() > 1) {
      output_index.push_back(i);
    }
    HloValueSet& value_set = GetValueSet(all_gather_start, output_index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}
bool HloDataflowAnalysis::UpdateAllGatherDoneValueSet(
    HloInstruction* all_gather_done) {
  CHECK_EQ(all_gather_done->opcode(), HloOpcode::kAllGatherDone);
  bool changed = false;
  for (auto& pair : GetInstructionValueSet(all_gather_done)) {
    const ShapeIndex& output_index = pair.first;
    HloValueSet& value_set = pair.second;
    if (!ShapeUtil::GetSubshape(all_gather_done->shape(), output_index)
             .IsArray()) {
      continue;
    }
    ShapeIndex operand_index = {1};
    for (int64_t i : output_index) {
      operand_index.push_back(i);
    }
    const HloValueSet& operand_value_set =
        GetValueSet(all_gather_done->operand(0), operand_index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}
bool HloDataflowAnalysis::UpdateAllReduceDoneValueSet(
    HloInstruction* all_reduce_done) {
  CHECK_EQ(all_reduce_done->opcode(), HloOpcode::kAllReduceDone);
  bool changed = false;
  for (auto& pair : GetInstructionValueSet(all_reduce_done)) {
    const ShapeIndex& output_index = pair.first;
    HloValueSet& value_set = pair.second;
    ShapeIndex operand_index = {};
    for (int64_t i : output_index) {
      operand_index.push_back(i);
    }
    const HloValueSet& operand_value_set =
        GetValueSet(all_reduce_done->operand(0), operand_index);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}
bool HloDataflowAnalysis::UpdateCollectivePermuteStartValueSet(
    HloInstruction* collective_permute_start) {
  CHECK_EQ(collective_permute_start->opcode(),
           HloOpcode::kCollectivePermuteStart);
  bool changed = false;
  if (collective_permute_start->operand(0)->shape().IsTuple()) {
    for (int i = 0; i < ShapeUtil::TupleElementCount(
                            collective_permute_start->operand(0)->shape());
         ++i) {
      const HloValueSet& operand_value_set =
          GetValueSet(collective_permute_start->operand(0), {i});
      HloValueSet& value_set = GetValueSet(collective_permute_start, {0, i});
      if (value_set != operand_value_set) {
        value_set = operand_value_set;
        changed = true;
      }
    }
  } else {
    const HloValueSet& operand_value_set =
        GetValueSet(collective_permute_start->operand(0));
    HloValueSet& value_set = GetValueSet(collective_permute_start, {0});
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}
bool HloDataflowAnalysis::UpdateCollectivePermuteDoneValueSet(
    HloInstruction* collective_permute_done) {
  CHECK_EQ(collective_permute_done->opcode(),
           HloOpcode::kCollectivePermuteDone);
  bool changed = false;
  if (collective_permute_done->shape().IsTuple()) {
    for (int i = 0;
         i < ShapeUtil::TupleElementCount(collective_permute_done->shape());
         ++i) {
      const HloValueSet& operand_value_set =
          GetValueSet(collective_permute_done->operand(0), {1, i});
      HloValueSet& value_set = GetValueSet(collective_permute_done, {i});
      if (value_set != operand_value_set) {
        value_set = operand_value_set;
        changed = true;
      }
    }
  } else {
    const HloValueSet& operand_value_set =
        GetValueSet(collective_permute_done->operand(0), {1});
    HloValueSet& value_set = GetValueSet(collective_permute_done);
    if (value_set != operand_value_set) {
      value_set = operand_value_set;
      changed = true;
    }
  }
  return changed;
}
bool HloDataflowAnalysis::UpdateInstructionValueSet(
    HloInstruction* instruction) {
  bool changed = false;
  switch (instruction->opcode()) {
    case HloOpcode::kAddDependency: {
      changed = UpdateAddDependencyValueSet(instruction);
      break;
    }
    case HloOpcode::kAllGatherStart: {
      changed = UpdateAllGatherStartValueSet(instruction);
      break;
    }
    case HloOpcode::kAllGatherDone: {
      changed = UpdateAllGatherDoneValueSet(instruction);
      break;
    }
    case HloOpcode::kAsyncStart: {
      changed = UpdateAsyncStartValueSet(instruction);
      break;
    }
    case HloOpcode::kAsyncUpdate: {
      changed = UpdateAsyncUpdateValueSet(instruction);
      break;
    }
    case HloOpcode::kAsyncDone: {
      changed = UpdateAsyncDoneValueSet(instruction);
      break;
    }
    case HloOpcode::kBitcast: {
      changed = UpdateBitcastValueSet(instruction);
      break;
    }
    case HloOpcode::kDomain: {
      changed = UpdateDomainValueSet(instruction);
      break;
    }
    case HloOpcode::kCopy: {
      changed = UpdateCopyValueSet(instruction);
      break;
    }
    case HloOpcode::kGetTupleElement: {
      changed = UpdateGetTupleElementValueSet(instruction);
      break;
    }
    case HloOpcode::kTuple: {
      changed = UpdateTupleValueSet(instruction);
      break;
    }
    case HloOpcode::kParameter: {
      changed = UpdateParameterValueSet(instruction);
      break;
    }
    case HloOpcode::kCall: {
      changed = UpdateCallValueSet(instruction);
      break;
    }
    case HloOpcode::kWhile: {
      changed = UpdateWhileValueSet(instruction);
      break;
    }
    case HloOpcode::kSend: {
      changed = UpdateSendValueSet(instruction);
      break;
    }
    case HloOpcode::kRecvDone: {
      changed = UpdateRecvDoneValueSet(instruction);
      break;
    }
    case HloOpcode::kCopyStart: {
      changed = UpdateCopyStartValueSet(instruction);
      break;
    }
    case HloOpcode::kCopyDone: {
      changed = UpdateCopyDoneValueSet(instruction);
      break;
    }
    case HloOpcode::kConditional: {
      changed = UpdateConditionalValueSet(instruction);
      break;
    }
    case HloOpcode::kAllReduceDone: {
      changed = UpdateAllReduceDoneValueSet(instruction);
      break;
    }
    case HloOpcode::kCollectivePermuteStart: {
      changed = UpdateCollectivePermuteStartValueSet(instruction);
      break;
    }
    case HloOpcode::kCollectivePermuteDone: {
      changed = UpdateCollectivePermuteDoneValueSet(instruction);
      break;
    }
    case HloOpcode::kOptimizationBarrier: {
      changed = UpdateOptimizationBarrierValueSet(instruction);
      break;
    }
    default:
      break;
  }
  if (forwards_value_ != nullptr) {
    for (auto& [index, value_set] : GetInstructionValueSet(instruction)) {
      if (std::optional<ForwardedOperand> forwarded_operand =
              forwards_value_(instruction, index);
          forwarded_operand.has_value()) {
        HloValueSet& operand_value_set =
            GetValueSet(instruction->operand(forwarded_operand->operand_number),
                        forwarded_operand->operand_index);
        if (value_set != operand_value_set) {
          value_set = operand_value_set;
          changed = true;
        }
      }
    }
  }
  return changed;
}
void HloDataflowAnalysis::Propagate() {
  using Work = std::pair<int64_t, HloInstruction*>;
  std::priority_queue<Work, std::vector<Work>, std::greater<Work>> worklist;
  absl::flat_hash_set<HloInstruction*> workset;
  auto priority_map = CalculatePostOrderSchedule(module_);
  auto add_to_worklist = [&priority_map, &worklist,
                          &workset](HloInstruction* instruction) {
    if (workset.insert(instruction).second) {
      worklist.emplace(priority_map[instruction], instruction);
    }
  };
  auto comps = module_.MakeComputationPostOrder();
  for (HloComputation* computation : comps) {
    if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                          execution_threads_)) {
      continue;
    }
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      add_to_worklist(instruction);
    }
  }
  VLOG(1) << "SSA_FORM_: " << ssa_form_;
  while (!worklist.empty()) {
    HloInstruction* instruction = worklist.top().second;
    worklist.pop();
    workset.erase(workset.find(instruction));
    VLOG(3) << "Worklist top: " << instruction->name();
    XLA_VLOG_LINES(3, ToString());
    if (!UpdateInstructionValueSet(instruction)) {
      VLOG(4) << "No change.";
      continue;
    }
    VLOG(4) << "New value set for " << instruction->name() << ": "
            << GetInstructionValueSet(instruction);
    for (HloInstruction* user : instruction->users()) {
      add_to_worklist(user);
      if (user->opcode() == HloOpcode::kConditional) {
        for (int j = 0; j < user->branch_count(); ++j) {
          if (user->operand(j + 1) == instruction) {
            add_to_worklist(
                user->branch_computation(j)->parameter_instruction(0));
          }
        }
      } else if (user->opcode() == HloOpcode::kAsyncUpdate ||
                 user->opcode() == HloOpcode::kAsyncDone) {
        if (HloInstruction::IsThreadIncluded(user->async_execution_thread(),
                                             execution_threads_)) {
          for (int64_t parameter_number = 0;
               parameter_number <
               user->async_wrapped_computation()->num_parameters();
               ++parameter_number) {
            add_to_worklist(
                user->async_wrapped_computation()->parameter_instruction(
                    parameter_number));
          }
        }
      } else {
        for (HloComputation* called_computation : user->called_computations()) {
          if (!HloInstruction::IsThreadIncluded(
                  called_computation->execution_thread(), execution_threads_)) {
            continue;
          }
          const CallGraphNode& call_graph_node =
              call_graph_->GetNode(called_computation);
          if (call_graph_node.context() == CallContext::kControlFlow) {
            for (int64_t operand_number : user->OperandIndices(instruction)) {
              add_to_worklist(
                  called_computation->parameter_instruction(operand_number));
            }
          }
        }
      }
    }
    if (instruction == instruction->parent()->root_instruction()) {
      const CallGraphNode& call_graph_node =
          call_graph_->GetNode(instruction->parent());
      for (const CallSite& callsite : call_graph_node.caller_callsites()) {
        if (callsite.instruction()->opcode() == HloOpcode::kWhile) {
          add_to_worklist(callsite.instruction());
          add_to_worklist(
              callsite.instruction()->while_body()->parameter_instruction(0));
          add_to_worklist(
              callsite.instruction()->while_condition()->parameter_instruction(
                  0));
        } else if (call_graph_node.context() == CallContext::kControlFlow) {
          add_to_worklist(callsite.instruction());
        }
      }
    }
  }
}
const InstructionValueSet& HloDataflowAnalysis::GetInstructionValueSet(
    const HloInstruction* instruction) const {
  DCHECK(value_sets_.contains(instruction))
      << "Instruction " << instruction->ToString() << " not found.";
  return *value_sets_.find(instruction)->second;
}
InstructionValueSet& HloDataflowAnalysis::GetInstructionValueSet(
    const HloInstruction* instruction) {
  DCHECK(value_sets_.contains(instruction))
      << "Instruction " << instruction->ToString() << " not found.";
  return *value_sets_.find(instruction)->second;
}
absl::Status HloDataflowAnalysis::InitializeInstructionValueSets() {
  for (const HloComputation* computation : module_.MakeComputationSorted()) {
    if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                          execution_threads_)) {
      continue;
    }
    const CallGraphNode& call_graph_node = call_graph_->GetNode(computation);
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      value_sets_.insert({instruction, std::make_unique<InstructionValueSet>(
                                           instruction->shape())});
      auto define_all_values =
          [this, &instruction](
              absl::FunctionRef<bool(const ShapeIndex&)> should_define =
                  [](const ShapeIndex&) { return true; }) {
            for (auto& pair : GetInstructionValueSet(instruction)) {
              const ShapeIndex& index = pair.first;
              bool defines_value;
              if (forwards_value_ != nullptr &&
                  forwards_value_(instruction, index).has_value()) {
                defines_value = false;
              } else {
                defines_value = should_define(index);
              }
              if (defines_value) {
                HloValue* value =
                    NewHloValue(instruction, index, false);
                GetValueSet(instruction, index).AddValue(value);
              }
            }
          };
      auto define_value_at = [this, &instruction](const ShapeIndex& index) {
        HloValue* value = NewHloValue(instruction, index, false);
        GetValueSet(instruction, index).AddValue(value);
      };
      switch (instruction->opcode()) {
        case HloOpcode::kBitcast:
          if (bitcast_defines_value_) {
            define_all_values();
          }
          break;
        case HloOpcode::kAddDependency:
        case HloOpcode::kWhile:
        case HloOpcode::kCall:
        case HloOpcode::kConditional:
        case HloOpcode::kGetTupleElement:
        case HloOpcode::kDomain:
        case HloOpcode::kOptimizationBarrier:
          break;
        case HloOpcode::kParameter:
          if (call_graph_node.context() == CallContext::kBoth) {
            return Unimplemented(
                "Computation %s is called in both a parallel (eg, kMap) and "
                "sequential (eg, kCall) context",
                computation->name());
          }
          if (call_graph_node.caller_callsites().empty() ||
              call_graph_node.context() == CallContext::kEmbedded) {
            define_all_values();
          }
          break;
        case HloOpcode::kCopy:
        case HloOpcode::kTuple:
          define_value_at({});
          break;
        case HloOpcode::kAsyncStart: {
          bool thread_included = HloInstruction::IsThreadIncluded(
              instruction->async_execution_thread(), execution_threads_);
          define_all_values([&](const ShapeIndex& index) {
            return ShapeUtil::GetSubshape(instruction->shape(), index)
                       .IsTuple() ||
                   (!thread_included && index.front() == 1) ||
                   (index.front() > 1);
          });
          break;
        }
        case HloOpcode::kAsyncUpdate:
          define_all_values([&](const ShapeIndex& index) {
            return ShapeUtil::GetSubshape(instruction->shape(), index)
                .IsTuple();
          });
          break;
        case HloOpcode::kAsyncDone:
          define_all_values([&](const ShapeIndex& index) {
            return ShapeUtil::GetSubshape(instruction->shape(), index)
                .IsTuple();
          });
          break;
        case HloOpcode::kCopyStart:
          define_value_at({});
          define_value_at({0});
          define_value_at({2});
          break;
        case HloOpcode::kCopyDone:
          break;
        case HloOpcode::kAllGatherStart:
          define_all_values([&](const ShapeIndex& index) {
            return ShapeUtil::GetSubshape(instruction->shape(), index)
                       .IsTuple() ||
                   index.front() == 1;
          });
          break;
        case HloOpcode::kAllGatherDone:
          if (instruction->shape().IsTuple()) {
            define_value_at({});
          }
          break;
        case HloOpcode::kAllReduceDone:
          break;
        case HloOpcode::kCollectivePermuteStart:
          define_value_at({});
          define_value_at({1});
          for (int i = 2; i < instruction->shape().tuple_shapes_size(); ++i) {
            define_value_at({i});
          }
          if (instruction->operand_count() > 1) {
            CHECK_EQ(instruction->operand_count(), 4);
            if (instruction->operand(1)->shape().IsTuple()) {
              for (int i = 0; i < ShapeUtil::TupleElementCount(
                                      instruction->operand(1)->shape());
                   ++i) {
                define_value_at({1, i});
              }
            }
          }
          break;
        case HloOpcode::kCollectivePermuteDone:
          if (instruction->shape().IsTuple()) {
            define_value_at({});
          }
          break;
        case HloOpcode::kRecvDone:
          define_value_at({});
          define_value_at({1});
          break;
        case HloOpcode::kSend:
          define_value_at({});
          define_value_at({1});
          define_value_at({2});
          break;
        default:
          define_all_values();
          break;
      }
    }
  }
  return absl::OkStatus();
}
void HloDataflowAnalysis::OptimizePhiValues() {
  if (!ssa_form_) {
    return;
  }
  VLOG(1) << "Before phi graph optimization";
  XLA_VLOG_LINES(1, phi_graph_.ToString());
  phi_graph_.Optimize();
  VLOG(1) << "After phi graph optimization";
  XLA_VLOG_LINES(1, phi_graph_.ToString());
  for (const HloComputation* computation : module_.computations()) {
    if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                          execution_threads_)) {
      continue;
    }
    for (HloInstruction* instruction : computation->instructions()) {
      InstructionValueSet& instruction_value_set =
          GetInstructionValueSet(instruction);
      VLOG(1) << "inst: " << instruction->name();
      VLOG(1) << instruction_value_set.ToString();
      instruction_value_set.ForEachMutableElement(
          [&](const xla::ShapeIndex& index, HloValueSet* value_set) {
            auto values = value_set->values();
            if (!(values.size() == 1 && values[0]->is_phi())) {
              return;
            }
            HloValue::Id phi_id = values[0]->id();
            HloValue::Id new_id = phi_graph_.FindOptimizedValue(phi_id);
            if (new_id != phi_id) {
              VLOG(1) << "Replacing " << values[0]->ToShortString() << " with "
                      << GetValue(new_id).ToShortString();
              value_set->Clear();
              const HloValue& new_value = GetValue(new_id);
              value_set->AddValue(&new_value);
              MarkValueForDeletion(phi_id);
            }
          });
    }
  }
}
absl::StatusOr<std::unique_ptr<HloDataflowAnalysis>> HloDataflowAnalysis::Run(
    const HloModule& module, bool ssa_form, bool bitcast_defines_value,
    const CanShareBuffer& can_share_buffer, const ForwardsValue& forwards_value,
    absl::flat_hash_set<absl::string_view> execution_threads) {
  VLOG(1) << "HloDataflowAnalysis::Run on module " << module.name();
  XLA_VLOG_LINES(2, module.ToString());
  auto dataflow_analysis = absl::WrapUnique(new HloDataflowAnalysis(
      module, ssa_form, bitcast_defines_value, can_share_buffer, forwards_value,
      execution_threads));
  TF_RETURN_IF_ERROR(dataflow_analysis->InitializeInstructionValueSets());
  dataflow_analysis->Propagate();
  dataflow_analysis->OptimizePhiValues();
  dataflow_analysis->DeleteMarkedValues();
  std::vector<std::vector<HloPosition>> value_positions(
      dataflow_analysis->next_value_id_);
  for (const HloComputation* computation : module.computations()) {
    if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                          execution_threads)) {
      continue;
    }
    for (HloInstruction* instruction : computation->instructions()) {
      for (const auto& pair :
           dataflow_analysis->GetInstructionValueSet(instruction)) {
        const ShapeIndex& index = pair.first;
        const HloValueSet& value_set = pair.second;
        for (const HloValue* value : value_set.values()) {
          if (value->defining_instruction() != instruction) {
            value_positions[value->id()].push_back(
                HloPosition{instruction, index});
          }
        }
      }
    }
  }
  for (auto& pair : dataflow_analysis->values_) {
    HloValue::Id value_id = pair.first;
    HloValue& value = *pair.second;
    value.SetPositions(value_positions[value_id]);
  }
  dataflow_analysis->values_vector_.reserve(dataflow_analysis->values_.size());
  for (const auto& pair : dataflow_analysis->values_) {
    dataflow_analysis->values_vector_.push_back(pair.second.get());
  }
  absl::c_sort(dataflow_analysis->values_vector_, HloValue::IdLessThan);
  TF_DCHECK_OK(dataflow_analysis->Verify());
  XLA_VLOG_LINES(1, dataflow_analysis->ToString());
  return std::move(dataflow_analysis);
}
absl::Status HloDataflowAnalysis::Verify() const {
  for (const HloValue* value : values()) {
    for (const HloPosition& position : value->positions()) {
      const HloValueSet& value_set = GetValueSet(position);
      TF_RET_CHECK(absl::c_linear_search(value_set.values(), value))
          << "Value set at position " << position << " does not contain value "
          << value->ToShortString();
    }
  }
  for (const auto& computation : module_.computations()) {
    if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                          execution_threads_)) {
      continue;
    }
    for (const auto& instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kAsyncStart &&
          (instruction->async_wrapped_opcode() == HloOpcode::kCall ||
           instruction->async_wrapped_opcode() == HloOpcode::kCustomCall)) {
        continue;
      }
      for (const auto& pair : GetInstructionValueSet(instruction)) {
        const ShapeIndex& index = pair.first;
        const HloValueSet& value_set = pair.second;
        const HloPosition position{instruction, index};
        for (const HloValue* value : value_set.values()) {
          TF_RET_CHECK(absl::c_linear_search(value->positions(), position))
              << "Value set at position " << position
              << " unexpectedly contains value " << value->ToShortString();
        }
      }
    }
  }
  return absl::OkStatus();
}
bool HloDataflowAnalysis::DoesNotUseOperandBuffer(
    const HloInstruction* operand, const ShapeIndex& index,
    const HloInstruction* user) const {
  for (const HloValue* value : GetValueSet(operand, index).values()) {
    for (const HloUse& use : value->GetUses()) {
      if (use.instruction == user) {
        if (user->IsLoopFusion()) {
          HloInstruction* fusion_param =
              user->fused_parameter(use.operand_number);
          const HloValue& value =
              GetValueDefinedAt(fusion_param, use.operand_index);
          return value.GetUses().empty();
        }
        return false;
      }
    }
  }
  return true;
}
 bool HloDataflowAnalysis::IsInPlaceOperation(HloOpcode opcode) {
  return opcode == HloOpcode::kDynamicUpdateSlice ||
         opcode == HloOpcode::kScatter;
}
 bool HloDataflowAnalysis::IsAsynchronousOperationStart(
    HloOpcode opcode) {
  return opcode == HloOpcode::kSend || opcode == HloOpcode::kRecv ||
         opcode == HloOpcode::kCopyStart ||
         opcode == HloOpcode::kAllReduceStart ||
         opcode == HloOpcode::kAllGatherStart ||
         opcode == HloOpcode::kCollectivePermuteStart ||
         opcode == HloOpcode::kAsyncStart;
}
 bool HloDataflowAnalysis::IsAsynchronousOperationDone(
    HloOpcode opcode) {
  return opcode == HloOpcode::kSendDone || opcode == HloOpcode::kRecvDone ||
         opcode == HloOpcode::kCopyDone ||
         opcode == HloOpcode::kAllReduceDone ||
         opcode == HloOpcode::kAllGatherDone ||
         opcode == HloOpcode::kCollectivePermuteDone ||
         opcode == HloOpcode::kAsyncDone;
}
namespace {
std::vector<std::pair<HloOperandIndex, ShapeIndex>>
GetFusionInstructionInPlaceInputOutputPairs(const HloInstruction* instruction) {
  std::vector<std::pair<HloOperandIndex, ShapeIndex>>
      in_place_input_output_pairs;
  ShapeUtil::ForEachLeafShape(
      instruction->shape(),
      [&](const Shape& sub_shape, const ShapeIndex& index) {
        const HloInstruction* output_source_instruction =
            instruction->fused_expression_root();
        ShapeIndex output_source_index = index;
        std::tie(output_source_instruction, output_source_index) =
            FollowTupleIndirection(output_source_instruction,
                                   output_source_index);
        auto in_place_pairs = HloDataflowAnalysis::GetInPlaceInputOutputPairs(
            output_source_instruction);
        ShapeIndex in_place_input_index;
        const HloInstruction* in_place_input_source = nullptr;
        for (const auto& output_source_in_place_pair : in_place_pairs) {
          const HloOperandIndex& input = output_source_in_place_pair.first;
          const ShapeIndex& output_index = output_source_in_place_pair.second;
          if (output_index == output_source_index) {
            CHECK(in_place_input_source == nullptr);
            in_place_input_source =
                output_source_instruction->operand(input.operand_number);
            in_place_input_index = input.operand_index;
            std::tie(in_place_input_source, in_place_input_index) =
                FollowTupleIndirection(in_place_input_source,
                                       in_place_input_index);
            if (in_place_input_source->opcode() == HloOpcode::kFusion) {
              auto nested_in_place_input_output_pairs =
                  HloDataflowAnalysis::GetInPlaceInputOutputPairs(
                      in_place_input_source);
              for (const auto& pair : nested_in_place_input_output_pairs) {
                if (pair.second == in_place_input_index) {
                  in_place_input_source =
                      in_place_input_source->operand(pair.first.operand_number);
                  in_place_input_index = pair.first.operand_index;
                  std::tie(in_place_input_source, in_place_input_index) =
                      FollowTupleIndirection(in_place_input_source,
                                             in_place_input_index);
                }
              }
            }
          }
        }
        if (in_place_input_source != nullptr &&
            in_place_input_source->opcode() == HloOpcode::kParameter) {
          in_place_input_output_pairs.emplace_back(
              HloOperandIndex{in_place_input_source->parameter_number(),
                              in_place_input_index},
              index);
        }
      });
  return in_place_input_output_pairs;
}
}  
 std::vector<std::pair<HloOperandIndex, ShapeIndex>>
HloDataflowAnalysis::GetInPlaceInputOutputPairs(
    const HloInstruction* instruction) {
  if (IsInPlaceOperation(instruction->opcode())) {
    const HloScatterInstruction* scatter =
        DynCast<HloScatterInstruction>(instruction);
    if (scatter && scatter->scatter_operand_count() > 1) {
      std::vector<std::pair<HloOperandIndex, ShapeIndex>> pairs;
      pairs.reserve(scatter->scatter_operand_count());
      for (int i = 0, n = scatter->scatter_operand_count(); i < n; ++i) {
        pairs.emplace_back(HloOperandIndex{i, {}}, ShapeIndex{i});
      }
      return pairs;
    }
    return {{HloOperandIndex{0, {}}, {}}};
  } else if (instruction->opcode() == HloOpcode::kCollectivePermute &&
             instruction->operands().size() == 4) {
    if (instruction->operand(1)->shape().IsTuple()) {
      std::vector<std::pair<HloOperandIndex, ShapeIndex>> in_place_pairs(
          {{HloOperandIndex{1, {}}, {}}});
      for (int i = 0; i < instruction->operand(1)->shape().tuple_shapes_size();
           i++) {
        in_place_pairs.push_back({HloOperandIndex{1, {i}}, {i}});
      }
      return in_place_pairs;
    } else {
      return {{HloOperandIndex{1, {}}, {}}};
    }
  } else if (instruction->opcode() == HloOpcode::kCollectivePermuteStart &&
             instruction->operands().size() == 4) {
    if (instruction->operand(1)->shape().IsTuple()) {
      std::vector<std::pair<HloOperandIndex, ShapeIndex>> in_place_pairs(
          {{HloOperandIndex{1, {}}, {1}}});
      for (int i = 0; i < instruction->operand(1)->shape().tuple_shapes_size();
           i++) {
        in_place_pairs.push_back({HloOperandIndex{1, {i}}, {1, i}});
      }
      return in_place_pairs;
    } else {
      return {{HloOperandIndex{1, {}}, {1}}};
    }
  } else if (instruction->opcode() == HloOpcode::kCustomCall) {
    const auto& aliasing_pairs = Cast<HloCustomCallInstruction>(instruction)
                                     ->output_to_operand_aliasing();
    std::vector<std::pair<HloOperandIndex, ShapeIndex>> in_place_pairs;
    in_place_pairs.reserve(aliasing_pairs.size());
    for (const auto& pair : aliasing_pairs) {
      ShapeIndex output_shape_index = pair.first;
      int64_t operand_index = pair.second.first;
      ShapeIndex operand_shape_index = pair.second.second;
      in_place_pairs.push_back(
          {HloOperandIndex{operand_index, {operand_shape_index}},
           output_shape_index});
    }
    return in_place_pairs;
  } else if (instruction->opcode() == HloOpcode::kAllReduceStart) {
    if (instruction->operands().size() == 1) {
      return {{HloOperandIndex{0, {}}, {}}};
    }
    std::vector<std::pair<HloOperandIndex, ShapeIndex>> in_place_pairs;
    in_place_pairs.reserve(instruction->operands().size());
    for (int i = 0; i < instruction->operands().size(); i++) {
      in_place_pairs.push_back({HloOperandIndex{i, {}}, {i}});
    }
    return in_place_pairs;
  } else if (instruction->opcode() == HloOpcode::kFusion) {
    const auto& aliasing_pairs =
        Cast<HloFusionInstruction>(instruction)->output_to_operand_aliasing();
    auto in_place_pairs =
        GetFusionInstructionInPlaceInputOutputPairs(instruction);
    if (!aliasing_pairs.empty()) {
      for (const auto& pair : aliasing_pairs) {
        ShapeIndex output_shape_index = pair.first;
        int64_t operand_index = pair.second.first;
        ShapeIndex operand_shape_index = pair.second.second;
        in_place_pairs.push_back(
            {HloOperandIndex{operand_index, {operand_shape_index}},
             output_shape_index});
      }
    }
    return in_place_pairs;
  } else if (instruction->opcode() == HloOpcode::kSetDimensionSize) {
    int64_t dimension = instruction->dimension();
    std::vector<std::pair<HloOperandIndex, ShapeIndex>> in_place_pairs;
    if (instruction->shape().is_dynamic_dimension(dimension) ==
        instruction->shape().is_dynamic_dimension(dimension)) {
      in_place_pairs.push_back({HloOperandIndex{0, {}}, {}});
    }
    return in_place_pairs;
  }
  return {};
}
bool HloDataflowAnalysis::CanShareOperandBufferWithUser(
    HloInstruction* operand, const ShapeIndex& operand_index,
    HloInstruction* user, const ShapeIndex& user_index) const {
  CHECK(user->IsUserOf(operand))
      << "user: " << user->ToString() << " operand: " << operand->ToString();
  if (operand->opcode() == HloOpcode::kConstant) {
    return false;
  }
  const Shape& operand_subshape =
      ShapeUtil::GetSubshape(operand->shape(), operand_index);
  const Shape& user_subshape =
      ShapeUtil::GetSubshape(user->shape(), user_index);
  if (IsSliceInputFusion(*user)) {
    HloInstruction* fusion_param =
        user->fused_parameter(user->operand_index(operand));
    return operand_subshape.IsArray() && user_subshape.IsArray() &&
           ShapeUtil::ElementsIn(operand_subshape) ==
               ShapeUtil::ElementsIn(user_subshape) &&
           ShapeUtil::SameElementType(operand_subshape, user_subshape) &&
           AreTransitiveUsesEffectivelyElementwise(
               fusion_param, user->fused_expression_root(), user_index);
  }
  auto shapes_equal = ShapeUtil::Equal(operand_subshape, user_subshape);
  if (shapes_equal) {
    for (const auto& operand_and_output_index :
         GetInPlaceInputOutputPairs(user)) {
      if (operand_and_output_index.second != user_index) {
        continue;
      }
      for (const HloUse& use :
           GetUniqueValueAt(operand, operand_index).GetUses()) {
        if (use == HloUse{user, operand_and_output_index.first.operand_number,
                          operand_and_output_index.first.operand_index}) {
          return true;
        }
      }
    }
  }
  if (can_share_buffer_ != nullptr) {
    if (std::optional<bool> hint =
            can_share_buffer_(user, operand, user_index)) {
      return *hint;
    }
  }
  if (!shapes_equal) {
    return false;
  }
  if (user->opcode() == HloOpcode::kFusion) {
    HloInstruction* fusion_param =
        user->fused_parameter(user->operand_index(operand));
    const HloValue& fusion_param_value =
        GetValueDefinedAt(fusion_param, operand_index);
    if (user->IsLoopFusion() || user->IsInputFusion()) {
      return AreTransitiveUsesElementwiseOrTuple(fusion_param);
    }
    if (user->IsOutputFusion() &&
        user->fused_expression_root()->opcode() == HloOpcode::kAdd) {
      auto* add = user->fused_expression_root();
      auto add_operand_it =
          absl::c_find_if(add->operands(), [&](HloInstruction* operand) {
            return operand->opcode() == HloOpcode::kConvolution ||
                   operand->opcode() == HloOpcode::kDot;
          });
      if (add_operand_it == add->operands().end()) {
        return false;
      }
      auto* matched_add_operand = *add_operand_it;
      const int64_t other_add_operand_index =
          matched_add_operand == add->operand(0) ? 1 : 0;
      if (fusion_param_value.GetUses().size() == 1) {
        const HloUse& use = fusion_param_value.GetUses()[0];
        return use.instruction == user->fused_expression_root() &&
               use.operand_number == other_add_operand_index;
      }
      return false;
    }
  }
  if (user->opcode() == HloOpcode::kWhile ||
      user->opcode() == HloOpcode::kConditional) {
    return true;
  }
  if (user->opcode() == HloOpcode::kDynamicUpdateSlice ||
      user->opcode() == HloOpcode::kScatter ||
      user->opcode() == HloOpcode::kTriangularSolve ||
      user->opcode() == HloOpcode::kSetDimensionSize) {
    const auto operand_indices = user->OperandIndices(operand);
    int64_t operand_no = user->opcode() == HloOpcode::kTriangularSolve ? 1 : 0;
    return operand_indices.size() == 1 && operand_indices[0] == operand_no;
  }
  if (user->opcode() == HloOpcode::kSort) {
    if (operand->users().size() != 1) {
      return false;
    }
    if (user->operand_count() == 1) {
      return true;
    }
    CHECK(!user_index.empty());
    const auto operand_indices = user->OperandIndices(operand);
    return operand_indices.size() == 1 && user_index[0] == operand_indices[0];
  }
  if (user->opcode() == HloOpcode::kCall) {
    auto uses = GetValueDefinedAt(operand, operand_index).GetUses();
    const bool found_caller_use =
        absl::c_find_if(uses, [user](const HloUse& use) {
          return use.instruction == user;
        }) != uses.end();
    auto* callee_root = user->to_apply()->root_instruction();
    const bool found_elementwise_callee_use =
        absl::c_find_if(uses, [callee_root](const HloUse& use) {
          return use.instruction == callee_root &&
                 callee_root->IsElementwiseOnOperand(use.operand_number);
        }) != uses.end();
    return uses.size() == 2 && found_caller_use && found_elementwise_callee_use;
  }
  return user->IsElementwiseOnOperand(user->operand_index(operand));
}
std::pair<const HloInstruction*, ShapeIndex> FollowTupleIndirection(
    const HloInstruction* instruction, ShapeIndex operand_index) {
  while (instruction->opcode() == HloOpcode::kTuple && !operand_index.empty()) {
    instruction = instruction->operand(operand_index.front());
    operand_index.pop_front();
  }
  while (instruction->opcode() == HloOpcode::kGetTupleElement) {
    operand_index.push_front(instruction->tuple_index());
    instruction = instruction->operand(0);
  }
  return {instruction, operand_index};
}
}  