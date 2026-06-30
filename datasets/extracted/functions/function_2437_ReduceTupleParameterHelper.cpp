#include "xla/tools/hlo_slicer.h"
#include <deque>
#include <memory>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/call_graph.h"
#include "xla/service/hlo_verifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tools/hlo_extractor.h"
#include "tsl/platform/status.h"
namespace xla {
namespace {
void ReduceTupleParameterHelper(HloModule* hlo_module,
                                HloInstruction* tuple_parameter) {
  for (HloInstruction* user_inst : tuple_parameter->users()) {
    if (user_inst->opcode() != HloOpcode::kGetTupleElement) {
      return;
    }
  }
  VLOG(1) << "Parameter instruction to be reduced: "
          << tuple_parameter->ToString()
          << " shape size: " << tuple_parameter->shape().tuple_shapes_size()
          << " users size: " << tuple_parameter->users().size();
  std::vector<Shape> used_shapes;
  for (HloInstruction* user_inst : tuple_parameter->users()) {
    used_shapes.push_back(user_inst->shape());
  }
  Shape new_tuple_shape =
      ShapeUtil::MakeTupleShape(absl::MakeSpan(used_shapes));
  tuple_parameter->mutable_shape()->mutable_tuple_shapes()->clear();
  for (const auto& shape : used_shapes) {
    tuple_parameter->mutable_shape()->mutable_tuple_shapes()->push_back(shape);
  }
  for (int i = 0; i < tuple_parameter->users().size(); ++i) {
    tuple_parameter->users()[i]->set_tuple_index(i);
  }
  hlo_module->mutable_config().SetComputationLayoutIfExists(
      hlo_module->entry_computation()->ComputeProgramShape());
}
void ReduceTupleParameter(HloModule* hlo_module) {
  std::vector<HloInstruction*> tuple_parameters;
  for (HloInstruction* parameter :
       hlo_module->entry_computation()->parameter_instructions()) {
    if (parameter->shape().IsTuple()) {
      tuple_parameters.push_back(parameter);
    }
  }
  for (HloInstruction* tuple_parameter : tuple_parameters) {
    ReduceTupleParameterHelper(hlo_module, tuple_parameter);
  }
}
HloInstruction* FindShardingInstruction(HloModule* hlo_module) {
  for (HloComputation* computation : hlo_module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kCustomCall &&
          instruction->custom_call_target() == "Sharding") {
        CHECK_EQ(instruction->operand_count(), 1);
        return instruction;
      }
    }
  }
  return nullptr;
}
void RemoveSharding(HloModule* hlo_module) {
  while (HloInstruction* custom_call_instruction =
             FindShardingInstruction(hlo_module)) {
    for (HloInstruction* user_instruction : custom_call_instruction->users()) {
      CHECK_OK(custom_call_instruction->ReplaceUseWith(
          user_instruction, custom_call_instruction->mutable_operand(0)));
    }
    custom_call_instruction->DetachFromOperandsAndUsers();
    CHECK_OK(custom_call_instruction->parent()->RemoveInstruction(
        custom_call_instruction));
    VLOG(1) << "Removed sharding custom-call: "
            << custom_call_instruction->ToString();
    HloVerifier verifier(false,
                         true);
    TF_CHECK_OK(verifier.Run(hlo_module).status());
  }
}
void IntraComputationSlicing(
    const HloComputation* computation,
    absl::flat_hash_set<const HloInstruction*>& sliced_instructions,
    absl::flat_hash_set<const HloInstruction*>& frontier_instructions,
    bool forward_slice, FrontierSelector frontier_selector,
    bool ignore_control_dependency) {
  std::deque<const HloInstruction*> worklist(sliced_instructions.begin(),
                                             sliced_instructions.end());
  while (!worklist.empty()) {
    const HloInstruction* inst = worklist.back();
    worklist.pop_back();
    if (frontier_selector && !frontier_selector(inst)) {
      frontier_instructions.insert(inst);
      continue;
    }
    std::vector<HloInstruction*> instructions_to_propagate =
        forward_slice ? std::vector<HloInstruction*>(inst->users().begin(),
                                                     inst->users().end())
                      : std::vector<HloInstruction*>(inst->operands().begin(),
                                                     inst->operands().end());
    if (!ignore_control_dependency) {
      if (forward_slice) {
        instructions_to_propagate.insert(instructions_to_propagate.end(),
                                         inst->control_successors().begin(),
                                         inst->control_successors().end());
      } else {
        instructions_to_propagate.insert(instructions_to_propagate.end(),
                                         inst->control_predecessors().begin(),
                                         inst->control_predecessors().end());
      }
    }
    for (auto next_inst : instructions_to_propagate) {
      if (!sliced_instructions.contains(next_inst)) {
        worklist.push_front(next_inst);
        sliced_instructions.insert(next_inst);
      }
    }
  }
}
SliceOutput SliceModuleHelper(
    const HloModule* hlo_module,
    absl::Span<const HloInstruction*> slice_starting_instructions,
    FrontierSelector frontier_selector, bool ignore_control_dependency,
    bool forward_slice, bool nearest_common_ancestor_as_root) {
  absl::flat_hash_map<const HloComputation*,
                      absl::flat_hash_set<const HloInstruction*>>
      sliced_computation_instructions_map;
  for (auto inst : slice_starting_instructions) {
    sliced_computation_instructions_map[inst->parent()].insert(inst);
  }
  absl::flat_hash_map<const HloComputation*,
                      absl::flat_hash_set<const HloInstruction*>>
      frontier_computation_instructions_map;
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(hlo_module);
  std::vector<HloComputation*> post_order_computations =
      hlo_module->MakeComputationPostOrder();
  std::vector<HloComputation*> computations_to_traverse =
      forward_slice
          ? post_order_computations
          : std::vector<HloComputation*>(post_order_computations.rbegin(),
                                         post_order_computations.rend());
  absl::flat_hash_set<const HloComputation*>
      nearest_common_ancestor_computations;
  if (nearest_common_ancestor_as_root) {
    std::vector<const HloComputation*> starting_computations;
    for (const auto& [computation, instructions] :
         sliced_computation_instructions_map) {
      starting_computations.push_back(computation);
    }
    nearest_common_ancestor_computations =
        call_graph->NearestCommonAncestorComputations(starting_computations);
    CHECK(!nearest_common_ancestor_computations.empty());
  }
  for (auto computation : computations_to_traverse) {
    if (sliced_computation_instructions_map.contains(computation)) {
      auto slicing_starting_instructions = std::vector<const HloInstruction*>(
          sliced_computation_instructions_map[computation].begin(),
          sliced_computation_instructions_map[computation].end());
      IntraComputationSlicing(
          computation, sliced_computation_instructions_map[computation],
          frontier_computation_instructions_map[computation], forward_slice,
          frontier_selector, ignore_control_dependency);
      if (forward_slice) {
        if (nearest_common_ancestor_as_root &&
            nearest_common_ancestor_computations.contains(computation)) {
          const HloInstruction* nearest_common_ancestor_instruction =
              *(call_graph->NearestCommonAncestorInstructions(
                    slicing_starting_instructions))
                   .begin();
          CHECK_NE(nearest_common_ancestor_instruction, nullptr);
          return SliceOutput{sliced_computation_instructions_map,
                             frontier_computation_instructions_map,
                             nearest_common_ancestor_instruction};
        }
        if (!sliced_computation_instructions_map[computation].contains(
                computation->root_instruction()) ||
            frontier_computation_instructions_map[computation].contains(
                computation->root_instruction())) {
          continue;
        }
        for (auto caller_inst :
             call_graph->GetComputationCallers(computation)) {
          sliced_computation_instructions_map[caller_inst->parent()].insert(
              caller_inst);
        }
      }
      if (!forward_slice) {
        for (const auto& callsite :
             call_graph->GetNode(computation).callsites()) {
          if (sliced_computation_instructions_map[computation].contains(
                  callsite.instruction())) {
            for (auto callee : callsite.called_computations()) {
              sliced_computation_instructions_map[callee].insert(
                  callee->root_instruction());
            }
          }
        }
      }
    }
  }
  return SliceOutput{sliced_computation_instructions_map,
                     frontier_computation_instructions_map};
}
}  
SliceOutput SliceModule(
    const HloModule* hlo_module,
    absl::Span<const HloInstruction*> slice_starting_instructions,
    FrontierSelector frontier_selector, bool ignore_control_dependency,
    bool forward_slice, bool nearest_common_ancestor_as_root) {
  if (forward_slice) {
    if (!nearest_common_ancestor_as_root) {
      return SliceModuleHelper(hlo_module, slice_starting_instructions,
                               frontier_selector, ignore_control_dependency,
                               true,
                               false);
    } else {
      CHECK(forward_slice) << "Option `nearest_common_ancestor_as_root` can "
                              "only be enabled when "
                              "forward slicing";
      CHECK((frontier_selector == nullptr))
          << "Option `nearest_common_ancestor_as_root` can not be specified "
             "with `frontier_selector`";
      SliceOutput forward_slice_output =
          SliceModuleHelper(hlo_module, slice_starting_instructions,
                            nullptr,
                            ignore_control_dependency, true,
                            true);
      std::vector<const HloInstruction*> nearest_common_ancestor(
          {forward_slice_output.nearest_common_ancestor_root()});
      CHECK_EQ(nearest_common_ancestor.size(), 1);
      SliceOutput backward_slice_output =
          SliceModuleHelper(hlo_module, 
                            absl::MakeSpan(nearest_common_ancestor),
                            nullptr,
                            ignore_control_dependency, false,
                            false);
      return SliceOutput{SliceOutput::IntersectSlicedInstructions(
                             forward_slice_output, backward_slice_output),
                         backward_slice_output.frontier_instructions(),
                         forward_slice_output.nearest_common_ancestor_root()};
    }
  } else {
    return SliceModuleHelper(hlo_module, slice_starting_instructions,
                             frontier_selector, ignore_control_dependency,
                             false,
                             false);
  }
}
std::vector<std::unique_ptr<HloModule>> SliceModuleAndExtract(
    const HloModule* hlo_module,
    absl::Span<const HloInstruction*> slice_starting_instructions,
    const SlicingConfiguration& slicing_configuration) {
  std::vector<std::unique_ptr<HloModule>> sliced_modules;
  int slicing_group = slicing_configuration.slicing_group;
  CHECK(slicing_group >= 1 || slicing_group == -1);
  std::vector<absl::Span<const HloInstruction*>> grouped_instructions;
  if (slicing_group == -1) {
    grouped_instructions = {slice_starting_instructions};
  } else {
    for (int i = 0; i < slice_starting_instructions.size();
         i += slicing_group) {
      grouped_instructions.push_back(
          slice_starting_instructions.subspan(i, slicing_group));
    }
  }
  for (const auto& grouped_slice_starting_instructions : grouped_instructions) {
    SliceOutput forward_slice_output;
    if (slicing_configuration.forward_slicing ==
        SlicingConfiguration::ForwardSlicingConfig::kRoot) {
      forward_slice_output = SliceModule(
          hlo_module, grouped_slice_starting_instructions,
          nullptr,
          false, true,
          false);
    } else if (slicing_configuration.forward_slicing ==
               SlicingConfiguration::ForwardSlicingConfig::kNca) {
      forward_slice_output = SliceModule(
          hlo_module, grouped_slice_starting_instructions,
          nullptr,
          false, true,
          true);
    }
    VLOG(1) << "[Num of forward sliced insts]: "
            << forward_slice_output.NumSlicedInstructions();
    SliceOutput backward_slice_output;
    if (slicing_configuration.backward_slicing) {
      backward_slice_output = SliceModule(
          hlo_module, grouped_slice_starting_instructions,
          nullptr,
          false, false);
    } else {
      backward_slice_output = SliceOutput();
    }
    auto sliced_result = SliceOutput(SliceOutput::UnionSlicedInstructions(
        forward_slice_output, backward_slice_output));
    const HloInstruction* extraction_root =
        slicing_configuration.forward_slicing ==
                SlicingConfiguration::ForwardSlicingConfig::kNca
            ? forward_slice_output.nearest_common_ancestor_root()
            : hlo_module->entry_computation()->root_instruction();
    VLOG(1) << "[Root instruction of the sliced module]: "
            << extraction_root->ToString();
    auto extract_selector = [&sliced_result](const HloInstruction* hlo_inst) {
      for (const auto& [computation, instructions] :
           sliced_result.sliced_instructions()) {
        if (instructions.contains(hlo_inst)) {
          return true;
        }
      }
      return false;
    };
    auto replace_type_selector =
        [](const HloInstruction* hlo_inst) -> ReplaceType {
      return ReplaceType::kReplaceZeroBroadcast;
    };
    auto extracted_module =
        ExtractModule(extraction_root, -1,
                      extract_selector,
                      replace_type_selector,
                      true);
    if (slicing_configuration.remove_sharding) {
      RemoveSharding(extracted_module.get());
    }
    if (slicing_configuration.reduce_tuple_parameter) {
      ReduceTupleParameter(extracted_module.get());
    }
    HloVerifier verifier(false,
                         true);
    TF_CHECK_OK(verifier.Run(extracted_module.get()).status());
    sliced_modules.emplace_back(std::move(extracted_module));
  }
  CHECK_EQ(sliced_modules.size(), grouped_instructions.size());
  return sliced_modules;
}
}  