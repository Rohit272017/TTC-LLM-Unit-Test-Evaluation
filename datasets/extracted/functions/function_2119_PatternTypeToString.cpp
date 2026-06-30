#include "xla/service/hlo_unstacker.h"
#include <algorithm>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/map_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/tuple_util.h"
#include "xla/service/while_loop_unroller.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace {
enum class PatternType {
  DSFusionNoBitcastPattern,
  DSFusionPattern,
  NestedDSFusionPattern,
  Other,
};
static std::string PatternTypeToString(PatternType pattern_type) {
  switch (pattern_type) {
    case PatternType::DSFusionNoBitcastPattern:
      return "DSFusionNoBitcastPattern";
    case PatternType::DSFusionPattern:
      return "DSFusionPattern";
    case PatternType::NestedDSFusionPattern:
      return "NestedDSFusionPattern";
    case PatternType::Other:
      return "Other";
  }
}
struct PatternInfo {
  PatternType type;
  std::vector<const HloInstruction*> unstacked_instrs;
  const HloInstruction* instr;
  Shape unstacked_shape;
  HloComputation* unstacking_computation;
  std::string ToString() const {
    if (unstacking_computation == nullptr) {
      return absl::StrCat("type: \n\t", PatternTypeToString(type), "\n",
                          "instr: \n\t", instr->name(), "\n", "shape: \n\t",
                          unstacked_shape.ToString(true));
    } else {
      return absl::StrCat("type: \n\t", PatternTypeToString(type), "\n",
                          "instr: \n\t", instr->name(), "\n", "shape: \n\t",
                          unstacked_shape.ToString(true), "\n", "comp: \n",
                          unstacking_computation->name());
    }
  }
};
struct UnstackerMetadata {
  static absl::StatusOr<UnstackerMetadata> Create(
      HloModule* module, std::function<bool(HloInstruction*)> unfuse_slice) {
    UnstackerMetadata metadata;
    TF_ASSIGN_OR_RETURN(
        bool prepared,
        WhileLoopUnroller::PrepareModuleForUnrolling(module, {}));
    if (prepared) {
      VLOG(3) << "Prepared module: " << module->name() << " for unstacking.";
    }
    std::vector<std::pair<HloInstruction*, WhileLoopConfig>> loops =
        WhileLoopUnroller::GetUnrollableLoops(module, {},
                                              std::nullopt);
    for (const auto& [instr, while_loop_config] : loops) {
      metadata.unrollable_loop_bodies[instr->while_body()] = while_loop_config;
      metadata.bodies[instr->while_body()] = instr;
    }
    metadata.unfuse_slice = unfuse_slice;
    return metadata;
  }
  absl::flat_hash_map<HloComputation*, WhileLoopConfig> unrollable_loop_bodies;
  absl::flat_hash_map<const HloComputation*, HloInstruction*> bodies;
  std::vector<
      std::pair<std::function<std::optional<PatternInfo>(
                    const UnstackerMetadata&, const HloInstruction*, int64_t)>,
                std::function<absl::Status(HloInstruction*, const Shape&)>>>
      custom_handlers;
  std::function<bool(HloInstruction*)> unfuse_slice;
};
class UnstackerTransformer {
 public:
  explicit UnstackerTransformer(const UnstackerMetadata& metadata)
      : metadata_(metadata) {}
  std::vector<const HloInstruction*> HandleInstruction(
      const HloInstruction* instr, int64_t changed_idx) {
    if (instr->opcode() != HloOpcode::kFusion) {
      return {};
    }
    VLOG(3) << "HandleInstruction(" << instr->shape().ToString()
            << instr->name() << ", " << changed_idx << ")";
    for (const auto& [custom_pattern, custom_handler] :
         metadata_.custom_handlers) {
      std::optional<PatternInfo> stacked_user =
          custom_pattern(metadata_, instr, changed_idx);
      if (!stacked_user.has_value()) {
        continue;
      }
      PatternInfo& pattern_info = stacked_user.value();
      pattern_type_ = pattern_info.type;
      VLOG(3) << "PatternInfo:" << "\n" << pattern_info.ToString();
      if (pattern_info.unstacking_computation != nullptr &&
          unstacking_computation_ != nullptr) {
        if (!absl::EqualsIgnoreCase(
                pattern_info.unstacking_computation->ToString(
                    HloPrintOptions::Fingerprint()),
                unstacking_computation_->ToString(
                    HloPrintOptions::Fingerprint()))) {
          VLOG(3) << "Seen multiple unstacking computations, cannot handle: "
                  << "\n previous computations: \n"
                  << unstacking_computation_->ToString(
                         HloPrintOptions::Fingerprint())
                  << "\n current computations: \n"
                  << pattern_info.unstacking_computation->ToString(
                         HloPrintOptions::Fingerprint());
          return {};
        }
      }
      if (pattern_info.unstacking_computation != nullptr) {
        unstacking_computation_ = pattern_info.unstacking_computation;
      }
      unstacked_shape_ = std::make_unique<Shape>(pattern_info.unstacked_shape);
      unstacked_instrs_.push_back(instr);
      std::function<absl::Status()> unstack_wrapper =
          [&custom_handler = custom_handler,
           pattern_info]() mutable -> absl::Status {
        HloInstruction* mutable_dynamic_slicing_fusion =
            const_cast<HloInstruction*>(pattern_info.instr);
        return custom_handler(mutable_dynamic_slicing_fusion,
                              pattern_info.unstacked_shape.tuple_shapes(0));
      };
      body_changes_.push_back(unstack_wrapper);
      return pattern_info.unstacked_instrs;
    }
    return {};
  }
  const UnstackerMetadata& GetMetadata() const { return metadata_; }
  std::vector<const HloInstruction*>& GetUnstackedInstructions() {
    return unstacked_instrs_;
  }
  const Shape* GetUnstackedShape() const { return unstacked_shape_.get(); }
  HloComputation* GetUnstackingComputation() const {
    return unstacking_computation_;
  }
  std::vector<std::function<void(const UnstackerTransformer&)>>&
  GetLoopChanges() {
    return loop_changes_;
  }
  std::vector<std::function<absl::Status()>>& GetBodyChanges() {
    return body_changes_;
  }
  absl::flat_hash_map<HloInstruction*, std::vector<int64_t>>&
  GetOperandChanges() {
    return operand_changes_;
  }
  void AddOperandChange(HloInstruction* instr, int64_t index) {
    operand_changes_[instr].push_back(index);
  }
  void AddLoopChange(
      std::function<void(const UnstackerTransformer&)> loop_change) {
    loop_changes_.push_back(loop_change);
  }
  PatternType GetPatternType() const { return pattern_type_; }
 private:
  PatternType pattern_type_;
  const UnstackerMetadata& metadata_;
  std::unique_ptr<Shape> unstacked_shape_ = nullptr;
  HloComputation* unstacking_computation_ = nullptr;
  std::vector<std::function<void(const UnstackerTransformer&)>> loop_changes_;
  std::vector<std::function<absl::Status()>> body_changes_;
  absl::flat_hash_map<HloInstruction*, std::vector<int64_t>> operand_changes_;
  std::vector<const HloInstruction*> unstacked_instrs_;
};
bool CanUnstackWhileOperand(const HloInstruction* while_instr,
                            UnstackerTransformer& unstacker, int64_t index);
bool UnstackWhileOperandAtIndex(
    const UnstackerMetadata& metadata, HloInstruction* while_instr,
    int64_t index, std::vector<const HloInstruction*>& unstacked_instructions);
bool PropagateGteShapeChange(HloInstruction* gte,
                             UnstackerTransformer& unstacker) {
  VLOG(5) << "PropagateGteShapeChange(" << gte->name() << ")";
  std::vector<const HloInstruction*> handled_instrs;
  absl::flat_hash_map<HloInstruction*, int64_t> visited;
  std::deque<HloInstruction*> worklist;
  worklist.push_back(gte);
  visited.insert({gte, gte->tuple_index()});
  while (!worklist.empty()) {
    HloInstruction* changed_instr_to_propagate = worklist.front();
    int64_t changed_operand_index =
        FindOrDie(visited, changed_instr_to_propagate);
    worklist.pop_front();
    for (HloInstruction* user : changed_instr_to_propagate->users()) {
      if (ContainsKey(visited, user)) {
        continue;
      }
      if (user->opcode() == HloOpcode::kGetTupleElement) {
        if (user->tuple_index() != changed_operand_index) {
          continue;
        }
        visited.insert({user, changed_operand_index});
        worklist.push_back(user);
      } else if (user->opcode() == HloOpcode::kTuple) {
        int64_t use_index = user->operand_index(changed_instr_to_propagate);
        visited.insert({user, {use_index}});
        worklist.push_back(user);
      } else if (user->opcode() == HloOpcode::kWhile) {
        bool changed_nested_while =
            CanUnstackWhileOperand(user, unstacker, changed_operand_index);
        if (!changed_nested_while) {
          return false;
        }
        visited.insert({user, changed_operand_index});
        worklist.push_back(user);
      } else {
        if (absl::c_find(handled_instrs, user) != handled_instrs.end()) {
          continue;
        }
        if (user->IsCustomCall("DynamicGte") ||
            user->IsCustomCall("DynamicTuple")) {
          continue;
        }
        int64_t use_index = user->operand_index(changed_instr_to_propagate);
        std::vector<const HloInstruction*> curr_handled_instrs =
            unstacker.HandleInstruction(user, use_index);
        if (curr_handled_instrs.empty()) {
          VLOG(3) << "Custom unstacker not found for " << user->name();
          return false;
        }
        for (const HloInstruction* instr : curr_handled_instrs) {
          for (HloInstruction* handled_instr_user : instr->users()) {
            if (user->shape() == gte->shape()) {
              visited.insert({handled_instr_user, changed_operand_index});
              worklist.push_back(handled_instr_user);
            }
          }
          handled_instrs.push_back(instr);
        }
      }
    }
  }
  for (const auto& [instr, index] : visited) {
    unstacker.AddOperandChange(instr, index);
  }
  return true;
}
bool CanPropagateGteShapeChangesInComputation(
    const HloComputation* comp, const HloInstruction* operand,
    UnstackerTransformer& shape_transformer, int64_t idx) {
  VLOG(3) << "Propagating shape change of index " << idx
          << " in : " << comp->name();
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    if (instr->opcode() == HloOpcode::kGetTupleElement &&
        instr->tuple_index() == idx) {
      if (instr->operand(0) != operand) {
        continue;
      }
      bool can_propagate = PropagateGteShapeChange(instr, shape_transformer);
      if (!can_propagate) {
        VLOG(3) << "Failed to propagate shape change for " << instr->name();
        return false;
      }
    }
  }
  VLOG(3) << "Finish propagating shape change of index " << idx
          << " in: " << comp->name();
  return true;
}
void UnstackWhileInput(const UnstackerTransformer& unstacker,
                       HloInstruction* while_instr, int64_t index) {
  VLOG(3) << "Unstacking while input: " << while_instr->name() << " at "
          << index;
  const Shape* new_shape = unstacker.GetUnstackedShape();
  HloComputation* unstacking_computation = unstacker.GetUnstackingComputation();
  const Shape& slice_shape = new_shape->tuple_shapes(0);
  HloInstruction* old_while_input =
      while_instr->while_init()->mutable_operand(index);
  if (old_while_input->shape().IsTuple()) {
    VLOG(3) << "Input is already unstacked: " << old_while_input->name();
    return;
  }
  std::vector<HloInstruction*> slices;
  if (old_while_input->IsCustomCall("AllocateBuffer")) {
    for (int64_t i = 0; i < new_shape->tuple_shapes_size(); ++i) {
      slices.push_back(while_instr->AddInstruction(
          HloInstruction::CreateCustomCall(slice_shape, {}, "AllocateBuffer")));
    }
  } else {
    for (int64_t i = 0; i < new_shape->tuple_shapes_size(); ++i) {
      HloInstruction* root_instr = unstacking_computation->root_instruction();
      HloInstruction* slice = nullptr;
      if (unstacker.GetPatternType() == PatternType::DSFusionPattern ||
          unstacker.GetPatternType() == PatternType::NestedDSFusionPattern ||
          unstacker.GetPatternType() == PatternType::DSFusionNoBitcastPattern) {
        HloInstruction* dynamic_slice = nullptr;
        if (unstacker.GetPatternType() == PatternType::DSFusionPattern ||
            unstacker.GetPatternType() == PatternType::NestedDSFusionPattern) {
          dynamic_slice = root_instr->mutable_operand(0);
        } else if (unstacker.GetPatternType() ==
                   PatternType::DSFusionNoBitcastPattern) {
          dynamic_slice = root_instr;
        }
        std::vector<int64_t> new_start_indices;
        new_start_indices.reserve(dynamic_slice->shape().rank());
        std::vector<int64_t> new_limit_indices;
        new_limit_indices.reserve(dynamic_slice->shape().rank());
        std::vector<int64_t> new_strides;
        new_strides.reserve(dynamic_slice->shape().rank());
        new_start_indices.push_back(i);
        new_limit_indices.push_back(i + 1);
        new_strides.push_back(1);
        for (int64_t j = 1; j < dynamic_slice->shape().rank(); ++j) {
          new_start_indices.push_back(0);
          new_limit_indices.push_back(
              dynamic_slice->mutable_operand(0)->shape().dimensions(j));
          new_strides.push_back(1);
        }
        slice = while_instr->AddInstruction(HloInstruction::CreateSlice(
            dynamic_slice->shape(), old_while_input, new_start_indices,
            new_limit_indices, new_strides));
      }
      if (slice == nullptr || !unstacker.GetMetadata().unfuse_slice(slice)) {
        std::vector<HloInstruction*> operands = {
            old_while_input,
            while_instr->AddInstruction(MakeScalarConstantWithShape(
                unstacking_computation->parameter_instruction(1)->shape(), i))};
        slice = while_instr->AddInstruction(HloInstruction::CreateFusion(
            slice_shape, HloInstruction::FusionKind::kLoop, operands,
            while_instr->GetModule()->AddEmbeddedComputation(
                unstacking_computation->Clone()),
            "hoisted"));
      }
      slices.push_back(slice);
    }
  }
  HloInstruction* new_operand_element =
      while_instr->AddInstruction(HloInstruction::CreateTuple(slices));
  HloInstruction* new_while_init =
      TupleUtil::ReplaceTupleWith(new_operand_element,
                                  while_instr->while_init(), {index}, false)
          .value();
  CHECK_OK(while_instr->ReplaceOperandWithDifferentShape(0, new_while_init));
}
bool CanUnstackWhileOperand(const HloInstruction* while_instr,
                            UnstackerTransformer& unstacker, int64_t index) {
  VLOG(5) << "ReplaceWhileOperandShape: " << while_instr->name() << " at "
          << index;
  bool body_changes_collected = CanPropagateGteShapeChangesInComputation(
      while_instr->while_body(),
      while_instr->while_body()->parameter_instruction(0), unstacker, index);
  if (!body_changes_collected) {
    return false;
  }
  bool condition_changes_collected = CanPropagateGteShapeChangesInComputation(
      while_instr->while_condition(),
      while_instr->while_condition()->parameter_instruction(0), unstacker,
      index);
  if (!condition_changes_collected) {
    return false;
  }
  bool parent_changes_collected = CanPropagateGteShapeChangesInComputation(
      while_instr->parent(), while_instr, unstacker, index);
  if (!parent_changes_collected) {
    VLOG(3) << "Failed: parent_changes_collected";
    return false;
  }
  HloInstruction* root_operand =
      while_instr->while_body()->root_instruction()->mutable_operand(index);
  if (root_operand == nullptr) {
    return false;
  }
  HloInstruction* gte_operand = nullptr;
  if (Match(root_operand, match::GetTupleElement(match::Op(&gte_operand)))) {
    if (Match(gte_operand, match::While())) {
      VLOG(3) << "Faced a gte originating from loop: "
              << root_operand->ToString();
      bool loop_feeding_root_changes_collected = CanUnstackWhileOperand(
          root_operand->operand(0), unstacker, root_operand->tuple_index());
      if (!loop_feeding_root_changes_collected) {
        VLOG(3) << "Failed: loop " << root_operand->operand(0)->name()
                << " output at " << index << " is not unstackable";
        return false;
      }
    } else if (!Match(gte_operand, match::Parameter().WithParameterNum(0))) {
      VLOG(3) << "Failed: root operand of while_body at " << index
              << " is not a parameter";
      return false;
    }
  }
  auto loop_change = [=](const UnstackerTransformer& unstacker,
                         HloInstruction* loop, int64_t idx) mutable {
    Shape old_shape = ShapeUtil::MakeStaticShape(
        loop->while_body()->parameter_instruction(0)->shape());
    ShapeUtil::UpdateTupleShape(*unstacker.GetUnstackedShape(), idx,
                                &old_shape);
    loop->while_body()->ReplaceParameter(
        0, HloInstruction::CreateParameter(0, old_shape, "unstacked"));
    loop->while_condition()->ReplaceParameter(
        0, HloInstruction::CreateParameter(0, old_shape, "unstacked"));
    CHECK_NE(unstacker.GetUnstackingComputation(), nullptr);
    UnstackWhileInput(unstacker, loop, idx);
    *loop->mutable_shape() = old_shape;
  };
  auto loop_change_wrapper = [&loop_change, while_instr,
                              index](const UnstackerTransformer& unstacker) {
    HloInstruction* mutable_loop = const_cast<HloInstruction*>(while_instr);
    loop_change(unstacker, mutable_loop, index);
  };
  unstacker.AddLoopChange(loop_change_wrapper);
  return true;
}
bool UnstackWhileOperandAtIndex(
    const UnstackerMetadata& metadata, HloInstruction* while_instr,
    int64_t index, std::vector<const HloInstruction*>& unstacked_instructions) {
  UnstackerTransformer unstacker = UnstackerTransformer(metadata);
  bool can_unstack = CanUnstackWhileOperand(while_instr, unstacker, index);
  if (!can_unstack) {
    VLOG(3) << "Unstacking failed for " << while_instr->name() << " at "
            << index;
    return false;
  }
  if (unstacker.GetUnstackedShape() == nullptr) {
    VLOG(3) << "Failed: unstacked shape is null";
    return false;
  }
  if (unstacker.GetUnstackingComputation() == nullptr) {
    VLOG(3) << "Failed: unstacking computation is null";
    return false;
  }
  for (auto& [instr, indices] : unstacker.GetOperandChanges()) {
    switch (instr->opcode()) {
      case HloOpcode::kGetTupleElement:
        VLOG(3) << "Changing shape of: " << instr->name();
        *instr->mutable_shape() = *unstacker.GetUnstackedShape();
        break;
      case HloOpcode::kTuple: {
        for (int64_t index : indices) {
          VLOG(3) << "Changing shape of: " << instr->name() << " at " << index;
          *instr->mutable_shape()->mutable_tuple_shapes(index) =
              *unstacker.GetUnstackedShape();
        }
        break;
      }
      case HloOpcode::kWhile:
        for (int64_t index : indices) {
          VLOG(3) << "Changing shape of: " << instr->name() << " at " << index;
          ShapeUtil::UpdateTupleShape(*unstacker.GetUnstackedShape(), index,
                                      instr->mutable_shape());
        }
        break;
      default:
        LOG(FATAL) << "Unsupported opcode: " << instr->name();
    }
  }
  for (const auto& body_change : unstacker.GetBodyChanges()) {
    CHECK_OK(body_change());
  }
  for (auto& loop_change : unstacker.GetLoopChanges()) {
    loop_change(unstacker);
  }
  for (const HloInstruction* instr : unstacker.GetUnstackedInstructions()) {
    unstacked_instructions.push_back(instr);
  }
  return true;
}
Shape MakeUnstackedShapeFromSlice(const Shape& slice_shape, int64_t layers) {
  std::vector<Shape> shapes;
  shapes.reserve(layers);
  for (int64_t i = 0; i < layers; ++i) {
    shapes.push_back(slice_shape);
  }
  return ShapeUtil::MakeTupleShape(shapes);
}
std::optional<WhileLoopConfig> IsFusionInsideUnrollableLoopWithNumParameter(
    const UnstackerMetadata& metadata, const HloInstruction* instr,
    int64_t num_fusion_params) {
  if (instr->opcode() != HloOpcode::kFusion) {
    return std::nullopt;
  }
  if (instr->fused_parameters().size() != num_fusion_params) {
    VLOG(3) << "Fusion has different number of parameters";
    return std::nullopt;
  }
  if (!metadata.unrollable_loop_bodies.contains(instr->parent())) {
    VLOG(5) << "Fusion not inside unrollable while body, " << instr->name()
            << " inside " << instr->parent()->name();
    return std::nullopt;
  }
  return metadata.unrollable_loop_bodies.at(instr->parent());
}
HloInstruction* GetMostMajorEffectivelyStaticDynamicSliceInFusion(
    const UnstackerMetadata& metadata, const HloInstruction* instr,
    int64_t num_fusion_params, int64_t stacked_operand_idx) {
  std::optional<WhileLoopConfig> while_instr_config =
      IsFusionInsideUnrollableLoopWithNumParameter(metadata, instr,
                                                   num_fusion_params);
  if (!while_instr_config.has_value()) {
    return nullptr;
  }
  for (HloInstruction* fused_instr :
       instr->fused_instructions_computation()->MakeInstructionPostOrder()) {
    std::optional<int64_t> dynamic_index =
        MatchEffectivelyStaticDynamicSliceInsideLoop(
            fused_instr,
            instr->fused_instructions_computation()->parameter_instruction(
                stacked_operand_idx),
            while_instr_config.value());
    if (dynamic_index.has_value() && dynamic_index.value() == 0) {
      return fused_instr;
    }
  }
  return nullptr;
}
HloInstruction* GetMostMajorShapeCoveringDynamicIndexInFusion(
    const UnstackerMetadata& metadata, const HloInstruction* instr,
    HloOpcode opcode, int64_t num_fusion_params, int64_t stacked_operand_idx) {
  std::optional<WhileLoopConfig> while_instr_config =
      IsFusionInsideUnrollableLoopWithNumParameter(metadata, instr,
                                                   num_fusion_params);
  if (!while_instr_config.has_value()) {
    return nullptr;
  }
  for (HloInstruction* fused_instr :
       instr->fused_instructions_computation()->MakeInstructionPostOrder()) {
    if (fused_instr->opcode() != opcode) {
      continue;
    }
    std::optional<int64_t> dynamic_index =
        MatchShapeCoveringDynamicIndexInstruction(
            fused_instr,
            instr->fused_instructions_computation()->parameter_instruction(
                stacked_operand_idx),
            opcode, while_instr_config.value());
    if (dynamic_index.has_value() && dynamic_index.value() == 0) {
      return fused_instr;
    }
  }
  return nullptr;
}
std::optional<PatternInfo> GetDSFusionPattern(const UnstackerMetadata& metadata,
                                              const HloInstruction* instr,
                                              int64_t stacked_operand_idx) {
  VLOG(3) << "Checking DSFusion";
  HloInstruction* shape_covering_instr =
      GetMostMajorEffectivelyStaticDynamicSliceInFusion(metadata, instr, 2,
                                                        stacked_operand_idx);
  if (shape_covering_instr == nullptr) {
    return std::nullopt;
  }
  HloInstruction* bitcast_operand = nullptr;
  if (Match(instr->fused_instructions_computation()->root_instruction(),
            match::Bitcast(match::Op(&bitcast_operand)))) {
    if (bitcast_operand == shape_covering_instr) {
      PatternInfo pattern_info;
      pattern_info.type = PatternType::DSFusionPattern;
      pattern_info.instr = instr;
      const Shape& slice_shape = shape_covering_instr->shape();
      const int64_t num_layers = instr->operand(0)->shape().dimensions(0);
      pattern_info.unstacked_shape =
          MakeUnstackedShapeFromSlice(slice_shape, num_layers);
      pattern_info.unstacking_computation =
          instr->fused_instructions_computation();
      pattern_info.unstacked_instrs.push_back(instr);
      return pattern_info;
    }
  }
  return std::nullopt;
}
absl::Status UnstackDSFusionPattern(
    HloInstruction* mutable_dynamic_slicing_fusion, const Shape& slice_shape) {
  HloComputation* parent_loop = mutable_dynamic_slicing_fusion->parent();
  HloInstruction* stacked = mutable_dynamic_slicing_fusion->mutable_operand(0);
  HloInstruction* offset = mutable_dynamic_slicing_fusion->mutable_operand(1);
  HloInstruction* new_operand =
      parent_loop->AddInstruction(HloInstruction::CreateCustomCall(
          slice_shape, {stacked, offset}, "DynamicGte"));
  HloInstruction* bitcast = mutable_dynamic_slicing_fusion->AddInstruction(
      HloInstruction::CreateBitcast(mutable_dynamic_slicing_fusion->shape(),
                                    new_operand));
  return mutable_dynamic_slicing_fusion->ReplaceAllUsesWithDifferentShape(
      bitcast);
}
std::optional<PatternInfo> GetDSFusionNoBitcastPattern(
    const UnstackerMetadata& metadata, const HloInstruction* instr,
    int64_t stacked_operand_idx) {
  VLOG(3) << "Checking DSFusionNoBitcast";
  HloInstruction* shape_covering_instr =
      GetMostMajorEffectivelyStaticDynamicSliceInFusion(metadata, instr, 2,
                                                        stacked_operand_idx);
  if (shape_covering_instr == nullptr) {
    return std::nullopt;
  }
  if (instr->fused_instructions_computation()->root_instruction() !=
      shape_covering_instr) {
    return std::nullopt;
  }
  PatternInfo pattern_info;
  pattern_info.type = PatternType::DSFusionNoBitcastPattern;
  pattern_info.instr = instr;
  const Shape& slice_shape = shape_covering_instr->shape();
  const int64_t num_layers = instr->operand(0)->shape().dimensions(0);
  pattern_info.unstacked_shape =
      MakeUnstackedShapeFromSlice(slice_shape, num_layers);
  pattern_info.unstacking_computation = instr->fused_instructions_computation();
  pattern_info.unstacked_instrs.push_back(instr);
  return pattern_info;
}
absl::Status UnstackDSFusionNoBitcastPattern(
    HloInstruction* mutable_dynamic_slicing_fusion, const Shape& slice_shape) {
  HloComputation* parent_loop = mutable_dynamic_slicing_fusion->parent();
  HloInstruction* stacked = mutable_dynamic_slicing_fusion->mutable_operand(0);
  HloInstruction* offset = mutable_dynamic_slicing_fusion->mutable_operand(1);
  HloInstruction* new_operand =
      parent_loop->AddInstruction(HloInstruction::CreateCustomCall(
          slice_shape, {stacked, offset}, "DynamicGte"));
  return mutable_dynamic_slicing_fusion->ReplaceAllUsesWithDifferentShape(
      new_operand);
}
std::optional<PatternInfo> GetDUSFusionPattern(
    const UnstackerMetadata& metadata, const HloInstruction* instr,
    int64_t stacked_operand_idx) {
  VLOG(3) << "Checking DUSFusion";
  HloInstruction* shape_covering_instr =
      GetMostMajorShapeCoveringDynamicIndexInFusion(
          metadata, instr, HloOpcode::kDynamicUpdateSlice, 3,
          stacked_operand_idx);
  if (shape_covering_instr == nullptr) {
    return std::nullopt;
  }
  if (Match(shape_covering_instr->operand(1),
            match::Bitcast(match::Parameter()))) {
    if (shape_covering_instr->parent()->root_instruction() ==
        shape_covering_instr) {
      PatternInfo pattern_info;
      pattern_info.type = PatternType::Other;
      pattern_info.instr = instr;
      pattern_info.unstacked_shape = MakeUnstackedShapeFromSlice(
          instr->operand(2)->shape(), instr->operand(0)->shape().dimensions(0));
      pattern_info.unstacking_computation = nullptr;
      pattern_info.unstacked_instrs.push_back(instr);
      return pattern_info;
    }
  }
  return std::nullopt;
}
absl::Status UnstackDUSFusionPattern(
    HloInstruction* mutable_dynamic_update_slicing_fusion,
    const Shape& slice_shape) {
  HloComputation* parent_loop = mutable_dynamic_update_slicing_fusion->parent();
  HloInstruction* stacked =
      mutable_dynamic_update_slicing_fusion->mutable_operand(0);
  HloInstruction* offset =
      mutable_dynamic_update_slicing_fusion->mutable_operand(1);
  HloInstruction* update =
      mutable_dynamic_update_slicing_fusion->mutable_operand(2);
  HloInstruction* new_operand =
      parent_loop->AddInstruction(HloInstruction::CreateCustomCall(
          stacked->shape(), {stacked, update, offset}, "DynamicTuple"));
  for (HloInstruction* user : mutable_dynamic_update_slicing_fusion->users()) {
    TF_RETURN_IF_ERROR(
        mutable_dynamic_update_slicing_fusion->ReplaceUseWithDifferentShape(
            user, new_operand));
  }
  return absl::OkStatus();
}
std::optional<PatternInfo> GetDUSFusionWithPadPattern(
    const UnstackerMetadata& metadata, const HloInstruction* instr,
    int64_t stacked_operand_idx) {
  VLOG(3) << "Checking DUSFusionWithPad";
  HloInstruction* shape_covering_instr =
      GetMostMajorShapeCoveringDynamicIndexInFusion(
          metadata, instr, HloOpcode::kDynamicUpdateSlice, 3,
          stacked_operand_idx);
  if (shape_covering_instr == nullptr) {
    return std::nullopt;
  }
  if (Match(
          shape_covering_instr->operand(1),
          match::Bitcast(match::Pad(match::Parameter(), match::Constant())))) {
    if (shape_covering_instr->parent()->root_instruction() ==
        shape_covering_instr) {
      const HloInstruction* pad_instr =
          shape_covering_instr->operand(1)->operand(0);
      PatternInfo pattern_info;
      pattern_info.type = PatternType::Other;
      pattern_info.instr = instr;
      pattern_info.unstacked_shape = MakeUnstackedShapeFromSlice(
          pad_instr->shape(),
          shape_covering_instr->operand(0)->shape().dimensions(0));
      pattern_info.unstacking_computation = nullptr;
      pattern_info.unstacked_instrs.push_back(instr);
      return pattern_info;
    }
  }
  return std::nullopt;
}
absl::Status UnstackDUSFusionWithPadPattern(
    HloInstruction* mutable_dynamic_update_slicing_fusion,
    const Shape& slice_shape) {
  HloComputation* parent_loop = mutable_dynamic_update_slicing_fusion->parent();
  HloComputation* fused_computation =
      mutable_dynamic_update_slicing_fusion->fused_instructions_computation();
  HloInstruction* stacked =
      mutable_dynamic_update_slicing_fusion->mutable_operand(
          fused_computation->root_instruction()
              ->mutable_operand(0)
              ->parameter_number());
  HloInstruction* offset =
      mutable_dynamic_update_slicing_fusion->mutable_operand(
          fused_computation->root_instruction()
              ->mutable_operand(2)
              ->parameter_number());
  HloInstruction* pad_instr = fused_computation->root_instruction()
                                  ->mutable_operand(1)
                                  ->mutable_operand(0);
  fused_computation->set_root_instruction(pad_instr, true);
  *mutable_dynamic_update_slicing_fusion->mutable_shape() = pad_instr->shape();
  HloInstruction* new_operand =
      parent_loop->AddInstruction(HloInstruction::CreateCustomCall(
          stacked->shape(),
          {stacked, mutable_dynamic_update_slicing_fusion, offset},
          "DynamicTuple"));
  for (HloInstruction* user : mutable_dynamic_update_slicing_fusion->users()) {
    if (user != new_operand) {
      TF_RETURN_IF_ERROR(
          mutable_dynamic_update_slicing_fusion->ReplaceUseWithDifferentShape(
              user, new_operand));
    }
  }
  return absl::OkStatus();
}
std::optional<PatternInfo> GetDSFusionWithAddPattern(
    const UnstackerMetadata& metadata, const HloInstruction* instr,
    int64_t stacked_operand_idx) {
  VLOG(3) << "Checking DSFusionWithAdd";
  HloInstruction* shape_covering_instr =
      GetMostMajorShapeCoveringDynamicIndexInFusion(
          metadata, instr, HloOpcode::kDynamicSlice, 2, stacked_operand_idx);
  if (shape_covering_instr == nullptr) {
    return std::nullopt;
  }
  HloComputation* fused_computation = instr->fused_instructions_computation();
  HloInstruction* fusion_root = fused_computation->root_instruction();
  HloInstruction* add_operand;
  if (Match(fusion_root,
            match::Reduce(match::Add(match::Op(&add_operand),
                                     match::Broadcast(match::Constant())),
                          match::Constant()))) {
    if (add_operand == shape_covering_instr) {
      const int64_t num_layers = instr->operand(0)->shape().dimensions(0);
      PatternInfo pattern_info;
      pattern_info.type = PatternType::Other;
      pattern_info.instr = instr;
      pattern_info.unstacked_shape =
          MakeUnstackedShapeFromSlice(instr->shape(), num_layers);
      HloComputation::Builder builder("unstack_add");
      HloInstruction* p0 =
          builder.AddInstruction(HloInstruction::CreateParameter(
              0, fused_computation->parameter_instruction(0)->shape(), "p0"));
      HloInstruction* p1 =
          builder.AddInstruction(HloInstruction::CreateParameter(
              1, fused_computation->parameter_instruction(1)->shape(), "p1"));
      HloInstruction* zero =
          builder.AddInstruction(MakeScalarConstantWithShape(p1->shape(), 0));
      std::vector<HloInstruction*> slice_starts;
      slice_starts.reserve(shape_covering_instr->shape().rank());
      slice_starts.push_back(p1);
      for (int64_t i = 0; i < shape_covering_instr->shape().rank() - 1; i++) {
        slice_starts.push_back(zero);
      }
      HloInstruction* slice =
          builder.AddInstruction(HloInstruction::CreateDynamicSlice(
              shape_covering_instr->shape(), p0, slice_starts,
              shape_covering_instr->dynamic_slice_sizes()));
      HloInstruction* zero_reduce =
          builder.AddInstruction(MakeScalarConstantWithShape(
              ShapeUtil::MakeScalarShape(slice->shape().element_type()), 0));
      HloInstruction* reduce =
          builder.AddInstruction(HloInstruction::CreateReduce(
              instr->shape(), slice, zero_reduce, fusion_root->dimensions(),
              fused_computation->root_instruction()->to_apply()));
      HloComputation* unstack_add =
          instr->GetModule()->AddEmbeddedComputation(builder.Build());
      unstack_add->set_root_instruction(reduce);
      pattern_info.unstacking_computation = unstack_add;
      pattern_info.unstacked_instrs.push_back(instr);
      return pattern_info;
    }
  }
  return std::nullopt;
}
absl::Status UnstackDSFusionWithAddPattern(
    HloInstruction* mutable_dynamic_slice_with_add_fusion,
    const Shape& slice_shape) {
  HloComputation* parent_loop = mutable_dynamic_slice_with_add_fusion->parent();
  HloInstruction* stacked =
      mutable_dynamic_slice_with_add_fusion->mutable_operand(0);
  HloInstruction* offset =
      mutable_dynamic_slice_with_add_fusion->mutable_operand(1);
  HloInstruction* new_operand =
      parent_loop->AddInstruction(HloInstruction::CreateCustomCall(
          slice_shape, {stacked, offset}, "DynamicGte"));
  HloInstruction* one = parent_loop->AddInstruction(MakeScalarConstantWithShape(
      ShapeUtil::MakeScalarShape(slice_shape.element_type()), 1));
  HloInstruction* broadcast = parent_loop->AddInstruction(
      HloInstruction::CreateBroadcast(slice_shape, one, {}));
  HloInstruction* add = mutable_dynamic_slice_with_add_fusion->AddInstruction(
      HloInstruction::CreateBinary(new_operand->shape(), HloOpcode::kAdd,
                                   new_operand, broadcast));
  TF_RETURN_IF_ERROR(
      mutable_dynamic_slice_with_add_fusion->ReplaceAllUsesWith(add));
  return absl::OkStatus();
}
std::optional<PatternInfo> GetNestedDSFusionPattern(
    const UnstackerMetadata& metadata, const HloInstruction* instr,
    int64_t stacked_operand_idx) {
  if (instr->opcode() != HloOpcode::kFusion) {
    return std::nullopt;
  }
  if (!metadata.unrollable_loop_bodies.contains(instr->parent())) {
    VLOG(5) << "Instruction not inside unrollable while body, " << instr->name()
            << " inside " << instr->parent()->name();
    return std::nullopt;
  }
  WhileLoopConfig while_instr_config =
      metadata.unrollable_loop_bodies.at(instr->parent());
  VLOG(3) << "Checking NestedDSFusionPattern";
  HloInstruction* inner_fusion_user = nullptr;
  for (HloInstruction* fused_instr :
       instr->fused_instructions_computation()->MakeInstructionPostOrder()) {
    if (Match(fused_instr, match::Parameter(stacked_operand_idx))) {
      if (fused_instr->user_count() != 1) {
        return std::nullopt;
      }
      if (Match(fused_instr->users()[0],
                match::Fusion(match::Op(), match::Op()))) {
        inner_fusion_user = fused_instr->users()[0];
        break;
      }
    }
  }
  if (inner_fusion_user == nullptr) {
    return std::nullopt;
  }
  for (HloInstruction* inner_fusion_instr :
       inner_fusion_user->fused_instructions_computation()
           ->MakeInstructionPostOrder()) {
    if (!Match(inner_fusion_instr, match::DynamicSlice())) {
      continue;
    }
    std::optional<int64_t> dynamic_index =
        MatchEffectivelyStaticDynamicSliceInsideLoop(
            inner_fusion_instr,
            inner_fusion_user->fused_instructions_computation()
                ->parameter_instruction(0),
            while_instr_config);
    if (dynamic_index.has_value() && dynamic_index.value() == 0) {
      const int64_t num_layers =
          inner_fusion_user->operand(0)->shape().dimensions(0);
      PatternInfo pattern_info;
      pattern_info.type = PatternType::NestedDSFusionPattern;
      pattern_info.instr = inner_fusion_user;
      pattern_info.unstacked_shape =
          MakeUnstackedShapeFromSlice(inner_fusion_instr->shape(), num_layers);
      pattern_info.unstacking_computation =
          inner_fusion_user->fused_instructions_computation();
      pattern_info.unstacked_instrs.push_back(inner_fusion_user);
      return pattern_info;
    }
  }
  return std::nullopt;
}
absl::Status UnstackNestedDSFusionPattern(
    HloInstruction* mutable_dynamic_slicing_fusion, const Shape& slice_shape) {
  HloInstruction* parent_fusion =
      mutable_dynamic_slicing_fusion->parent()->FusionInstruction();
  HloInstruction* stacked_in_ds_fusion =
      mutable_dynamic_slicing_fusion->mutable_operand(0);
  CHECK_EQ(stacked_in_ds_fusion->opcode(), HloOpcode::kParameter);
  int64_t stacked_param_number = stacked_in_ds_fusion->parameter_number();
  HloInstruction* stacked =
      parent_fusion->mutable_operand(stacked_param_number);
  HloInstruction* offset_in_ds_fusion =
      mutable_dynamic_slicing_fusion->mutable_operand(1);
  CHECK_EQ(offset_in_ds_fusion->opcode(), HloOpcode::kParameter);
  HloInstruction* offset =
      parent_fusion->mutable_operand(offset_in_ds_fusion->parameter_number());
  HloInstruction* sliced_param =
      parent_fusion->fused_instructions_computation()->ReplaceParameter(
          stacked_param_number,
          HloInstruction::CreateParameter(stacked_param_number, slice_shape,
                                          "sliced"));
  HloInstruction* bitcast = mutable_dynamic_slicing_fusion->AddInstruction(
      HloInstruction::CreateBitcast(mutable_dynamic_slicing_fusion->shape(),
                                    sliced_param));
  HloInstruction* bitcast_fusion =
      mutable_dynamic_slicing_fusion->AddInstruction(
          HloInstruction::CreateFusion(mutable_dynamic_slicing_fusion->shape(),
                                       HloInstruction::FusionKind::kLoop,
                                       bitcast));
  TF_RETURN_IF_ERROR(
      mutable_dynamic_slicing_fusion->ReplaceAllUsesWith(bitcast_fusion));
  HloInstruction* new_operand =
      parent_fusion->AddInstruction(HloInstruction::CreateCustomCall(
          slice_shape, {stacked, offset}, "DynamicGte"));
  return parent_fusion->ReplaceOperandWithDifferentShape(
      sliced_param->parameter_number(), new_operand);
}
std::optional<PatternInfo> GetDSAndDUSPattern(const UnstackerMetadata& metadata,
                                              const HloInstruction* instr,
                                              int64_t stacked_operand_idx) {
  VLOG(3) << "Checking DSAndDUSPattern";
  if (instr->opcode() != HloOpcode::kFusion) {
    return std::nullopt;
  }
  const HloInstruction* stacked = instr->operand(stacked_operand_idx);
  if (stacked->user_count() != 2) {
    return std::nullopt;
  }
  HloInstruction* shape_covering_ds_instr =
      GetMostMajorShapeCoveringDynamicIndexInFusion(
          metadata, instr, HloOpcode::kDynamicSlice, 2, stacked_operand_idx);
  if (shape_covering_ds_instr == nullptr) {
    return std::nullopt;
  }
  HloInstruction* bitcast_operand = nullptr;
  if (!Match(instr->fused_instructions_computation()->root_instruction(),
             match::Bitcast(match::Op(&bitcast_operand)))) {
    return std::nullopt;
  }
  if (bitcast_operand != shape_covering_ds_instr) {
    return std::nullopt;
  }
  if (!GetDUSFusionPattern(metadata, stacked->users()[1],
                           stacked->users()[1]->operand_index(stacked))) {
    return std::nullopt;
  }
  PatternInfo pattern_info;
  pattern_info.type = PatternType::Other;
  pattern_info.instr = instr;
  const Shape& slice_shape = instr->shape();
  const int64_t num_layers = instr->operand(0)->shape().dimensions(0);
  pattern_info.unstacked_shape =
      MakeUnstackedShapeFromSlice(slice_shape, num_layers);
  pattern_info.unstacking_computation = instr->fused_instructions_computation();
  pattern_info.unstacked_instrs.push_back(instr);
  pattern_info.unstacked_instrs.push_back(stacked->users()[1]);
  return pattern_info;
}
absl::Status UnstackDSAndDUSPattern(HloInstruction* mutable_dynamic_slice,
                                    const Shape& slice_shape) {
  HloInstruction* stacked_gte = mutable_dynamic_slice->mutable_operand(0);
  int64_t stacked_gte_index = stacked_gte->tuple_index();
  HloComputation* parent = stacked_gte->parent();
  ShapeUtil::UpdateTupleShape(stacked_gte->shape(), stacked_gte_index,
                              parent->root_instruction()->mutable_shape());
  HloComputation* parent_loop = mutable_dynamic_slice->parent();
  HloInstruction* stacked = mutable_dynamic_slice->mutable_operand(0);
  HloInstruction* offset = mutable_dynamic_slice->mutable_operand(1);
  HloInstruction* new_operand =
      parent_loop->AddInstruction(HloInstruction::CreateCustomCall(
          slice_shape, {stacked, offset}, "DynamicGte"));
  TF_RETURN_IF_ERROR(
      mutable_dynamic_slice->ReplaceAllUsesWithDifferentShape(new_operand));
  HloInstruction* mutable_dynamic_update_slice = stacked_gte->users()[1];
  TF_RETURN_IF_ERROR(
      UnstackDUSFusionPattern(mutable_dynamic_update_slice, slice_shape));
  return absl::OkStatus();
}
std::optional<PatternInfo> GetReduceFusionPattern(
    const UnstackerMetadata& metadata, const HloInstruction* instr,
    int64_t stacked_operand_idx) {
  VLOG(3) << "Checking ReduceFusion";
  HloInstruction* shape_covering_instr =
      GetMostMajorShapeCoveringDynamicIndexInFusion(
          metadata, instr, HloOpcode::kDynamicSlice, 2, stacked_operand_idx);
  if (shape_covering_instr == nullptr) {
    return std::nullopt;
  }
  HloInstruction* reduce_operand = nullptr;
  HloInstruction* fusion_root =
      instr->fused_instructions_computation()->root_instruction();
  if (Match(fusion_root, match::Reduce(match::Op(&reduce_operand),
                                       match::ConstantScalar())) &&
      Match(fusion_root->to_apply()->root_instruction(),
            match::Add(match::Parameter(), match::Parameter()))) {
    if (reduce_operand == shape_covering_instr) {
      PatternInfo pattern_info;
      pattern_info.type = PatternType::Other;
      pattern_info.instr = instr;
      const Shape& slice_shape = instr->shape();
      const int64_t num_layers = instr->operand(0)->shape().dimensions(0);
      pattern_info.unstacked_shape =
          MakeUnstackedShapeFromSlice(slice_shape, num_layers);
      pattern_info.unstacking_computation =
          instr->fused_instructions_computation();
      pattern_info.unstacked_instrs.push_back(instr);
      return pattern_info;
    }
  }
  return std::nullopt;
}
absl::Status UnstackReduceFusionPattern(HloInstruction* mutable_reduce_fusion,
                                        const Shape& slice_shape) {
  HloComputation* parent_loop = mutable_reduce_fusion->parent();
  HloInstruction* stacked = mutable_reduce_fusion->mutable_operand(0);
  HloInstruction* offset = mutable_reduce_fusion->mutable_operand(1);
  HloInstruction* new_operand =
      parent_loop->AddInstruction(HloInstruction::CreateCustomCall(
          slice_shape, {stacked, offset}, "DynamicGte"));
  return mutable_reduce_fusion->ReplaceAllUsesWithDifferentShape(new_operand);
}
};  
absl::StatusOr<bool> HloUnstacker::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_ASSIGN_OR_RETURN(auto metadata,
                      UnstackerMetadata::Create(module, unfuse_slice_));
  metadata.custom_handlers.push_back(
      std::make_pair(GetDSAndDUSPattern, UnstackDSAndDUSPattern));
  metadata.custom_handlers.push_back(
      std::make_pair(GetDSFusionPattern, UnstackDSFusionPattern));
  metadata.custom_handlers.push_back(
      std::make_pair(GetDUSFusionPattern, UnstackDUSFusionPattern));
  metadata.custom_handlers.push_back(std::make_pair(
      GetDUSFusionWithPadPattern, UnstackDUSFusionWithPadPattern));
  metadata.custom_handlers.push_back(
      std::make_pair(GetDSFusionWithAddPattern, UnstackDSFusionWithAddPattern));
  metadata.custom_handlers.push_back(
      std::make_pair(GetReduceFusionPattern, UnstackReduceFusionPattern));
  metadata.custom_handlers.push_back(
      std::make_pair(GetNestedDSFusionPattern, UnstackNestedDSFusionPattern));
  metadata.custom_handlers.push_back(std::make_pair(
      GetDSFusionNoBitcastPattern, UnstackDSFusionNoBitcastPattern));
  std::vector<HloInstruction*> entry_loops;
  for (HloInstruction* instr :
       module->entry_computation()->MakeInstructionPostOrder()) {
    if (Match(instr, match::While(match::Tuple())) &&
        Match(instr->while_body()->root_instruction(), match::Tuple())) {
      entry_loops.push_back(instr);
    }
  }
  bool unstacked = false;
  std::vector<const HloInstruction*> unstacked_instructions;
  for (HloInstruction* loop : entry_loops) {
    for (int64_t i = 0; i < loop->shape().tuple_shapes_size(); ++i) {
      if (loop->while_init()->operand(i)->shape().IsTuple()) {
        continue;
      }
      VLOG(3) << "Attempting to unstack " << loop->name() << " at " << i
              << " = " << loop->while_init()->operand(i)->shape().ToString(true)
              << loop->while_init()->operand(i)->ToShortString();
      unstacked |=
          UnstackWhileOperandAtIndex(metadata, loop, i, unstacked_instructions);
      VLOG(3) << "###################";
    }
  }
  if (!unstacked) {
    return false;
  }
  TF_RETURN_IF_ERROR(module->RemoveUnusedComputations());
  std::vector<HloInstruction*> loops_to_unroll;
  for (const HloInstruction* instr : unstacked_instructions) {
    HloInstruction* loop = metadata.bodies[instr->parent()];
    if (std::find(loops_to_unroll.begin(), loops_to_unroll.end(), loop) ==
        loops_to_unroll.end()) {
      loops_to_unroll.push_back(loop);
    }
  }
  for (int64_t i = loops_to_unroll.size() - 1; i >= 0; --i) {
    HloInstruction* loop = loops_to_unroll[i];
    TF_ASSIGN_OR_RETURN(UnrollResult unroll_result,
                        WhileLoopUnroller::UnrollAndReturnReplacement(
                            loop, -1,
                            false,
                            true, false));
    bool unrolled = unroll_result.unrolled;
    CHECK(unrolled);
  }
  VLOG(3) << "after unstacking \n" << module->ToString();
  return true;
}
}  