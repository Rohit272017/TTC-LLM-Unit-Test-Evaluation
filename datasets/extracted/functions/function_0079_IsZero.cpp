#include "xla/service/while_loop_all_reduce_code_motion.h"
#include <memory>
#include <optional>
#include <stack>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal_util.h"
#include "xla/map_util.h"
#include "xla/service/call_graph.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/hlo_replication_analysis.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace {
struct AccumulationContext {
  HloInstruction* accumulation_instruction;
  HloInstruction* accumulation_buffer;
  int64_t param_tuple_index;
  std::optional<HloInstruction*> dynamic_slice;
  std::optional<HloInstruction*> dynamic_update_slice;
};
struct MovableAllReduceContext {
  bool is_movable;
  std::vector<AccumulationContext> accumulation_contexts;
};
bool IsZero(const HloInstruction* hlo) {
  if (hlo->IsConstant() && hlo->shape().rank() == 0 &&
      hlo->literal().IsZero({})) {
    return true;
  }
  if (hlo->opcode() == HloOpcode::kBroadcast) {
    return IsZero(hlo->operand(0));
  }
  return false;
}
bool IsValueReplicatedWithinEachAllReduceGroup(
    const HloInstruction& instruction, const ShapeIndex& index,
    CollectiveOpGroupMode all_reduce_group_mode,
    absl::Span<const ReplicaGroup> replica_groups, int num_replicas,
    int num_partitions,
    const std::unique_ptr<HloReplicationAnalysis>&
        cross_replica_replication_analysis,
    const std::unique_ptr<HloReplicationAnalysis>&
        cross_partition_replication_analysis) {
  VLOG(5) << "IsValueReplicatedWithinEachAllReduceGroup,"
          << " all_reduce_group_mode: "
          << CollectiveOpGroupModeToString(all_reduce_group_mode);
  switch (all_reduce_group_mode) {
    case CollectiveOpGroupMode::kCrossReplica: {
      return cross_replica_replication_analysis == nullptr ||
             cross_replica_replication_analysis->HloInstructionIsReplicatedAt(
                 &instruction, index, replica_groups);
    }
    case CollectiveOpGroupMode::kCrossPartition: {
      return cross_partition_replication_analysis == nullptr ||
             cross_partition_replication_analysis->HloInstructionIsReplicatedAt(
                 &instruction, index, replica_groups);
    }
    case CollectiveOpGroupMode::kCrossReplicaAndPartition: {
      return (cross_replica_replication_analysis == nullptr ||
              cross_replica_replication_analysis->HloInstructionIsReplicatedAt(
                  &instruction, index, replica_groups)) &&
             (cross_partition_replication_analysis == nullptr ||
              cross_partition_replication_analysis
                  ->HloInstructionIsReplicatedAt(&instruction, index));
    }
    case CollectiveOpGroupMode::kFlattenedID: {
      if (num_replicas == 1) {
        return cross_partition_replication_analysis == nullptr ||
               cross_partition_replication_analysis
                   ->HloInstructionIsReplicatedAt(&instruction, index,
                                                  replica_groups);
      }
      if (num_partitions == 1) {
        return cross_replica_replication_analysis == nullptr ||
               cross_replica_replication_analysis->HloInstructionIsReplicatedAt(
                   &instruction, index, replica_groups);
      }
      return (cross_replica_replication_analysis == nullptr ||
              cross_replica_replication_analysis->HloInstructionIsReplicatedAt(
                  &instruction, index)) &&
             (cross_partition_replication_analysis == nullptr ||
              cross_partition_replication_analysis
                  ->HloInstructionIsReplicatedAt(&instruction, index));
    }
  }
}
HloInstruction* GetEffectiveScalar(HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kBroadcast) {
    return nullptr;
  }
  HloInstruction* operand = instruction->mutable_operand(0);
  if (!ShapeUtil::IsScalar(operand->shape())) {
    return nullptr;
  }
  return operand;
}
MovableAllReduceContext IsAllReduceMovable(
    HloAllReduceInstructionBase* all_reduce, HloComputation* while_body,
    const std::unique_ptr<HloReplicationAnalysis>&
        cross_replica_replication_analysis,
    const std::unique_ptr<HloReplicationAnalysis>&
        cross_partition_replication_analysis) {
  VLOG(4) << "IsAllReduceMovable: " << all_reduce->ToString();
  std::optional<ReductionKind> reduction_type =
      MatchReductionComputation(all_reduce->to_apply());
  const bool all_reduce_is_summation =
      reduction_type.has_value() && *reduction_type == ReductionKind::SUM;
  const absl::InlinedVector<PrimitiveType, 12> kSupportedTypes{
      BF16, F16, F32, F64, S8, S16, S32, S64, U8, U16, U32, U64};
  if (!absl::c_linear_search(kSupportedTypes,
                             all_reduce->shape().element_type()) ||
      !all_reduce_is_summation) {
    return MovableAllReduceContext{false,
                                   {}};
  }
  CollectiveOpGroupMode all_reduce_group_mode =
      GetCollectiveOpGroupMode(all_reduce->channel_id().has_value(),
                               all_reduce->use_global_device_ids())
          .value();
  auto is_value_replicated_within_replica_group =
      [&cross_replica_replication_analysis,
       &cross_partition_replication_analysis, &all_reduce_group_mode,
       all_reduce](const HloInstruction& instruction,
                   const ShapeIndex& index) -> bool {
    bool is_replicated = IsValueReplicatedWithinEachAllReduceGroup(
        instruction, index, all_reduce_group_mode, all_reduce->replica_groups(),
        all_reduce->GetModule()->config().replica_count(),
        all_reduce->GetModule()->config().num_partitions(),
        cross_replica_replication_analysis,
        cross_partition_replication_analysis);
    VLOG(5) << "instruction: " << instruction.name()
            << " is_replicate: " << is_replicated;
    return is_replicated;
  };
  struct BufferTupleIndex {
    bool unsupported_operation{false};
    std::optional<int64_t> tuple_index;
    bool returned_from_computation{false};
    std::optional<HloInstruction*> dynamic_slice;
    std::optional<HloInstruction*> dynamic_update_slice;
  };
  const bool is_reduce_scatter =
      all_reduce->opcode() == HloOpcode::kReduceScatter;
  auto get_origin_tuple_index =
      [is_reduce_scatter](HloInstruction* instruction) -> BufferTupleIndex {
    VLOG(4) << "get_origin_tuple_index called on " << instruction->ToString();
    BufferTupleIndex result;
    while (!result.unsupported_operation) {
      switch (instruction->opcode()) {
        default: {
          VLOG(4) << "get_origin_tuple_index, instruction: ("
                  << instruction->ToString()
                  << ") is an unsupported operation on accumulation buffer.";
          result.unsupported_operation = true;
          break;
        }
        case HloOpcode::kBitcast:
        case HloOpcode::kConvert:
        case HloOpcode::kReshape:
        case HloOpcode::kTranspose:
          if (is_reduce_scatter) {
            VLOG(4) << "get_origin_tuple_index, instruction: ("
                    << instruction->ToString()
                    << ") is an unsupported operation on accumulation buffer.";
            result.unsupported_operation = true;
          } else {
            instruction = instruction->mutable_operand(0);
          }
          break;
        case HloOpcode::kGetTupleElement: {
          if (result.tuple_index.has_value()) {
            result.unsupported_operation = true;
          } else {
            result.tuple_index =
                Cast<HloGetTupleElementInstruction>(instruction)->tuple_index();
            instruction = instruction->mutable_operand(0);
          }
          break;
        }
        case HloOpcode::kDynamicSlice: {
          if (is_reduce_scatter) {
            VLOG(4) << "get_origin_tuple_index, instruction: ("
                    << instruction->ToString()
                    << ") is an unsupported operation on accumulation buffer.";
            result.unsupported_operation = true;
          } else if (result.dynamic_slice.has_value()) {
            VLOG(4) << "get_origin_tuple_index, instruction: ("
                    << instruction->ToString()
                    << "), we do not yet support more than 1 dynamic-slices on"
                    << " the accumulation buffer.";
            result.unsupported_operation = true;
          } else {
            result.dynamic_slice = instruction;
            instruction = instruction->mutable_operand(0);
          }
          break;
        }
        case HloOpcode::kParameter: {
          int parameter_number =
              Cast<HloParameterInstruction>(instruction)->parameter_number();
          CHECK_EQ(parameter_number, 0);
          break;
        }
      }
      if (instruction->opcode() == HloOpcode::kParameter) {
        break;
      }
    }
    return result;
  };
  auto get_output_tuple_index =
      [is_reduce_scatter](HloInstruction* instruction,
                          HloComputation* while_body) -> BufferTupleIndex {
    VLOG(4) << "get_output_tuple_index called on " << instruction->ToString();
    BufferTupleIndex result;
    std::stack<HloInstruction*> to_visit;
    to_visit.push(instruction);
    while (!to_visit.empty() && !result.unsupported_operation) {
      HloInstruction* instruction = to_visit.top();
      to_visit.pop();
      for (HloInstruction* user : instruction->users()) {
        switch (user->opcode()) {
          case HloOpcode::kBitcast:
          case HloOpcode::kConvert:
          case HloOpcode::kReshape:
          case HloOpcode::kGetTupleElement:
          case HloOpcode::kTranspose:
          case HloOpcode::kSlice: {
            if (is_reduce_scatter) {
              result.unsupported_operation = true;
            } else {
              to_visit.push(user);
            }
            break;
          }
          case HloOpcode::kDynamicUpdateSlice: {
            if (result.dynamic_update_slice.has_value() || is_reduce_scatter) {
              result.unsupported_operation = true;
            } else {
              result.dynamic_update_slice = user;
              to_visit.push(user);
            }
            break;
          }
          case HloOpcode::kTuple: {
            if (result.tuple_index.has_value()) {
              result.unsupported_operation = true;
            } else {
              result.tuple_index = user->operand_index(instruction);
              if (while_body->root_instruction() == user) {
                if (result.returned_from_computation) {
                  result.unsupported_operation = true;
                }
                result.returned_from_computation = true;
              } else {
                to_visit.push(user);
              }
            }
            break;
          }
          default: {
            VLOG(4) << "get_output_tuple_index, instruction: ("
                    << instruction->ToString()
                    << ") is an unsupported operation on accumulation buffer.";
            result.unsupported_operation = true;
          }
        }
        if (result.unsupported_operation) {
          break;
        }
      }
    }
    return result;
  };
  auto is_buffer_used =
      [&is_value_replicated_within_replica_group, is_reduce_scatter](
          absl::Span<const AccumulationContext> accumulation_contexts,
          HloComputation* while_body_computation) -> bool {
    CHECK_EQ(while_body_computation->num_parameters(), 1);
    HloInstruction* parameter_instruction =
        while_body_computation->parameter_instruction(0);
    for (const auto& accumulation : accumulation_contexts) {
      HloInstruction* accumulation_instruction =
          accumulation.accumulation_instruction;
      int64_t tuple_index = accumulation.param_tuple_index;
      std::stack<HloInstruction*> to_visit;
      for (HloInstruction* user : parameter_instruction->users()) {
        if (auto* gte = DynCast<HloGetTupleElementInstruction>(user)) {
          if (gte->tuple_index() == tuple_index) {
            to_visit.push(user);
          }
        } else {
          return true;
        }
      }
      while (!to_visit.empty()) {
        HloInstruction* instruction = to_visit.top();
        to_visit.pop();
        for (HloInstruction* user : instruction->users()) {
          VLOG(5) << "is_buffer_used, user: " << user->name();
          switch (user->opcode()) {
            case HloOpcode::kBitcast:
            case HloOpcode::kConvert:
            case HloOpcode::kReshape:
            case HloOpcode::kTranspose:
              if (is_reduce_scatter) {
                VLOG(4) << "buffer is used by " << user->ToString()
                        << ", preventing the motion of reduce-scatter.";
                return true;
              }
              to_visit.push(user);
              break;
            case HloOpcode::kSelect: {
              if (((user->operand_index(instruction) == 1 &&
                    IsZero(user->operand(2))) ||
                   (user->operand_index(instruction) == 2 &&
                    IsZero(user->operand(1)))) &&
                  is_value_replicated_within_replica_group(*(user->operand(0)),
                                                           {})) {
                to_visit.push(user);
              } else {
                return true;
              }
              break;
            }
            case HloOpcode::kAdd: {
              if (user != accumulation_instruction) {
                return true;
              }
              break;
            }
            case HloOpcode::kDynamicSlice: {
              if (!accumulation.dynamic_slice.has_value() ||
                  user != *accumulation.dynamic_slice) {
                return true;
              }
              break;
            }
            case HloOpcode::kDynamicUpdateSlice: {
              if (!accumulation.dynamic_update_slice.has_value() ||
                  user != *accumulation.dynamic_update_slice) {
                return true;
              }
              break;
            }
            default: {
              VLOG(4) << "buffer is used by " << user->ToString()
                      << ", preventing the motion of all-reduce.";
              return true;
            }
          }
        }
      }
    }
    return false;
  };
  auto dus_matches_ds_offsets =
      [](const HloInstruction& dynamic_slice,
         const HloInstruction& dynamic_update_slice) -> bool {
    if (dynamic_slice.operand_count() + 1 !=
        dynamic_update_slice.operand_count()) {
      return false;
    }
    for (int i = 1; i < dynamic_slice.operand_count(); ++i) {
      if (dynamic_slice.operand(i) != dynamic_update_slice.operand(i + 1)) {
        return false;
      }
    }
    return true;
  };
  auto dus_indices_are_replicated =
      [&is_value_replicated_within_replica_group](
          const HloInstruction& dynamic_update_slice) -> bool {
    for (int i = 2; i < dynamic_update_slice.operand_count(); ++i) {
      if (!is_value_replicated_within_replica_group(
              *dynamic_update_slice.operand(i), {})) {
        return false;
      }
    }
    return true;
  };
  std::vector<AccumulationContext> accumulation_contexts;
  std::stack<HloInstruction*> to_visit;
  bool is_all_reduce_movable = true;
  to_visit.push(all_reduce);
  while (!to_visit.empty() && is_all_reduce_movable) {
    HloInstruction* instruction = to_visit.top();
    to_visit.pop();
    for (HloInstruction* user : instruction->users()) {
      switch (user->opcode()) {
        case HloOpcode::kConvert:
          to_visit.push(user);
          break;
        case HloOpcode::kBitcast:
        case HloOpcode::kReshape:
        case HloOpcode::kGetTupleElement:
        case HloOpcode::kTranspose:
        case HloOpcode::kSlice: {
          if (is_reduce_scatter) {
            is_all_reduce_movable = false;
          } else {
            to_visit.push(user);
          }
          break;
        }
        case HloOpcode::kSelect: {
          bool is_select_ok = [&]() {
            bool operand_1_match = user->operand_index(instruction) == 1 &&
                                   IsZero(user->operand(2));
            bool operand_2_match = user->operand_index(instruction) == 2 &&
                                   IsZero(user->operand(1));
            if (!operand_1_match && !operand_2_match) {
              return false;
            }
            if (!is_reduce_scatter) {
              return true;
            }
            HloInstruction* predicate = user->mutable_operand(0);
            return GetEffectiveScalar(predicate) != nullptr;
          }();
          if (is_select_ok) {
            to_visit.push(user);
          } else {
            is_all_reduce_movable = false;
          }
          break;
        }
        case HloOpcode::kAdd: {
          int64_t buffer_index = 1 - user->operand_index(instruction);
          HloInstruction* accumulation_buffer =
              user->mutable_operand(buffer_index);
          auto origin_buffer_tuple_index =
              get_origin_tuple_index(accumulation_buffer);
          if (origin_buffer_tuple_index.unsupported_operation) {
            is_all_reduce_movable = false;
            break;
          }
          auto output_buffer_tuple_index =
              get_output_tuple_index(user, while_body);
          if (!output_buffer_tuple_index.unsupported_operation &&
              output_buffer_tuple_index.returned_from_computation &&
              origin_buffer_tuple_index.tuple_index.has_value() &&
              output_buffer_tuple_index.tuple_index.has_value() &&
              origin_buffer_tuple_index.tuple_index ==
                  output_buffer_tuple_index.tuple_index &&
              (origin_buffer_tuple_index.dynamic_slice.has_value() ==
               output_buffer_tuple_index.dynamic_update_slice.has_value()) &&
              (!origin_buffer_tuple_index.dynamic_slice.has_value() ||
               (dus_matches_ds_offsets(
                    **origin_buffer_tuple_index.dynamic_slice,
                    **output_buffer_tuple_index.dynamic_update_slice) &&
                dus_indices_are_replicated(
                    **output_buffer_tuple_index.dynamic_update_slice)))) {
            accumulation_contexts.push_back(AccumulationContext{
                user, accumulation_buffer,
                *output_buffer_tuple_index.tuple_index,
                origin_buffer_tuple_index.dynamic_slice,
                output_buffer_tuple_index.dynamic_update_slice});
          } else {
            is_all_reduce_movable = false;
          }
          break;
        }
        default: {
          VLOG(4) << "get_accumulation_contexts, all-reduce result is used "
                  << " by " << user->ToString() << ", not movable.";
          is_all_reduce_movable = false;
        }
      }
    }
  }
  if (is_buffer_used(accumulation_contexts, while_body)) {
    is_all_reduce_movable = false;
  }
  return MovableAllReduceContext{is_all_reduce_movable, accumulation_contexts};
}
struct WhileInitContext {
  HloInstruction* while_init{nullptr};
  absl::flat_hash_map<int, HloInstruction*> tuple_index_to_old_buffer;
};
WhileInitContext CreateNewWhileInit(
    HloInstruction* old_while_instruction,
    const HloInstructionMap<std::vector<AccumulationContext>>&
        all_reduce_to_accumulations) {
  HloInstruction* old_while_init = old_while_instruction->mutable_operand(0);
  HloComputation* while_parent = old_while_instruction->parent();
  std::vector<HloInstruction*> new_while_init_elements(
      old_while_init->operand_count(), nullptr);
  for (const auto& all_reduce_and_accumulations_pair :
       all_reduce_to_accumulations) {
    const std::vector<AccumulationContext>& accumulations =
        all_reduce_and_accumulations_pair.second;
    HloInstruction* loop_all_reduce = all_reduce_and_accumulations_pair.first;
    for (auto& accumulation_context : accumulations) {
      int64_t tuple_index = accumulation_context.param_tuple_index;
      HloInstruction* old_buffer = old_while_init->mutable_operand(tuple_index);
      const Shape& accumulation_shape =
          loop_all_reduce->opcode() == HloOpcode::kAllReduce
              ? old_buffer->shape()
              : loop_all_reduce->operand(0)->shape();
      HloInstruction* new_buffer = while_parent->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateFromDimensions(
              accumulation_shape.element_type(),
              accumulation_shape.dimensions())));
      new_while_init_elements[tuple_index] = new_buffer;
    }
  }
  absl::flat_hash_map<int, HloInstruction*> tuple_index_to_old_buffer;
  for (int i = 0; i < old_while_init->operand_count(); i++) {
    if (!new_while_init_elements[i]) {
      new_while_init_elements[i] = old_while_init->mutable_operand(i);
    } else {
      tuple_index_to_old_buffer[i] = old_while_init->mutable_operand(i);
    }
  }
  HloInstruction* new_while_init = while_parent->AddInstruction(
      HloInstruction::CreateTuple(new_while_init_elements));
  return WhileInitContext{new_while_init, tuple_index_to_old_buffer};
}
absl::Status ChangeAccumulatorShapesInLoopBodies(
    HloInstruction* old_while_instruction,
    const HloInstructionMap<std::vector<AccumulationContext>>&
        all_reduce_to_accumulations) {
  HloComputation* body = old_while_instruction->while_body();
  HloComputation* cond = old_while_instruction->while_condition();
  absl::flat_hash_map<Shape, HloInstruction*> zeros;
  auto create_zero_of_shape = [&zeros, body](const Shape& shape) {
    auto it = zeros.find(shape);
    if (it != zeros.end()) {
      return it->second;
    }
    HloInstruction* zero = body->AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateFromShape(shape)));
    zeros[shape] = zero;
    return zero;
  };
  for (const auto& [loop_reduce_scatter, accumulations] :
       all_reduce_to_accumulations) {
    if (loop_reduce_scatter->opcode() != HloOpcode::kReduceScatter) {
      continue;
    }
    const Shape& accumulation_shape = loop_reduce_scatter->operand(0)->shape();
    for (auto& accumulation_context : accumulations) {
      const int64_t tuple_index = accumulation_context.param_tuple_index;
      HloInstruction* param_body = body->parameter_instruction(0);
      std::vector<Shape> element_shapes = param_body->shape().tuple_shapes();
      element_shapes[tuple_index] = accumulation_shape;
      *param_body->mutable_shape() = ShapeUtil::MakeTupleShape(element_shapes);
      for (HloInstruction* user : param_body->users()) {
        if (user->opcode() != HloOpcode::kGetTupleElement) {
          continue;
        }
        HloGetTupleElementInstruction* gte =
            Cast<HloGetTupleElementInstruction>(user);
        if (gte->tuple_index() != tuple_index) {
          continue;
        }
        *gte->mutable_shape() = accumulation_shape;
        for (HloInstruction* gte_user : gte->users()) {
          CHECK_EQ(gte_user->opcode(), HloOpcode::kAdd);
          *gte_user->mutable_shape() = accumulation_shape;
        }
      }
      std::vector<HloInstruction*> reduce_scatter_users =
          loop_reduce_scatter->users();
      while (!reduce_scatter_users.empty()) {
        HloInstruction* user = reduce_scatter_users.back();
        reduce_scatter_users.pop_back();
        if (user->opcode() == HloOpcode::kSelect) {
          HloInstruction* zero = create_zero_of_shape(accumulation_shape);
          HloInstruction* scalar_predicate =
              GetEffectiveScalar(user->mutable_operand(0));
          Shape pred_shape =
              ShapeUtil::ChangeElementType(accumulation_shape, PRED);
          HloInstruction* pred =
              body->AddInstruction(HloInstruction::CreateBroadcast(
                  pred_shape, scalar_predicate, {}));
          TF_RETURN_IF_ERROR(user->ReplaceOperandWithDifferentShape(0, pred));
          HloInstruction *new_operand_1, *new_operand_2;
          if (user->operand_index(loop_reduce_scatter) == 1) {
            new_operand_1 = loop_reduce_scatter->mutable_operand(0);
            new_operand_2 = zero;
          } else {
            new_operand_1 = zero;
            new_operand_2 = loop_reduce_scatter->mutable_operand(0);
          }
          TF_RETURN_IF_ERROR(
              user->ReplaceOperandWithDifferentShape(1, new_operand_1));
          TF_RETURN_IF_ERROR(
              user->ReplaceOperandWithDifferentShape(2, new_operand_2));
          *user->mutable_shape() = accumulation_shape;
        } else {
          TF_RET_CHECK(user->opcode() == HloOpcode::kAdd);
          TF_RET_CHECK(user->shape() == accumulation_shape);
        }
      }
      HloInstruction* root = body->root_instruction();
      *root->mutable_shape() = param_body->shape();
      HloInstruction* param_cond = cond->parameter_instruction(0);
      *param_cond->mutable_shape() = param_body->shape();
    }
  }
  return absl::OkStatus();
}
absl::flat_hash_map<int, HloInstruction*> CreateSinkedAllReduces(
    HloInstruction* new_while_instruction,
    const HloInstructionMap<std::vector<AccumulationContext>>&
        all_reduce_to_accumulations,
    const absl::flat_hash_map<int, HloInstruction*>&
        tuple_index_to_old_buffer) {
  HloComputation* while_parent = new_while_instruction->parent();
  absl::flat_hash_map<int, HloInstruction*> tuple_index_to_new_buffer;
  for (const auto& all_reduce_and_accumulations_pair :
       all_reduce_to_accumulations) {
    HloInstruction* loop_all_reduce = all_reduce_and_accumulations_pair.first;
    const std::vector<AccumulationContext>& accumulations =
        all_reduce_and_accumulations_pair.second;
    for (const auto& accumulation_context : accumulations) {
      int64_t tuple_index = accumulation_context.param_tuple_index;
      const Shape& accumulation_buffer_shape =
          new_while_instruction->shape().tuple_shapes(tuple_index);
      HloInstruction* accumulation_buffer =
          while_parent->AddInstruction(HloInstruction::CreateGetTupleElement(
              accumulation_buffer_shape, new_while_instruction, tuple_index));
      HloInstruction* all_reduce_operand = accumulation_buffer;
      if (!ShapeUtil::SameElementType(loop_all_reduce->shape(),
                                      accumulation_buffer_shape)) {
        Shape all_reduce_shape =
            ShapeUtil::MakeShape(loop_all_reduce->shape().element_type(),
                                 accumulation_buffer_shape.dimensions());
        all_reduce_operand =
            while_parent->AddInstruction(HloInstruction::CreateConvert(
                all_reduce_shape, accumulation_buffer));
      }
      HloInstruction* all_reduced_delta;
      if (loop_all_reduce->opcode() == HloOpcode::kAllReduce) {
        auto* old_all_reduce = Cast<HloAllReduceInstruction>(loop_all_reduce);
        all_reduced_delta =
            while_parent->AddInstruction(HloInstruction::CreateAllReduce(
                all_reduce_operand->shape(), {all_reduce_operand},
                old_all_reduce->called_computations()[0],
                old_all_reduce->device_list(),
                old_all_reduce->constrain_layout(),
                hlo_query::NextChannelId(*(while_parent->parent())),
                old_all_reduce->use_global_device_ids()));
      } else {
        auto* old_reduce_scatter =
            Cast<HloReduceScatterInstruction>(loop_all_reduce);
        all_reduced_delta =
            while_parent->AddInstruction(HloInstruction::CreateReduceScatter(
                old_reduce_scatter->shape(), {all_reduce_operand},
                old_reduce_scatter->called_computations()[0],
                old_reduce_scatter->device_list(),
                old_reduce_scatter->constrain_layout(),
                hlo_query::NextChannelId(*(while_parent->parent())),
                old_reduce_scatter->use_global_device_ids(),
                old_reduce_scatter->scatter_dimension()));
      }
      if (!ShapeUtil::SameElementType(all_reduced_delta->shape(),
                                      accumulation_buffer_shape)) {
        all_reduced_delta =
            while_parent->AddInstruction(HloInstruction::CreateConvert(
                accumulation_buffer_shape, all_reduced_delta));
      }
      CHECK(ContainsKey(tuple_index_to_old_buffer, tuple_index));
      HloInstruction* old_buffer = tuple_index_to_old_buffer.at(tuple_index);
      CHECK(Shape::Equal().IgnoreLayout()(old_buffer->shape(),
                                          all_reduced_delta->shape()));
      HloInstruction* add_to_old_buffer =
          while_parent->AddInstruction(HloInstruction::CreateBinary(
              all_reduced_delta->shape(), HloOpcode::kAdd, old_buffer,
              all_reduced_delta));
      tuple_index_to_new_buffer[tuple_index] = add_to_old_buffer;
    }
  }
  return tuple_index_to_new_buffer;
}
HloInstruction* CreateNewWhileResult(
    HloInstruction* new_while_instruction,
    const absl::flat_hash_map<int, HloInstruction*>&
        tuple_index_to_new_buffer) {
  HloComputation* while_parent = new_while_instruction->parent();
  CHECK(new_while_instruction->shape().IsTuple());
  std::vector<HloInstruction*> new_while_result_elements(
      new_while_instruction->shape().tuple_shapes_size(), nullptr);
  for (int i = 0; i < new_while_result_elements.size(); i++) {
    if (ContainsKey(tuple_index_to_new_buffer, i)) {
      new_while_result_elements[i] = tuple_index_to_new_buffer.at(i);
    } else {
      HloInstruction* gte =
          while_parent->AddInstruction(HloInstruction::CreateGetTupleElement(
              new_while_instruction->shape().tuple_shapes(i),
              new_while_instruction, i));
      new_while_result_elements[i] = gte;
    }
  }
  HloInstruction* new_while_result = while_parent->AddInstruction(
      HloInstruction::CreateTuple(new_while_result_elements));
  return new_while_result;
}
absl::Status AddSinkedAllReducesAndReplaceWhile(
    HloInstruction* while_instruction,
    const HloInstructionMap<std::vector<AccumulationContext>>&
        all_reduce_to_accumulations) {
  auto new_while_init_context =
      CreateNewWhileInit(while_instruction, all_reduce_to_accumulations);
  TF_RETURN_IF_ERROR(ChangeAccumulatorShapesInLoopBodies(
      while_instruction, all_reduce_to_accumulations));
  HloInstruction* new_while_instruction =
      while_instruction->parent()->AddInstruction(HloInstruction::CreateWhile(
          new_while_init_context.while_init->shape(),
          while_instruction->while_condition(), while_instruction->while_body(),
          new_while_init_context.while_init));
  absl::flat_hash_map<int, HloInstruction*> tuple_index_to_new_buffer =
      CreateSinkedAllReduces(new_while_instruction, all_reduce_to_accumulations,
                             new_while_init_context.tuple_index_to_old_buffer);
  HloInstruction* new_while_result =
      CreateNewWhileResult(new_while_instruction, tuple_index_to_new_buffer);
  TF_RETURN_IF_ERROR(while_instruction->parent()->ReplaceInstruction(
      while_instruction, new_while_result));
  return absl::OkStatus();
}
}  
absl::StatusOr<bool> WhileLoopAllReduceCodeMotion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool is_changed = false;
  if (module->config().num_partitions() > 1 &&
      !module->config().use_spmd_partitioning()) {
    return false;
  }
  std::unique_ptr<HloReplicationAnalysis> cross_replica_replication_analysis;
  if (module->config().replica_count() > 1) {
    VLOG(5) << "num_replicas: " << module->config().replica_count()
            << " run HloReplicationAnalysis across replicas";
    TF_ASSIGN_OR_RETURN(cross_replica_replication_analysis,
                        HloReplicationAnalysis::RunWithPartialReplication(
                            module, false));
  }
  std::unique_ptr<HloReplicationAnalysis> cross_partition_replication_analysis;
  if (module->config().use_spmd_partitioning() &&
      module->config().num_partitions() > 1) {
    VLOG(5) << "num_partitions: " << module->config().num_partitions()
            << " run HloReplicationAnalysis across partitions";
    TF_ASSIGN_OR_RETURN(cross_partition_replication_analysis,
                        HloReplicationAnalysis::RunWithPartialReplication(
                            module, true));
  }
  uint32_t count_all_reduce = 0, count_reduce_scatter = 0;
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    std::vector<HloInstruction*> computation_callers =
        call_graph->GetComputationCallers(computation);
    std::vector<HloInstruction*> while_caller_instructions;
    for (HloInstruction* caller_instruction : computation_callers) {
      if (caller_instruction->opcode() == HloOpcode::kWhile &&
          caller_instruction->shape().IsTuple() &&
          caller_instruction->while_body() == computation) {
        while_caller_instructions.push_back(caller_instruction);
      }
    }
    if (while_caller_instructions.empty()) {
      continue;
    }
    std::vector<HloAllReduceInstructionBase*> while_body_all_reduces;
    for (HloInstruction* while_body_instruction :
         computation->MakeInstructionPostOrder()) {
      HloOpcode op = while_body_instruction->opcode();
      const bool is_candidate =
          (op == HloOpcode::kAllReduce) ||
          (enable_reduce_scatter_ && op == HloOpcode::kReduceScatter);
      if (!is_candidate) {
        continue;
      }
      auto* all_reduce_instruction =
          Cast<HloAllReduceInstructionBase>(while_body_instruction);
      if (all_reduce_instruction->constrain_layout()) {
        return false;
      } else {
        while_body_all_reduces.push_back(all_reduce_instruction);
      }
    }
    HloInstructionMap<std::vector<AccumulationContext>>
        all_reduce_to_accumulations;
    for (HloAllReduceInstructionBase* all_reduce : while_body_all_reduces) {
      auto movable_all_reduce_context = IsAllReduceMovable(
          all_reduce, computation, cross_replica_replication_analysis,
          cross_partition_replication_analysis);
      if (movable_all_reduce_context.is_movable) {
        all_reduce_to_accumulations[all_reduce] =
            std::move(movable_all_reduce_context.accumulation_contexts);
      }
      VLOG(3) << "WhileLoopAllReduceCodeMotion, all-reduce: "
              << all_reduce->ToString()
              << " is_movable: " << movable_all_reduce_context.is_movable
              << " while loop: " << while_caller_instructions.front()->name()
              << " num_accumulations: "
              << (movable_all_reduce_context.is_movable
                      ? all_reduce_to_accumulations[all_reduce].size()
                      : 0);
    }
    if (all_reduce_to_accumulations.empty()) {
      continue;
    }
    for (HloInstruction* while_instruction : while_caller_instructions) {
      TF_RETURN_IF_ERROR(AddSinkedAllReducesAndReplaceWhile(
          while_instruction, all_reduce_to_accumulations));
      is_changed = true;
    }
    for (const auto& all_reduce_accumulations_pair :
         all_reduce_to_accumulations) {
      HloInstruction* all_reduce = all_reduce_accumulations_pair.first;
      if (all_reduce->opcode() == HloOpcode::kAllReduce) {
        count_all_reduce++;
      } else {
        count_reduce_scatter++;
      }
      TF_RETURN_IF_ERROR(computation->ReplaceInstructionWithDifferentShape(
          all_reduce, all_reduce->mutable_operand(0)));
    }
    if (!all_reduce_to_accumulations.empty()) {
      call_graph = CallGraph::Build(module);
    }
  }
  VLOG(2) << "Hoisted " << count_all_reduce << " all-reduce and "
          << count_reduce_scatter << " reduce-scatter out of while loops";
  return is_changed;
}
}  