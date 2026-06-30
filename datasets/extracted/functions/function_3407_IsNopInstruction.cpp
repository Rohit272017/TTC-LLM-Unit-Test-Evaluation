#include "xla/service/latency_hiding_scheduler.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_reachability.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/map_util.h"
#include "xla/service/dump.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace {
const int64_t kDefaultMemorySpace = 0;
bool IsNopInstruction(const HloInstruction& hlo) {
  HloOpcode op = hlo.opcode();
  return op == HloOpcode::kGetTupleElement || op == HloOpcode::kBitcast ||
         op == HloOpcode::kConstant || op == HloOpcode::kParameter ||
         op == HloOpcode::kBroadcast || op == HloOpcode::kIota ||
         hlo.IsEffectiveBitcast() ||
         (op == HloOpcode::kTuple && hlo.user_count() == 1 &&
          hlo.users().front()->opcode() == HloOpcode::kWhile);
}
bool InstructionDefinesValue(const HloInstruction* instruction,
                             const HloValue* value) {
  if (value->defining_instruction() == instruction) {
    return true;
  }
  if (value->shape().has_layout() &&
      value->shape().layout().memory_space() != kDefaultMemorySpace) {
    return false;
  }
  if (instruction->opcode() == HloOpcode::kAsyncStart) {
    if (instruction->async_wrapped_opcode() == HloOpcode::kCall) {
      return instruction->async_wrapped_instruction()
                 ->called_computations()[0]
                 ->root_instruction() == value->defining_instruction();
    }
    return instruction->async_wrapped_instruction() ==
           value->defining_instruction();
  }
  return false;
}
bool InstructionFirstDefinesBuffer(
    const HloInstruction* instruction,
    const BufferInfoTracker::ValueInfo& buffer_value_info) {
  if (buffer_value_info.first_definition == instruction) {
    return true;
  }
  if (buffer_value_info.value->values()[0]->shape().has_layout() &&
      buffer_value_info.value->values()[0]->shape().layout().memory_space() !=
          kDefaultMemorySpace) {
    return false;
  }
  if (instruction->opcode() == HloOpcode::kAsyncStart) {
    if (instruction->async_wrapped_opcode() == HloOpcode::kCall) {
      return instruction->async_wrapped_instruction()
                 ->called_computations()[0]
                 ->root_instruction() == buffer_value_info.first_definition;
    }
    return instruction->async_wrapped_instruction() ==
           buffer_value_info.first_definition;
  }
  return false;
}
}  
CanonicalAsyncOp DefaultGetCanonicalAsyncOp(const HloInstruction& hlo) {
  switch (hlo.opcode()) {
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncDone:
      if (hlo.async_wrapped_opcode() == HloOpcode::kCall) {
        return {hlo.opcode(), hlo.async_wrapped_instruction()
                                  ->called_computations()[0]
                                  ->root_instruction()
                                  ->opcode()};
      }
      return {hlo.opcode(), hlo.async_wrapped_opcode()};
    case HloOpcode::kAllReduceStart:
      return {HloOpcode::kAsyncStart, HloOpcode::kAllReduce};
    case HloOpcode::kAllGatherStart:
      return {HloOpcode::kAsyncStart, HloOpcode::kAllGather};
    case HloOpcode::kCollectivePermuteStart:
      return {HloOpcode::kAsyncStart, HloOpcode::kCollectivePermute};
    case HloOpcode::kCopyStart:
      return {HloOpcode::kAsyncStart, HloOpcode::kCopy};
    case HloOpcode::kCopyDone:
      return {HloOpcode::kAsyncDone, HloOpcode::kCopy};
    case HloOpcode::kAllReduceDone:
      return {HloOpcode::kAsyncDone, HloOpcode::kAllReduce};
    case HloOpcode::kAllGatherDone:
      return {HloOpcode::kAsyncDone, HloOpcode::kAllGather};
    case HloOpcode::kCollectivePermuteDone:
      return {HloOpcode::kAsyncDone, HloOpcode::kCollectivePermute};
    default:
      return {hlo.opcode(), hlo.opcode()};
  }
}
bool LatencyEstimator::IsAsyncPair(const HloGraphNode& from,
                                   const HloGraphNode& target) const {
  CanonicalAsyncOp from_op = GetCanonicalAsyncOp(from.GetInstr());
  CanonicalAsyncOp target_op = GetCanonicalAsyncOp(target.GetInstr());
  return from_op.outer == HloOpcode::kAsyncStart &&
         target_op.outer == HloOpcode::kAsyncDone &&
         from_op.inner == target_op.inner;
}
bool LatencyEstimator::IsP2pPair(const HloGraphNode& from,
                                 const HloGraphNode& target) const {
  return (from.GetInstr().opcode() == HloOpcode::kSend &&
          target.GetInstr().opcode() == HloOpcode::kSendDone) ||
         (from.GetInstr().opcode() == HloOpcode::kRecv &&
          target.GetInstr().opcode() == HloOpcode::kRecvDone);
}
LatencyEstimator::TimeCost ApproximateLatencyEstimator::GetLatencyBetween(
    const HloGraphNode& from, const HloGraphNode& target) const {
  if (IsAsyncPair(from, target)) {
    return kHighLatency;
  }
  return kLowLatency;
}
LatencyEstimator::TimeCost ApproximateLatencyEstimator::NodeCost(
    const HloInstruction* instr) const {
  if (instr->IsLoopFusion()) {
    return kMediumCost;
  }
  if (instr->IsOutputFusion() || instr->opcode() == HloOpcode::kConvolution) {
    return kHighCost;
  }
  return kLowCost;
}
bool AsyncTracker::IsSupportedAsyncDone(const HloInstruction& hlo) const {
  CanonicalAsyncOp op = GetCanonicalAsyncOp(hlo);
  if (op.outer == HloOpcode::kSendDone || op.outer == HloOpcode::kRecvDone) {
    return config_.schedule_send_recvs;
  }
  if (op.outer == HloOpcode::kAsyncDone) {
    if (hlo.IsAsynchronous() &&
        hlo.async_execution_thread() != hlo.parent()->execution_thread()) {
      return true;
    }
    switch (op.inner) {
      case HloOpcode::kAllToAll:
      case HloOpcode::kAllGather:
      case HloOpcode::kAllReduce:
      case HloOpcode::kCollectiveBroadcast:
      case HloOpcode::kCollectivePermute:
      case HloOpcode::kCopy:
      case HloOpcode::kReduceScatter:
        return true;
      default:
        return false;
    }
  }
  return false;
}
bool AsyncTracker::IsSupportedAsyncStart(const HloInstruction& hlo) const {
  CanonicalAsyncOp op = GetCanonicalAsyncOp(hlo);
  if (op.outer == HloOpcode::kSend || op.outer == HloOpcode::kRecv) {
    return config_.schedule_send_recvs;
  }
  if (op.outer == HloOpcode::kAsyncStart) {
    if (hlo.IsAsynchronous() &&
        hlo.async_execution_thread() != hlo.parent()->execution_thread()) {
      return true;
    }
    switch (op.inner) {
      case HloOpcode::kAllToAll:
      case HloOpcode::kAllGather:
      case HloOpcode::kAllReduce:
      case HloOpcode::kCollectiveBroadcast:
      case HloOpcode::kCollectivePermute:
      case HloOpcode::kCopy:
      case HloOpcode::kReduceScatter:
        return true;
      default:
        return false;
    }
  }
  return false;
}
ResourcesVector AsyncTracker::GetResourcesFromInstructionImpl(
    const HloInstruction& hlo) const {
  CanonicalAsyncOp op = GetCanonicalAsyncOp(hlo);
  auto get_resource_for_op = [](HloOpcode op) -> ResourceType {
    switch (op) {
      case HloOpcode::kAllReduce:
        return ResourceType::kAllReduce;
      case HloOpcode::kAllGather:
        return ResourceType::kAllGather;
      case HloOpcode::kAllToAll:
        return ResourceType::kAllToAll;
      case HloOpcode::kCollectiveBroadcast:
        return ResourceType::kCollectiveBroadcast;
      case HloOpcode::kCollectivePermute:
        return ResourceType::kCollectivePermute;
      case HloOpcode::kCopy:
        return ResourceType::kCopy;
      case HloOpcode::kReduceScatter:
        return ResourceType::kReduceScatter;
      default:
        return ResourceType::kNoResource;
    }
  };
  if (op.outer == HloOpcode::kAsyncStart || op.outer == HloOpcode::kAsyncDone) {
    ResourceType type = get_resource_for_op(op.inner);
    if (type == ResourceType::kNoResource) {
      return {};
    }
    ResourceUsageType usage = op.outer == HloOpcode::kAsyncStart
                                  ? ResourceUsageType::kResourceRelease
                                  : ResourceUsageType::kResourceOccupy;
    return {std::make_pair(ResourceTypeToIndex(type), usage)};
  }
  switch (hlo.opcode()) {
    case HloOpcode::kAfterAll:
      return ResourcesVector{
          std::make_pair(ResourceTypeToIndex(ResourceType::kSendHost),
                         ResourceUsageType::kNoResource)};
    case HloOpcode::kRecv:
      return ResourcesVector{
          static_cast<const HloSendRecvInstruction*>(&hlo)->is_host_transfer()
              ? std::make_pair(
                    config_.force_send_recv_to_use_same_resource
                        ? ResourceTypeToIndex(ResourceType::kSendHost)
                        : ResourceTypeToIndex(ResourceType::kRecvHost),
                    ResourceUsageType::kResourceRelease)
              : std::make_pair(ResourceTypeToIndex(ResourceType::kSendRecv),
                               ResourceUsageType::kResourceRelease)};
    case HloOpcode::kSend:
      return ResourcesVector{
          static_cast<const HloSendRecvInstruction*>(&hlo)->is_host_transfer()
              ? std::make_pair(ResourceTypeToIndex(ResourceType::kSendHost),
                               ResourceUsageType::kResourceRelease)
              : std::make_pair(ResourceTypeToIndex(ResourceType::kSendRecv),
                               ResourceUsageType::kResourceRelease)};
    case HloOpcode::kRecvDone:
      return ResourcesVector{
          static_cast<const HloSendRecvInstruction*>(hlo.operand(0))
                  ->is_host_transfer()
              ? std::make_pair(
                    config_.force_send_recv_to_use_same_resource
                        ? ResourceTypeToIndex(ResourceType::kSendHost)
                        : ResourceTypeToIndex(ResourceType::kRecvHost),
                    ResourceUsageType::kResourceOccupy)
              : std::make_pair(ResourceTypeToIndex(ResourceType::kSendRecv),
                               ResourceUsageType::kResourceOccupy)};
    case HloOpcode::kSendDone:
      return ResourcesVector{
          static_cast<const HloSendRecvInstruction*>(hlo.operand(0))
                  ->is_host_transfer()
              ? std::make_pair(ResourceTypeToIndex(ResourceType::kSendHost),
                               ResourceUsageType::kResourceOccupy)
              : std::make_pair(ResourceTypeToIndex(ResourceType::kSendRecv),
                               ResourceUsageType::kResourceOccupy)};
    default:
      return ResourcesVector{};
  }
}
ResourcesVector AsyncTracker::GetResourcesFromInstruction(
    const HloInstruction& hlo) const {
  if (!resources_cache_.contains(&hlo)) {
    resources_cache_.insert({&hlo, GetResourcesFromInstructionImpl(hlo)});
  }
  return resources_cache_.at(&hlo);
}
int64_t AsyncTracker::GetNumResourcesPerInstruction(
    ResourceType resource_type, const HloInstruction& instr) const {
  return GetNumResourcesPerInstruction(ResourceTypeToIndex(resource_type),
                                       instr);
}
int64_t AsyncTracker::GetNumResourcesPerInstruction(
    int64_t resource_type, const HloInstruction& instr) const {
  if (instr.called_computations().empty() ||
      instr.opcode() == HloOpcode::kAsyncStart ||
      instr.opcode() == HloOpcode::kAsyncDone) {
    return absl::c_any_of(GetResourcesFromInstruction(instr),
                          [resource_type](const ResourcePair& resource) {
                            return resource.second ==
                                       ResourceUsageType::kResourceOccupy &&
                                   (resource_type == resource.first);
                          })
               ? 1
               : 0;
  }
  std::function<void(const HloComputation*)> recursively_compute_resource_map =
      [this,
       &recursively_compute_resource_map](const HloComputation* computation) {
        absl::flat_hash_map<int64_t, int64_t> per_opcode_map;
        for (HloInstruction* instr : computation->instructions()) {
          if (IsSupportedAsyncDone(*instr)) {
            for (auto& resource : GetResourcesFromInstruction(*instr)) {
              ++per_opcode_map[resource.first];
            }
          }
          for (const HloComputation* called_comp :
               instr->called_computations()) {
            auto it = async_in_computation_cache_.find(called_comp);
            if (it == async_in_computation_cache_.end()) {
              recursively_compute_resource_map(called_comp);
              it = async_in_computation_cache_.find(called_comp);
              CHECK(it != async_in_computation_cache_.end());
            }
            for (auto& called_per_opcode_pair : it->second) {
              per_opcode_map[called_per_opcode_pair.first] +=
                  called_per_opcode_pair.second;
            }
          }
        }
        async_in_computation_cache_[computation] = std::move(per_opcode_map);
      };
  int64_t num_resources = 0;
  for (const HloComputation* computation : instr.called_computations()) {
    auto it = async_in_computation_cache_.find(computation);
    if (it == async_in_computation_cache_.end()) {
      recursively_compute_resource_map(computation);
      it = async_in_computation_cache_.find(computation);
      CHECK(it != async_in_computation_cache_.end());
    }
    auto opcode_it = it->second.find(resource_type);
    if (opcode_it == it->second.end()) {
      continue;
    }
    num_resources += opcode_it->second;
  }
  return num_resources;
}
void AsyncTracker::SetConcurrentResourceLimits(
    absl::flat_hash_map<int64_t, int64_t>& max_concurrent_resource) const {
  max_concurrent_resource[ResourceTypeToIndex(
      ResourceType::kCollectiveBroadcast)] =
      config_.collective_broadcast_overlap_limit;
  max_concurrent_resource[ResourceTypeToIndex(
      ResourceType::kCollectivePermute)] =
      config_.collective_permute_overlap_limit;
  max_concurrent_resource[ResourceTypeToIndex(ResourceType::kCopy)] =
      config_.copy_overlap_limit;
  max_concurrent_resource[ResourceTypeToIndex(ResourceType::kAllToAll)] =
      config_.all_to_all_overlap_limit;
  max_concurrent_resource[ResourceTypeToIndex(ResourceType::kAllGather)] =
      config_.all_gather_overlap_limit;
  max_concurrent_resource[ResourceTypeToIndex(ResourceType::kAllReduce)] =
      config_.all_reduce_overlap_limit;
  max_concurrent_resource[ResourceTypeToIndex(ResourceType::kReduceScatter)] =
      config_.reduce_scatter_overlap_limit;
  max_concurrent_resource[ResourceTypeToIndex(ResourceType::kSendRecv)] =
      config_.send_recv_overlap_limit;
  max_concurrent_resource[ResourceTypeToIndex(ResourceType::kSendHost)] =
      config_.send_recv_host_overlap_limit;
  max_concurrent_resource[ResourceTypeToIndex(ResourceType::kRecvHost)] =
      config_.send_recv_host_overlap_limit;
  const int64_t first_target_resource =
      AsyncTracker::GetFirstTargetDefinedResource();
  for (int64_t i = 0; i < GetNumTargetDefinedResources(); ++i) {
    max_concurrent_resource[first_target_resource + i] =
        GetNumAvailableResources(first_target_resource + i);
  }
}
absl::string_view AsyncTracker::GetResourceName(int64_t resource_type) const {
  switch (resource_type) {
    case ResourceTypeToIndex(ResourceType::kNoResource):
      return "kNoResource";
    case ResourceTypeToIndex(ResourceType::kAllToAll):
      return "kAllToAll";
    case ResourceTypeToIndex(ResourceType::kAllGather):
      return "kAllGather";
    case ResourceTypeToIndex(ResourceType::kAllReduce):
      return "kAllReduce";
    case ResourceTypeToIndex(ResourceType::kCollectiveBroadcast):
      return "kCollectiveBroadcast";
    case ResourceTypeToIndex(ResourceType::kCollectivePermute):
      return "kCollectivePermute";
    case ResourceTypeToIndex(ResourceType::kCopy):
      return "kCopy";
    case ResourceTypeToIndex(ResourceType::kSendRecv):
      return "kSendRecv";
    case ResourceTypeToIndex(ResourceType::kSendHost):
      return "kSendHost";
    case ResourceTypeToIndex(ResourceType::kRecvHost):
      return "kRecvHost";
    case ResourceTypeToIndex(ResourceType::kReduceScatter):
      return "kReduceScatter";
    default:
      return "Not a valid default resource";
  }
}
absl::string_view AsyncTracker::GetResourceUsageName(
    ResourceUsageType resource_usage_type) const {
  return GetResourceUsageName(ResourceUsageTypeToIndex(resource_usage_type));
}
ResourceHazardType AsyncTracker::GetResourceHazardType(
    int64_t resource_type) const {
  return ResourceHazardType::kUnshareable;
}
absl::string_view AsyncTracker::GetResourceUsageName(
    int64_t resource_usage_type) const {
  switch (resource_usage_type) {
    case ResourceUsageTypeToIndex(ResourceUsageType::kNoResource):
      return "kNoResource";
    case ResourceUsageTypeToIndex(ResourceUsageType::kResourceOccupy):
      return "kResourceOccupy";
    case ResourceUsageTypeToIndex(ResourceUsageType::kResourceRelease):
      return "kResourceRelease";
    default:
      return "Not a valid resource usage type";
  }
}
int64_t AsyncTracker::GetNumTargetDefinedResources() const { return 0; }
int64_t AsyncTracker::GetNumAvailableResources(int64_t resource_type) const {
  return 0;
}
absl::InlinedVector<int64_t, 1>
AsyncTracker::GetReleasedShareableResourcesFromVector(
    const ResourcesVector& resources) const {
  return {};
}
absl::InlinedVector<int64_t, 1>
AsyncTracker::GetOccupiedShareableResourcesFromVector(
    const ResourcesVector& resources) const {
  return {};
}
absl::InlinedVector<int64_t, 1>
AsyncTracker::GetOccupiedSerialResourcesFromVector(
    const ResourcesVector& resources) const {
  return {};
}
absl::InlinedVector<int64_t, 1>
AsyncTracker::GetReleasedNonextendableResourcesFromVector(
    const ResourcesVector& resources) const {
  return {};
}
bool AsyncTracker::ReleasesSelectiveResource(const HloGraphNode* node) const {
  return absl::c_any_of(
      node->GetResources(), [&](const ResourcePair& resource) {
        return resource.second == ResourceUsageType::kResourceRelease &&
               GetResourceHazardType(resource.first) ==
                   ResourceHazardType::kSelective;
      });
}
bool AsyncTracker::OccupiesSelectiveResource(const HloGraphNode* node) const {
  return absl::c_any_of(
      node->GetResources(), [&](const ResourcePair& resource) {
        return resource.second == ResourceUsageType::kResourceOccupy &&
               GetResourceHazardType(resource.first) ==
                   ResourceHazardType::kSelective;
      });
}
BufferInfoTracker::BufferInfoTracker(
    const HloModule* module, const HloAliasAnalysis* alias_analysis,
    const HloCostAnalysis::ShapeSizeFunction& shape_size_bytes) {
  buffer_infos_.resize(alias_analysis->buffers().back().id() + 1);
  std::function<void(const HloComputation*)> process_computation =
      [&process_computation, module, alias_analysis, this,
       &shape_size_bytes](const HloComputation* computation) {
        const HloInstructionSequence& sequence =
            module->schedule().sequence(computation);
        for (int idx = 0; idx < sequence.size(); ++idx) {
          const HloInstruction* instruction = sequence.instructions()[idx];
          for (auto* called_computation : instruction->called_computations()) {
            if (called_computation->IsFusionComputation()) {
              continue;
            }
            process_computation(called_computation);
          }
          ShapeUtil::ForEachSubshape(
              instruction->shape(),
              [&](const Shape& subshape, const ShapeIndex& index) {
                for (const HloBuffer* buffer :
                     alias_analysis->ComputeBuffersAt(instruction, index)) {
                  if (buffer_infos_[buffer->id()].value == nullptr) {
                    buffer_infos_[buffer->id()] =
                        CreateBufferInfo(buffer, instruction, shape_size_bytes);
                  }
                }
              });
        }
      };
  process_computation(module->entry_computation());
}
void ModulePressureState::InitializePressureStates() {
  memory_pressure_states_.clear();
  std::function<void(HloComputation*,
                     const MemoryPressureTracker::LiveBufferSet&)>
      process_computation = [this, &process_computation](
                                HloComputation* computation,
                                const MemoryPressureTracker::LiveBufferSet&
                                    initial_live_buffers) {
        const HloInstructionSequence& sequence =
            module_->schedule().sequence(computation);
        MemoryPressureTracker tracker(hlo_alias_analysis_, buffer_tracker_,
                                      memory_pressure_states_);
        tracker.Initialize(computation, initial_live_buffers);
        VLOG(6) << "Pressure at bottom for " << computation->name() << ": "
                << tracker.memory_usage();
        for (int idx = sequence.size() - 1; idx >= 0; --idx) {
          const HloInstruction* instruction = sequence.instructions()[idx];
          if (!instruction->called_computations().empty()) {
            for (auto* called_computation :
                 instruction->called_computations()) {
              if (called_computation->IsFusionComputation()) {
                continue;
              }
              process_computation(called_computation, tracker.live_buffers());
            }
          }
          VLOG(10) << "Instruction: " << instruction->ToString();
          VLOG(10) << "Pressure change: "
                   << tracker.MemoryPressureDifference(instruction).first;
          VLOG(10) << "Current usage: " << tracker.memory_usage();
          tracker.UpdateBuffers(instruction);
          VLOG(10) << "Current usage after update: " << tracker.memory_usage();
          VLOG(10) << "Current peak after update: "
                   << tracker.pressure_state().memory_peak;
        }
        VLOG(6) << "Pressure peak for " << computation->name() << ": "
                << tracker.pressure_state().memory_peak;
        UpdatePressureStateForComputation(computation,
                                          tracker.pressure_state());
      };
  process_computation(module_->entry_computation(), {});
}
void MemoryPressureTracker::Initialize(
    const HloComputation* computation,
    const LiveBufferSet& initial_live_buffers) {
  live_memory_usage_ = 0;
  initial_memory_pressure_ = 0;
  pressure_state_ = MemoryPressureState{};
  output_buffers_.clear();
  defined_buffers_.clear();
  live_buffers_set_.clear();
  for (auto* instruction : computation->instructions()) {
    auto& output_values = this->output_buffers_[instruction];
    auto& defined_values = this->defined_buffers_[instruction];
    ShapeUtil::ForEachSubshape(
        instruction->shape(),
        [&](const Shape& subshape, const ShapeIndex& index) {
          for (const HloBuffer* buffer :
               hlo_alias_analysis_->ComputeBuffersAt(instruction, index)) {
            output_values.push_back(std::make_pair(
                buffer_tracker_.GetBufferInfo(buffer->id()), index));
            if (absl::c_any_of(buffer->values(), [&](const HloValue* value) {
                  return InstructionDefinesValue(instruction, value);
                })) {
              defined_values.push_back(
                  buffer_tracker_.GetBufferInfo(buffer->id()));
            }
          }
        });
  }
  if (!initial_live_buffers.empty()) {
    for (HloBuffer::Id id : initial_live_buffers) {
      auto& buffer = buffer_tracker_.GetBufferInfo(id);
      if (buffer.value->values()[0]->shape().has_layout() &&
          buffer.value->values()[0]->shape().layout().memory_space() != 0) {
        continue;
      }
      live_buffers_[buffer.value->id()] = 1;
      initial_memory_pressure_ += buffer.buffer_size;
    }
    live_buffers_set_ = initial_live_buffers;
  } else {
    absl::c_fill(live_buffers_, 0);
  }
  pressure_state_.live_ids_at_bottom = live_buffers_set_;
}
void MemoryPressureTracker::UpdateBuffers(const HloInstruction* instruction) {
  int64_t computations_peak = 0;
  for (auto* called_comp : instruction->called_computations()) {
    if (called_comp->IsFusionComputation()) {
      continue;
    }
    auto it = pressure_state_cache_.find(called_comp);
    CHECK(it != pressure_state_cache_.end());
    computations_peak = std::max(computations_peak, it->second.memory_peak);
  }
  if (pressure_state_.memory_peak < live_memory_usage_ + computations_peak) {
    pressure_state_.memory_peak = live_memory_usage_ + computations_peak;
  }
  for (auto* op : instruction->operands()) {
    auto& output_values = output_buffers_[op];
    for (auto& info : output_values) {
      if (ShouldSkipBufferAllocations(instruction, info.second,
                                      info.first.first_definition) ||
          (info.first.value->values()[0]->shape().has_layout() &&
           info.first.value->values()[0]->shape().layout().memory_space() !=
               kDefaultMemorySpace)) {
        continue;
      }
      if (live_buffers_[info.first.value->id()] == 0) {
        live_buffers_[info.first.value->id()] = 1;
        live_buffers_set_.insert(info.first.value->id());
        live_memory_usage_ += info.first.buffer_size;
      }
    }
  }
  pressure_state_.memory_peak =
      std::max(live_memory_usage_, pressure_state_.memory_peak);
  auto it = defined_buffers_.find(instruction);
  CHECK(it != defined_buffers_.end());
  if (!ShouldSkipBufferReleases(instruction)) {
    for (auto& b : it->second) {
      if (b.value->values()[0]->shape().has_layout() &&
          b.value->values()[0]->shape().layout().memory_space() !=
              kDefaultMemorySpace) {
        continue;
      }
      if (live_buffers_[b.value->id()] != 0) {
        if (InstructionFirstDefinesBuffer(instruction, b)) {
          live_memory_usage_ -= b.buffer_size;
          live_buffers_set_.erase(b.value->id());
        }
      }
    }
  }
}
std::pair<int64_t, int64_t> MemoryPressureTracker::MemoryPressureDifference(
    const HloInstruction* instruction) const {
  int64_t increase = 0;
  int64_t peak = 0;
  if (!instruction->called_computations().empty()) {
    int64_t called_comp_peak = 0;
    for (auto* called_comp : instruction->called_computations()) {
      if (called_comp->IsFusionComputation()) {
        continue;
      }
      auto it = pressure_state_cache_.find(called_comp);
      CHECK(it != pressure_state_cache_.end());
      peak = called_comp_peak =
          std::max(called_comp_peak, it->second.memory_peak);
    }
  }
  for (auto* op : instruction->operands()) {
    auto it = output_buffers_.find(op);
    CHECK(it != output_buffers_.end());
    for (auto& b : it->second) {
      if (ShouldSkipBufferAllocations(instruction, b.second,
                                      b.first.first_definition) ||
          (b.first.value->values()[0]->shape().has_layout() &&
           b.first.value->values()[0]->shape().layout().memory_space() !=
               kDefaultMemorySpace)) {
        continue;
      }
      if (!live_buffers_[b.first.value->id()]) {
        increase += b.first.buffer_size;
      }
    }
  }
  peak = std::max(increase, peak);
  auto it = defined_buffers_.find(instruction);
  CHECK(it != defined_buffers_.end());
  if (!ShouldSkipBufferReleases(instruction)) {
    for (auto& b : it->second) {
      if (b.value->values()[0]->shape().has_layout() &&
          b.value->values()[0]->shape().layout().memory_space() !=
              kDefaultMemorySpace) {
        continue;
      }
      if (live_buffers_[b.value->id()]) {
        if (InstructionFirstDefinesBuffer(instruction, b)) {
          increase -= b.buffer_size;
        }
      }
    }
  }
  return std::make_pair(increase, peak);
}
DefaultSchedulerCore::ScheduleCandidate InitializeCandidate(
    HloGraphNode* node,
    const DefaultSchedulerCore::SchedulingState& sched_state) {
  DefaultSchedulerCore::ScheduleCandidate cand;
  cand.node = node;
  return cand;
}
namespace {
int64_t GetNumHopsToClosestSelectiveOverlap(
    const DefaultSchedulerCore::ReadyQueueSet& ready_set,
    const HloGraphNode* node) {
  int64_t num_hops_to_closest_selective_resource_occupier =
      std::numeric_limits<int64_t>::max();
  for (const HloGraphNode* n : ready_set) {
    if (n == node) {
      continue;
    }
    num_hops_to_closest_selective_resource_occupier =
        std::min(num_hops_to_closest_selective_resource_occupier,
                 n->GetNumHopsToClosestSelectiveResourceOccupier());
  }
  return num_hops_to_closest_selective_resource_occupier;
}
class ReadySetLt {
 public:
  explicit ReadySetLt(
      const DefaultSchedulerCore::SchedulingState* sched_state,
      DefaultSchedulerCore::TargetSchedulingRule target_scheduling_rule,
      DefaultSchedulerCore::TargetSchedulingRule early_target_scheduling_rule)
      : sched_state_(*sched_state),
        target_scheduling_rule_(target_scheduling_rule),
        early_target_scheduling_rule_(early_target_scheduling_rule) {}
  DefaultSchedulerCore::CandidateResult operator()(
      DefaultSchedulerCore::ScheduleCandidate& a,
      DefaultSchedulerCore::ScheduleCandidate& b) const {
    if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
            a.node->GetForceEarly(), a, b.node->GetForceEarly(), b,
            "kForceEarly")) {
      return *value;
    }
    if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
            !a.node->GetForceDelay(), a, !b.node->GetForceDelay(), b,
            "kForceDelay")) {
      return *value;
    }
    if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
            IsNop(*a.node), a, IsNop(*b.node), b, "kIsNop")) {
      return *value;
    }
    std::pair<int64_t, int64_t> a_increase = std::make_pair(0LL, 0LL);
    std::pair<int64_t, int64_t> b_increase = std::make_pair(0LL, 0LL);
    if (sched_state_.config.memory_limit != UINT64_MAX &&
        sched_state_.memory_pressure_tracker->memory_usage() >
            (sched_state_.config.memory_limit / 2)) {
      a_increase = GetMemoryPressureChanges(a);
      b_increase = GetMemoryPressureChanges(b);
      if (sched_state_.memory_pressure_tracker->memory_usage() >=
          sched_state_.config.memory_limit) {
        if (sched_state_.config.depth_based_memory_pressure_reduction) {
          if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
                  a_increase.first < 0 && a_increase.first < b_increase.first,
                  a,
                  b_increase.first < 0 && b_increase.first < a_increase.first,
                  b, "kOnlyDecreaseMemoryOverLimit")) {
            return *value;
          }
          if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
                  a.node->GetGraphDepth() > b.node->GetGraphDepth(), a,
                  b.node->GetGraphDepth() > a.node->GetGraphDepth(), b,
                  "kDepthOverLimit")) {
            return *value;
          }
        }
        if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
                a_increase.first < b_increase.first, a,
                b_increase.first < a_increase.first, b,
                "kDecreaseMemoryOverLimit")) {
          return *value;
        }
      }
      if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
              a_increase.second +
                      sched_state_.memory_pressure_tracker->memory_usage() <=
                  sched_state_.config.memory_limit,
              a,
              b_increase.second +
                      sched_state_.memory_pressure_tracker->memory_usage() <=
                  sched_state_.config.memory_limit,
              b, "kMemoryPeakOverLimit")) {
        return *value;
      }
    }
    if (early_target_scheduling_rule_) {
      if (auto value = early_target_scheduling_rule_(a, b)) {
        return *value;
      }
    }
    if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
            ShouldScheduleAsyncDone(a), a, ShouldScheduleAsyncDone(b), b,
            "kScheduleDone")) {
      return *value;
    }
    if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
            PastDueCyclesForNonextendableResource(a) >
                PastDueCyclesForNonextendableResource(b),
            a,
            PastDueCyclesForNonextendableResource(b) >
                PastDueCyclesForNonextendableResource(a),
            b, "kReleaseNonextendable")) {
      return *value;
    }
    if (sched_state_.config.enable_release_start_policy) {
      const ApproximateLatencyEstimator::TimeCost a_ready_interval =
          a.node->GetReadyTime() - sched_state_.current_time;
      const ApproximateLatencyEstimator::TimeCost b_ready_interval =
          b.node->GetReadyTime() - sched_state_.current_time;
      bool a_ready_and_release =
          a_ready_interval <= 0 &&
          a.node->DoesReleaseResource(ResourceType::kCollectivePermute);
      bool b_ready_and_release =
          b_ready_interval <= 0 &&
          b.node->DoesReleaseResource(ResourceType::kCollectivePermute);
      if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
              a_ready_and_release, a, b_ready_and_release, b,
              "kScheduleStart")) {
        return *value;
      }
      if (a_ready_and_release && b_ready_and_release) {
        if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
                a_ready_interval < b_ready_interval, a,
                b_ready_interval < a_ready_interval, b, "kScheduleStart")) {
          return *value;
        }
      }
    }
    auto async_depth_0_candidate =
        [this](DefaultSchedulerCore::ScheduleCandidate& a,
               DefaultSchedulerCore::ScheduleCandidate& b)
        -> std::optional<DefaultSchedulerCore::CandidateResult> {
      if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
              !(a.node->DoesReleaseAnyResource() &&
                               a.node->GetAsyncDepth() == 0 &&
                               !IsResourceConstrained(a)),
              a,
              !(b.node->DoesReleaseAnyResource() &&
                b.node->GetAsyncDepth() == 0 && !IsResourceConstrained(b)),
              b, "kStartAtZeroDepth")) {
        return value;
      }
      return std::nullopt;
    };
    if (sched_state_.config.aggressive_scheduling_policies &&
        sched_state_.config.prioritize_async_depth_over_stall) {
      if (auto value = async_depth_0_candidate(a, b)) {
        return *value;
      }
    }
    const ApproximateLatencyEstimator::TimeCost a_ready_interval =
        std::max(a.node->GetReadyTime() - sched_state_.current_time, 0.0);
    const ApproximateLatencyEstimator::TimeCost b_ready_interval =
        std::max(b.node->GetReadyTime() - sched_state_.current_time, 0.0);
    if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
            a_ready_interval < b_ready_interval, a,
            b_ready_interval < a_ready_interval, b, "kLessStall")) {
      return *value;
    }
    if (sched_state_.config.resource_serializing) {
      const int64_t a_num_conflicting_resources =
          GetNumConflictingSerialResources(a);
      const int64_t b_num_conflicting_resources =
          GetNumConflictingSerialResources(b);
      if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
              a_num_conflicting_resources < b_num_conflicting_resources, a,
              b_num_conflicting_resources < a_num_conflicting_resources, b,
              "kLessSerialResourceConflict")) {
        return *value;
      }
    }
    if (sched_state_.config.aggressive_scheduling_policies &&
        !sched_state_.config.prioritize_async_depth_over_stall) {
      if (auto value = async_depth_0_candidate(a, b)) {
        return *value;
      }
    }
    if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
            a.node->DoesReleaseAnyResource() && IsResourceConstrained(a), a,
            b.node->DoesReleaseAnyResource() && IsResourceConstrained(b), b,
            "kFreeBackedupResource")) {
      return *value;
    }
    if (sched_state_.config.aggressive_scheduling_policies) {
      if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
              a.node->GetAsyncDepth() > b.node->GetAsyncDepth(), a,
              b.node->GetAsyncDepth() > a.node->GetAsyncDepth(), b,
              "kAsyncDepth")) {
        return *value;
      }
      if (!sched_state_.next_ready_stack.empty()) {
        HloGraphNode::TimeCost latest_ready =
            sched_state_.next_ready_stack.front()->GetReadyTime();
        HloGraphNode::TimeCost a_cost_diff = std::abs(
            latest_ready - sched_state_.current_time - a.node->GetCost());
        HloGraphNode::TimeCost b_cost_diff = std::abs(
            latest_ready - sched_state_.current_time - b.node->GetCost());
        if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
                !a.node->DoesReleaseAnyResource() && a_cost_diff < b_cost_diff,
                a,
                !b.node->DoesReleaseAnyResource() && b_cost_diff < a_cost_diff,
                b, "kAvoidWaste")) {
          return *value;
        }
      }
    }
    bool a_operands = absl::c_any_of(
        a.node->GetInstr().operands(),
        [async_tracker = sched_state_.async_tracker](const HloInstruction* i) {
          return async_tracker->IsSupportedAsyncDone(*i);
        });
    bool b_operands = absl::c_any_of(
        b.node->GetInstr().operands(),
        [async_tracker = sched_state_.async_tracker](const HloInstruction* i) {
          return async_tracker->IsSupportedAsyncDone(*i);
        });
    if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
            a_operands, a, b_operands, b, "kUnlockDone")) {
      return *value;
    }
    if (target_scheduling_rule_) {
      if (auto value = target_scheduling_rule_(a, b)) {
        return *value;
      }
    }
    if (sched_state_.config.enable_selective_resources &&
        sched_state_.selective_resource_releasers.empty()) {
      int64_t distance_to_selective_overlap_for_a =
          GetNumHopsToClosestSelectiveOverlap(sched_state_.ready_set, a.node);
      int64_t distance_to_selective_overlap_for_b =
          GetNumHopsToClosestSelectiveOverlap(sched_state_.ready_set, b.node);
      int64_t max_distance =
          sched_state_.config.max_hops_to_closest_selective_overlap;
      if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
              (a.node->GetValuableForSelectiveOverlap() &&
               distance_to_selective_overlap_for_a <= max_distance),
              b,
              (b.node->GetValuableForSelectiveOverlap() &&
               distance_to_selective_overlap_for_b <= max_distance),
              a, "kNotValuableForSelectiveOverlap")) {
        return *value;
      }
    }
    if (sched_state_.config.aggressive_scheduling_policies) {
      int ready_if_a_scheduled = ReadyIfScheduled(*a.node);
      int ready_if_b_scheduled = ReadyIfScheduled(*b.node);
      if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
              ready_if_a_scheduled > ready_if_b_scheduled, a,
              ready_if_b_scheduled > ready_if_a_scheduled, b,
              "kCreatesMoreReadyNodes")) {
        return *value;
      }
    }
    if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
            a_increase.first < 0, a, b_increase.first < 0, b,
            "kDecreaseMemory")) {
      return *value;
    }
    if (sched_state_.sched_graph.OriginalInstructionPosition(
            &a.node->GetInstr()) >
        sched_state_.sched_graph.OriginalInstructionPosition(
            &b.node->GetInstr())) {
      return {a, "kOriginalOrder"};
    }
    return {b, "kOriginalOrder"};
  }
 private:
  const DefaultSchedulerCore::SchedulingState& sched_state_;
  DefaultSchedulerCore::TargetSchedulingRule target_scheduling_rule_;
  DefaultSchedulerCore::TargetSchedulingRule early_target_scheduling_rule_;
  int ReadyIfScheduled(const HloGraphNode& gn) const {
    int ready_nodes_if_scheduled = 0;
    for (auto& pred : gn.GetPredecessors()) {
      if (pred.Target().GetOutdegree() == 1) {
        ++ready_nodes_if_scheduled;
      }
    }
    return ready_nodes_if_scheduled;
  }
  static bool IsNop(const HloGraphNode& gn) {
    return IsNopInstruction(gn.GetInstr());
  }
  bool IsResourceConstrained(
      DefaultSchedulerCore::ScheduleCandidate& cand) const {
    if (cand.resource_constrained) {
      return *cand.resource_constrained;
    }
    if (cand.node->GetResources().empty()) {
      cand.resource_constrained = false;
      return *(cand.resource_constrained);
    }
    cand.resource_constrained = false;
    for (const auto& [resource_type, usage_type] : cand.node->GetResources()) {
      auto max_it = sched_state_.max_concurrent_resource.find(resource_type);
      auto res_it = sched_state_.resource_users_in_queue.find(resource_type);
      cand.resource_constrained =
          max_it != sched_state_.max_concurrent_resource.end() &&
          max_it->second == 0 &&
          res_it != sched_state_.resource_users_in_queue.end() &&
          res_it->second > 0;
      if (*cand.resource_constrained) {
        return *cand.resource_constrained;
      }
    }
    return *cand.resource_constrained;
  }
  bool ShouldScheduleAsyncDone(
      DefaultSchedulerCore::ScheduleCandidate& gn_cand) const {
    if (!gn_cand.node->DoesOccupyAnyResource()) {
      return false;
    }
    return !ShouldDelaySendHostDone(gn_cand);
  }
  HloGraphNode::TimeCost PastDueCyclesForNonextendableResource(
      DefaultSchedulerCore::ScheduleCandidate& cand) const {
    if (sched_state_.async_tracker
            ->GetReleasedNonextendableResourcesFromVector(
                cand.node->GetResources())
            .empty()) {
      return 0.0;
    }
    return std::max(sched_state_.current_time - cand.node->GetReadyTime(), 0.0);
  }
  bool ShouldDelaySendHostDone(
      DefaultSchedulerCore::ScheduleCandidate& gn_cand) const {
    const HloGraphNode& gn = *gn_cand.node;
    if (!gn.UsesResourceType(ResourceType::kSendHost).has_value() ||
        gn.GetInstr().opcode() != HloOpcode::kSendDone) {
      return false;
    }
    const HloGraphNode& start =
        sched_state_.sched_graph.GetNode(gn.GetInstr().operand(0));
    const LatencyEstimator::TimeCost latency =
        sched_state_.latency_estimator->GetLatencyBetween(start, gn);
    if (!gn_cand.estimated_connected_send_ready_time.has_value()) {
      HloGraphNode::TimeCost start_ready_time = 0;
      for (const auto& succ : start.GetSuccessors()) {
        if (succ.Target().GetReadyTime() >=
            std::numeric_limits<HloGraphNode::TimeCost>::max()) {
          return false;
        }
        start_ready_time = std::max(
            start_ready_time, succ.Latency() + succ.Target().GetReadyTime());
      }
      gn_cand.estimated_connected_send_ready_time = start_ready_time;
    }
    if (*gn_cand.estimated_connected_send_ready_time -
            sched_state_.current_time <=
        latency) {
      return false;
    }
    return true;
  }
  std::pair<int64_t, int64_t> GetMemoryPressureChanges(
      DefaultSchedulerCore::ScheduleCandidate& cand) const {
    if (cand.pressure_change) {
      return *cand.pressure_change;
    }
    std::optional<std::pair<int64_t, int64_t>> start_result;
    if (this->sched_state_.async_tracker->IsSupportedAsyncDone(
            cand.node->GetInstr())) {
      const HloInstruction* start = cand.node->GetInstr().operand_count() > 0
                                        ? cand.node->GetInstr().operand(0)
                                        : nullptr;
      if (start != nullptr &&
          this->sched_state_.async_tracker->IsSupportedAsyncStart(*start)) {
        start_result =
            sched_state_.memory_pressure_tracker->MemoryPressureDifference(
                start);
      }
    }
    cand.pressure_change =
        sched_state_.memory_pressure_tracker->MemoryPressureDifference(
            &cand.node->GetInstr());
    if (start_result.has_value()) {
      cand.pressure_change->first =
          std::min(start_result->first, cand.pressure_change->first);
      cand.pressure_change->second =
          std::max(start_result->second, cand.pressure_change->second);
    }
    return *cand.pressure_change;
  }
  int64_t GetNumConflictingSerialResources(
      DefaultSchedulerCore::ScheduleCandidate& cand) const {
    auto resources =
        sched_state_.async_tracker->GetOccupiedSerialResourcesFromVector(
            cand.node->GetResources());
    int64_t num_conflicting_resources = 0;
    for (int64_t resource : resources) {
      if (!sched_state_.resources_in_flight.contains(resource)) continue;
      num_conflicting_resources +=
          sched_state_.resources_in_flight.at(resource);
    }
    return num_conflicting_resources;
  }
};
enum SkipNodeReason {
  kShouldSkipNodeFunction,
  kExceedsOverlapLimit,
};
absl::string_view SkipNodeReasonString(SkipNodeReason reason) {
  switch (reason) {
    case SkipNodeReason::kShouldSkipNodeFunction:
      return "Skipped due to kShouldSkipNodeFunction.";
    case SkipNodeReason::kExceedsOverlapLimit:
      return "Skipped due to kExceedsOverlapLimit.";
  }
}
}  
absl::StatusOr<HloGraphNode*>
DefaultSchedulerCore::FindAndExtractBestNodeAvailable(
    DefaultSchedulerCore::SchedulingState& sched_state,
    DefaultSchedulerCore::ShouldSkipNodeFunction should_skip_node) {
  absl::InlinedVector<std::pair<HloGraphNode*, SkipNodeReason>, 2>
      skipped_nodes_and_reasons;
  auto scheduling_instruction_crosses_overlap_limit =
      [&sched_state](const HloInstruction& instr) {
        for (const auto& [resource, limit] :
             sched_state.max_concurrent_resource) {
          auto it = sched_state.resources_in_flight.find(resource);
          if (it == sched_state.resources_in_flight.end() || it->second == 0) {
            continue;
          }
          const int64_t num_resources_needed =
              sched_state.async_tracker->GetNumResourcesPerInstruction(resource,
                                                                       instr);
          if (limit < num_resources_needed) {
            return true;
          }
        }
        return false;
      };
  VLOG(2) << "Current time: " << sched_state.current_time;
  ReadySetLt ready_lt{&sched_state, target_scheduling_rule_,
                      early_target_scheduling_rule_};
  ScheduleCandidate ready_chosen;
  auto chosen_it = sched_state.ready_set.end();
  for (auto ready_node_it = sched_state.ready_set.begin(),
            e = sched_state.ready_set.end();
       ready_node_it != e; ++ready_node_it) {
    if (should_skip_node && should_skip_node(*ready_node_it)) {
      if (ready_chosen.node == nullptr) {
        skipped_nodes_and_reasons.push_back(
            {*ready_node_it, SkipNodeReason::kShouldSkipNodeFunction});
      }
      continue;
    }
    if (scheduling_instruction_crosses_overlap_limit(
            (*ready_node_it)->GetInstr())) {
      if (ready_chosen.node == nullptr) {
        skipped_nodes_and_reasons.push_back(
            {*ready_node_it, SkipNodeReason::kExceedsOverlapLimit});
      }
      continue;
    }
    ScheduleCandidate ready_candidate =
        InitializeCandidate(*ready_node_it, sched_state);
    if (ready_chosen.node == nullptr) {
      ready_chosen = ready_candidate;
      chosen_it = ready_node_it;
      VLOG(2) << "Choosing from ready (" << ready_chosen.node->GetInstr().name()
              << ") Reason: First Candidate";
      continue;
    }
    CandidateResult cand_result = ready_lt(ready_candidate, ready_chosen);
    const bool new_candidate_selected =
        cand_result.result.node == *ready_node_it;
    auto print_pressure_change =
        [](const std::optional<std::pair<int64_t, int64_t>>& p) {
          if (p.has_value()) {
            return std::to_string(p.value().first);
          }
          return std::string("N/A");
        };
    VLOG(2) << "Choosing from ready ("
            << (new_candidate_selected ? ready_candidate.node->GetInstr().name()
                                       : ready_chosen.node->GetInstr().name())
            << ") vs ("
            << (new_candidate_selected
                    ? ready_chosen.node->GetInstr().name()
                    : ready_candidate.node->GetInstr().name())
            << ") Reason: " << cand_result.reason << " mem pressure chosen "
            << print_pressure_change(
                   (new_candidate_selected ? ready_candidate : ready_chosen)
                       .pressure_change)
            << " mem pressure other "
            << print_pressure_change(
                   (new_candidate_selected ? ready_chosen : ready_candidate)
                       .pressure_change);
    if (new_candidate_selected) {
      ready_chosen = cand_result.result;
      chosen_it = ready_node_it;
    }
  }
  if (ready_chosen.node == nullptr) {
    return absl::InternalError(absl::StrCat(
        "FindAndExtractBestNodeAvailable failed to find a node to "
        "schedule, skipped nodes: ",
        absl::StrJoin(skipped_nodes_and_reasons, "; ",
                      [](std::string* out, const auto& pair) {
                        absl::StrAppend(out, pair.first->GetInstr().name(),
                                        ": ",
                                        SkipNodeReasonString(pair.second));
                      })));
  }
  CHECK(chosen_it != sched_state.ready_set.end());
  std::swap(*chosen_it, sched_state.ready_set.back());
  sched_state.ready_set.pop_back();
  return ready_chosen.node;
}
void DefaultSchedulerCore::LogInstruction(const HloInstruction* instr) const {
  VLOG(5) << instr->ToString();
}
void PrintOccupierList(
    std::vector<std::pair<HloEdge*, HloGraphNode::TimeCost>>& occupiers) {
  for (int64_t i = 0; i < occupiers.size(); i++) {
    VLOG(3) << "\tOccupier " << i << ": "
            << occupiers[i].first->Target().GetInstr().name()
            << ", projected finish time: " << occupiers[i].second
            << " original latency: " << occupiers[i].first->OriginalLatency()
            << " latency: " << occupiers[i].first->Latency();
  }
}
bool DefaultSchedulerCore::DeleteOccupierFromResource(
    HloGraphNode::TimeCost current_time, HloEdge& edge,
    std::vector<std::pair<HloEdge*, HloGraphNode::TimeCost>>& occupiers) {
  if (absl::c_any_of(
          occupiers,
          [&edge](const std::pair<HloEdge*, HloGraphNode::TimeCost>& element) {
            return element.first == &edge;
          }) == false) {
    return false;
  }
  std::vector<std::pair<HloEdge*, HloGraphNode::TimeCost>>::iterator it =
      occupiers.begin();
  int64_t num_occupiers = occupiers.size();
  HloGraphNode::TimeCost prev_time = current_time;
  HloGraphNode::TimeCost accumulated_saved_time = 0;
  while (it != occupiers.end() && it->first != &edge) {
    if (it->second <= current_time) {
      num_occupiers--;
      it++;
      continue;
    }
    HloGraphNode::TimeCost remaining_time_of_edge = it->second - prev_time;
    prev_time = it->second;
    CHECK_GT(num_occupiers, 0);
    HloGraphNode::TimeCost current_saved_time =
        remaining_time_of_edge / num_occupiers;
    accumulated_saved_time += current_saved_time;
    CHECK_GE(it->second, accumulated_saved_time);
    it->second -= accumulated_saved_time;
    num_occupiers--;
    it++;
  }
  CHECK(it != occupiers.end());  
  if (it->second > current_time) {
    HloGraphNode::TimeCost remaining_time_of_edge = it->second - prev_time;
    HloGraphNode::TimeCost current_saved_time =
        remaining_time_of_edge / num_occupiers;
    accumulated_saved_time += current_saved_time;
  }
  it = occupiers.erase(it);
  for (; it != occupiers.end(); it++) {
    it->second -= accumulated_saved_time;
  }
  return true;
}
bool DefaultSchedulerCore::AddOccupierToResource(
    HloGraphNode::TimeCost current_time, HloEdge& new_edge,
    std::vector<std::pair<HloEdge*, HloGraphNode::TimeCost>>& occupiers) {
  CHECK(new_edge.OriginalLatency() > 0 && current_time >= 0);
  auto new_edge_remaining = new_edge.OriginalLatency();
  std::vector<std::pair<HloEdge*, HloGraphNode::TimeCost>>::iterator it =
      occupiers.begin();
  int64_t num_occupiers = occupiers.size();
  HloGraphNode::TimeCost prev_time = current_time;
  HloGraphNode::TimeCost accumulated_delay = 0;
  while (it != occupiers.end() &&
         it->second - prev_time <= new_edge_remaining * num_occupiers) {
    if (it->second <= current_time) {
      num_occupiers--;
      it++;
      continue;
    }
    HloGraphNode::TimeCost remaining_time_of_edge = it->second - prev_time;
    prev_time = it->second;
    CHECK_GT(num_occupiers, 0);
    HloGraphNode::TimeCost current_delay =
        remaining_time_of_edge / num_occupiers;
    new_edge_remaining -= current_delay;
    accumulated_delay += current_delay;
    it->second += accumulated_delay;
    num_occupiers--;
    it++;
  }
  num_occupiers++;
  HloGraphNode::TimeCost adjusted_remaining_time =
      new_edge_remaining * num_occupiers;
  it = occupiers.insert(
      it, std::make_pair(&new_edge, prev_time + accumulated_delay +
                                        adjusted_remaining_time));
  it++;
  accumulated_delay += new_edge_remaining;
  CHECK(new_edge.OriginalLatency() - 0.0001 < accumulated_delay &&
        accumulated_delay < new_edge.OriginalLatency() + 0.0001);
  for (; it != occupiers.end(); it++) {
    it->second += accumulated_delay;
  }
  return true;
}
absl::StatusOr<HloGraphNode::TimeCost> DefaultSchedulerCore::ScheduleNode(
    HloGraphNode* n, DefaultSchedulerCore::SchedulingState* sched_state) const {
  sched_state->new_sequence_reversed.push_back(
      const_cast<HloInstruction*>(&n->GetInstr()));
  n->SetScheduled();
  if (sched_state->config.enable_selective_resources &&
      n->ReleasesSelectiveResource()) {
    auto it = std::find(sched_state->selective_resource_releasers.begin(),
                        sched_state->selective_resource_releasers.end(), n);
    if (it == sched_state->selective_resource_releasers.end()) {
      LOG(WARNING) << "Selective resource releasers list does not contain node "
                      "that releases a selective resource: "
                   << n->ToString();
    } else {
      sched_state->selective_resource_releasers.erase(it);
    }
  }
  if (sched_state->config.enable_selective_resources &&
      !n->GetValuableForSelectiveOverlap()) {
    for (HloGraphNode* node : sched_state->selective_resource_releasers) {
      node->SetReadyTime(node->GetReadyTime() + n->GetCost());
    }
  }
  for (auto& resource : n->GetResources()) {
    if (resource.second == ResourceUsageType::kResourceRelease) {
      ++(sched_state->max_concurrent_resource[resource.first]);
    } else if (resource.second == ResourceUsageType::kResourceOccupy) {
      --(sched_state->max_concurrent_resource[resource.first]);
      --(sched_state->resource_users_in_queue[resource.first]);
    }
  }
  HloGraphNode::TimeCost schedule_time = sched_state->current_time;
  for (const HloEdge& pred : n->GetSuccessors()) {
    const HloGraphNode::TimeCost time_from_edge =
        pred.Target().GetReadyTime() + pred.Latency();
    schedule_time = std::max(schedule_time, time_from_edge);
    if (sched_state->config.resource_sharing) {
      auto occupied_resources = n->GetShareableResourcesOnEdge(pred);
      for (const int64_t resource : occupied_resources) {
        auto occupiers = sched_state->shareable_resource_occupiers[resource];
        for (auto [occupier_edge, edge_pft] : occupiers) {
          if (occupier_edge == &pred) {
            VLOG(3) << "Ready time of scheduled node " << n->GetInstr().name()
                    << " before update with pft: " << edge_pft
                    << ", ready_time: " << schedule_time;
            schedule_time = std::max(schedule_time, edge_pft);
            VLOG(3) << "Ready time of scheduled node " << n->GetInstr().name()
                    << " after update with pft: " << edge_pft
                    << ", ready_time: " << schedule_time;
          }
        }
      }
    }
  }
  n->SetReadyTime(schedule_time);
  HloGraphNode::TimeCost current_time = schedule_time + n->GetCost();
  if (sched_state->config.resource_sharing) {
    for (HloEdge& edge : n->GetSuccessors()) {
      auto released_resources = n->GetShareableResourcesOnEdge(edge);
      for (const int64_t resource : released_resources) {
        CHECK(DeleteOccupierFromResource(
            schedule_time, edge,
            sched_state->shareable_resource_occupiers[resource]));
        if (VLOG_IS_ON(2)) {
          VLOG(3) << "Occupier list for "
                  << sched_state->async_tracker->GetResourceName(resource)
                  << ": ";
          PrintOccupierList(
              sched_state->shareable_resource_occupiers[resource]);
        }
      }
    }
    for (HloEdge& edge : n->GetPredecessors()) {
      for (HloEdge& inverse_edge : edge.Target().GetSuccessors()) {
        if (&(inverse_edge.Target()) == n) {
          auto occupied_resources =
              edge.Target().GetShareableResourcesOnEdge(inverse_edge);
          for (const int64_t resource : occupied_resources) {
            CHECK(AddOccupierToResource(
                current_time, inverse_edge,
                sched_state->shareable_resource_occupiers[resource]));
            if (VLOG_IS_ON(2)) {
              VLOG(3) << "Occupier list for "
                      << sched_state->async_tracker->GetResourceName(resource)
                      << ": ";
              PrintOccupierList(
                  sched_state->shareable_resource_occupiers[resource]);
            }
          }
          break;
        }
      }
    }
  }
  auto ready_time_cmp = [](const HloGraphNode* a, const HloGraphNode* b) {
    return a->GetReadyTime() > b->GetReadyTime();
  };
  while (!sched_state->next_ready_stack.empty()) {
    const HloGraphNode* node = sched_state->next_ready_stack.front();
    if (node->GetReadyTime() < current_time) {
      std::pop_heap(sched_state->next_ready_stack.begin(),
                    sched_state->next_ready_stack.end(), ready_time_cmp);
      sched_state->next_ready_stack.pop_back();
      continue;
    }
    break;
  }
  for (HloEdge& edge : n->GetPredecessors()) {
    const int64_t current_outdegree = edge.Target().GetOutdegree();
    if (current_outdegree != 1) {
      edge.Target().SetOutdegree(current_outdegree - 1);
      continue;
    }
    edge.Target().SetOutdegree(0);
    LatencyEstimator::TimeCost ready_time = current_time;
    for (const HloEdge& pred : edge.Target().GetSuccessors()) {
      const LatencyEstimator::TimeCost edge_time =
          pred.Target().GetReadyTime() + pred.Latency();
      ready_time = std::max(ready_time, edge_time);
      if (sched_state->config.resource_sharing) {
        auto occupied_resources =
            edge.Target().GetShareableResourcesOnEdge(pred);
        for (const int64_t resource : occupied_resources) {
          auto occupiers = sched_state->shareable_resource_occupiers[resource];
          for (auto [occupier_edge, edge_pft] : occupiers) {
            if (occupier_edge == &pred) {
              VLOG(3) << "Ready time of predecessor "
                      << edge.Target().GetInstr().name()
                      << " before update with pft: " << edge_pft
                      << ", ready_time: " << ready_time;
              ready_time = std::max(ready_time, edge_pft);
              VLOG(3) << "Ready time of predecessor "
                      << edge.Target().GetInstr().name()
                      << " after update with pft: " << edge_pft
                      << ", ready_time: " << ready_time;
            }
          }
        }
      }
    }
    for (auto& resource : edge.Target().GetResources()) {
      if (resource.second == ResourceUsageType::kResourceOccupy) {
        ++(sched_state->resource_users_in_queue[resource.first]);
      }
    }
    edge.Target().SetReadyTime(ready_time);
    sched_state->ready_set.push_back(&edge.Target());
    if (edge.Target().GetReadyTime() > current_time) {
      sched_state->next_ready_stack.push_back(&edge.Target());
      std::push_heap(sched_state->next_ready_stack.begin(),
                     sched_state->next_ready_stack.end(), ready_time_cmp);
    }
    if (sched_state->config.enable_selective_resources &&
        edge.Target().ReleasesSelectiveResource()) {
      sched_state->selective_resource_releasers.push_back(&edge.Target());
    }
  }
  ++sched_state->scheduled_count;
  for (auto& resource : n->GetResources()) {
    if (resource.second == ResourceUsageType::kResourceRelease) {
      --sched_state->resources_in_flight[resource.first];
    } else if (resource.second == ResourceUsageType::kResourceOccupy) {
      ++sched_state->resources_in_flight[resource.first];
    }
  }
  VLOG(10) << "Memory pressure before schedule: "
           << sched_state->memory_pressure_tracker->memory_usage();
  VLOG(10)
      << "Memory peak before schedule: "
      << sched_state->memory_pressure_tracker->pressure_state().memory_peak;
  sched_state->memory_pressure_tracker->UpdateBuffers(&n->GetInstr());
  VLOG(10) << "Memory pressure after schedule: "
           << sched_state->memory_pressure_tracker->memory_usage();
  VLOG(10)
      << "Memory peak after schedule: "
      << sched_state->memory_pressure_tracker->pressure_state().memory_peak;
  return current_time;
}
std::string HloEdge::ToString() const {
  return absl::StrCat("\tEdge: ", target_->GetInstr().name(),
                      " latency: ", Latency(), "\n");
}
bool HloScheduleGraph::IsPredecessorTransitively(
    const HloGraphNode* node, const HloGraphNode* possible_predecessor) {
  absl::flat_hash_set<const HloGraphNode*> visited = {possible_predecessor};
  std::vector<const HloGraphNode*> to_visit_queue = {node};
  while (!to_visit_queue.empty()) {
    const HloGraphNode* curr = to_visit_queue.back();
    to_visit_queue.pop_back();
    if (curr == possible_predecessor) {
      return true;
    }
    if (visited.contains(curr)) {
      continue;
    }
    visited.insert(curr);
    for (const auto& edge : curr->GetPredecessors()) {
      auto user_node_it = nodes_.find(&edge.Target().GetInstr());
      to_visit_queue.push_back(user_node_it->second.get());
    }
  }
  return false;
}
HloScheduleGraph::HloScheduleGraph(
    const std::vector<HloInstruction*>* post_order_instructions,
    HloAliasAnalysis* alias_analysis, const LatencyEstimator* latency_estimator,
    const AsyncTracker* async_tracker)
    : original_order_(post_order_instructions->begin(),
                      post_order_instructions->end()) {
  HloComputation* comp = (*post_order_instructions)[0]->parent();
  auto reachability = HloReachabilityMap::Build(comp);
  int64_t current_pos = 0;
  std::vector<const HloInstruction*> while_instrs;
  for (HloInstruction* instr : *post_order_instructions) {
    auto [new_node_it, inserted] = nodes_.try_emplace(
        instr, std::make_unique<HloGraphNode>(instr, current_pos));
    CHECK(inserted) << "Expected the value to not be already in the map";
    instr_order_map_[instr] = current_pos++;
    new_node_it->second->predecessors_.reserve(instr->operand_count());
    new_node_it->second->successors_.reserve(instr->user_count());
    new_node_it->second->cost_ = latency_estimator->NodeCost(instr);
    new_node_it->second->resources_ =
        async_tracker->GetResourcesFromInstruction(*instr);
    new_node_it->second->released_shareable_resources_ =
        async_tracker->GetReleasedShareableResourcesFromVector(
            new_node_it->second->GetResources());
    new_node_it->second->occupied_shareable_resources_ =
        async_tracker->GetOccupiedShareableResourcesFromVector(
            new_node_it->second->GetResources());
    new_node_it->second->releases_selective_resource_ =
        async_tracker->ReleasesSelectiveResource(new_node_it->second.get());
    new_node_it->second->occupies_selective_resource_ =
        async_tracker->OccupiesSelectiveResource(new_node_it->second.get());
    if (instr->opcode() == HloOpcode::kWhile) {
      while_instrs.push_back(instr);
    }
  }
  auto add_dependency_helper = [latency_estimator](HloGraphNode* from,
                                                   HloGraphNode* to) {
    const LatencyEstimator::TimeCost latency =
        latency_estimator->GetLatencyBetween(*from, *to);
    from->successors_.push_back(HloEdge(latency, to));
    to->predecessors_.push_back(HloEdge(latency, from));
    ++to->indegree_;
    ++from->outdegree_;
  };
  for (const HloInstruction* instr : *post_order_instructions) {
    auto node_it = nodes_.find(instr);
    CHECK(node_it != nodes_.end()) << "We should have just allocated a node";
    HloGraphNode* instr_node = node_it->second.get();
    VLOG(10) << "Adding users for " << instr_node->GetInstr().ToString();
    for (const HloInstruction* user : instr->users()) {
      VLOG(10) << "\tUser: " << user->ToString();
      auto user_node_it = nodes_.find(user);
      CHECK(user_node_it != nodes_.end());
      HloGraphNode* user_node = user_node_it->second.get();
      add_dependency_helper(instr_node, user_node);
    }
    for (const HloInstruction* ctrl_succ : instr->control_successors()) {
      VLOG(10) << "\tCtrl Successor: " << ctrl_succ->ToString();
      auto ctrl_succ_node_it = nodes_.find(ctrl_succ);
      CHECK(ctrl_succ_node_it != nodes_.end());
      HloGraphNode* ctrl_succ_node = ctrl_succ_node_it->second.get();
      add_dependency_helper(instr_node, ctrl_succ_node);
    }
    if (async_tracker->IsSupportedAsyncDone(*instr)) {
      const HloInstruction* async_start = instr->operand(0);
      if (alias_analysis != nullptr) {
        for (const HloBuffer* buffer :
             alias_analysis->ComputeBuffersAt(instr, {})) {
          for (const HloValue* value : buffer->values()) {
            if (value->defining_instruction() == instr) {
              continue;
            }
            for (const HloUse& use : value->GetUses()) {
              if (ContainsKey(instr_order_map_, use.instruction)) {
                if (use.instruction == async_start ||
                    reachability->IsReachable(instr, use.instruction)) {
                  continue;
                }
                auto it = nodes_.find(use.instruction);
                CHECK(it != nodes_.end());
                HloGraphNode* pred_node = it->second.get();
                it = nodes_.find(async_start);
                CHECK(it != nodes_.end());
                HloGraphNode* start_node = it->second.get();
                if (IsPredecessorTransitively(pred_node, start_node)) {
                  continue;
                }
                pred_node->successors_.push_back(HloEdge(1, start_node));
                start_node->predecessors_.push_back(HloEdge(1, pred_node));
                ++pred_node->outdegree_;
                ++start_node->indegree_;
              }
            }
          }
        }
      }
    }
    if (instr->opcode() == HloOpcode::kSendDone) {
      for (const auto* ctrl_pred : instr->control_predecessors()) {
        if (ctrl_pred->opcode() != HloOpcode::kRecvDone) {
          continue;
        }
        const HloInstruction* dependent_while_instr = nullptr;
        for (const auto* while_hlo : while_instrs) {
          if (!reachability->IsReachable(ctrl_pred, while_hlo)) {
            continue;
          }
          if (dependent_while_instr == nullptr) {
            dependent_while_instr = while_hlo;
            continue;
          }
          if (OriginalInstructionPosition(while_hlo) <
              OriginalInstructionPosition(dependent_while_instr)) {
            dependent_while_instr = while_hlo;
          }
        }
        if (dependent_while_instr != nullptr) {
          auto send_done_it = nodes_.find(instr);
          CHECK(send_done_it != nodes_.end());
          HloGraphNode* send_done_node = send_done_it->second.get();
          auto while_it = nodes_.find(dependent_while_instr);
          CHECK(while_it != nodes_.end());
          HloGraphNode* while_node = while_it->second.get();
          send_done_node->successors_.push_back(HloEdge(1, while_node));
          while_node->predecessors_.push_back(HloEdge(1, send_done_node));
          ++send_done_node->outdegree_;
          ++while_node->indegree_;
        }
        break;
      }
    }
  }
}
std::string HloScheduleGraph::ToString(
    const AsyncTracker* async_tracker) const {
  std::string result;
  std::vector<std::pair<const HloGraphNode*, int>> stack;
  for (const auto& node : nodes_) {
    if (node.second->predecessors_.empty()) {
      stack.push_back(std::make_pair(node.second.get(), 0));
    }
  }
  std::vector<const HloGraphNode*> order;
  absl::flat_hash_set<const HloGraphNode*> visited;
  while (!stack.empty()) {
    auto& val = stack.back();
    if (val.second == val.first->successors_.size()) {
      order.push_back(val.first);
      stack.pop_back();
      continue;
    }
    const int64_t next_child = val.second++;
    if (visited.insert(&val.first->successors_[next_child].Target()).second) {
      stack.push_back(
          std::make_pair(&val.first->successors_[next_child].Target(), 0));
    }
  }
  for (auto it = order.rbegin(), e = order.rend(); it != e; ++it) {
    absl::StrAppend(&result, (*it)->ToString(async_tracker));
  }
  return result;
}
HloGraphNode& HloScheduleGraph::GetNode(const HloInstruction* instr) const {
  auto it = nodes_.find(instr);
  CHECK(it != nodes_.end());
  return *it->second;
}
std::vector<HloGraphNode*> HloScheduleGraph::FindBottomRoots() const {
  std::vector<HloGraphNode*> roots;
  for (const HloInstruction* instr : original_order_) {
    HloGraphNode& node = GetNode(instr);
    if (node.GetOutdegree() == 0) {
      roots.push_back(&node);
    }
  }
  return roots;
}
std::vector<HloGraphNode*> HloScheduleGraph::FindTopRoots() const {
  std::vector<HloGraphNode*> roots;
  for (const HloInstruction* instr : original_order_) {
    HloGraphNode& node = GetNode(instr);
    if (node.GetIndegree() == 0) {
      roots.push_back(&node);
    }
  }
  return roots;
}
void HloScheduleGraph::InitializeGraphAnalysis(
    const AsyncTracker* async_tracker) {
  absl::flat_hash_map<HloGraphNode*, int> current_rank;
  std::vector<HloGraphNode*> stack;
  for (const HloInstruction* instr : original_order_) {
    HloGraphNode& node = GetNode(instr);
    current_rank[&node] = node.GetIndegree();
    node.SetAsyncDepth(0.0);
    node.SetDepth(0.0);
    node.SetGraphDepth(0);
    if (node.GetIndegree() == 0) {
      stack.push_back(&node);
    }
  }
  while (!stack.empty()) {
    auto* node = stack.back();
    stack.pop_back();
    if (async_tracker->OccupiesSelectiveResource(node)) {
      node->num_hops_to_closest_selective_resource_occupier_ = 0;
    } else {
      int64_t closest_predecessor_distance =
          std::numeric_limits<int64_t>::max();
      for (auto& pred : node->GetPredecessors()) {
        closest_predecessor_distance = std::min(
            closest_predecessor_distance,
            pred.Target().num_hops_to_closest_selective_resource_occupier_);
      }
      if (closest_predecessor_distance != std::numeric_limits<int64_t>::max()) {
        node->num_hops_to_closest_selective_resource_occupier_ =
            closest_predecessor_distance + 1;
      }
    }
    if (async_tracker->IsSupportedAsyncDone(node->GetInstr())) {
      for (auto& pred : node->GetPredecessors()) {
        node->SetAsyncDepth(
            std::max(pred.Target().GetAsyncDepth() + pred.Latency(),
                     node->GetAsyncDepth()));
        node->SetDepth(std::max(
            pred.Target().GetDepth() + pred.Target().GetCost() + pred.Latency(),
            node->GetDepth()));
        node->SetGraphDepth(
            std::max(pred.Target().GetGraphDepth() + 1, node->GetGraphDepth()));
      }
    } else {
      for (auto& pred : node->GetPredecessors()) {
        node->SetAsyncDepth(
            std::max(pred.Target().GetAsyncDepth(), node->GetAsyncDepth()));
        node->SetDepth(std::max(
            pred.Target().GetDepth() + pred.Target().GetCost() + pred.Latency(),
            node->GetDepth()));
        node->SetGraphDepth(
            std::max(pred.Target().GetGraphDepth() + 1, node->GetGraphDepth()));
      }
    }
    for (auto& succ : node->GetSuccessors()) {
      if (--current_rank[&succ.Target()] == 0) {
        stack.push_back(&succ.Target());
      }
    }
  }
}
absl::Status DefaultSchedulerCore::InitializeScheduler(
    const HloModule* module) {
  TF_ASSIGN_OR_RETURN(alias_analysis_, HloAliasAnalysis::Run(module));
  module_pressure_state_ = std::make_unique<ModulePressureState>(
      module, alias_analysis_.get(), shape_size_bytes_);
  module_pressure_state_->InitializePressureStates();
  module_pressure_state_->SetMemoryPeak(0);
  return absl::OkStatus();
}
absl::Status DefaultSchedulerCore::SchedulingStep(
    SchedulingState* sched_state) {
  TF_ASSIGN_OR_RETURN(HloGraphNode * node,
                      FindAndExtractBestNodeAvailable(
                          *sched_state, nullptr));
  CHECK(node != nullptr);
  TF_ASSIGN_OR_RETURN(sched_state->current_time,
                      ScheduleNode(node, sched_state));
  VLOG(2) << "Scheduled: " << node->GetInstr().name();
  XLA_VLOG_LINES(5, node->ToString());
  return absl::OkStatus();
}
absl::StatusOr<std::vector<HloInstruction*>>
DefaultSchedulerCore::ScheduleComputation(const HloComputation* computation) {
  const HloSchedule& module_schedule = computation->parent()->schedule();
  MemoryPressureTracker memory_pressure_tracker(
      alias_analysis_.get(), module_pressure_state_->buffer_tracker(),
      module_pressure_state_->pressure_state_cache());
  memory_pressure_tracker.Initialize(
      computation,
      module_pressure_state_->GetPressureStateForComputation(computation)
          .live_ids_at_bottom);
  SchedulingState sched_state(
      &module_schedule.sequence(computation), alias_analysis_.get(),
      latency_estimator_, async_tracker_, &memory_pressure_tracker, config_);
  async_tracker_->PostProcessScheduleGraph(&sched_state.sched_graph,
                                           latency_estimator_);
  sched_state.sched_graph.InitializeGraphAnalysis(async_tracker_);
  VLOG(5) << "Just built graph:";
  XLA_VLOG_LINES(5, sched_state.sched_graph.ToString(async_tracker_));
  async_tracker_->SetConcurrentResourceLimits(
      sched_state.max_concurrent_resource);
  auto roots = sched_state.sched_graph.FindBottomRoots();
  for (HloGraphNode* root : roots) {
    root->SetReadyTime(0.0);
  }
  VLOG(5) << "Initial memory pressure for " << computation->name() << ": "
          << memory_pressure_tracker.memory_usage();
  sched_state.ready_set.insert(sched_state.ready_set.end(), roots.begin(),
                               roots.end());
  while (!sched_state.ready_set.empty()) {
    VLOG(10) << "Current ready time: " << sched_state.current_time;
    VLOG(2) << "Current ready queue:";
    XLA_VLOG_LINES(2, [&sched_state]() {
      struct LogFormatter {
        void operator()(std::string* out, const HloGraphNode* n) const {
          out->append(absl::StrCat("\t", n->GetInstr().name(),
                                   " Ready time: ", n->GetReadyTime(),
                                   " Depth: ", n->GetGraphDepth()));
        }
      };
      return absl::StrJoin(sched_state.ready_set, "\n", LogFormatter());
    }());
    TF_RETURN_IF_ERROR(SchedulingStep(&sched_state));
  }
  if (VLOG_IS_ON(5)) {
    VLOG(5) << "New order";
    for (auto r_it = sched_state.new_sequence_reversed.rbegin(),
              e_it = sched_state.new_sequence_reversed.rend();
         r_it != e_it; ++r_it) {
      LogInstruction(*r_it);
    }
  }
  module_pressure_state_->UpdatePressureStateForComputation(
      computation, memory_pressure_tracker.pressure_state());
  absl::c_reverse(sched_state.new_sequence_reversed);
  if (post_processing_fn_) {
    post_processing_fn_(sched_state);
  }
  CHECK_EQ(sched_state.new_sequence_reversed.size(),
           sched_state.sched_graph.GetOriginalInstrList().size())
      << "Not all instructions have been scheduled "
      << sched_state.new_sequence_reversed.size() << " vs "
      << sched_state.sched_graph.GetOriginalInstrList().size();
  VLOG(2) << "Total time: "
          << sched_state.sched_graph
                 .GetNode(sched_state.new_sequence_reversed.front())
                 .GetReadyTime();
  const auto& debug_options = xla::GetDebugOptionsFromFlags();
  if (debug_options.xla_dump_latency_hiding_schedule() &&
      computation->IsEntryComputation()) {
    int core_freq = latency_estimator_->CyclesPerMicrosecond();
    DumpLatencyHidingSchedule(computation, sched_state.sched_graph,
                              sched_state.new_sequence_reversed, core_freq,
                              debug_options);
  }
  return std::move(sched_state.new_sequence_reversed);
}
void DefaultSchedulerCore::DumpLatencyHidingSchedule(
    const HloComputation* computation, const HloScheduleGraph& schedule_graph,
    const std::vector<HloInstruction*>& instructions,
    const int cycles_per_microsecond, const DebugOptions& debug_options) {
  ScheduleProto proto;
  proto.set_computation_id(computation->unique_id());
  proto.set_cycles_per_microsecond(cycles_per_microsecond);
  const HloGraphNode& first_node = schedule_graph.GetNode(instructions.front());
  const double total_time = first_node.GetReadyTime() + first_node.GetCost();
  for (const HloInstruction* instr : instructions) {
    const HloGraphNode& instr_node = schedule_graph.GetNode(instr);
    const double start_time =
        total_time - (instr_node.GetReadyTime() + instr_node.GetCost());
    const double end_time = start_time + instr_node.GetCost();
    ScheduleProto::Instruction* instr_msg = proto.add_instructions();
    instr_msg->set_id(instr->unique_id());
    instr_msg->set_start_timestamp_cycles(start_time);
    instr_msg->set_end_timestamp_cycles(end_time);
  }
  *proto.mutable_hlo_module() = computation->parent()->ToProto();
  const std::string fn = absl::StrFormat("%s.schedule", computation->name());
  DumpProtobufToFile(proto, debug_options, fn);
}
LatencyHidingScheduler::SchedulerStatistics
LatencyHidingScheduler::LatencyHidingStatistics(
    const HloComputation* computation,
    const LatencyEstimator* latency_estimator,
    const AsyncTracker* async_tracker,
    const HloCostAnalysis::ShapeSizeFunction& shape_size_bytes) {
  const HloModule* module = computation->parent();
  absl::flat_hash_map<
      HloOpcode,
      std::vector<std::tuple<const HloInstruction*, int64_t, int64_t>>>
      outstanding_collectives;
  double current_time = 0;
  enum class AsyncKind {
    kNotAsync,
    kAllGather,
    kAllReduce,
    kCollectivePermute,
    kAllToAll,
    kReduceScatter,
    kSend,
    kRecv,
    kCollectiveBroadcast,
  };
  auto opcode_to_async_kind = [](HloOpcode opcode) {
    switch (opcode) {
      case HloOpcode::kAllGather:
        return AsyncKind::kAllGather;
      case HloOpcode::kAllReduce:
        return AsyncKind::kAllReduce;
      case HloOpcode::kCollectiveBroadcast:
        return AsyncKind::kCollectiveBroadcast;
      case HloOpcode::kCollectivePermute:
        return AsyncKind::kCollectivePermute;
      case HloOpcode::kAllToAll:
        return AsyncKind::kAllToAll;
      case HloOpcode::kReduceScatter:
        return AsyncKind::kReduceScatter;
      case HloOpcode::kSend:
        return AsyncKind::kSend;
      case HloOpcode::kRecv:
        return AsyncKind::kRecv;
      default:
        return AsyncKind::kNotAsync;
    }
  };
  auto find_node_successor_edge = [](const HloGraphNode& graph_node,
                                     const HloGraphNode& successor_node) {
    auto edge_it = std::find_if(graph_node.GetSuccessors().begin(),
                                graph_node.GetSuccessors().end(),
                                [&successor_node](const HloEdge& edge) {
                                  return &edge.Target() == &successor_node;
                                });
    CHECK(edge_it != graph_node.GetSuccessors().end());
    return edge_it;
  };
  auto find_outstanding_async = [&outstanding_collectives,
                                 async_tracker](const HloInstruction* instr) {
    const auto& collective_vec =
        outstanding_collectives[async_tracker->GetCanonicalAsyncOp(*instr)
                                    .inner];
    auto it = absl::c_find_if(
        collective_vec,
        [instr](const std::tuple<const HloInstruction*, int64_t, int64_t>& p) {
          return instr == std::get<0>(p);
        });
    CHECK(it != collective_vec.end());
    return it;
  };
  absl::flat_hash_map<AsyncKind, double> wasted_time_per_collective;
  SchedulerConfig config;
  config.schedule_send_recvs = true;
  config.use_real_cost_model = true;
  std::unique_ptr<HloAliasAnalysis> hlo_alias_analysis =
      HloAliasAnalysis::Run(module).value();
  auto instructions_post_order = computation->MakeInstructionPostOrder();
  HloScheduleGraph schedule_graph(&instructions_post_order,
                                  nullptr, latency_estimator,
                                  async_tracker);
  async_tracker->PostProcessScheduleGraph(&schedule_graph, latency_estimator);
  int64_t curr_pos = 0;
  for (const HloInstruction* instr :
       module->schedule().sequence(computation).instructions()) {
    const HloGraphNode& instr_node = schedule_graph.GetNode(instr);
    current_time += instr_node.GetCost();
    if (async_tracker->IsSupportedAsyncStart(*instr)) {
      outstanding_collectives[async_tracker->GetCanonicalAsyncOp(*instr).inner]
          .push_back({instr, current_time, curr_pos});
    } else if (async_tracker->IsSupportedAsyncDone(*instr)) {
      const HloInstruction* start_instr = instr->operand(0);
      if (async_tracker->IsSupportedAsyncStart(*start_instr)) {
        auto it = find_outstanding_async(start_instr);
        const HloGraphNode& start_node =
            schedule_graph.GetNode(std::get<0>(*it));
        auto edge_it = find_node_successor_edge(start_node, instr_node);
        const double async_wasted_cycles = std::max(
            0.0, edge_it->Latency() - (current_time - std::get<1>(*it)));
        AsyncKind kind = opcode_to_async_kind(
            async_tracker->GetCanonicalAsyncOp(*start_instr).inner);
        wasted_time_per_collective[kind] += async_wasted_cycles;
        current_time += async_wasted_cycles;
      }
    }
    curr_pos++;
  }
  ModulePressureState module_pressure_state(module, hlo_alias_analysis.get(),
                                            shape_size_bytes);
  module_pressure_state.InitializePressureStates();
  const MemoryPressureTracker::MemoryPressureState* memory_pressure_state =
      module_pressure_state.ComputationIsMemoryTracked(computation)
          ? &module_pressure_state.GetPressureStateForComputation(computation)
          : nullptr;
  MemoryPressureTracker mem_pressure_tracker(
      hlo_alias_analysis.get(), module_pressure_state.buffer_tracker(),
      module_pressure_state.pressure_state_cache());
  if (memory_pressure_state != nullptr) {
    mem_pressure_tracker.Initialize(computation,
                                    memory_pressure_state->live_ids_at_bottom);
  }
  return LatencyHidingScheduler::SchedulerStatistics{
      computation,
      wasted_time_per_collective[AsyncKind::kAllGather],
      wasted_time_per_collective[AsyncKind::kAllReduce],
      wasted_time_per_collective[AsyncKind::kCollectiveBroadcast],
      wasted_time_per_collective[AsyncKind::kCollectivePermute],
      wasted_time_per_collective[AsyncKind::kAllToAll],
      wasted_time_per_collective[AsyncKind::kReduceScatter],
      wasted_time_per_collective[AsyncKind::kSend],
      wasted_time_per_collective[AsyncKind::kRecv],
      current_time,
      memory_pressure_state ? mem_pressure_tracker.initial_memory_pressure() +
                                  memory_pressure_state->memory_peak
                            : 0};
}
std::string LatencyHidingScheduler::SchedulerStatisticsString(
    const SchedulerStatistics& sched_stats) {
  std::string result;
  if (const HloComputation* comp = sched_stats.computation) {
    absl::StrAppend(&result, "For computation: ", comp->name(), ", module ",
                    comp->parent()->name(), "(", comp->parent()->unique_id(),
                    ")\n");
  }
  absl::StrAppend(&result, "Total wasted cycles: ",
                  sched_stats.all_gather_wasted_cycles +
                      sched_stats.all_reduce_wasted_cycles +
                      sched_stats.collective_broadcast_wasted_cycles +
                      sched_stats.collective_permute_wasted_cycles +
                      sched_stats.all_to_all_wasted_cycles +
                      sched_stats.reduce_scatter_wasted_cycles +
                      sched_stats.send_wasted_cycles +
                      sched_stats.recv_wasted_cycles,
                  "\n");
  absl::StrAppend(&result, "Wasted cycles for all-reduce: ",
                  sched_stats.all_reduce_wasted_cycles, "\n");
  absl::StrAppend(&result, "Wasted cycles for all-gather: ",
                  sched_stats.all_gather_wasted_cycles, "\n");
  absl::StrAppend(&result, "Wasted cycles for collective-broadcast: ",
                  sched_stats.collective_broadcast_wasted_cycles, "\n");
  absl::StrAppend(&result, "Wasted cycles for collective-permute: ",
                  sched_stats.collective_permute_wasted_cycles, "\n");
  absl::StrAppend(&result, "Wasted cycles for all-to-all: ",
                  sched_stats.all_to_all_wasted_cycles, "\n");
  absl::StrAppend(&result, "Wasted cycles for reduce-scatter: ",
                  sched_stats.reduce_scatter_wasted_cycles, "\n");
  absl::StrAppend(&result,
                  "Wasted cycles for send: ", sched_stats.send_wasted_cycles,
                  "\n");
  absl::StrAppend(&result,
                  "Wasted cycles for recv: ", sched_stats.recv_wasted_cycles,
                  "\n");
  absl::StrAppend(&result, "Total cycles: ", sched_stats.total_cycles, "\n");
  absl::StrAppend(&result, "Memory pressure peak (bytes): ",
                  sched_stats.memory_pressure_peak, "\n");
  return result;
}
void LatencyHidingScheduler::LogScheduleStatistics(
    const HloComputation* computation) {
  XLA_VLOG_LINES(1, SchedulerStatisticsString(LatencyHidingStatistics(
                        computation, latency_estimator_.get(),
                        async_tracker_.get(), shape_size_bytes_)));
}
absl::StatusOr<bool> LatencyHidingScheduler::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(5) << "Original module:";
  XLA_VLOG_LINES(5, module->ToString());
  std::vector<HloComputation*> computations_to_schedule;
  computations_to_schedule.reserve(module->computation_count());
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (auto* instr : computation->instructions()) {
      if (async_tracker_->IsSupportedAsyncStart(*instr) ||
          async_tracker_->IsSupportedAsyncDone(*instr)) {
        computations_to_schedule.push_back(computation);
        break;
      }
    }
  }
  if (computations_to_schedule.empty()) {
    return false;
  }
  absl::flat_hash_map<HloComputation*, std::vector<HloInstruction*>>
      saved_schedules;
  TF_RETURN_IF_ERROR(scheduler_core_->InitializeScheduler(module));
  for (HloComputation* computation : computations_to_schedule) {
    TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> new_schedule,
                        scheduler_core_->ScheduleComputation(computation));
    saved_schedules[computation] = std::move(new_schedule);
  }
  uint64_t initial_memory_limit = scheduler_core_->GetMemoryLimit();
  for (int64_t iter = 0;
       iter < scheduler_core_->GetRerunTimes() &&
       scheduler_core_->GetMemoryPeak() > initial_memory_limit;
       iter++) {
    LOG(INFO) << "LatencyHidingScheduler current memory usage: "
              << scheduler_core_->GetMemoryPeak()
              << " bytes, does not fit in limit: "
              << scheduler_core_->GetMemoryLimit()
              << ". Setting the new limit to "
              << scheduler_core_->GetMemoryLimit() * 0.9;
    TF_RETURN_IF_ERROR(scheduler_core_->InitializeScheduler(module));
    scheduler_core_->SetMemoryLimit(scheduler_core_->GetMemoryLimit() * 0.9);
    for (HloComputation* computation : computations_to_schedule) {
      TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> new_schedule,
                          scheduler_core_->ScheduleComputation(computation));
      saved_schedules[computation] = std::move(new_schedule);
    }
  }
  LOG(INFO) << "LatencyHidingScheduler current memory usage: "
            << scheduler_core_->GetMemoryPeak()
            << " bytes. Current limit: " << scheduler_core_->GetMemoryLimit();
  for (HloComputation* computation : computations_to_schedule) {
    VLOG(1) << "Statistics before scheduling:";
    LogScheduleStatistics(computation);
    module->schedule().set_sequence(
        computation, absl::MakeConstSpan(saved_schedules[computation]));
    VLOG(1) << "Statistics after scheduling:";
    LogScheduleStatistics(computation);
  }
  return true;
}
}  