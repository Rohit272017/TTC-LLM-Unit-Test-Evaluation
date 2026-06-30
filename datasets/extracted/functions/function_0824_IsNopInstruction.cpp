#include "xla/service/gpu/gpu_latency_hiding_scheduler.h"
#include <cstdint>
#include <tuple>
#include <utility>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
namespace xla {
namespace gpu {
namespace {
static constexpr int64_t kCostlyAllReduceThreshold = 30 * 1024 * 1024;
static constexpr int64_t kCostlyAllReduceMultiplier = 4;
bool IsNopInstruction(const HloInstruction& hlo) {
  HloOpcode op = hlo.opcode();
  return op == HloOpcode::kGetTupleElement || op == HloOpcode::kBitcast ||
         op == HloOpcode::kConstant || op == HloOpcode::kParameter ||
         op == HloOpcode::kTuple || op == HloOpcode::kPartitionId ||
         op == HloOpcode::kReplicaId || hlo.IsEffectiveBitcast() ||
         op == HloOpcode::kOptimizationBarrier;
}
bool IsAsyncComputeOp(const HloInstruction& hlo) {
  return (hlo.opcode() == HloOpcode::kAsyncStart ||
          hlo.opcode() == HloOpcode::kAsyncDone) &&
         !hlo_query::IsCollectiveCommunicationOp(hlo.async_wrapped_opcode()) &&
         hlo.async_execution_thread() != hlo.parent()->execution_thread();
}
int64_t GetPipelineStream(const HloInstruction& start) {
  auto it = start.frontend_attributes().map().find(kSendRecvPipelineAttr);
  if (it != start.frontend_attributes().map().end() && it->second == "1") {
    return 1;
  }
  return 0;
}
std::pair<GpuResourceType, ResourceUsageType> GetP2PResourceAndUsage(
    const HloInstruction& instr, const CanonicalAsyncOp& op) {
  ResourceUsageType usage = op.outer == HloOpcode::kAsyncStart
                                ? ResourceUsageType::kResourceRelease
                                : ResourceUsageType::kResourceOccupy;
  int64_t pipeline = GetPipelineStream(instr);
  HloOpcode opcode = op.inner;
  GpuResourceType resource;
  if (pipeline == 0) {
    resource = opcode == HloOpcode::kSend
                   ? GpuResourceType::kGpuAsyncStreamSend0
                   : GpuResourceType::kGpuAsyncStreamRecv0;
  } else {
    resource = opcode == HloOpcode::kSend
                   ? GpuResourceType::kGpuAsyncStreamSend1
                   : GpuResourceType::kGpuAsyncStreamRecv1;
  }
  return {resource, usage};
}
bool IsGpuAsyncStart(const HloInstruction& hlo) {
  return (hlo_query::IsAsyncCollectiveStartOp(&hlo,
                                              true) &&
          !IsSyncCollective(&hlo)) ||
         IsAsyncComputeOp(hlo);
}
bool IsGpuAsyncDone(const HloInstruction& hlo) {
  return (hlo_query::IsAsyncCollectiveDoneOp(&hlo,
                                             true) &&
          !IsSyncCollective(hlo.operand(0))) ||
         IsAsyncComputeOp(hlo);
}
bool IsAsyncPair(const HloInstruction& from, const HloInstruction& target) {
  return IsGpuAsyncStart(from) && IsGpuAsyncDone(target);
}
}  
int64_t GetSizeOfShape(const Shape& shape, int pointer_size) {
  int64_t size = ShapeUtil::ByteSizeOf(shape, pointer_size);
  if (shape.IsTuple() || shape.is_static()) {
    return size;
  }
  int64_t metadata_size = sizeof(int32_t) * shape.dimensions_size();
  return size + metadata_size;
}
CanonicalAsyncOp GpuGetCanonicalAsyncOp(const HloInstruction& hlo) {
  switch (hlo.opcode()) {
    case HloOpcode::kSend:
      return {HloOpcode::kAsyncStart, HloOpcode::kSend};
    case HloOpcode::kSendDone:
      return {HloOpcode::kAsyncDone, HloOpcode::kSend};
    case HloOpcode::kRecv:
      return {HloOpcode::kAsyncStart, HloOpcode::kRecv};
    case HloOpcode::kRecvDone:
      return {HloOpcode::kAsyncDone, HloOpcode::kRecv};
    default:
      return DefaultGetCanonicalAsyncOp(hlo);
  }
}
GpuAsyncTrackerBase::GpuAsyncTrackerBase(const SchedulerConfig& config,
                                         GetCanonicalAsyncOpFunc func)
    : AsyncTracker(config, func) {}
bool GpuAsyncTrackerBase::IsSupportedAsyncDone(
    const HloInstruction& hlo) const {
  return IsGpuAsyncDone(hlo);
}
bool GpuAsyncTrackerBase::IsSupportedAsyncStart(
    const HloInstruction& hlo) const {
  return IsGpuAsyncStart(hlo);
}
void GpuAsyncTrackerBase::PostProcessScheduleGraph(
    HloScheduleGraph* schedule_graph,
    const LatencyEstimator* latency_estimator) const {
  for (auto inst : schedule_graph->GetOriginalInstrList()) {
    if (inst->opcode() == HloOpcode::kRecv) {
      if (inst->frontend_attributes().map().count(kSendRecvPipelineAttr) > 0) {
        HloGraphNode& node = schedule_graph->GetNode(inst);
        node.SetForceEarly(true);
        VLOG(5) << "Setting force early for instruction: " << inst->ToString();
      }
    }
    if (inst->has_backend_config()) {
      auto gpu_config = inst->backend_config<GpuBackendConfig>();
      if (gpu_config.ok()) {
        HloGraphNode& node = schedule_graph->GetNode(inst);
        node.SetForceDelay(gpu_config->force_earliest_schedule());
        VLOG(5) << "Setting force delay for instruction: " << inst->ToString();
      }
    }
  }
}
GpuAsyncTracker::GpuAsyncTracker(const SchedulerConfig& config)
    : GpuAsyncTrackerBase(config) {}
ResourcesVector GpuAsyncTracker::GetResourcesFromInstruction(
    const HloInstruction& instr) const {
  CanonicalAsyncOp op = GetCanonicalAsyncOp(instr);
  if (op.outer == HloOpcode::kAsyncStart || op.outer == HloOpcode::kAsyncDone) {
    ResourceUsageType usage;
    GpuResourceType resource;
    if (op.inner == HloOpcode::kSend || op.inner == HloOpcode::kRecv) {
      std::tie(resource, usage) = GetP2PResourceAndUsage(instr, op);
    } else {
      usage = op.outer == HloOpcode::kAsyncStart
                  ? ResourceUsageType::kResourceRelease
                  : ResourceUsageType::kResourceOccupy;
      resource = hlo_query::IsCollectiveCommunicationOp(op.inner)
                     ? GpuResourceType::kGpuAsyncStreamCollectives
                     : GpuResourceType::kGpuAsyncStreamComputes;
    }
    return {std::make_pair(
        GetFirstTargetDefinedResource() + static_cast<int64_t>(resource),
        usage)};
  }
  return GpuAsyncTrackerBase::GetResourcesFromInstruction(instr);
}
int64_t GpuAsyncTracker::GetNumTargetDefinedResources() const {
  return static_cast<int64_t>(GpuResourceType::kNumTargetResources);
};
int64_t GpuAsyncTracker::GetNumAvailableResources(int64_t resource_type) const {
  const int64_t first_target_resource = GetFirstTargetDefinedResource();
  if (resource_type < first_target_resource) {
    return GpuAsyncTrackerBase::GetNumAvailableResources(resource_type);
  }
  CHECK_LT(resource_type,
           first_target_resource +
               static_cast<int64_t>(GpuResourceType::kNumTargetResources));
  if ((resource_type - first_target_resource) ==
      static_cast<int64_t>(GpuResourceType::kGpuAsyncStreamComputes)) {
    return 2;
  }
  return 1;
}
absl::string_view GpuAsyncTracker::GetResourceName(
    int64_t resource_type) const {
  const int64_t first_target_resource = GetFirstTargetDefinedResource();
  if (resource_type < first_target_resource) {
    return GpuAsyncTrackerBase::GetResourceName(resource_type);
  }
  CHECK_LE(resource_type,
           first_target_resource + GetNumTargetDefinedResources());
  switch (static_cast<GpuResourceType>(resource_type - first_target_resource)) {
    case GpuResourceType::kGpuAsyncStreamSend0:
      return "kGpuAsyncStreamSend0";
    case GpuResourceType::kGpuAsyncStreamSend1:
      return "kGpuAsyncStreamSend1";
    case GpuResourceType::kGpuAsyncStreamRecv0:
      return "kGpuAsyncStreamRecv0";
    case GpuResourceType::kGpuAsyncStreamRecv1:
      return "kGpuAsyncStreamRecv1";
    case GpuResourceType::kGpuAsyncStreamCollectives:
      return "kGpuAsyncStreamCollectives";
    case GpuResourceType::kGpuAsyncStreamComputes:
      return "kGpuAsyncStreamComputes";
    default:
      return "kUnsupportedResource";
  }
}
ResourceHazardType GpuAsyncTracker::GetResourceHazardType(
    int64_t resource_type) const {
  const int64_t first_target_resource = GetFirstTargetDefinedResource();
  if (resource_type < first_target_resource) {
    return GpuAsyncTrackerBase::GetResourceHazardType(resource_type);
  }
  CHECK_LE(resource_type,
           first_target_resource + GetNumTargetDefinedResources());
  return ResourceHazardType::kUnshareable;
}
int64_t GpuAsyncTracker::GetNumResourcesPerInstruction(
    int64_t resource_type, const HloInstruction& instr) const {
  int64_t num_resources =
      GpuAsyncTrackerBase::GetNumResourcesPerInstruction(resource_type, instr);
  if (num_resources <= 0 || instr.opcode() != HloOpcode::kWhile) {
    return num_resources;
  }
  int64_t first_p2p_resource =
      GetFirstTargetDefinedResource() +
      static_cast<int64_t>(GpuResourceType::kGpuAsyncStreamSend0);
  if (resource_type < first_p2p_resource ||
      resource_type > first_p2p_resource + 4) {
    return num_resources;
  }
  auto find_instruction_for_pipeline = [&](HloOpcode opcode, int64_t pipeline) {
    for (auto user1 : instr.users()) {
      if (user1->opcode() == HloOpcode::kGetTupleElement) {
        for (auto user2 : user1->users()) {
          if (user2->opcode() == opcode) {
            if (GetPipelineStream(*user2) == pipeline) {
              return true;
            }
          }
        }
      }
    }
    return false;
  };
  bool found;
  if (resource_type == first_p2p_resource) {
    found = find_instruction_for_pipeline(HloOpcode::kSendDone, 0);
  } else if (resource_type == first_p2p_resource + 1) {
    found = find_instruction_for_pipeline(HloOpcode::kSendDone, 1);
  } else if (resource_type == first_p2p_resource + 2) {
    found = find_instruction_for_pipeline(HloOpcode::kRecvDone, 0);
  } else {
    found = find_instruction_for_pipeline(HloOpcode::kRecvDone, 1);
  }
  return num_resources - (found ? 1 : 0);
}
GpuLatencyEstimator::GpuLatencyEstimator(int64_t pointer_size,
                                         GetCanonicalAsyncOpFunc func)
    : ApproximateLatencyEstimator(func), pointer_size_(pointer_size) {}
ApproximateLatencyEstimator::TimeCost GpuLatencyEstimator::NodeCost(
    const HloInstruction* instr) const {
  if (IsNopInstruction(*instr)) {
    return 0.0;
  }
  if (instr->opcode() == HloOpcode::kCustomCall) {
    if (IsCublasGemm(*instr) || IsCustomCallToDnnConvolution(*instr)) {
      return ApproximateLatencyEstimator::kMediumCost;
    }
    return ApproximateLatencyEstimator::kMediumCost;
  }
  return ApproximateLatencyEstimator::NodeCost(instr);
}
ApproximateLatencyEstimator::TimeCost GpuLatencyEstimator::GetLatencyBetween(
    const HloGraphNode& from, const HloGraphNode& to) const {
  if (IsAsyncPair(from, to)) {
    if (from.GetInstr().opcode() == HloOpcode::kRecv) {
      return ApproximateLatencyEstimator::kLowLatency;
    } else if (from.GetInstr().opcode() == HloOpcode::kSend) {
      return ApproximateLatencyEstimator::kHighLatency * 10;
    }
    bool enable_approx_collectives =
        from.GetInstr()
            .GetModule()
            ->config()
            .debug_options()
            .xla_gpu_enable_approx_costly_collectives();
    bool is_all_reduce = from.GetInstr().opcode() == HloOpcode::kAllReduceStart;
    bool collective_size_exceeds_threshold =
        GetSizeOfShape(from.GetInstr().shape(), pointer_size_) >
        kCostlyAllReduceThreshold;
    if (enable_approx_collectives && is_all_reduce &&
        collective_size_exceeds_threshold) {
      return ApproximateLatencyEstimator::kHighLatency *
             kCostlyAllReduceMultiplier;
    }
    return ApproximateLatencyEstimator::kHighLatency;
  }
  return ApproximateLatencyEstimator::kLowLatency;
}
void GPUProfileStatisticsAggregator::HandleMissingInstructionCost(
    const HloInstruction& instruction) {
  if (!IsNopInstruction(instruction) &&
      instruction.opcode() != HloOpcode::kWhile) {
    missing_instructions_.insert(&instruction);
  }
}
void GPUProfileStatisticsAggregator::HandleFoundInstructionCost(
    const HloInstruction& instruction) {
  found_instructions_count_++;
}
void GPUProfileStatisticsAggregator::HandleMissingInstructionLatency(
    const HloInstruction& from, const HloInstruction& to) {
  if (IsAsyncPair(from, to)) {
    missing_instructions_.insert(&from);
  }
}
void GPUProfileStatisticsAggregator::HandleFoundInstructionLatency(
    const HloInstruction& from, const HloInstruction& to) {
  found_instructions_count_++;
}
}  
}  