#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/fusions/fusions.h"
#include "xla/service/gpu/fusions/triton.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
namespace xla {
namespace gpu {
namespace {
bool FusionUsesParameterElementwiseFromRoot(
    const HloInstruction* fusion, int parameter_index,
    const GpuHloCostAnalysis* cost_analysis) {
  return cost_analysis->CommonElementwiseUtilization(
             fusion->fused_parameter(parameter_index),
             fusion->fused_expression_root()) == 1.f;
}
int GetCoalescingWasteFactor(PrimitiveType element_type,
                             const se::DeviceDescription& gpu_device_info) {
  int64_t element_size_bytes =
      element_type == PrimitiveType::TUPLE ||
              element_type == PrimitiveType::TOKEN
          ? 4 
          : ShapeUtil::ByteSizeOfPrimitiveType(element_type);
  return gpu_device_info.dram_to_l2_transaction_size_bytes() /
         element_size_bytes;
}
float AdjustBandwidth(const se::DeviceDescription& gpu_device_info,
                      float bandwidth, int64_t num_blocks) {
  float per_block_bandwidth = gpu_device_info.clock_rate_ghz() * 1.0e9f *
                              gpu_device_info.memory_transactions_per_clock();
  float max_bandwidth = num_blocks * per_block_bandwidth;
  return std::min(bandwidth, max_bandwidth);
}
}  
std::optional<EstimateRunTimeData> GpuPerformanceModelCache::Get(
    const HloInstruction& instruction) {
  auto it = instruction_runtime_data_.find(&instruction);
  if (it != instruction_runtime_data_.end()) {
    return it->second;
  }
  return std::nullopt;
}
std::optional<absl::Duration> GpuPerformanceModelCache::Get(
    const HloInstruction& producer, const HloInstruction& consumer) {
  absl::MutexLock lock(&mutex_);
  auto it = fusion_runtime_data_.find(&producer);
  if (it != fusion_runtime_data_.end()) {
    auto jt = it->second.find(&consumer);
    if (jt != it->second.end()) {
      return jt->second;
    }
  }
  return std::nullopt;
}
const absl::flat_hash_map<const HloInstruction*, absl::Duration>&
GpuPerformanceModelCache::GetAllConsumers(const HloInstruction& producer) {
  return fusion_runtime_data_[&producer];
}
bool GpuPerformanceModelCache::ContainsConsumers(
    const HloInstruction& producer) {
  return fusion_runtime_data_.contains(&producer);
}
void GpuPerformanceModelCache::Set(const HloInstruction& instruction,
                                   const EstimateRunTimeData& runtime_data) {
  instruction_runtime_data_[&instruction] = runtime_data;
}
void GpuPerformanceModelCache::Set(const HloInstruction& producer,
                                   const HloInstruction& consumer,
                                   absl::Duration runtime) {
  absl::MutexLock lock(&mutex_);
  fusion_runtime_data_[&producer][&consumer] = runtime;
}
void GpuPerformanceModelCache::Invalidate(const HloInstruction& instruction) {
  instruction_runtime_data_.erase(&instruction);
  fusion_runtime_data_.erase(&instruction);
  for (auto* operand : instruction.operands()) {
    if (operand->opcode() == HloOpcode::kGetTupleElement) {
      operand = operand->mutable_operand(0);
    }
    auto it = fusion_runtime_data_.find(operand);
    if (it != fusion_runtime_data_.end()) {
      it->second.erase(&instruction);
    }
  }
}
LaunchDimensions GpuPerformanceModelBase::EstimateFusionLaunchDimensions(
    const HloFusionAnalysis& fusion_analysis) {
  auto emitter =
      GetFusionEmitter(PreBufferAssignmentFusionInfo{fusion_analysis});
  if (const auto* kernel_emitter =
          dynamic_cast<const KernelFusionInterface*>(emitter.get())) {
    return kernel_emitter->launch_dimensions();
  }
  if (const auto* triton_emitter =
          dynamic_cast<const TritonFusion*>(emitter.get())) {
    if (auto launch_config = triton_emitter->launch_config()) {
      return launch_config->launch_dimensions;
    }
  }
  VLOG(5) << "Using fallback launch dimensions estimate for "
          << fusion_analysis.fusion().ToString();
  int64_t num_threads_per_block = 128;
  int64_t estimated_num_threads =
      ShapeUtil::ElementsInRecursive(fusion_analysis.fusion_root(0).shape());
  int64_t num_blocks =
      CeilOfRatio(estimated_num_threads, num_threads_per_block);
  return LaunchDimensions(num_blocks, num_threads_per_block);
}
int64_t GpuPerformanceModelBase::GetOperandBytesAccessed(
    const GpuHloCostAnalysis* cost_analysis, const HloInstruction* instr,
    const HloInstruction* operand) {
  if (!instr->IsUserOf(operand)) {
    return 0;
  }
  return cost_analysis->operand_bytes_accessed(*instr,
                                               instr->operand_index(operand));
}
float GpuPerformanceModelBase::GetOperandUtilization(
    const GpuHloCostAnalysis* cost_analysis, const HloInstruction* instr,
    const HloInstruction* operand) {
  if (operand->IsMultiOutputFusion()) {
    float res = 0.f;
    for (int64_t i = 0; i < instr->operand_count(); ++i) {
      if (instr->operand(i)->opcode() == HloOpcode::kGetTupleElement &&
          instr->operand(i)->operand(0) == operand) {
        res += cost_analysis->operand_utilization(*instr, i);
      }
    }
    return res;
  }
  if (!instr->IsUserOf(operand)) {
    return 0.f;
  }
  return cost_analysis->operand_utilization(*instr,
                                            instr->operand_index(operand));
}
float GpuPerformanceModelBase::GetCommonUtilization(
    const GpuHloCostAnalysis* cost_analysis, const HloInstruction* producer,
    int64_t producer_idx_of_operand, const HloInstruction* consumer) {
  const auto* operand = producer->operand(producer_idx_of_operand);
  if (!consumer || !consumer->IsUserOf(operand)) {
    return 0.f;
  }
  if (producer->IsElementwise() ||
      (producer->opcode() == HloOpcode::kFusion &&
       FusionUsesParameterElementwiseFromRoot(producer, producer_idx_of_operand,
                                              cost_analysis))) {
    if (consumer->opcode() == HloOpcode::kFusion) {
      int64_t consumer_idx_of_common_operand = consumer->operand_index(operand);
      float res = 0.f;
      std::vector<int64_t> consumer_indices_of_producer;
      if (producer->IsMultiOutputFusion()) {
        for (int64_t i = 0; i < consumer->operand_count(); ++i) {
          if (consumer->operand(i)->opcode() == HloOpcode::kGetTupleElement &&
              consumer->operand(i)->operand(0) == producer) {
            consumer_indices_of_producer.push_back(i);
          }
        }
      } else {
        consumer_indices_of_producer.push_back(
            consumer->operand_index(producer));
      }
      for (int64_t consumer_idx_of_producer : consumer_indices_of_producer) {
        res += cost_analysis->CommonElementwiseUtilization(
            consumer->fused_parameter(consumer_idx_of_common_operand),
            consumer->fused_parameter(consumer_idx_of_producer));
      }
      return res;
    } else if (consumer->IsElementwise()) {
      return 1.f;
    }
  }
  return 0.f;
}
int64_t GpuPerformanceModelBase::GetSharedOperandBytesAccessed(
    const GpuHloCostAnalysis* cost_analysis, const HloInstruction* producer,
    const HloInstruction* consumer, const HloInstruction* operand) {
  float producer_utilization_by_consumer =
      GetOperandUtilization(cost_analysis, consumer, producer);
  int64_t bytes_accessed_by_producer =
      GetOperandBytesAccessed(cost_analysis, producer, operand);
  int64_t bytes_accessed_by_consumer =
      GetOperandBytesAccessed(cost_analysis, consumer, operand);
  float common_utilization =
      producer->IsUserOf(operand)
          ? GetCommonUtilization(cost_analysis, producer,
                                 producer->operand_index(operand), consumer)
          : 0.f;
  int64_t operand_size = cost_analysis->GetShapeSize(operand->shape());
  int64_t common_bytes_accessed =
      std::llround(operand_size * common_utilization);
  return std::llround(bytes_accessed_by_producer *
                      producer_utilization_by_consumer) +
         bytes_accessed_by_consumer - common_bytes_accessed;
}
absl::Duration GpuPerformanceModelBase::ReadTime(
    const se::DeviceDescription& gpu_device_info, int64_t num_blocks,
    int64_t n_bytes_net, int64_t n_bytes_total) {
  float bandwidth = gpu_device_info.memory_bandwidth();
  if (n_bytes_net < gpu_device_info.l2_cache_size()) {
    bandwidth *= kL2CacheSpeedup;
    if (n_bytes_net <
        gpu_device_info.l1_cache_size_per_SM() * gpu_device_info.core_count()) {
      bandwidth *= kL1CacheSpeedup;
    }
  }
  bandwidth = AdjustBandwidth(gpu_device_info, bandwidth, num_blocks);
  return absl::Seconds(n_bytes_total / bandwidth);
}
absl::Duration GpuPerformanceModelBase::ReadTimeWithDRAMHeuristic(
    const se::DeviceDescription& gpu_device_info, int64_t num_blocks,
    int64_t n_bytes_net, int64_t n_bytes_total, PrimitiveType element_type,
    bool coalesced) {
  int waste_factor =
      coalesced ? 1 : GetCoalescingWasteFactor(element_type, gpu_device_info);
  float dram_bandwidth = gpu_device_info.memory_bandwidth() / waste_factor;
  float rest_bandwidth = gpu_device_info.memory_bandwidth();
  if (n_bytes_net < gpu_device_info.l2_cache_size()) {
    rest_bandwidth *= kL2CacheSpeedup;
    if (n_bytes_net <
        gpu_device_info.l1_cache_size_per_SM() * gpu_device_info.core_count()) {
      rest_bandwidth *= kL1CacheSpeedup;
    }
  } else {
    rest_bandwidth /= waste_factor;
  }
  dram_bandwidth = AdjustBandwidth(gpu_device_info, dram_bandwidth, num_blocks);
  rest_bandwidth = AdjustBandwidth(gpu_device_info, rest_bandwidth, num_blocks);
  int64_t n_bytes_read_dram = std::min(n_bytes_net, n_bytes_total);
  int64_t n_bytes_read_cache = n_bytes_total - n_bytes_read_dram;
  return absl::Seconds(n_bytes_read_dram / dram_bandwidth) +
         absl::Seconds(n_bytes_read_cache / rest_bandwidth);
}
 absl::Duration GpuPerformanceModelBase::ProducerInputAccessTime(
    const GpuHloCostAnalysis* cost_analysis,
    const se::DeviceDescription& gpu_device_info, int64_t num_blocks,
    const HloInstruction* producer, const HloFusionAnalysis& fusion_analysis,
    const GpuPerformanceModelOptions& config,
    const HloInstruction* fused_consumer) {
  absl::Duration ret = absl::ZeroDuration();
  float producer_output_utilization =
      fused_consumer
          ? GetOperandUtilization(cost_analysis, fused_consumer, producer)
          : 1.f;
  for (int i = 0; i < producer->operand_count(); ++i) {
    int64_t operand_bytes_accessed =
        cost_analysis->operand_bytes_accessed(*producer, i);
    float operand_utilization =
        cost_analysis->operand_utilization(*producer, i);
    int64_t n_bytes_net = std::llround(operand_bytes_accessed /
                                       std::max(operand_utilization, 1.0f));
    float common_utilization = GetCommonUtilization(
        cost_analysis, producer, i, fused_consumer);
    CHECK_LE(common_utilization, producer_output_utilization);
    float n_bytes_total = operand_bytes_accessed *
                          (producer_output_utilization - common_utilization);
    ret += ReadTime(gpu_device_info, num_blocks, n_bytes_net, n_bytes_total);
  }
  return ret;
}
absl::Duration GpuPerformanceModelBase::WriteTime(
    const se::DeviceDescription& gpu_device_info, int64_t bytes_written) {
  return absl::Seconds(1.0f * bytes_written /
                       gpu_device_info.memory_bandwidth());
}
absl::Duration GpuPerformanceModelBase::ComputeTime(
    const se::DeviceDescription& gpu_device_info, int64_t flops,
    int64_t num_blocks, int64_t num_threads_per_block) {
  int64_t n_active_fpus_per_core =
      std::min<int64_t>(num_threads_per_block, gpu_device_info.fpus_per_core());
  int64_t n_active_core =
      std::min<int64_t>(num_blocks, gpu_device_info.core_count());
  int64_t fpu_count = n_active_core * n_active_fpus_per_core;
  int64_t flop_per_ns_per_fpu = gpu_device_info.clock_rate_ghz() *  2;
  int64_t flop_per_ns_effective = flop_per_ns_per_fpu * fpu_count;
  return absl::Nanoseconds(1.0f * flops / flop_per_ns_effective);
}
absl::Duration GpuPerformanceModelBase::CombineComputeAndMemoryAccessTime(
    absl::Duration compute_time, absl::Duration memory_access_time,
    const GpuPerformanceModelOptions& config) {
  return compute_time + memory_access_time -
         std::min(compute_time, memory_access_time) *
             config.memory_compute_parallelism;
}
void GpuPerformanceModelBase::VLogOperandRead(const HloInstruction* operand,
                                              int64_t n_bytes_total,
                                              int64_t n_bytes_net,
                                              bool coalesced) {
  VLOG(8) << "operand " << operand->name()
          << ", n_bytes_total: " << n_bytes_total
          << ", n_bytes_net: " << n_bytes_net << ", coalesced: " << coalesced;
}
}  
}  