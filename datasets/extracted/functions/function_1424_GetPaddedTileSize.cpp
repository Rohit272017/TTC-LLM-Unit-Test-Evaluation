#include "xla/service/gpu/model/gpu_indexing_performance_model.h"
#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "llvm/Support/MathExtras.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/fusions/triton.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/coalescing_analysis.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/model/symbolic_tile_analysis.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/model/triton_emitter_constraints.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace gpu {
namespace {
struct OperandReadInfo {
  int64_t total_bytes_read = 0;
  int64_t is_coalesced = true;
};
int64_t GetPaddedTileSize(absl::Span<int64_t const> tile_sizes) {
  int64_t result = 1;
  for (int64_t tile_size : tile_sizes) {
    result *= llvm::PowerOf2Ceil(tile_size);
  }
  return result;
}
bool DoesTileFitsInRegisters(int64_t tile_size,
                             const se::DeviceDescription& device_info) {
  constexpr double kFractionOfRegistersAvailableToStoreTile = 0.4;
  return tile_size <= kFractionOfRegistersAvailableToStoreTile *
                          device_info.registers_per_block_limit();
}
int64_t GetNumWarps(int64_t tile_size) {
  if (tile_size <= 512) return 1;
  if (tile_size <= 1024) return 2;
  if (tile_size <= 16384) return 4;
  if (tile_size <= 32768) return 8;
  if (tile_size <= 65536) return 16;
  return 32;
}
}  
int64_t GpuPerformanceModelWithIndexingAnalysis::FlopsPerElement(
    const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kBitcast:
    case HloOpcode::kBroadcast:
    case HloOpcode::kConstant:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kGather:
    case HloOpcode::kIota:
    case HloOpcode::kPad:
    case HloOpcode::kParameter:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
    case HloOpcode::kTuple:
      return 0;
    default:
      break;
  };
  if (instr->IsElementwise()) {
    return cost_analysis_.GetFlopsPerElementwiseOpElement(
        instr->shape().element_type(), instr->opcode());
  }
  if (instr->opcode() == HloOpcode::kReduce) {
    int64_t flops_per_reduce_computation = 0;
    for (const HloInstruction* reducer_instr :
         instr->called_computations()[0]->instructions()) {
      flops_per_reduce_computation += FlopsPerElement(reducer_instr);
    }
    auto operand_shape = instr->operand(0)->shape();
    auto output_shape = instr->shape().IsArray()
                            ? instr->shape()
                            : instr->shape().tuple_shapes(0);
    int64_t reduction_factor = ShapeUtil::ElementsIn(operand_shape) /
                               ShapeUtil::ElementsIn(output_shape);
    return (reduction_factor - 1) * flops_per_reduce_computation;
  }
  TF_CHECK_OK(
      cost_analysis_.RevisitInstruction(const_cast<HloInstruction*>(instr)));
  return cost_analysis_.flop_count(*instr) /
         ShapeUtil::ElementsInRecursive(instr->shape());
}
int64_t GpuPerformanceModelWithIndexingAnalysis::GetShapeSizeRecursive(
    const Shape& shape) const {
  CHECK(shape.IsArray() || shape.IsTuple());
  if (shape.IsArray()) {
    return shape_size_(shape);
  }
  int64_t total_size = 0;
  for (const auto& element_shape : shape.tuple_shapes()) {
    total_size += GetShapeSizeRecursive(element_shape);
  }
  return total_size;
}
int64_t GetIterationSpaceSize(const IndexingMap& indexing_map,
                              const HloInstruction* instr) {
  if (indexing_map.IsUndefined()) {
    return ShapeUtil::ElementsInRecursive(instr->shape());
  }
  if (indexing_map.IsKnownEmpty()) {
    return 0;
  }
  auto get_ranges_iteration_space_size =
      [](const std::vector<Interval>& ranges) {
        int64_t num_iters = 1;
        for (const Interval& range : ranges) {
          num_iters *= range.upper - range.lower + 1;
        }
        return num_iters;
      };
  return get_ranges_iteration_space_size(indexing_map.GetSymbolBounds()) *
         get_ranges_iteration_space_size(indexing_map.GetDimensionBounds());
}
EstimateRunTimeData
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimeForFusion(
    const HloFusionAnalysis& fusion_analysis, bool is_coalesced) {
  auto& fusion_adaptor = fusion_analysis.fusion();
  VLOG(5) << "EstimateRunTimeForFusion: " << fusion_adaptor.ToString();
  auto roots = fusion_adaptor.GetRoots();
  CHECK_EQ(roots.size(), 1)
      << "Indexing cost model doesn't support multi-output fusions.";
  auto root_shape = roots.front().shape();
  LaunchDimensions launch_dimensions =
      EstimateFusionLaunchDimensions(fusion_analysis);
  int64_t num_blocks = launch_dimensions.num_blocks();
  auto grouped_fusion_indexing = ComputeGroupedOutputToInputIndexing(
      fusion_adaptor, roots[0], mlir_context_);
  int64_t flops = 0;
  int64_t bytes_read = 0;
  absl::Duration read_time = absl::ZeroDuration();
  for (const auto& [instr, indexing_maps] : grouped_fusion_indexing) {
    VLOG(10) << "instr: " << instr->name();
    bool is_operand = !fusion_adaptor.ContainsInstruction(instr);
    auto element_type = instr->shape().element_type();
    int64_t n_bytes_total = 0;
    for (const auto& indexing_map : indexing_maps) {
      VLOG(10) << indexing_map;
      int64_t num_iters = GetIterationSpaceSize(indexing_map, instr);
      if (is_operand) {
        int64_t type_size = ShapeUtil::ByteSizeOfPrimitiveType(element_type);
        n_bytes_total += type_size * num_iters;
      } else {
        int64_t flops_per_element = FlopsPerElement(instr);
        flops += flops_per_element * num_iters;
      }
    }
    if (is_operand) {
      int64_t operand_size = shape_size_(instr->shape());
      int64_t n_bytes_net = std::min(operand_size, n_bytes_total);
      bytes_read += n_bytes_total;
      VLogOperandRead(instr, n_bytes_total, n_bytes_net, is_coalesced);
      read_time +=
          ReadTimeWithDRAMHeuristic(*device_info_, num_blocks, n_bytes_net,
                                    n_bytes_total, element_type, is_coalesced);
    }
  }
  int64_t bytes_written = GetShapeSizeRecursive(root_shape);
  absl::Duration compute_time =
      ComputeTime(*device_info_, flops, num_blocks,
                  launch_dimensions.num_threads_per_block());
  absl::Duration write_time = WriteTime(*device_info_, bytes_written);
  absl::Duration memory_access_time = read_time + write_time;
  absl::Duration exec_time = CombineComputeAndMemoryAccessTime(
      compute_time, memory_access_time,
      GpuPerformanceModelOptions::PriorityFusion());
  EstimateRunTimeData runtime_data = {flops,     bytes_read, bytes_written,
                                      read_time, write_time, compute_time,
                                      exec_time};
  VLOG(3) << "Runtime data for HLO fusion: " << fusion_adaptor.ToString()
          << "\n"
          << launch_dimensions.ToString() << "\n"
          << runtime_data.ToString();
  return runtime_data;
}
EstimateRunTimeData
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimeForInstruction(
    const HloInstruction* producer) {
  if (producer->opcode() == HloOpcode::kBitcast) {
    return EstimateRunTimeData::Zero();
  }
  auto fusion_analysis = HloFusionAnalysis::Create(*producer, *device_info_);
  bool is_coalesced = IsReadCoalescedHeuristic(
      fusion_analysis.GetEmitterFusionKind(), producer);
  return EstimateRunTimeForFusion(fusion_analysis, is_coalesced);
}
EstimateRunTimeData
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimeForProducerConsumer(
    const HloInstruction* producer, const HloInstruction* consumer) {
  auto fusion_analysis =
      HloFusionAnalysis::Create(*producer, *consumer, *device_info_);
  bool is_coalesced = IsReadCoalescedHeuristic(
      fusion_analysis.GetEmitterFusionKind(), producer, consumer);
  return EstimateRunTimeForFusion(fusion_analysis, is_coalesced);
}
GpuPerformanceModelWithIndexingAnalysis::RunTimes
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimes(
    const HloInstruction* producer,
    absl::Span<const HloInstruction* const> fused_consumers) {
  auto producer_runtime = EstimateRunTimeForInstruction(producer);
  absl::Duration time_unfused =
      kKernelLaunchOverhead * (fused_consumers.size() + 1) +
      producer_runtime.exec_time;
  absl::Duration time_fused = kKernelLaunchOverhead * fused_consumers.size();
  for (const auto& consumer : fused_consumers) {
    time_unfused += EstimateRunTimeForInstruction(consumer).exec_time;
    time_fused +=
        EstimateRunTimeForProducerConsumer(producer, consumer).exec_time;
  }
  return {time_unfused, time_fused};
}
absl::StatusOr<EstimateRunTimeData>
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimeForTiledHloComputation(
    const HloFusionAdaptor& fusion_adaptor,
    const TiledHloComputation& tiled_hlo_computation,
    const LaunchDimensions& launch_dimensions) {
  absl::flat_hash_map<const HloInstruction*, OperandReadInfo> n_bytes_total_map;
  int64_t flops = 0;
  int64_t bytes_read = 0;
  int64_t num_blocks = launch_dimensions.num_blocks();
  for (const auto& tiled_hlo : tiled_hlo_computation.instructions()) {
    int64_t padded_tile_size = GetPaddedTileSize(tiled_hlo->tile_sizes());
    if (!DoesTileFitsInRegisters(padded_tile_size, *device_info_)) {
      return EstimateRunTimeData::Infinite();
    }
    const HloInstruction* hlo = tiled_hlo->hlo();
    if (fusion_adaptor.ContainsInstruction(hlo)) {
      if (hlo->opcode() == HloOpcode::kConcatenate) {
        return absl::FailedPreconditionError(
            "Concatenate is not supported by the indexing cost model.");
      }
      int64_t num_elements = num_blocks * padded_tile_size;
      flops += FlopsPerElement(hlo) * num_elements;
    } else {
      int64_t tile_size = Product(tiled_hlo->tile_sizes());
      int64_t num_elements = num_blocks * tile_size;
      int64_t element_type_size =
          ShapeUtil::ByteSizeOfPrimitiveType(hlo->shape().element_type());
      int64_t tile_bytes_read = element_type_size * num_elements;
      bytes_read += tile_bytes_read;
      bool is_coalesced =
          IsTiledReadCoalescedHeuristic(*tiled_hlo, *device_info_);
      OperandReadInfo& operand_read_info = n_bytes_total_map[hlo];
      operand_read_info.total_bytes_read += tile_bytes_read;
      operand_read_info.is_coalesced &= is_coalesced;
    }
  }
  absl::Duration read_time = absl::ZeroDuration();
  for (const auto& [hlo, operand_read_info] : n_bytes_total_map) {
    int64_t operand_size = shape_size_(hlo->shape());
    int64_t n_bytes_net =
        std::min(operand_size, operand_read_info.total_bytes_read);
    read_time +=
        ReadTimeWithDRAMHeuristic(*device_info_, num_blocks, n_bytes_net,
                                  operand_read_info.total_bytes_read,
                                  hlo->shape().element_type(),
                                  operand_read_info.is_coalesced);
  }
  int64_t bytes_written =
      GetShapeSizeRecursive(tiled_hlo_computation.GetRoot()->hlo()->shape());
  absl::Duration compute_time =
      ComputeTime(*device_info_, flops, launch_dimensions.num_blocks(),
                  launch_dimensions.num_threads_per_block());
  absl::Duration write_time = WriteTime(*device_info_, bytes_written);
  absl::Duration memory_access_time = read_time + write_time;
  absl::Duration exec_time = CombineComputeAndMemoryAccessTime(
      compute_time, memory_access_time,
      GpuPerformanceModelOptions::PriorityFusion());
  return EstimateRunTimeData{flops,
                             bytes_read,
                             bytes_written,
                             read_time,
                             write_time,
                             compute_time,
                             exec_time};
}
absl::StatusOr<EstimateRunTimeData>
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimeForTiledFusion(
    const HloFusionAdaptor& fusion_adaptor,
    const LaunchDimensions& launch_dimensions,
    absl::Span<const int64_t> tile_sizes) {
  SymbolicTileAnalysisOrError analysis_or_error =
      SymbolicTileAnalysis::AnalyzeFusion(
          fusion_adaptor, mlir_context_,
          nullptr);
  if (const auto* fusion_decision =
          std::get_if<FusionDecision>(&analysis_or_error)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "SymbolicTileAnalysis failed. ", fusion_decision->Explain()));
  }
  SymbolicTileAnalysis analysis =
      std::get<SymbolicTileAnalysis>(std::move(analysis_or_error));
  TF_ASSIGN_OR_RETURN(TiledHloComputation tiled_hlo_computation,
                      analysis.ComputeTiledHloInstructions(tile_sizes));
  return EstimateRunTimeForTiledHloComputation(
      fusion_adaptor, tiled_hlo_computation, launch_dimensions);
}
absl::StatusOr<EstimateRunTimeData>
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimeForTriton(
    const HloInstruction* producer, const HloInstruction* consumer) {
  const auto& fusion_analysis =
      (consumer == nullptr) ? fusion_analysis_cache_->Get(*producer)
                            : fusion_analysis_cache_->Get(*producer, *consumer);
  auto launch_config = TritonFusion(fusion_analysis).launch_config();
  if (!launch_config.has_value()) {
    return absl::InvalidArgumentError(
        "Could not get launch config for Triton fusion.");
  }
  return EstimateRunTimeForTiledFusion(
      fusion_analysis.fusion(), launch_config->launch_dimensions,
      launch_config->block_level_parameters.output_tile_sizes);
}
LaunchDimensions
GpuPerformanceModelWithIndexingAnalysis::GetLaunchDimensionsForTiledFusion(
    const TiledHloComputation& tiled_hlo_computation) {
  int64_t num_blocks = tiled_hlo_computation.num_output_tiles();
  int64_t largest_live_tile_size = 1;
  for (const auto& tiled_hlo : tiled_hlo_computation.instructions()) {
    largest_live_tile_size = std::max(
        largest_live_tile_size, GetPaddedTileSize(tiled_hlo->tile_sizes()));
  }
  int64_t num_warps = GetNumWarps(largest_live_tile_size);
  return {static_cast<uint64_t>(num_blocks),
          static_cast<uint64_t>(num_warps * WarpSize())};
}
absl::StatusOr<TiledRunTimeDataOrError>
GpuPerformanceModelWithIndexingAnalysis::TryFindBestTilingForFusion(
    const HloFusionAdaptor& fusion_adaptor) {
  SymbolicTileAnalysisOrError analysis_or_error =
      SymbolicTileAnalysis::AnalyzeFusion(
          fusion_adaptor, mlir_context_,
          TritonEmitterConstraints::GetBuilder(*device_info_));
  if (const auto* fusion_decision =
          std::get_if<FusionDecision>(&analysis_or_error)) {
    return *fusion_decision;
  }
  SymbolicTileAnalysis analysis =
      std::get<SymbolicTileAnalysis>(std::move(analysis_or_error));
  TF_ASSIGN_OR_RETURN(auto tilings, analysis.GetGoodTilings());
  std::optional<TiledRunTimeData> best_tiled_run_time_data;
  for (const auto& tiling : tilings) {
    TF_ASSIGN_OR_RETURN(TiledHloComputation tiled_hlo_computation,
                        analysis.ComputeTiledHloInstructions(tiling));
    LaunchDimensions launch_dimensions =
        GetLaunchDimensionsForTiledFusion(tiled_hlo_computation);
    TF_ASSIGN_OR_RETURN(
        EstimateRunTimeData estimate_run_time_data,
        EstimateRunTimeForTiledHloComputation(
            fusion_adaptor, tiled_hlo_computation, launch_dimensions));
    if (!best_tiled_run_time_data.has_value() ||
        estimate_run_time_data.exec_time <
            best_tiled_run_time_data->runtime_data.exec_time) {
      BlockLevelParameters block_level_parameters;
      block_level_parameters.output_tile_sizes =
          std::vector<int64_t>(tiling.begin(), tiling.end());
      block_level_parameters.num_warps =
          launch_dimensions.num_threads_per_block() / WarpSize();
      best_tiled_run_time_data =
          TiledRunTimeData{estimate_run_time_data, block_level_parameters};
    }
  }
  if (!best_tiled_run_time_data.has_value()) {
    return FusionDecision::Forbid("No valid tilings found.");
  }
  return *best_tiled_run_time_data;
}
}  
}  