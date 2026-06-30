#include "xla/service/gpu/transforms/priority_fusion.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/fusion_deduplication_cache.h"
#include "xla/service/gpu/fusion_process_dump.pb.h"
#include "xla/service/gpu/fusions/triton/triton_support.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_indexing_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/model/triton_emitter_constraints.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/blocking_counter.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"
namespace xla {
namespace gpu {
namespace {
bool IsFusible(const HloInstruction& instr) {
  if (!instr.IsFusible()) {
    return false;
  }
  if (instr.IsElementwise()) {
    return true;
  }
  switch (instr.opcode()) {
    case HloOpcode::kFusion:
      return instr.fusion_kind() != HloInstruction::FusionKind::kCustom;
    case HloOpcode::kCopy:
    case HloOpcode::kIota:
    case HloOpcode::kConstant:
    case HloOpcode::kReduce:
    case HloOpcode::kBitcast:
    case HloOpcode::kBroadcast:
    case HloOpcode::kConcatenate:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kGather:
    case HloOpcode::kPad:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kScatter:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
      return true;
    default:
      return false;
  }
}
GpuBackendConfig GetTritonGpuBackendConfig(
    const BlockLevelParameters& block_level_parameters) {
  GpuBackendConfig gpu_backend_config;
  gpu_backend_config.mutable_fusion_backend_config()->set_kind(
      std::string(kTritonFusionKind));
  *gpu_backend_config.mutable_fusion_backend_config()
       ->mutable_block_level_fusion_config() =
      block_level_parameters.ToBlockLevelFusionConfig();
  return gpu_backend_config;
}
class PriorityFusionQueue {
  using Priority = absl::Duration;
  using CanFuseCallback = std::function<FusionDecision(
      HloInstruction* , int64_t )>;
 public:
  PriorityFusionQueue(HloComputation* computation,
                      const GpuHloCostAnalysis::Options& cost_analysis_options,
                      const se::DeviceDescription* device_info,
                      FusionProcessDumpProto* fusion_process_dump,
                      tsl::thread::ThreadPool* thread_pool,
                      mlir::MLIRContext* mlir_context,
                      HloFusionAnalysisCache& fusion_analysis_cache,
                      FusionDeduplicationCache& fusion_deduplication_cache,
                      bool triton_softmax_priority_fusion_enabled)
      : computation_(computation),
        device_info_(device_info),
        cost_analysis_(cost_analysis_options, *device_info),
        gpu_indexing_performance_model_(device_info, &fusion_analysis_cache,
                                        cost_analysis_options.shape_size,
                                        mlir_context),
        fusion_process_dump_(fusion_process_dump),
        thread_pool_(thread_pool),
        mlir_context_(mlir_context),
        fusion_analysis_cache_(fusion_analysis_cache),
        fusion_deduplication_cache_(fusion_deduplication_cache),
        triton_softmax_priority_fusion_enabled_(
            triton_softmax_priority_fusion_enabled) {
    VLOG(2) << "Running full HLO cost analysis for " << computation_->name();
    TF_CHECK_OK(computation_->Accept(&cost_analysis_));
    dump_fusion_visualization_ = computation->parent()
                                     ->config()
                                     .debug_options()
                                     .xla_dump_fusion_visualization();
    std::vector<HloInstruction*> instructions;
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      TF_CHECK_OK(UpdatePerformanceModelCache(instruction));
      if (instruction->opcode() == HloOpcode::kParameter ||
          instruction->user_count() == 0 || !instruction->IsFusible() ||
          instruction->opcode() == HloOpcode::kTuple ||
          instruction->opcode() == HloOpcode::kGetTupleElement) {
        continue;
      }
      instructions.push_back(instruction);
    }
    ComputeAndSetPriorities(instructions);
  }
  void ComputeAndSetPriorities(
      const std::vector<HloInstruction*>& instructions) {
    std::vector<Priority> priorities = ComputePriorities(instructions);
    for (auto [instruction, priority] : llvm::zip(instructions, priorities)) {
      auto key = std::make_pair(priority, instruction->unique_id());
      auto reverse_it = reverse_map_.find(instruction);
      if (reverse_it != reverse_map_.end()) {
        const PriorityQueue::iterator& queue_it = reverse_it->second;
        if (key == queue_it->first) {
          continue;
        }
        producer_priority_queue_.erase(queue_it);
        reverse_map_.erase(reverse_it);
      }
      if (priority < absl::ZeroDuration()) {
        continue;
      }
      auto emplace_result = producer_priority_queue_.emplace(key, instruction);
      reverse_map_.emplace(instruction, emplace_result.first);
    }
  }
  std::vector<Priority> ComputePriorities(
      const std::vector<HloInstruction*>& instructions) {
    auto schedule_or_run = [this](std::function<void()> fn) {
      if (thread_pool_) {
        thread_pool_->Schedule(std::move(fn));
      } else {
        fn();
      }
    };
    tsl::BlockingCounter counter(instructions.size());
    std::vector<Priority> priorities(instructions.size());
    for (size_t i = 0; i < instructions.size(); ++i) {
      schedule_or_run([&, i] {
        priorities[i] = CalculateProducerPriority(instructions[i]);
        counter.DecrementCount();
      });
    }
    counter.Wait();
    return priorities;
  }
  bool DequeueNextProducer() {
    current_producer_ = nullptr;
    current_consumers_.clear();
    while (!producer_priority_queue_.empty() && current_consumers_.empty()) {
      auto next_it = std::prev(producer_priority_queue_.end());
      current_producer_ = next_it->second;
      producer_priority_queue_.erase(next_it);
      reverse_map_.erase(current_producer_);
      current_consumers_ = current_producer_->users();
      if (current_producer_->opcode() == HloOpcode::kBitcast) {
        llvm::erase_if(current_consumers_, [&](HloInstruction* consumer) {
          return !CanFuseCached(current_producer_, consumer);
        });
      }
    }
    return !current_consumers_.empty();
  }
  absl::Status UpdatePerformanceModelCache(HloInstruction* producer) {
    bool is_triton_fusion = IsGenericTritonFusion(*producer);
    if (!IsFusible(*producer) && !is_triton_fusion) {
      return absl::OkStatus();
    }
    if (gpu_performance_model_cache_.Get(*producer)) {
      return absl::OkStatus();
    }
    EstimateRunTimeData runtime_data;
    if (is_triton_fusion) {
      TF_ASSIGN_OR_RETURN(
          runtime_data,
          gpu_indexing_performance_model_.EstimateRunTimeForTriton(producer));
    } else {
      auto config = GpuPerformanceModelOptions::PriorityFusion(
          &fusion_analysis_cache_, &gpu_performance_model_cache_);
      runtime_data = GpuPerformanceModel::EstimateRunTimeForInstruction(
          producer, *device_info_, &cost_analysis_, config);
    }
    gpu_performance_model_cache_.Set(*producer, runtime_data);
    return absl::OkStatus();
  }
  absl::Status UpdatePriorities() {
    for (auto instruction : to_update_priority_) {
      TF_RETURN_IF_ERROR(cost_analysis_.RevisitInstruction(instruction));
    }
    for (auto producer : to_update_priority_) {
      TF_RETURN_IF_ERROR(UpdatePerformanceModelCache(producer));
    }
    ComputeAndSetPriorities(std::vector<HloInstruction*>{
        to_update_priority_.begin(), to_update_priority_.end()});
    to_update_priority_.clear();
    operands_to_new_consumers_.clear();
    operands_to_removed_consumers_runtimes_.clear();
    return absl::OkStatus();
  }
  void PreFusion(HloInstruction* producer, HloInstruction* consumer) {
    if (dump_fusion_visualization_) {
      RegisterFusionState(
          *computation_,
          absl::StrCat("About to fuse |", producer->name(), "| into |",
                       consumer->name(), "| inside PriorityFusion"),
          *consumer, producer);
    }
  }
  void InvalidateCaches(HloInstruction* instruction) {
    can_fuse_cache_.erase(instruction);
    for (const HloInstruction* operand : instruction->operands()) {
      auto it = can_fuse_cache_.find(operand);
      if (it != can_fuse_cache_.end()) {
        it->second.erase(instruction);
      }
    }
    block_level_parameters_cache_.erase(instruction);
    for (const HloInstruction* operand : instruction->operands()) {
      auto it = block_level_parameters_cache_.find(operand);
      if (it != block_level_parameters_cache_.end()) {
        it->second.erase(instruction);
      }
    }
    gpu_performance_model_cache_.Invalidate(*instruction);
    fusion_analysis_cache_.Invalidate(*instruction);
    fusion_info_cache_.Invalidate(instruction);
  }
  void UpdateRuntimes(
      GpuPerformanceModel::RunTimes& runtimes, const HloInstruction* consumer,
      const absl::flat_hash_map<const HloInstruction*, absl::Duration>&
          original_consumers) {
    auto it = original_consumers.find(consumer);
    if (it != original_consumers.end()) {
      runtimes.time_fused += it->second;
      auto consumer_cache_result = gpu_performance_model_cache_.Get(*consumer);
      CHECK(consumer_cache_result.has_value());
      runtimes.time_unfused += (*consumer_cache_result).exec_time;
    }
  }
  void ComputeRuntimesOfRemovedConsumers() {
    for (const auto& pair : operands_to_new_consumers_) {
      auto operand = pair.first;
      if (!reverse_map_.contains(operand)) {
        continue;
      }
      if (!gpu_performance_model_cache_.ContainsConsumers(*operand)) {
        continue;
      }
      const auto& original_consumers =
          gpu_performance_model_cache_.GetAllConsumers(*operand);
      GpuPerformanceModel::RunTimes runtimes;
      for (auto consumer : current_consumers()) {
        UpdateRuntimes(runtimes, consumer, original_consumers);
      }
      UpdateRuntimes(runtimes, current_producer(), original_consumers);
      auto operand_cache_result = gpu_performance_model_cache_.Get(*operand);
      runtimes.time_unfused += (*operand_cache_result).exec_time +
                               GpuPerformanceModel::kKernelLaunchOverhead;
      operands_to_removed_consumers_runtimes_.emplace(operand, runtimes);
    }
  }
  void OnFusingInstruction(HloInstruction* fusion,
                           HloInstruction* original_producer,
                           HloInstruction* original_consumer) {
    if (fusion_process_dump_) {
      auto* fusion_step =
          fusion_process_dump_->add_fusion_steps()->mutable_fusion();
      fusion_step->set_fusion_name(std::string(fusion->name()));
      fusion_step->set_producer_name(std::string(original_producer->name()));
      fusion_step->set_consumer_name(std::string(original_consumer->name()));
    }
    if (dump_fusion_visualization_) {
      RegisterFusionState(
          *computation_,
          absl::StrCat("Fused |", original_producer->name(), "| into |",
                       fusion->name(), "| inside PriorityFusion"),
          *fusion);
    }
    if (fusion != original_consumer) {
      RemoveInstruction(original_consumer);
    }
    for (HloInstruction* operand : fusion->operands()) {
      if (operand == original_producer ||
          operand->opcode() == HloOpcode::kConstant ||
          operand->opcode() == HloOpcode::kGetTupleElement) {
        continue;
      }
      if (!operand->IsFusible()) {
        continue;
      }
      to_update_priority_.insert(operand);
      operands_to_new_consumers_[operand].push_back(fusion);
    }
    to_update_priority_.insert(fusion);
  }
  void RemoveInstruction(HloInstruction* instruction) {
    to_update_priority_.erase(instruction);
    fusion_analysis_cache_.Invalidate(*instruction);
    auto reverse_it = reverse_map_.find(instruction);
    if (reverse_it == reverse_map_.end()) {
      return;
    }
    producer_priority_queue_.erase(reverse_it->second);
    reverse_map_.erase(reverse_it);
  }
  absl::flat_hash_map<const HloInstruction*, BlockLevelParameters>
  GetBlockLevelParametersMap(const HloInstruction* producer) {
    auto it = block_level_parameters_cache_.find(producer);
    if (it == block_level_parameters_cache_.end()) {
      return {};
    }
    return it->second;
  }
  HloInstruction* current_producer() { return current_producer_; }
  const std::vector<HloInstruction*>& current_consumers() {
    return current_consumers_;
  }
 private:
  Priority CalculateProducerPriority(HloInstruction* producer) {
    if (producer->opcode() == HloOpcode::kBitcast) {
      return absl::InfiniteDuration();
    }
    if (producer->opcode() == HloOpcode::kConstant) {
      return -absl::InfiniteDuration();
    }
    if (auto fusion_decision = CanFuseWithAllNonBitcastUsers(producer);
        !fusion_decision) {
      if (fusion_process_dump_) {
        absl::MutexLock lock(&fusion_process_dump_mutex_);
        auto* step = fusion_process_dump_->add_fusion_steps()
                         ->mutable_producer_ineligible();
        step->set_producer_name(std::string(producer->name()));
        step->set_reason(fusion_decision.Explain());
      }
      return -absl::InfiniteDuration();
    }
    auto removed_consumers_runtime_it =
        operands_to_removed_consumers_runtimes_.find(producer);
    bool is_incremental_update = removed_consumers_runtime_it !=
                                 operands_to_removed_consumers_runtimes_.end();
    absl::Span<HloInstruction* const> fused_consumers =
        is_incremental_update
            ? operands_to_new_consumers_.find(producer)->second
            : absl::MakeConstSpan(producer->users());
    GpuPerformanceModel::RunTimes run_times =
        GpuPerformanceModel::EstimateRunTimesForPriorityFusion(
            producer, *device_info_, &cost_analysis_,
            GpuPerformanceModelOptions::PriorityFusion(
                &fusion_analysis_cache_, &gpu_performance_model_cache_),
            fused_consumers);
    Priority current_priority;
    if (is_incremental_update) {
      const GpuPerformanceModel::RunTimes& removed_consumers_runtime =
          removed_consumers_runtime_it->second;
      run_times.time_unfused -= removed_consumers_runtime.time_unfused;
      run_times.time_fused -= removed_consumers_runtime.time_fused;
      const PriorityQueue::iterator& queue_it =
          FindOrDie(reverse_map_, producer);
      current_priority = queue_it->first.first;
    }
    if (fusion_process_dump_) {
      absl::MutexLock lock(&fusion_process_dump_mutex_);
      auto* step =
          fusion_process_dump_->add_fusion_steps()->mutable_update_priority();
      step->set_producer_name(std::string(producer->name()));
      for (auto* consumer : producer->users()) {
        step->add_consumer_names(std::string(consumer->name()));
      }
      step->set_us_fused(absl::ToDoubleMicroseconds(run_times.time_fused));
      step->set_us_unfused(absl::ToDoubleMicroseconds(run_times.time_unfused));
    }
    return current_priority + run_times.time_unfused - run_times.time_fused;
  }
  FusionDecision IsTritonSupported(const HloInstruction& instruction) {
    if (instruction.opcode() != HloOpcode::kFusion) {
      return IsTritonSupportedInstruction(
          instruction, device_info_->gpu_compute_capability());
    }
    for (const HloInstruction* instruction :
         instruction.fused_instructions_computation()->instructions()) {
      if (auto codegen_decision = IsTritonSupportedInstruction(
              *instruction, device_info_->gpu_compute_capability());
          !codegen_decision) {
        return codegen_decision;
      }
    }
    return FusionDecision::Allow();
  }
  TiledRunTimeDataOrError GetTiledRunTimeDataCached(
      const HloInstruction* producer, const HloInstruction* consumer) {
    FusionDeduplicationCache::FusionId fusion_id = [&]() {
      absl::MutexLock lock(&fusion_deduplication_cache_mutex_);
      return fusion_deduplication_cache_.GetFusionId(*producer, *consumer);
    }();
    {
      absl::MutexLock lock(&tiled_run_time_data_cache_mutex_);
      auto it = tiled_run_time_data_cache_.find(fusion_id);
      if (it != tiled_run_time_data_cache_.end()) {
        return it->second;
      }
    }
    auto fusion = HloFusionAdaptor::ForProducerConsumer(producer, consumer);
    absl::StatusOr<TiledRunTimeDataOrError> result_or_status =
        gpu_indexing_performance_model_.TryFindBestTilingForFusion(*fusion);
    TiledRunTimeDataOrError tiled_run_time_data_or_error =
        [&]() -> TiledRunTimeDataOrError {
      if (result_or_status.ok()) {
        return *result_or_status;
      } else {
        return FusionDecision::Forbid(
            absl::StrCat("TiledRunTimeDataOrError return status: ",
                         result_or_status.status().message()));
      }
    }();
    if (const auto* fusion_decision =
            std::get_if<FusionDecision>(&tiled_run_time_data_or_error)) {
      tiled_run_time_data_or_error = FusionDecision::Forbid(
          absl::StrCat("Fusion can not be tiled with SymbolicTileAnalysis: ",
                       fusion_decision->Explain()));
    }
    absl::MutexLock lock(&tiled_run_time_data_cache_mutex_);
    tiled_run_time_data_cache_.emplace(fusion_id, tiled_run_time_data_or_error);
    return tiled_run_time_data_or_error;
  }
  FusionDecision CanFuseTriton(HloInstruction* producer,
                               HloInstruction* consumer) {
    if (!triton_softmax_priority_fusion_enabled_) {
      return FusionDecision::Forbid("triton softmax fusion is not enabled");
    }
    if (IsGenericTritonFusion(*producer)) {
      if (!IsFusible(*consumer)) {
        return FusionDecision::Forbid("the consumer is not fusible");
      }
      if (auto fusion_decision = IsTritonSupported(*consumer);
          !fusion_decision) {
        return fusion_decision;
      }
    } else {
      if (!IsFusible(*producer)) {
        return FusionDecision::Forbid("the producer is not fusible");
      }
      if (auto fusion_decision = IsTritonSupported(*producer);
          !fusion_decision) {
        return fusion_decision;
      }
    }
    TiledRunTimeDataOrError tiled_run_time_data_or_error =
        GetTiledRunTimeDataCached(producer, consumer);
    if (const auto* fusion_decision =
            std::get_if<FusionDecision>(&tiled_run_time_data_or_error)) {
      return *fusion_decision;
    }
    TiledRunTimeData tiled_run_time_data =
        std::get<TiledRunTimeData>(std::move(tiled_run_time_data_or_error));
    gpu_performance_model_cache_.Set(
        *producer, *consumer, tiled_run_time_data.runtime_data.exec_time);
    {
      absl::MutexLock lock(&block_level_parameters_cache_mutex_);
      block_level_parameters_cache_[producer][consumer] =
          tiled_run_time_data.block_level_parameters;
    }
    return FusionDecision::Allow();
  }
  FusionDecision CanFuse(HloInstruction* producer, HloInstruction* consumer) {
    if (IsGenericTritonFusion(*producer) || IsGenericTritonFusion(*consumer)) {
      return CanFuseTriton(producer, consumer);
    }
    if (!IsFusible(*producer)) {
      return FusionDecision::Forbid("the producer is not fusible");
    }
    if (!IsFusible(*consumer)) {
      return FusionDecision::Forbid("the consumer is not fusible");
    }
    if (consumer->opcode() == HloOpcode::kBitcast) {
      return FusionDecision::Forbid(
          "not fusing into a single bitcast as consumer");
    }
    if (auto can_fuse = CanEmitInputFusedScatter(*producer, *consumer);
        !can_fuse) {
      return can_fuse;
    }
    auto contains_significant_reduce = [&](const HloInstruction* instr) {
      auto fusion = HloFusionAdaptor::ForInstruction(instr);
      return HloAnyOf(*fusion, [](auto node) {
        if (!(node.opcode() == HloOpcode::kReduce && node.shape().IsArray())) {
          return false;
        }
        int64_t reduction_size =
            ShapeUtil::ElementsIn(node.instruction().operand(0)->shape()) /
            ShapeUtil::ElementsIn(node.shape());
        return reduction_size >= 16;
      });
    };
    if (contains_significant_reduce(producer) &&
        contains_significant_reduce(consumer)) {
      return FusionDecision::Forbid(
          "both the producer and the consumer contain a reduce");
    }
    const auto& analysis = fusion_analysis_cache_.Get(*producer);
    if (analysis.GetEmitterFusionKind() ==
        HloFusionAnalysis::EmitterFusionKind::kReduction) {
      const auto& analysis_fused =
          fusion_analysis_cache_.Get(*producer, *consumer);
      if (analysis_fused.GetEmitterFusionKind() ==
          HloFusionAnalysis::EmitterFusionKind::kLoop) {
        return FusionDecision::Forbid(
            "fusion into output of a reduce fusion would create a loop fusion");
      }
    }
    if (auto fits_budget = FusionFitsInBudget(
            *consumer, *producer, *device_info_,
            true, &fusion_info_cache_);
        !fits_budget) {
      return fits_budget;
    }
    if (cost_analysis_.ProducerConsumerMergedTooLarge(*producer, *consumer)) {
      return FusionDecision::Forbid(
          "the fusion would result in an overly large code duplication");
    }
    if (producer == producer->parent()->root_instruction()) {
      return FusionDecision::Forbid(
          "not fusing into the output of the root instruction");
    }
    return InstructionFusion::ShouldFuseInPlaceOp(producer, consumer);
  }
  FusionDecision CanFuseCached(HloInstruction* producer,
                               HloInstruction* consumer) {
    {
      absl::MutexLock lock(&can_fuse_cache_mutex_);
      auto& producer_cache = can_fuse_cache_[producer];
      auto it = producer_cache.find(consumer);
      if (it != producer_cache.end()) {
        return it->second;
      }
    }
    auto fusion_decision = CanFuse(producer, consumer);
    {
      absl::MutexLock lock(&can_fuse_cache_mutex_);
      can_fuse_cache_[producer].insert_or_assign(consumer, fusion_decision);
    }
    return fusion_decision;
  }
  FusionDecision CanFuseWithAllNonBitcastUsers(HloInstruction* producer) {
    if (producer->users().empty()) {
      return FusionDecision::Forbid("No users to fuse");
    }
    bool has_non_bitcast_user = false;
    for (const auto& user : producer->users()) {
      if (user->opcode() == HloOpcode::kBitcast) {
        continue;
      }
      has_non_bitcast_user = true;
      if (auto fusion_decision = CanFuseCached(producer, user);
          !fusion_decision) {
        VLOG(10) << "Cannot fuse " << producer->name() << " with "
                 << user->name() << ", because: " << fusion_decision.Explain();
        return fusion_decision;
      }
    }
    if (!has_non_bitcast_user) {
      return FusionDecision::Forbid(
          "not fusing because there are only bitcast users");
    }
    return FusionDecision::Allow();
  }
  HloComputation* computation_;
  const se::DeviceDescription* device_info_;
  GpuHloCostAnalysis cost_analysis_;
  GpuPerformanceModelWithIndexingAnalysis gpu_indexing_performance_model_;
  using PriorityQueue = std::map<std::pair<Priority, int>, HloInstruction*>;
  PriorityQueue producer_priority_queue_;
  absl::flat_hash_map<HloInstruction*, PriorityQueue::iterator> reverse_map_;
  HloInstruction* current_producer_;
  std::vector<HloInstruction*> current_consumers_;
  absl::flat_hash_set<HloInstruction*> to_update_priority_;
  absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>
      operands_to_new_consumers_;
  absl::flat_hash_map<HloInstruction*, GpuPerformanceModel::RunTimes>
      operands_to_removed_consumers_runtimes_;
  FusionProcessDumpProto* fusion_process_dump_;
  absl::Mutex fusion_process_dump_mutex_;
  tsl::thread::ThreadPool* thread_pool_;
  mlir::MLIRContext* mlir_context_;
  HloFusionAnalysisCache& fusion_analysis_cache_;
  FusionDeduplicationCache& fusion_deduplication_cache_;
  absl::Mutex fusion_deduplication_cache_mutex_;
  absl::flat_hash_map<
      const HloInstruction*,
      absl::flat_hash_map<const HloInstruction*, FusionDecision>>
      can_fuse_cache_;
  absl::Mutex can_fuse_cache_mutex_;
  absl::flat_hash_map<
      const HloInstruction*,
      absl::flat_hash_map<const HloInstruction*, BlockLevelParameters>>
      block_level_parameters_cache_;
  absl::Mutex block_level_parameters_cache_mutex_;
  absl::flat_hash_map<FusionDeduplicationCache::FusionId,
                      TiledRunTimeDataOrError>
      tiled_run_time_data_cache_;
  absl::Mutex tiled_run_time_data_cache_mutex_;
  GpuPerformanceModelCache gpu_performance_model_cache_;
  FusionInfoCache fusion_info_cache_;
  bool triton_softmax_priority_fusion_enabled_;
  bool dump_fusion_visualization_;
};
}  
bool IsSmallConstant(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kConstant && instr->shape().IsArray() &&
         ShapeUtil::ElementsIn(instr->shape()) <= 1;
}
bool PriorityFusion::ConsumeFuel(HloInstruction* producer,
                                 HloInstruction* consumer) {
  return xla::ConsumeFuel(name(), [&] {
    return absl::StrFormat("Not fusing producer %s with consumer %s",
                           producer->name(), consumer->name());
  });
};
absl::StatusOr<bool> PriorityFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool dump_enabled =
      DumpingEnabledForHloPass(name(), module->config().debug_options());
  if (dump_enabled) {
    fusion_process_dump_ = std::make_unique<FusionProcessDumpProto>();
    *fusion_process_dump_->mutable_gpu_device_info() =
        device_info_.ToGpuProto();
  }
  auto fusible_computations =
      GetFusibleComputations(*module, execution_threads);
  for (auto* computation : fusible_computations) {
    for (auto* instruction : computation->instructions()) {
      module->SetAndUniquifyInstrName(instruction,
                                      absl::StrCat(instruction->name(), ".0"));
    }
  }
  if (dump_enabled) {
    fusion_process_dump_->set_hlo_module_before_fusion(
        module->ToString(HloPrintOptions::ShortParsable()));
  }
  bool triton_softmax_priority_fusion_enabled =
      module->config()
          .debug_options()
          .xla_gpu_experimental_enable_triton_softmax_priority_fusion();
  FusionDeduplicationCache fusion_deduplication_cache =
      FusionDeduplicationCache::Create(*module);
  int changed = false;
  for (auto* computation : fusible_computations) {
    CHECK(!computation->IsFusionComputation());
    auto fusion_queue = std::make_unique<PriorityFusionQueue>(
        computation, cost_analysis_options_, &device_info_,
        fusion_process_dump_.get(), thread_pool_, &mlir_context_,
        fusion_analysis_cache_, fusion_deduplication_cache,
        triton_softmax_priority_fusion_enabled);
    while (fusion_queue->DequeueNextProducer()) {
      auto producer = fusion_queue->current_producer();
      absl::flat_hash_map<const HloInstruction*, BlockLevelParameters>
          block_level_parameters_map =
              fusion_queue->GetBlockLevelParametersMap(producer);
      for (auto* consumer : fusion_queue->current_consumers()) {
        if (consumer->opcode() == HloOpcode::kBitcast) {
          continue;
        }
        if (!ConsumeFuel(producer, consumer)) continue;
        VLOG(5) << "next: " << consumer->name() << "(" << consumer << ") + "
                << producer->name() << "(" << producer << ")";
        int64_t consumer_operand_index = consumer->operand_index(producer);
        fusion_queue->PreFusion(producer, consumer);
        auto fusion_instruction = Fuse(producer, consumer);
        fusion_deduplication_cache.UpdateFusedInstructionId(
            *fusion_instruction, *producer, *consumer, consumer_operand_index);
        fusion_queue->OnFusingInstruction(fusion_instruction, producer,
                                          consumer);
        auto backend_config_it = block_level_parameters_map.find(consumer);
        if (backend_config_it != block_level_parameters_map.end()) {
          TF_RETURN_IF_ERROR(fusion_instruction->set_backend_config(
              GetTritonGpuBackendConfig(backend_config_it->second)));
        }
        changed = true;
      }
      fusion_queue->ComputeRuntimesOfRemovedConsumers();
      if (producer->user_count() == 0) {
        fusion_queue->InvalidateCaches(producer);
        producer->DetachFromOperandsAndUsers();
        fusion_queue->RemoveInstruction(producer);
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(producer));
      }
      for (auto* consumer : fusion_queue->current_consumers()) {
        fusion_queue->InvalidateCaches(consumer);
      }
      TF_RETURN_IF_ERROR(fusion_queue->UpdatePriorities());
    }
    std::vector<HloInstruction*> constants;
    for (auto* instruction : computation->instructions()) {
      if (IsSmallConstant(instruction)) {
        constants.push_back(instruction);
      }
    }
    for (auto* constant : constants) {
      auto users = constant->users();
      for (auto* user : users) {
        if ((IsFusible(*user) || IsGenericTritonFusion(*user)) &&
            CanEmitInputFusedScatter(*constant, *user)) {
          Fuse(constant, user);
          changed = true;
        }
      }
    }
  }
  fusion_analysis_cache_.Clear();
  if (dump_enabled) {
    DumpPerModuleProtobufToFile(*module, *fusion_process_dump_,
                                module->config().debug_options(),
                                "priority_fusion_dump");
  }
  return changed;
}
HloInstruction::FusionKind PriorityFusion::ChooseKind(
    const HloInstruction* producer, const HloInstruction* consumer) {
  const auto& analysis = fusion_analysis_cache_.Get(*producer, *consumer);
  switch (analysis.GetEmitterFusionKind()) {
    case HloFusionAnalysis::EmitterFusionKind::kLoop:
      return HloInstruction::FusionKind::kLoop;
    case HloFusionAnalysis::EmitterFusionKind::kTriton:
    case HloFusionAnalysis::EmitterFusionKind::kCustomFusion:
    case HloFusionAnalysis::EmitterFusionKind::kCuDnn:
      return HloInstruction::FusionKind::kCustom;
    case HloFusionAnalysis::EmitterFusionKind::kConcatenate:
    case HloFusionAnalysis::EmitterFusionKind::kReduction:
    case HloFusionAnalysis::EmitterFusionKind::kTranspose:
    case HloFusionAnalysis::EmitterFusionKind::kInputSlices:
    case HloFusionAnalysis::EmitterFusionKind::kScatter:
      return HloInstruction::FusionKind::kInput;
  }
}
HloInstruction* PriorityFusion::Fuse(HloInstruction* producer,
                                     HloInstruction* consumer) {
  VLOG(2) << "Fusing " << producer->ToString() << " into "
          << consumer->ToString();
  HloComputation* computation = consumer->parent();
  auto kind = ChooseKind(producer, consumer);
  HloInstruction* fusion_instruction = consumer;
  if (fusion_instruction->opcode() != HloOpcode::kFusion) {
    fusion_instruction = computation->AddInstruction(
        HloInstruction::CreateFusion(consumer->shape(), kind, consumer));
    TF_CHECK_OK(computation->ReplaceInstruction(consumer, fusion_instruction));
  } else if (kind != fusion_instruction->fusion_kind()) {
    fusion_instruction->set_fusion_kind(kind);
  }
  fusion_instruction->set_called_computations_execution_thread(
      computation->execution_thread(),
      false);
  if (producer->opcode() == HloOpcode::kFusion) {
    fusion_instruction->MergeFusionInstruction(producer);
  } else {
    fusion_instruction->FuseInstruction(producer);
  }
  if (fusion_instruction != consumer) {
    VLOG(2) << "       created new fusion: " << fusion_instruction->ToString();
  }
  return fusion_instruction;
}
}  
}  