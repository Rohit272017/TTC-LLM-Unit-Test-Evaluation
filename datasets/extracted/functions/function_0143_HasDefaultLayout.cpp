#include "xla/service/gpu/transforms/softmax_rewriter_triton.h"
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/fusion_pipeline.h"
#include "xla/service/gpu/fusions/triton/triton_support.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_indexing_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/gpu/model/symbolic_tile_analysis.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/model/triton_emitter_constraints.h"
#include "xla/service/gpu/transforms/reduction_dimension_grouper.h"
#include "xla/service/gpu/transforms/reduction_splitter.h"
#include "xla/service/gpu/transforms/tree_reduction_rewriter.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace gpu {
namespace {
using hlo_query::IsBroadcastOfParameter;
using hlo_query::IsBroadcastOfScalarConstant;
bool HasDefaultLayout(const Shape& shape) {
  return shape.has_layout() &&
         LayoutUtil::IsMonotonicWithDim0Major(shape.layout());
}
bool TrivialEdge(HloInstruction** producer, HloInstruction* consumer,
                 HloOpcode opcode, const se::GpuComputeCapability& gpu_version);
bool BitcastIsTilingNoop(HloInstruction* bitcast,
                         const se::GpuComputeCapability& gpu_version) {
  CHECK_EQ(bitcast->opcode(), HloOpcode::kBitcast);
  if (ShapeUtil::IsEffectiveScalar(bitcast->shape())) {
    return true;
  }
  auto last_dimension = [](const HloInstruction* instr) {
    return instr->shape().dimensions().back();
  };
  HloInstruction* reduce = nullptr;
  TrivialEdge(&reduce, bitcast->mutable_operand(0), HloOpcode::kReduce,
              gpu_version);
  return (HasDefaultLayout(bitcast->shape()) &&
          HasDefaultLayout(bitcast->operand(0)->shape()) &&
          (reduce != nullptr ||
           last_dimension(bitcast->operand(0)) == last_dimension(bitcast)));
}
inline bool HasOneUse(const HloInstruction* instr) {
  return instr->user_count() == 1;
}
bool IsBatchOrReductionDimBroadcast(const HloInstruction& hlo) {
  CHECK_EQ(hlo.opcode(), HloOpcode::kBroadcast)
      << "Expected broadcast " << hlo.ToShortString();
  CHECK_EQ(hlo.operand(0)->opcode(), HloOpcode::kParameter)
      << "Expected parameter " << hlo.operand(0)->ToShortString();
  const HloBroadcastInstruction* broadcast =
      Cast<HloBroadcastInstruction>(&hlo);
  const HloParameterInstruction* parameter =
      Cast<HloParameterInstruction>(hlo.operand(0));
  if (broadcast->dimensions().empty() ||
      parameter->shape().dimensions_size() + 1 !=
          broadcast->shape().dimensions_size()) {
    return false;
  }
  bool preserve_first_dim = broadcast->dimensions().front() == 0;
  bool preserve_last_dim = broadcast->dimensions().back() ==
                           broadcast->shape().dimensions_size() - 1;
  return !(preserve_first_dim && preserve_last_dim);
}
bool IsBroadcastOfAScalar(const HloInstruction& hlo) {
  CHECK_EQ(hlo.opcode(), HloOpcode::kBroadcast)
      << "Expected broadcast " << hlo.ToShortString();
  return ShapeUtil::IsScalar(hlo.operand(0)->shape());
}
bool IsSingleRowParameterBroadcast(const HloInstruction& hlo) {
  CHECK_EQ(hlo.opcode(), HloOpcode::kBroadcast)
      << "Expected broadcast " << hlo.ToShortString();
  CHECK_EQ(hlo.operand(0)->opcode(), HloOpcode::kParameter)
      << "Expected parameter " << hlo.operand(0)->ToShortString();
  const HloBroadcastInstruction* broadcast =
      Cast<HloBroadcastInstruction>(&hlo);
  const HloParameterInstruction* parameter =
      Cast<HloParameterInstruction>(hlo.operand(0));
  if (parameter->shape().dimensions_size() != 1) {
    return false;
  }
  return broadcast->dimensions()[0] == broadcast->shape().dimensions_size() - 1;
}
bool IsSupportedBroadcastOfParameter(const HloInstruction& hlo) {
  return IsBroadcastOfParameter(hlo) &&
         (IsBatchOrReductionDimBroadcast(hlo) || IsBroadcastOfAScalar(hlo) ||
          IsSingleRowParameterBroadcast(hlo));
}
HloInstruction* ChooseOperandForFusionProcessing(HloInstruction* instr) {
  CHECK_GT(instr->operand_count(), 0);
  CHECK_LE(instr->operand_count(), 2);
  if (instr->operand_count() > 1 &&
      (IsBroadcastOfScalarConstant(*instr->operand(0)) ||
       IsSupportedBroadcastOfParameter(*instr->operand(0)))) {
    return instr->mutable_operand(1);
  }
  return instr->mutable_operand(0);
}
bool IsTriviallyFusible(HloInstruction* instr,
                        const se::GpuComputeCapability& gpu_version,
                        int num_allowed_users = 1) {
  if (instr->user_count() > num_allowed_users ||
      !HasDefaultLayout(instr->shape())) {
    return false;
  }
  if (instr->opcode() == HloOpcode::kBitcast &&
      BitcastIsTilingNoop(instr, gpu_version)) {
    return true;
  }
  if (instr->IsElementwise() && instr->operand_count() == 1) {
    return static_cast<bool>(IsTritonSupportedInstruction(*instr, gpu_version));
  }
  if (instr->IsElementwiseBinary()) {
    const HloInstruction* operand_0 = instr->operand(0);
    const HloInstruction* operand_1 = instr->operand(1);
    if (operand_0 == operand_1) {
      return static_cast<bool>(
          IsTritonSupportedInstruction(*instr, gpu_version));
    }
    if ((IsBroadcastOfScalarConstant(*operand_0) ||
         IsSupportedBroadcastOfParameter(*operand_0)) ^
        (IsBroadcastOfScalarConstant(*operand_1) ||
         IsSupportedBroadcastOfParameter(*operand_1))) {
      return static_cast<bool>(
          IsTritonSupportedInstruction(*instr, gpu_version));
    }
  }
  return false;
}
bool TrivialEdge(HloInstruction** producer, HloInstruction* consumer,
                 HloOpcode opcode,
                 const se::GpuComputeCapability& gpu_version) {
  while (consumer->opcode() != opcode) {
    if (IsTriviallyFusible(consumer, gpu_version)) {
      consumer = ChooseOperandForFusionProcessing(consumer);
    } else {
      return false;
    }
  }
  *producer = consumer;
  return true;
}
bool IsTriviallyConnectedProducerOf(
    HloInstruction* producer, HloInstruction* consumer,
    const se::GpuComputeCapability& gpu_version) {
  if (producer == consumer) {
    return true;
  }
  HloInstruction* found_producer = consumer;
  while (
      TrivialEdge(&found_producer, consumer, producer->opcode(), gpu_version)) {
    if (found_producer == producer) {
      return true;
    }
    if (!IsTriviallyFusible(found_producer, gpu_version)) {
      return false;
    }
    consumer = found_producer->mutable_operand(0);
  }
  return false;
}
HloInstruction* FindFirstNonFusibleDiamondProducer(
    HloInstruction* diamond_producer,
    const se::GpuComputeCapability& gpu_version) {
  if (IsTriviallyFusible(diamond_producer, gpu_version,
                         2)) {
    diamond_producer = ChooseOperandForFusionProcessing(diamond_producer);
    while (IsTriviallyFusible(diamond_producer, gpu_version)) {
      diamond_producer = ChooseOperandForFusionProcessing(diamond_producer);
    }
  }
  return diamond_producer;
}
absl::StatusOr<HloFusionInstruction*> MakeFusionForDiamondChain(
    const DiamondChainDescriptor& diamond_chain) {
  auto [root, producer] = diamond_chain;
  std::string suggested_name = "triton_softmax";
  HloComputation::Builder builder(absl::StrCat(suggested_name, "_computation"));
  absl::flat_hash_map<const HloInstruction*, HloInstruction*>
      old_to_new_mapping;
  int param = 0;
  old_to_new_mapping[producer] =
      builder.AddInstruction(HloInstruction::CreateParameter(
          param, producer->shape(), absl::StrCat("parameter_", param)));
  param++;
  std::vector<HloInstruction*> parameters = {producer};
  std::function<void(HloInstruction*)> create_computation =
      [&](HloInstruction* instr) -> void {
    if (old_to_new_mapping.contains(instr)) {
      return;
    }
    std::vector<HloInstruction*> new_operands;
    for (HloInstruction* operand : instr->mutable_operands()) {
      create_computation(operand);
      new_operands.push_back(old_to_new_mapping[operand]);
    }
    if (instr->opcode() == HloOpcode::kParameter) {
      old_to_new_mapping[instr] =
          builder.AddInstruction(HloInstruction::CreateParameter(
              param, instr->shape(), absl::StrCat("parameter_", param)));
      parameters.push_back(instr);
      param++;
    } else {
      old_to_new_mapping[instr] = builder.AddInstruction(
          instr->CloneWithNewOperands(instr->shape(), new_operands));
    }
  };
  create_computation(root);
  HloComputation* computation =
      root->GetModule()->AddComputationAndUnifyNamesAndIds(builder.Build(),
                                                           false);
  HloInstruction* softmax_fusion =
      root->parent()->AddInstruction(HloInstruction::CreateFusion(
          root->shape(), HloInstruction::FusionKind::kCustom, parameters,
          computation));
  softmax_fusion->GetModule()->SetAndUniquifyInstrName(softmax_fusion,
                                                       "triton_softmax");
  TF_ASSIGN_OR_RETURN(auto gpu_config,
                      softmax_fusion->backend_config<GpuBackendConfig>());
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();
  backend_config.set_kind(std::string(kTritonFusionKind));
  TF_RETURN_IF_ERROR(softmax_fusion->set_backend_config(gpu_config));
  return xla::Cast<HloFusionInstruction>(softmax_fusion);
}
absl::Status RunFusionPipeline(
    HloModule* module, const se::DeviceDescription& device_info,
    const HloCostAnalysis::ShapeSizeFunction& shape_size) {
  HloPassPipeline reduction_pipeline("reduction_pipeline");
  reduction_pipeline.AddPass<ReductionDimensionGrouper>();
  reduction_pipeline.AddPass<HloPassFix<ReductionSplitter>>(
      false);
  reduction_pipeline.AddPass<HloPassFix<TreeReductionRewriter>>(
      device_info.gpu_compute_capability());
  TF_RETURN_IF_ERROR(reduction_pipeline.Run(module).status());
  return FusionPipeline(module->config().debug_options(), shape_size,
                        nullptr, device_info)
      .Run(module)
      .status();
}
absl::StatusOr<absl::Duration>
EstimateOptimizedHloRunTimeWithoutSoftMaxRewriterTriton(
    const HloFusionInstruction* fusion,
    const se::DeviceDescription& device_info,
    const HloCostAnalysis::ShapeSizeFunction& shape_size) {
  auto new_module = ExtractComputationIntoNewModule(
      *fusion->fused_instructions_computation());
  TF_RETURN_IF_ERROR(
      RunFusionPipeline(new_module.get(), device_info, shape_size));
  VLOG(10) << "priority fusion module: " << new_module->ToString();
  HloComputation* entry_computation = new_module->entry_computation();
  GpuHloCostAnalysis::Options cost_analysis_options{
      shape_size,
      {},
      {},
      true};
  GpuHloCostAnalysis cost_analysis(cost_analysis_options, device_info);
  TF_RETURN_IF_ERROR(entry_computation->Accept(&cost_analysis));
  absl::Duration total_run_time = absl::ZeroDuration();
  for (const HloInstruction* instr : entry_computation->instructions()) {
    total_run_time += GpuPerformanceModel::EstimateRunTimeForInstruction(
                          instr, device_info, &cost_analysis,
                          GpuPerformanceModelOptions::PriorityFusion())
                          .exec_time;
  }
  return total_run_time;
}
absl::StatusOr<FusionDecision>
DecideIfShouldFuseAndMaybeSetBlockLevelParameters(
    HloFusionInstruction* softmax_fusion,
    GpuPerformanceModelWithIndexingAnalysis& indexing_performance_model,
    const se::DeviceDescription& device_info,
    const HloCostAnalysis::ShapeSizeFunction& shape_size,
    bool use_cost_model_to_evaluate_fusions) {
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(softmax_fusion);
  TF_ASSIGN_OR_RETURN(
      TiledRunTimeDataOrError tiled_runtime_data_or,
      indexing_performance_model.TryFindBestTilingForFusion(*fusion_adaptor));
  if (const auto* fusion_decision =
          std::get_if<FusionDecision>(&tiled_runtime_data_or)) {
    return FusionDecision::Forbid(absl::StrCat("SymbolicTileAnalysis failed: ",
                                               fusion_decision->Explain()));
  }
  TiledRunTimeData tiled_runtime_data =
      std::get<TiledRunTimeData>(std::move(tiled_runtime_data_or));
  if (use_cost_model_to_evaluate_fusions) {
    TF_ASSIGN_OR_RETURN(absl::Duration run_time_without_softmax_rewriter,
                        EstimateOptimizedHloRunTimeWithoutSoftMaxRewriterTriton(
                            softmax_fusion, device_info, shape_size));
    VLOG(5) << "run time estimate if normalization diamond fused together: "
            << tiled_runtime_data.runtime_data.exec_time;
    VLOG(5)
        << "run time estimate if normalization diamond is not fused together: "
        << run_time_without_softmax_rewriter;
    if (run_time_without_softmax_rewriter <
        tiled_runtime_data.runtime_data.exec_time) {
      return FusionDecision::Forbid(
          "Run time estimate for without applying the custom normalization "
          "rewrite is faster.");
    }
  }
  TF_ASSIGN_OR_RETURN(auto backend_config,
                      softmax_fusion->backend_config<GpuBackendConfig>());
  *backend_config.mutable_fusion_backend_config()
       ->mutable_block_level_fusion_config() =
      tiled_runtime_data.block_level_parameters.ToBlockLevelFusionConfig();
  TF_RETURN_IF_ERROR(softmax_fusion->set_backend_config(backend_config));
  VLOG(5) << "Fusing with backend config: " << backend_config.DebugString();
  return FusionDecision::Allow();
}
absl::StatusOr<bool> MaybeFuseDiamondChainImpl(
    const DiamondChainDescriptor& diamond_chain,
    GpuPerformanceModelWithIndexingAnalysis& indexing_performance_model,
    const se::DeviceDescription& device_info,
    const HloCostAnalysis::ShapeSizeFunction& shape_size,
    bool use_cost_model_to_evaluate_fusions) {
  TF_ASSIGN_OR_RETURN(HloFusionInstruction * softmax_fusion,
                      MakeFusionForDiamondChain(diamond_chain));
  HloInstruction* root = diamond_chain.root;
  VLOG(5) << "MaybeFuseDiamondChainImpl: " << softmax_fusion->ToString();
  TF_ASSIGN_OR_RETURN(
      FusionDecision fusion_decision,
      DecideIfShouldFuseAndMaybeSetBlockLevelParameters(
          softmax_fusion, indexing_performance_model, device_info, shape_size,
          use_cost_model_to_evaluate_fusions));
  if (!fusion_decision.CanFuse()) {
    VLOG(5) << "Not fusing: " << fusion_decision.Explain();
    softmax_fusion->DetachFromOperandsAndUsers();
    TF_RETURN_IF_ERROR(
        softmax_fusion->parent()->RemoveInstruction(softmax_fusion));
    return false;
  }
  if (root->IsRoot()) {
    root->parent()->set_root_instruction(softmax_fusion);
    TF_RETURN_IF_ERROR(
        root->parent()->RemoveInstructionAndUnusedOperands(root));
  } else {
    TF_RETURN_IF_ERROR(
        root->parent()->ReplaceInstruction(root, softmax_fusion));
  }
  return true;
}
absl::StatusOr<bool> CanSymbolicTileAnalysisTileDiamondChain(
    const DiamondChainDescriptor& diamond_chain,
    const se::DeviceDescription& device_info) {
  TF_ASSIGN_OR_RETURN(HloFusionInstruction * softmax_fusion,
                      MakeFusionForDiamondChain(diamond_chain));
  mlir::MLIRContext context;
  SymbolicTileAnalysisOrError symbolic_tile_analysis_or_error =
      SymbolicTileAnalysis::AnalyzeComputation(
          *softmax_fusion->called_computation(), &context,
          TritonEmitterConstraints::GetBuilder(device_info));
  bool can_tile = std::holds_alternative<SymbolicTileAnalysis>(
      symbolic_tile_analysis_or_error);
  TF_RETURN_IF_ERROR(diamond_chain.root->GetModule()->RemoveEmbeddedComputation(
      softmax_fusion->called_computation()));
  TF_RETURN_IF_ERROR(
      diamond_chain.root->parent()->RemoveInstruction(softmax_fusion));
  return can_tile;
}
FusionDecision ShouldFuseReduction(const HloInstruction& reduce,
                                   const se::GpuComputeCapability& cc) {
  if (CodegenDecision is_supported = IsTritonSupportedInstruction(reduce, cc);
      !is_supported) {
    return FusionDecision::Forbid(is_supported.Explain());
  }
  if (reduce.dimensions().size() != 1 ||
      reduce.dimensions(0) != reduce.operand(0)->shape().rank() - 1) {
    return FusionDecision::Forbid(
        "The reductions in the diamond must reduce 1 dimension and that "
        "dimension must be the last dimension of the operand.");
  }
  const HloInstruction* identity = reduce.operand(1);
  bool should_fuse_identity =
      identity->opcode() == HloOpcode::kConstant ||
      (identity->opcode() == HloOpcode::kConvert &&
       identity->operand(0)->opcode() == HloOpcode::kConstant &&
       IsTritonSupportedInstruction(*identity, cc));
  if (!should_fuse_identity) {
    return FusionDecision::Forbid(
        "Reduction identity is not a constant or a supported convert of a "
        "constant.");
  }
  return FusionDecision::Allow();
}
DiamondMatchingDecision MatchesTritonCompatibleClosedReductionDiamondImpl(
    HloInstruction* instr, const se::GpuComputeCapability& cc) {
  if (!instr->IsElementwiseBinary()) {
    return FusionDecision::Forbid("Root is not elementwise binary.");
  }
  if (!IsTritonSupportedInstruction(*instr, cc)) {
    return FusionDecision::Forbid(
        "Root is not supported for Triton instruction.");
  }
  HloInstruction* producer;
  HloInstruction* broadcast;
  HloInstruction* reduce;
  if (!TrivialEdge(&broadcast, instr->mutable_operand(1), HloOpcode::kBroadcast,
                   cc)) {
    return FusionDecision::Forbid(
        "Could not find a trivial connection from root to a broadcast.");
  }
  if (!TrivialEdge(&reduce, broadcast->mutable_operand(0), HloOpcode::kReduce,
                   cc)) {
    return FusionDecision::Forbid(
        "Could not find a trivial connection from matched broadcast to a "
        "reduction.");
  }
  if (!(HasDefaultLayout(broadcast->shape()) &&
        HasDefaultLayout(reduce->shape()))) {
    return FusionDecision::Forbid(
        "Broadcast or reduce have non-default layouts.");
  }
  if (FusionDecision should_fuse_reduction = ShouldFuseReduction(*reduce, cc);
      !should_fuse_reduction) {
    VLOG(2) << should_fuse_reduction.Explain();
    return should_fuse_reduction;
  }
  const HloInstruction* identity = reduce->operand(1);
  bool should_fuse_identity =
      identity->opcode() == HloOpcode::kConstant ||
      (identity->opcode() == HloOpcode::kConvert &&
       identity->operand(0)->opcode() == HloOpcode::kConstant &&
       IsTritonSupportedInstruction(*identity, cc));
  if (!should_fuse_identity) {
    return FusionDecision::Forbid(
        "Reduction identity is not a constant or a supported convert of a "
        "constant.");
  }
  if (!HasOneUse(broadcast) || !HasOneUse(reduce)) {
    return FusionDecision::Forbid("More than one use of broadcast or reduce.");
  }
  producer = reduce->mutable_operand(0);
  if (absl::c_linear_search(broadcast->dimensions(),
                            broadcast->shape().rank() - 1)) {
    return FusionDecision::Forbid(
        "Broadcast is not along the reduction dimension.");
  }
  while (IsTriviallyFusible(producer, cc)) {
    producer = ChooseOperandForFusionProcessing(producer);
  }
  if (!HasDefaultLayout(producer->shape())) {
    return FusionDecision::Forbid("Producer has non-default layout.");
  }
  if (!IsTriviallyConnectedProducerOf(producer, instr->mutable_operand(0),
                                      cc)) {
    return FusionDecision::Forbid("Producer is not trivially connected.");
  }
  if (producer != instr->operand(0) && instr->operand(0)->user_count() != 1) {
    return FusionDecision::Forbid("Unsupported root-producer connection.");
  }
  VLOG(5) << "Matched Softmax diamond with: ";
  VLOG(5) << "root: " << instr->ToString();
  VLOG(5) << "producer: " << producer->ToString();
  VLOG(5) << "broadcast: " << broadcast->ToString();
  VLOG(5) << "reduce: " << reduce->ToString();
  return producer;
}
absl::StatusOr<std::vector<DiamondChainDescriptor>> FindAllFusibleDiamonds(
    HloModule& module,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    const se::DeviceDescription& device_info) {
  const se::GpuComputeCapability& cc = device_info.gpu_compute_capability();
  std::vector<DiamondChainDescriptor> matched_diamonds;
  for (HloComputation* comp :
       module.MakeNonfusionComputations(execution_threads)) {
    if (comp->IsCustomCallComputation()) {
      continue;
    }
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      auto producer =
          MatchesTritonCompatibleClosedReductionDiamondImpl(instr, cc);
      if (std::holds_alternative<HloInstruction*>(producer)) {
        DiamondChainDescriptor diamond_chain{
            instr, std::get<HloInstruction*>(producer)};
        TF_ASSIGN_OR_RETURN(bool can_tile_diamond_chain,
                            CanSymbolicTileAnalysisTileDiamondChain(
                                diamond_chain, device_info));
        if (can_tile_diamond_chain) {
          matched_diamonds.push_back(diamond_chain);
        } else {
          VLOG(5) << "Cannot tile the diamond pattern described by "
                  << "instructions " << instr->ToString() << " and "
                  << std::get<HloInstruction*>(producer)->ToString() << ".";
          continue;
        }
      } else {
        VLOG(5) << "Cannot match the diamond pattern for instruction "
                << instr->ToString()
                << ". Reason: " << std::get<FusionDecision>(producer).Explain();
      }
    }
  }
  return std::move(matched_diamonds);
}
int64_t GetReductionDimensionSizeForDiamond(
    const DiamondChainDescriptor& diamond_chain) {
  HloInstruction* diamond_root = diamond_chain.root;
  HloInstruction* instr = diamond_root->mutable_operand(1);
  while (instr->opcode() != HloOpcode::kReduce) {
    instr = ChooseOperandForFusionProcessing(instr);
  }
  int operand_rank = instr->operand(0)->shape().rank();
  CHECK_EQ(instr->dimensions().size(), 1);
  CHECK_EQ(instr->dimensions(0), operand_rank - 1);
  return instr->operand(0)->shape().dimensions(operand_rank - 1);
}
HloInstruction* GetLastTriviallyFusibleUser(
    HloInstruction* instr, const se::GpuComputeCapability& cc) {
  while (HasOneUse(instr) && !instr->IsRoot() &&
         IsTriviallyFusible(instr->users().front(), cc)) {
    instr = instr->users().front();
  }
  if (HasOneUse(instr) && !instr->IsRoot() &&
      IsTriviallyFusible(
          instr->users().front(), cc,
          instr->users().front()->user_count())) {
    instr = instr->users().front();
  }
  return instr;
}
}  
DiamondMatchingDecision
SoftmaxRewriterTriton::MatchesTritonCompatibleClosedReductionDiamond(
    HloInstruction* instr) const {
  return MatchesTritonCompatibleClosedReductionDiamondImpl(
      instr, device_info_.gpu_compute_capability());
}
absl::StatusOr<std::vector<DiamondChainDescriptor>>
SoftmaxRewriterTriton::FindAllFusibleDiamondChains(
    HloModule& module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) const {
  TF_ASSIGN_OR_RETURN(
      std::vector<DiamondChainDescriptor> matched_diamonds,
      FindAllFusibleDiamonds(module, execution_threads, device_info_));
  if (matched_diamonds.empty()) {
    return std::vector<DiamondChainDescriptor>();
  }
  std::vector<DiamondChainDescriptor> diamond_chains;
  diamond_chains.reserve(matched_diamonds.size());
  const se::GpuComputeCapability& cc = device_info_.gpu_compute_capability();
  HloInstruction* current_fusion_producer =
      FindFirstNonFusibleDiamondProducer(matched_diamonds.front().producer, cc);
  int current_reduce_dimension_size =
      GetReductionDimensionSizeForDiamond(matched_diamonds.front());
  for (int diamond_idx = 1; diamond_idx < matched_diamonds.size();
       ++diamond_idx) {
    HloInstruction* diamond_producer = matched_diamonds[diamond_idx].producer;
    HloInstruction* previous_diamond_root =
        matched_diamonds[diamond_idx - 1].root;
    HloInstruction* first_non_fusible_diamond_producer =
        FindFirstNonFusibleDiamondProducer(diamond_producer, cc);
    int diamond_reduce_dimension_size =
        GetReductionDimensionSizeForDiamond(matched_diamonds[diamond_idx]);
    if (first_non_fusible_diamond_producer == previous_diamond_root &&  
        ((first_non_fusible_diamond_producer != diamond_producer &&
          HasOneUse(first_non_fusible_diamond_producer)) ||  
         (first_non_fusible_diamond_producer == diamond_producer &&
          first_non_fusible_diamond_producer->user_count() == 2)) &&  
        diamond_reduce_dimension_size == current_reduce_dimension_size) {  
      continue;
    }
    diamond_chains.push_back(DiamondChainDescriptor{
        GetLastTriviallyFusibleUser(previous_diamond_root, cc),
        current_fusion_producer,
    });
    current_fusion_producer = first_non_fusible_diamond_producer;
    current_reduce_dimension_size = diamond_reduce_dimension_size;
  }
  diamond_chains.push_back(DiamondChainDescriptor{
      GetLastTriviallyFusibleUser(matched_diamonds.back().root, cc),
      current_fusion_producer});
  std::vector<DiamondChainDescriptor> filtered_diamond_chains;
  for (const DiamondChainDescriptor& diamond_chain : diamond_chains) {
    TF_ASSIGN_OR_RETURN(
        bool can_tile_diamond_chain,
        CanSymbolicTileAnalysisTileDiamondChain(diamond_chain, device_info_));
    if (can_tile_diamond_chain) {
      filtered_diamond_chains.push_back(diamond_chain);
    }
  }
  return filtered_diamond_chains;
}
absl::StatusOr<bool> SoftmaxRewriterTriton::MaybeFuseDiamondChain(
    const DiamondChainDescriptor& diamond_chain) {
  HloFusionAnalysisCache fusion_analysis_cache(device_info_);
  GpuPerformanceModelWithIndexingAnalysis indexing_performance_model(
      &device_info_, &fusion_analysis_cache, shape_size_, &mlir_context_);
  return MaybeFuseDiamondChainImpl(diamond_chain, indexing_performance_model,
                                   device_info_, shape_size_,
                                   use_cost_model_to_evaluate_fusions_);
}
absl::StatusOr<bool> SoftmaxRewriterTriton::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_RETURN_IF_ERROR(EnsureTritonSupportsComputeCapability(
      device_info_.gpu_compute_capability()));
  TF_ASSIGN_OR_RETURN(std::vector<DiamondChainDescriptor> diamond_chains,
                      FindAllFusibleDiamondChains(*module, execution_threads));
  bool changed = false;
  for (auto diamond_chain = diamond_chains.rbegin();
       diamond_chain != diamond_chains.rend(); ++diamond_chain) {
    TF_ASSIGN_OR_RETURN(bool fused, MaybeFuseDiamondChain(*diamond_chain));
    changed |= fused;
  }
  return changed;
}
}  
}  