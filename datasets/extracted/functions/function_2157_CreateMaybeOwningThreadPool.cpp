#include "xla/service/gpu/gpu_compiler.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>
#include "absl/base/call_once.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/maybe_owning.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/all_gather_broadcast_reorder.h"
#include "xla/service/all_gather_combiner.h"
#include "xla/service/all_reduce_combiner.h"
#include "xla/service/all_reduce_contiguous.h"
#include "xla/service/all_reduce_folder.h"
#include "xla/service/all_reduce_promotion.h"
#include "xla/service/all_reduce_reassociate.h"
#include "xla/service/async_collective_creator.h"
#include "xla/service/batched_gather_scatter_normalizer.h"
#include "xla/service/batchnorm_expander.h"
#include "xla/service/bitcast_dtypes_expander.h"
#include "xla/service/broadcast_canonicalizer.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/call_inliner.h"
#include "xla/service/collective_permute_decomposer.h"
#include "xla/service/collective_pipeliner.h"
#include "xla/service/collective_quantizer.h"
#include "xla/service/collectives_schedule_linearizer.h"
#include "xla/service/comparison_expander.h"
#include "xla/service/compiler.h"
#include "xla/service/conditional_canonicalizer.h"
#include "xla/service/conditional_simplifier.h"
#include "xla/service/convert_memory_placement_to_internal_annotations.h"
#include "xla/service/convert_mover.h"
#include "xla/service/convolution_4d_expander.h"
#include "xla/service/convolution_pred_expander.h"
#include "xla/service/copy_insertion.h"
#include "xla/service/cpu_gpu_shape_verifier.h"
#include "xla/service/dot_decomposer.h"
#include "xla/service/dot_merger.h"
#include "xla/service/dump.h"
#include "xla/service/dynamic_dimension_inference.h"
#include "xla/service/dynamic_dimension_simplifier.h"
#include "xla/service/dynamic_index_splitter.h"
#include "xla/service/dynamic_padder.h"
#include "xla/service/eigh_expander.h"
#include "xla/service/executable.h"
#include "xla/service/export_hlo.h"
#include "xla/service/flatten_call_graph.h"
#include "xla/service/float_normalization.h"
#include "xla/service/float_support.h"
#include "xla/service/gather_expander.h"
#include "xla/service/gather_simplifier.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/autotuning/custom_kernel_fusion_autotuner.h"
#include "xla/service/gpu/compile_module_to_llvm_ir.h"
#include "xla/service/gpu/conv_layout_normalization.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/execution_stream_assignment.h"
#include "xla/service/gpu/fusion_pipeline.h"
#include "xla/service/gpu/fusions/triton/triton_support.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_float_support.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include "xla/service/gpu/gpu_latency_hiding_scheduler.h"
#include "xla/service/gpu/gpu_p2p_pipeliner.h"
#include "xla/service/gpu/gpu_spmd_pipeline.h"
#include "xla/service/gpu/hlo_fusion_stats.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/ir_emitter_unnested.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/metrics.h"
#include "xla/service/gpu/model/gpu_cost_model_stats_collection.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/prepare_hlo_for_ir_emitting_pipeline.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/service/gpu/runtime_intrinsics.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/transforms/algebraic_simplifier.h"
#include "xla/service/gpu/transforms/algorithm_checker.h"
#include "xla/service/gpu/transforms/all_gather_optimizer.h"
#include "xla/service/gpu/transforms/all_reduce_blueconnect.h"
#include "xla/service/gpu/transforms/all_reduce_splitter.h"
#include "xla/service/gpu/transforms/async_collective_annotator.h"
#include "xla/service/gpu/transforms/async_wrapper.h"
#include "xla/service/gpu/transforms/collective_permute_cycle_decomposer.h"
#include "xla/service/gpu/transforms/collective_permute_valid_iteration_annotator.h"
#include "xla/service/gpu/transforms/command_buffer_scheduling.h"
#include "xla/service/gpu/transforms/conv_rewriter.h"
#include "xla/service/gpu/transforms/convert_async_collectives_to_sync.h"
#include "xla/service/gpu/transforms/cudnn_custom_call_converter.h"
#include "xla/service/gpu/transforms/custom_kernel_fusion_rewriter.h"
#include "xla/service/gpu/transforms/dot_dimension_sorter.h"
#include "xla/service/gpu/transforms/dot_operand_converter.h"
#include "xla/service/gpu/transforms/double_buffer_loop_unrolling.h"
#include "xla/service/gpu/transforms/dynamic_slice_fusion_rewriter.h"
#include "xla/service/gpu/transforms/fusion_block_level_rewriter.h"
#include "xla/service/gpu/transforms/fusion_wrapper.h"
#include "xla/service/gpu/transforms/gemm_broadcast_folding_rewriter.h"
#include "xla/service/gpu/transforms/gemm_fusion.h"
#include "xla/service/gpu/transforms/gemm_rewriter.h"
#include "xla/service/gpu/transforms/gemv_rewriter.h"
#include "xla/service/gpu/transforms/layout_assignment.h"
#include "xla/service/gpu/transforms/move_copy_to_users.h"
#include "xla/service/gpu/transforms/pipelined_p2p_rewriter.h"
#include "xla/service/gpu/transforms/reduce_scatter_creator.h"
#include "xla/service/gpu/transforms/reduction_degenerate_dim_remover.h"
#include "xla/service/gpu/transforms/reduction_dimension_grouper.h"
#include "xla/service/gpu/transforms/reduction_layout_normalizer.h"
#include "xla/service/gpu/transforms/reduction_splitter.h"
#include "xla/service/gpu/transforms/rename_fusions.h"
#include "xla/service/gpu/transforms/sanitize_constant_names.h"
#include "xla/service/gpu/transforms/scatter_expander.h"
#include "xla/service/gpu/transforms/scatter_slice_simplifier.h"
#include "xla/service/gpu/transforms/softmax_rewriter_triton.h"
#include "xla/service/gpu/transforms/stream_attribute_annotator.h"
#include "xla/service/gpu/transforms/stream_attribute_async_wrapper.h"
#include "xla/service/gpu/transforms/topk_specializer.h"
#include "xla/service/gpu/transforms/topk_splitter.h"
#include "xla/service/gpu/transforms/transpose_dimension_grouper.h"
#include "xla/service/gpu/transforms/tree_reduction_rewriter.h"
#include "xla/service/gpu/transforms/triton_fusion_numerics_verifier.h"
#include "xla/service/gpu/transforms/windowed_einsum_handler.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_computation_deduplicator.h"
#include "xla/service/hlo_constant_folding.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_dataflow_analysis.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_rematerialization.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/host_memory_transfer_asyncifier.h"
#include "xla/service/host_offload_legalize.h"
#include "xla/service/host_offloader.h"
#include "xla/service/layout_assignment.h"
#include "xla/service/layout_normalization.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/logistic_expander.h"
#include "xla/service/operand_upcaster.h"
#include "xla/service/optimization_barrier_expander.h"
#include "xla/service/optimize_input_output_buffer_alias.h"
#include "xla/service/qr_expander.h"
#include "xla/service/real_imag_expander.h"
#include "xla/service/reduce_decomposer.h"
#include "xla/service/reduce_scatter_combiner.h"
#include "xla/service/reduce_scatter_reassociate.h"
#include "xla/service/reduce_window_rewriter.h"
#include "xla/service/reshape_decomposer.h"
#include "xla/service/reshape_mover.h"
#include "xla/service/result_caster.h"
#include "xla/service/rng_bit_generator_expander.h"
#include "xla/service/rng_expander.h"
#include "xla/service/scatter_expander.h"
#include "xla/service/scatter_simplifier.h"
#include "xla/service/sharding_remover.h"
#include "xla/service/simplify_fp_conversions.h"
#include "xla/service/slice_sinker.h"
#include "xla/service/slow_operation_alarm.h"
#include "xla/service/sort_simplifier.h"
#include "xla/service/stable_sort_expander.h"
#include "xla/service/stochastic_convert_decomposer.h"
#include "xla/service/sub_byte_normalization.h"
#include "xla/service/topk_rewriter.h"
#include "xla/service/transpose_folding.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/service/while_loop_all_reduce_code_motion.h"
#include "xla/service/while_loop_constant_sinking.h"
#include "xla/service/while_loop_simplifier.h"
#include "xla/service/while_loop_trip_count_annotator.h"
#include "xla/service/zero_sized_hlo_elimination.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/blocking_counter.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"  
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"
#ifdef PLATFORM_GOOGLE
#include "xla/hlo/experimental/auto_sharding/auto_sharding.h"
#endif  
namespace xla {
namespace gpu {
namespace {
using MaybeOwningThreadPool = MaybeOwning<tsl::thread::ThreadPool>;
MaybeOwningThreadPool CreateMaybeOwningThreadPool(
    int parallelism, tsl::thread::ThreadPool* default_thread_pool,
    int default_parallelism) {
  CHECK_GE(parallelism, 0);
  CHECK_GE(default_parallelism, 1);
  CHECK(default_thread_pool == nullptr ||
        default_thread_pool->CurrentThreadId() == -1);
  auto create_thread_pool = [&](int num_threads) {
    CHECK_GE(num_threads, 1);
    return std::make_unique<tsl::thread::ThreadPool>(tsl::Env::Default(), "",
                                                     num_threads);
  };
  switch (parallelism) {
    case 0:
      if (default_thread_pool == nullptr && default_parallelism > 1) {
        return MaybeOwningThreadPool(create_thread_pool(default_parallelism));
      }
      return MaybeOwningThreadPool(default_thread_pool);
    case 1:
      return MaybeOwningThreadPool(nullptr);
    default:
      return MaybeOwningThreadPool(create_thread_pool(parallelism));
  }
}
absl::StatusOr<AutotuneConfig> GetAutotuneConfig(
    se::StreamExecutor* stream_exec, const DebugOptions& debug_options,
    const GpuCompiler::CompileOptions& options,
    const Compiler::TargetConfig& gpu_target_config) {
  if (stream_exec) {
    return AutotuneConfig{DeviceConfig{stream_exec, options.device_allocator},
                          debug_options};
  }
  return AutotuneConfig{DevicelessConfig{gpu_target_config.device_description},
                        debug_options};
}
se::GpuComputeCapability GetGpuVersion(const se::StreamExecutor* stream_exec) {
  return stream_exec->GetDeviceDescription().gpu_compute_capability();
}
class GpuThunkAotCompilationResult : public AotCompilationResult {
 public:
  static absl::StatusOr<std::unique_ptr<GpuThunkAotCompilationResult>>
  FromModule(const HloModule* hlo_module,
             const BufferAssignment* buffer_assignment,
             std::string_view asm_text, absl::Span<const uint8_t> binary,
             const BinaryMap& dnn_compiled_graphs) {
    CompilationResultProto proto;
    *proto.mutable_hlo_module_with_config() = hlo_module->ToProtoWithConfig();
    *proto.mutable_buffer_assignment() = buffer_assignment->ToProto();
    proto.set_asm_text(std::string(asm_text));
    proto.set_binary(binary.data(), binary.size());
    proto.mutable_dnn_compiled_graphs()->insert(dnn_compiled_graphs.cbegin(),
                                                dnn_compiled_graphs.cend());
    return std::unique_ptr<GpuThunkAotCompilationResult>(
        new GpuThunkAotCompilationResult(hlo_module->Clone(),
                                         std::move(proto)));
  }
  static absl::StatusOr<std::unique_ptr<GpuThunkAotCompilationResult>>
  FromString(const std::string& serialized) {
    CompilationResultProto proto;
    if (!proto.ParseFromString(serialized)) {
      return Internal(
          "Failed to parse serialized GpuThunkAotCompilationResult.");
    }
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModule> module,
        HloModule::CreateFromProtoWithConfig(proto.hlo_module_with_config()));
    return std::unique_ptr<GpuThunkAotCompilationResult>(
        new GpuThunkAotCompilationResult(std::move(module), std::move(proto)));
  }
  absl::StatusOr<std::string> SerializeAsString() const override {
    return proto_.SerializeAsString();
  }
  absl::StatusOr<std::unique_ptr<Executable>> LoadExecutable(
      Compiler* compiler, const se::StreamExecutor* stream_exec) const override;
  const HloModule* optimized_module() const override { return module_.get(); }
  std::unique_ptr<HloModule> consume_optimized_module() override {
    return std::move(module_);
  }
 private:
  GpuThunkAotCompilationResult(std::unique_ptr<HloModule> module,
                               CompilationResultProto proto)
      : module_(std::move(module)), proto_(std::move(proto)) {}
  std::unique_ptr<HloModule> module_;
  CompilationResultProto proto_;
};
}  
absl::StatusOr<std::unique_ptr<Executable>>
GpuThunkAotCompilationResult::LoadExecutable(
    Compiler* compiler, const se::StreamExecutor* stream_exec) const {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      HloModule::CreateFromProtoWithConfig(proto_.hlo_module_with_config()));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<BufferAssignment> buffer_assignment,
      BufferAssignment::FromProto(proto_.buffer_assignment(), hlo_module.get(),
                                  compiler->BufferSizeBytesFunction(),
                                  nullptr));
  ExecutionStreamAssignment execution_stream_assignment(hlo_module.get());
  std::vector<uint8_t> binary(proto_.binary().begin(), proto_.binary().end());
  TF_ASSIGN_OR_RETURN(
      se::Platform * platform,
      se::PlatformManager::PlatformWithId(compiler->PlatformId()));
  std::string platform_name = platform->Name();
  const se::DeviceDescription& gpu_device_info =
      stream_exec->GetDeviceDescription();
  mlir::DialectRegistry registry;
  auto mlir_context = std::make_unique<mlir::MLIRContext>(registry);
  llvm::LLVMContext llvm_context;
  auto* gpu_compiler = dynamic_cast<GpuCompiler*>(compiler);
  if (gpu_compiler == nullptr) {
    return Internal("Compiler is not a GpuCompiler.");
  }
  auto llvm_module = std::make_unique<llvm::Module>("", llvm_context);
  llvm_module->setTargetTriple(gpu_compiler->target_triple());
  llvm_module->setDataLayout(gpu_compiler->data_layout());
  IrEmitterContext ir_emitter_context(
      hlo_module.get(), buffer_assignment.get(), &execution_stream_assignment,
      platform_name, gpu_device_info, mlir_context.get(), llvm_module.get(),
      nullptr,
      false);
  absl::string_view cache_file_path =
      hlo_module->config().debug_options().xla_gpu_kernel_cache_file();
  if (!cache_file_path.empty() &&
      hlo_module->config()
          .debug_options()
          .xla_gpu_enable_llvm_module_compilation_parallelism()) {
    TF_RETURN_IF_ERROR(LoadCache(ir_emitter_context, cache_file_path));
  }
  auto ir_emitter = IrEmitterUnnested::Create(&ir_emitter_context);
  TF_RETURN_IF_ERROR(
      ir_emitter->EmitHloComputation(hlo_module->entry_computation()));
  std::vector<GpuExecutable::ConstantInfo> constants =
      std::move(ir_emitter_context.constants());
  TF_ASSIGN_OR_RETURN(auto output_info,
                      GetOutputInfo(*hlo_module, *buffer_assignment));
  const Shape& output_shape = hlo_module->result_shape();
  int64_t debug_buffer_assignment_show_max =
      hlo_module->config()
          .debug_options()
          .xla_debug_buffer_assignment_show_max();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<GpuExecutable> executable,
      GpuExecutable::Create(GpuExecutable::Params{
          proto_.asm_text(),
          binary,
          BinaryMap(proto_.dnn_compiled_graphs().cbegin(),
                    proto_.dnn_compiled_graphs().cend()),
          gpu_device_info.gpu_compute_capability(),
          ir_emitter->ConsumeThunkSequence(),
          std::move(constants),
          std::move(output_info),
          std::move(hlo_module->name()),
          std::move(output_shape),
          std::nullopt,
          std::move(buffer_assignment),
          debug_buffer_assignment_show_max,
          std::move(hlo_module),
          true}));
  return executable;
}
GpuCompiler::GpuCompiler(se::Platform::Id platform_id,
                         const char* target_triple, const char* data_layout)
    : platform_id_(platform_id),
      target_triple_(target_triple),
      data_layout_(data_layout),
      pointer_size_(llvm::DataLayout(data_layout)
                        .getPointerSize(0 )) {}
namespace {
void AddHloVerifier(HloPassPipeline* pipeline,
                    bool verify_unique_channel_ids = false,
                    HloVerifierOpts&& opts = {}, bool debug_only = false) {
  opts.verify_unique_channel_ids = verify_unique_channel_ids;
  std::unique_ptr<TargetVerifierMetadata> verifier_metadata =
      std::make_unique<CpuGpuVerifierMetadata>(std::move(opts));
  if (debug_only) {
    pipeline->AddInvariantCheckerDebug<HloVerifier>(
        std::move(verifier_metadata), "hlo verifier (debug)");
  } else {
    pipeline->AddInvariantChecker<HloVerifier>(std::move(verifier_metadata),
                                               "hlo verifier");
  }
}
void CheckNotScheduled(HloModule* hlo_module) {
  if (hlo_module->has_schedule() &&
      !hlo_module->config().debug_options().xla_disable_all_hlo_passes()) {
    LOG(WARNING) << "\nThe current HLO module " << hlo_module->name()
                 << " is scheduled and optimized. \n"
                 << "It is not expected to run optimization passes again.\n"
                    "Use a test method like RunAndCompareNoHloPasses() or "
                 << "the xla_disable_all_hlo_passes flag.";
  }
}
void LogDebugOptions(HloModule* hlo_module) {
  XLA_VLOG_LINES(
      1, absl::StrFormat("GpuCompilationEnvironment of hlo_module %s:\n%s",
                         hlo_module->name(),
                         hlo_module->config().debug_options().DebugString()));
}
AlgebraicSimplifierOptions LayoutInsensitiveAlgebraicSimplifierOptions(
    const HloModuleConfig& hlo_module_config,
    const Compiler::TargetConfig& gpu_target_config,
    AlgebraicSimplifierOptions opts_from_compiler) {
  AlgebraicSimplifierOptions layout_insensitive_algsimp_opts =
      opts_from_compiler;
  layout_insensitive_algsimp_opts.set_conv_is_lowerable_callback(
      ConvRewriter::ConvIsLowerable);
  layout_insensitive_algsimp_opts.set_enable_dot_strength_reduction(
      hlo_module_config.debug_options()
          .xla_gpu_enable_dot_strength_reduction());
  layout_insensitive_algsimp_opts.set_supports_non_canonical_dots(false);
  layout_insensitive_algsimp_opts.set_minmax_propagate_nan(
      !hlo_module_config.debug_options().xla_gpu_enable_fast_min_max());
  layout_insensitive_algsimp_opts
      .set_unconditionally_simplify_reduce_of_transpose_or_reshape(true);
  if (gpu_target_config.platform_name == "ROCM") {
    layout_insensitive_algsimp_opts.set_enable_conv_operand_swap(false);
  }
  layout_insensitive_algsimp_opts
      .set_enable_unconditional_reduce_of_concat_replacement(false);
  return layout_insensitive_algsimp_opts;
}
absl::Status RunPreSPMDPartitionerPasses(HloModule* hlo_module) {
  HloPassPipeline pre_spmd_pipeline("pre-spmd-partitioner");
  pre_spmd_pipeline.AddPass<BatchedGatherScatterNormalizer>();
  pre_spmd_pipeline.AddPass<CuDnnCustomCallConverter>();
  pre_spmd_pipeline.AddPass<ConvertMemoryPlacementToInternalAnnotations>();
  pre_spmd_pipeline.AddPass<CallInliner>();
  pre_spmd_pipeline.AddPass<ZeroSizedHloElimination>();
  pre_spmd_pipeline.AddPass<ConditionalCanonicalizer>();
  pre_spmd_pipeline.AddPass<TopkDecomposer>([&](const HloInstruction* instr) {
    return instr->opcode() == HloOpcode::kTopK;
  });
  pre_spmd_pipeline.AddPass<TopkRewriter>(
      [](const HloSortInstruction*, int64_t) { return true; });
  return pre_spmd_pipeline.Run(hlo_module).status();
}
absl::Status RunSPMDPasses(
    HloModule* hlo_module, const Compiler::TargetConfig& gpu_target_config,
    const AlgebraicSimplifierOptions& layout_insensitive_algsimp_opts) {
  bool auto_sharding = hlo_module->config().use_auto_spmd_partitioning();
#ifndef PLATFORM_GOOGLE
  if (auto_sharding) {
    LOG(ERROR) << "GPU autosharding is not yet available in open source.";
  }
#endif
  const int64_t num_partitions = hlo_module->config().num_partitions();
  if (num_partitions > 1) {
    if (!hlo_module->config().use_spmd_partitioning()) {
      return InvalidArgument(
          "num_partitions=%d but SPMD partitioning not enabled.",
          num_partitions);
    }
    HloPassPipeline spmd_pipeline("spmd-partitioner");
    AddSPMDPasses(
        hlo_module, layout_insensitive_algsimp_opts,
        gpu_target_config.device_description.gpu_compute_capability(),
        spmd_pipeline,
#ifdef PLATFORM_GOOGLE
        [&](HloPassPipeline& pipeline) {
          if (auto_sharding) {
            AutoShardingOption option;
            option.enable = true;
            if (!hlo_module->config()
                     .auto_spmd_partitioning_mesh_shape()
                     .empty()) {
              option.device_mesh_shape =
                  hlo_module->config().auto_spmd_partitioning_mesh_shape();
            } else {
              option.device_mesh_shape = {
                  gpu_target_config.device_description.core_count(), 1};
            }
            if (!hlo_module->config()
                     .auto_spmd_partitioning_mesh_ids()
                     .empty()) {
              option.device_mesh_ids =
                  hlo_module->config().auto_spmd_partitioning_mesh_ids();
            }
            option.memory_budget_per_device =
                hlo_module->config()
                    .debug_options()
                    .xla_gpu_auto_spmd_partitioning_memory_budget_gb() *
                1024 * 1024 * 1024;
            option.memory_budget_ratio =
                hlo_module->config()
                    .debug_options()
                    .xla_gpu_auto_spmd_partitioning_memory_budget_ratio();
            spmd_pipeline.AddPass<AutoSharding>(option);
          }
        });
#else
        std::nullopt);
#endif  
    if (hlo_module->config()
            .debug_options()
            .xla_gpu_unsafe_pipelined_loop_annotator()) {
      spmd_pipeline.AddPass<WhileLoopTripCountAnnotator>();
      spmd_pipeline.AddPass<CollectivePermuteValidIterationAnnotator>();
    }
    return spmd_pipeline.Run(hlo_module).status();
  } else {
    HloPassPipeline sharding_removal_pipeline("sharding-removal");
    sharding_removal_pipeline.AddPass<ShardingRemover>();
    sharding_removal_pipeline.AddPass<HloDCE>();
    return sharding_removal_pipeline.Run(hlo_module).status();
  }
}
absl::Status RunOptimizationPasses(
    HloModule* hlo_module, const Compiler::TargetConfig& gpu_target_config,
    const AlgebraicSimplifierOptions& layout_insensitive_algsimp_opts) {
  const DebugOptions& debug_options = hlo_module->config().debug_options();
  HloPassPipeline pipeline("optimization");
  AddHloVerifier(&pipeline,
                 !debug_options.xla_experimental_ignore_channel_id());
  if (debug_options.xla_gpu_multi_streamed_windowed_einsum()) {
    pipeline.AddPass<WindowedEinsumHandler>();
  }
  pipeline.AddPass<TopKSplitter>();
  pipeline.AddPass<TopkSpecializer>();
  pipeline.AddPass<TopkDecomposer>();
  HloPredicate upcaster_filter = [&](const HloInstruction* instr) {
    const auto* cuda_cc = std::get_if<se::CudaComputeCapability>(
        &gpu_target_config.device_description.gpu_compute_capability());
    if (cuda_cc != nullptr &&
        !cuda_cc->IsAtLeast(se::CudaComputeCapability::VOLTA)) {
      return true;
    }
    return !gpu::IsMatrixMultiplication(*instr);
  };
  pipeline.AddPass<DotDimensionSorter>();
  pipeline.AddPass<DotDecomposer>();
  pipeline.AddPass<ResultCaster>(upcaster_filter);
  pipeline.AddPass<OperandUpcaster>(upcaster_filter);
  pipeline.AddPass<DotOperandConverter>();
  pipeline.AddPass<SubByteNormalization>(
      SubByteNormalization::SET_ELEMENT_SIZE);
  pipeline.AddPass<RngExpander>();
  pipeline.AddPass<RngBitGeneratorExpander>(RandomAlgorithm::RNG_PHILOX);
  pipeline.AddPass<ComparisonExpander>(std::array{std::make_pair(BF16, F32)});
  pipeline.AddPass<ZeroSizedHloElimination>();
  if (RequireDeterminism(hlo_module->config())) {
    pipeline.AddPass<ScatterExpander>(
        ScatterExpander::kEliminateIndeterministicScatters);
  }
  pipeline.AddPass<GpuScatterExpander>();
  pipeline.AddPass<QrExpander>();
  pipeline.AddPass<EighExpander>();
  pipeline.AddPass<DynamicIndexSplitter>();
  pipeline.AddPass<CallInliner>();
  pipeline.AddPass<StochasticConvertDecomposer>();
  pipeline.AddPass<Convolution4DExpander>();
  pipeline.AddPass<ConvolutionPredExpander>();
  pipeline.AddPass<StableSortExpander>();
  pipeline.AddPass<BatchNormExpander>(
      true,
      true,
      true);
  pipeline.AddPass<LogisticExpander>();
  pipeline.AddPass<ConditionalCanonicalizer>();
  pipeline.AddPass<DynamicDimensionSimplifier>();
  if (debug_options.xla_reduce_window_rewrite_base_length() != 0) {
    pipeline.AddPass<HloPassFix<ReduceWindowRewriter>>(
        debug_options.xla_reduce_window_rewrite_base_length());
  }
  DynamicPadderOptions dynamic_padder_options;
  switch (debug_options.xla_gpu_shape_checks()) {
    case DebugOptions::IGNORE:
      dynamic_padder_options.shape_check_mode =
          DynamicDimensionInference::ShapeCheckMode::kIgnore;
      break;
    case DebugOptions::RUNTIME: {
      dynamic_padder_options.shape_check_mode =
          DynamicDimensionInference::ShapeCheckMode::kRuntime;
      dynamic_padder_options.assertion_generator = [&](HloInstruction* inst) {
        auto created = Cast<HloCustomCallInstruction>(
            inst->parent()->AddInstruction(HloInstruction::CreateCustomCall(
                ShapeUtil::MakeTokenShape(), {inst}, kXlaGpuAssertCustomCallTag,
                "Buffers have different size at runtime",
                API_VERSION_STATUS_RETURNING)));
        created->set_custom_call_has_side_effect(true);
      };
      break;
    }
    case DebugOptions::COMPILE_TIME:
      dynamic_padder_options.shape_check_mode =
          DynamicDimensionInference::ShapeCheckMode::kCompileTime;
      break;
    default:
      LOG(FATAL) << "Unreachable";
  }
  pipeline.AddPass<DynamicPadder>(dynamic_padder_options);
  se::GpuComputeCapability gpu_version =
      gpu_target_config.device_description.gpu_compute_capability();
  [&, &pipeline =
          pipeline.AddPass<HloPassFix<HloPassPipeline>>("simplification")] {
    AddHloVerifier(&pipeline,
                   !debug_options.xla_experimental_ignore_channel_id(),
                   HloVerifierOpts{}, true);
    pipeline.AddPass<ZeroSizedHloElimination>();
    pipeline.AddPass<GatherSimplifier>();
    pipeline.AddPass<GatherExpander>(GatherExpander::kEliminateSimpleGathers);
    pipeline.AddPass<ScatterSimplifier>();
    pipeline.AddPass<ScatterExpander>(
        ScatterExpander::kEliminateSimpleScatters);
    pipeline.AddPass<ScatterSliceSimplifier>();
    pipeline.AddPass<GpuAlgebraicSimplifier>(layout_insensitive_algsimp_opts,
                                             gpu_version);
    pipeline.AddPass<BitcastDtypesExpander>();
    pipeline.AddPass<DotDimensionSorter>();
    pipeline.AddPass<DotDecomposer>();
    pipeline.AddPass<DotMerger>(
        int64_t{
            debug_options.xla_gpu_dot_merger_threshold_mb()}
        << 20);
    pipeline.AddPass<SortSimplifier>();
    pipeline.AddPass<TupleSimplifier>();
    pipeline.AddPass<WhileLoopConstantSinking>();
    pipeline.AddPass<WhileLoopSimplifier>();
    pipeline.AddPass<SliceSinker>();
    ReshapeMoverOptions reshape_mover_options;
    reshape_mover_options.reshape_of_1d_broadcast_is_cheap = true;
    pipeline.AddPass<ReshapeMover>(reshape_mover_options);
    pipeline.AddPass<HloConstantFolding>();
    pipeline.AddPass<ConditionalSimplifier>();
    pipeline.AddPass<RealImagExpander>();
    pipeline.AddPass<TransposeFolding>(CanFoldTransposeOperandIntoDot);
    pipeline.AddPass<HloCSE>(false);
    pipeline.AddPass<HloDCE>();
  }();
  [&, &pipeline =
          pipeline.AddPass<HloPassFix<HloPassPipeline>>("simplification-2")] {
    pipeline.AddPass<ConvertMover>();
    pipeline.AddPass<GpuAlgebraicSimplifier>(layout_insensitive_algsimp_opts,
                                             gpu_version);
  }();
  pipeline.AddPass<HloComputationDeduplicator>(
      false);
  return pipeline.Run(hlo_module).status();
}
absl::Status AddCollectivePipelinerPasses(
    const DebugOptions& debug_options, HloPassPipeline& collectives_pipeline) {
  if (debug_options.xla_gpu_enable_pipelined_collectives() ||
      debug_options.xla_gpu_enable_pipelined_all_reduce()) {
    CollectivePipeliner::Config config{
        0,
        INT64_MAX,
        true,
        false,
        true,
        CollectivePipeliner::PipeliningDirection::kForward,
        HloPredicateIsOp<HloOpcode::kAllReduce>,
        HloPredicateTrue,
        HloPredicateFalse};
    collectives_pipeline.AddPass<CollectivePipeliner>(config);
  }
  if (debug_options.xla_gpu_enable_pipelined_collectives() ||
      debug_options.xla_gpu_enable_pipelined_all_gather()) {
    CollectivePipeliner::Config config{
        0,
        INT64_MAX,
        true,
        false,
        true,
        CollectivePipeliner::PipeliningDirection::kBackward,
        HloPredicateIsOp<HloOpcode::kAllGather>,
        HloPredicateTrue,
        HloPredicateFalse,
        HloPredicateFalse,
        false,
        std::nullopt,
        std::nullopt,
        true,
    };
    collectives_pipeline.AddPass<CollectivePipeliner>(config);
  }
  if (debug_options.xla_gpu_enable_pipelined_collectives() ||
      debug_options.xla_gpu_enable_pipelined_reduce_scatter()) {
    CollectivePipeliner::Config config{
        0,
        INT64_MAX,
        true,
        false,
        true,
        CollectivePipeliner::PipeliningDirection::kForward,
        HloPredicateIsOp<HloOpcode::kReduceScatter>,
        HloPredicateTrue,
        HloPredicateFalse};
    collectives_pipeline.AddPass<CollectivePipeliner>(config);
  }
  return absl::OkStatus();
}
absl::Status RunPostLayoutCollectivePipelinerPasses(HloModule* hlo_module) {
  const DebugOptions& debug_options = hlo_module->config().debug_options();
  HloPassPipeline collectives_pipeline("collective-pipeliner-optimizations");
  if (debug_options.xla_gpu_run_post_layout_collective_pipeliner()) {
    TF_RETURN_IF_ERROR(
        AddCollectivePipelinerPasses(debug_options, collectives_pipeline));
    collectives_pipeline.AddPass<WhileLoopTripCountAnnotator>();
    collectives_pipeline.AddPass<FlattenCallGraph>();
  }
  return collectives_pipeline.Run(hlo_module).status();
}
absl::Status RunCollectiveOptimizationPasses(
    HloModule* hlo_module,
    const AlgebraicSimplifierOptions& layout_insensitive_algsimp_opts,
    se::GpuComputeCapability gpu_version) {
  const DebugOptions& debug_options = hlo_module->config().debug_options();
  HloPassPipeline collectives_pipeline("collective-optimizations");
  collectives_pipeline.AddPass<AllReduceFolder>();
  collectives_pipeline.AddPass<AllReduceSplitter>();
  collectives_pipeline.AddPass<AllGatherOptimizer>();
  collectives_pipeline.AddPass<AllReduceReassociate>(
      debug_options.xla_gpu_enable_reassociation_for_converted_ar());
  collectives_pipeline.AddPass<ReduceScatterReassociate>();
  collectives_pipeline.AddPass<WhileLoopAllReduceCodeMotion>(
      debug_options
          .xla_gpu_enable_while_loop_reduce_scatter_code_motion());
  if (!debug_options.xla_gpu_run_post_layout_collective_pipeliner()) {
    TF_RETURN_IF_ERROR(
        AddCollectivePipelinerPasses(debug_options, collectives_pipeline));
  }
  collectives_pipeline.AddPass<ReduceScatterCreator>();
  collectives_pipeline.AddPass<CollectivePermuteCycleDecomposer>(
      hlo_module->config()
          .debug_options()
          .xla_gpu_collective_permute_decomposer_threshold());
  collectives_pipeline.AddPass<CollectivePermuteDecomposer>(
      hlo_module->config()
          .debug_options()
          .xla_gpu_collective_permute_decomposer_threshold());
  if (hlo_module->config()
          .debug_options()
          .xla_gpu_enable_pipelined_collectives() ||
      hlo_module->config().debug_options().xla_gpu_enable_pipelined_p2p()) {
    AddP2PPipeliner(collectives_pipeline);
  }
  collectives_pipeline.AddPass<GpuAlgebraicSimplifier>(
      layout_insensitive_algsimp_opts, gpu_version);
  collectives_pipeline.AddPass<AllGatherBroadcastReorder>();
  const std::pair<PrimitiveType, PrimitiveType> ar_promoted_types[] = {
      {U16, U32}, {S16, S32}};
  collectives_pipeline.AddPass<AllReducePromotion>(ar_promoted_types);
  collectives_pipeline.AddPass<HloDCE>();
  collectives_pipeline.AddPass<CollectiveQuantizer>();
  collectives_pipeline.AddPass<HloDCE>();
  collectives_pipeline.AddPass<WhileLoopTripCountAnnotator>();
  return collectives_pipeline.Run(hlo_module).status();
}
absl::Status RunLayoutAssignmentPasses(HloModule* hlo_module,
                                       se::GpuComputeCapability gpu_version,
                                       se::dnn::VersionInfo dnn_version) {
  HloPassPipeline pipeline("layout assignment");
  pipeline.AddPass<FlattenCallGraph>();
  ChannelLayoutConstraints layout_constraints;
  pipeline.AddPass<GpuLayoutAssignment>(
      hlo_module->mutable_entry_computation_layout(), gpu_version, dnn_version,
      &layout_constraints);
  pipeline.AddPass<SubByteNormalization>(
      SubByteNormalization::SET_ELEMENT_SIZE);
  pipeline.AddPass<OptimizeInputOutputBufferAlias>(true);
  pipeline.AddPass<HostOffloadLegalize>(
      static_cast<int64_t>(stream_executor::MemoryType::kHost),
       true);
  return pipeline.Run(hlo_module).status();
}
absl::Status RunFusionPasses(HloModule* hlo_module,
                             const Compiler::TargetConfig& gpu_target_config,
                             tsl::thread::ThreadPool* thread_pool,
                             HloCostAnalysis::ShapeSizeFunction shape_size_fn) {
  const se::DeviceDescription& gpu_device_info =
      gpu_target_config.device_description;
  TF_RETURN_IF_ERROR(FusionPipeline(hlo_module->config().debug_options(),
                                    shape_size_fn, thread_pool, gpu_device_info)
                         .Run(hlo_module)
                         .status());
  if (hlo_module->config().debug_options().xla_gpu_collect_cost_model_stats()) {
    GpuHloCostAnalysis::Options cost_analysis_options{
        shape_size_fn,
        {},
        {},
        true};
    HloPassPipeline post_fusion_analysis("post_fusion_analysis");
    post_fusion_analysis.AddPass<GpuCostModelStatsCollection>(
        gpu_device_info, cost_analysis_options);
    TF_RETURN_IF_ERROR(post_fusion_analysis.Run(hlo_module).status());
  }
  TF_RETURN_IF_ERROR(
      HorizontalFusionPipeline(gpu_device_info).Run(hlo_module).status());
  if (VLOG_IS_ON(2)) {
    HloFusionStatsVisitor stats;
    TF_RETURN_IF_ERROR(hlo_module->entry_computation()->Accept(&stats));
    VLOG(2) << stats.ToString();
  }
  return absl::OkStatus();
}
void AddDoubleBufferingPasses(const DebugOptions& opts,
                              HloPassPipeline& pipeline) {
  std::optional<DoubleBufferLoopUnrolling::UnrollStrategy> unroll_strategy =
      std::nullopt;
  if (opts.xla_gpu_enable_while_loop_double_buffering()) {
    unroll_strategy = DoubleBufferLoopUnrolling::UnrollStrategy::kDoubleBuffer;
  }
  if (opts.xla_gpu_enable_while_loop_unrolling() ==
      DebugOptions::WHILE_LOOP_UNROLLING_DOUBLE_BUFFER) {
    unroll_strategy = DoubleBufferLoopUnrolling::UnrollStrategy::kDoubleBuffer;
  }
  if (opts.xla_gpu_enable_while_loop_unrolling() ==
      DebugOptions::WHILE_LOOP_UNROLLING_FULL_UNROLL) {
    LOG_IF(WARNING, unroll_strategy != std::nullopt)
        << "Overriding double buffering set via "
           "`xla_gpu_enable_while_loop_double_buffering` flag.";
    unroll_strategy = DoubleBufferLoopUnrolling::UnrollStrategy::kFullUnroll;
  }
  if (opts.xla_gpu_enable_while_loop_unrolling() ==
          DebugOptions::WHILE_LOOP_UNROLLING_AUTO_UNROLL &&
      opts.xla_gpu_enable_heuristic_pass_configuration() &&
      !opts.xla_gpu_enable_while_loop_double_buffering()) {
    unroll_strategy = DoubleBufferLoopUnrolling::UnrollStrategy::kAuto;
  }
  if (unroll_strategy != std::nullopt) {
    pipeline.AddPass<WhileLoopSimplifier>();
    pipeline.AddPass<DoubleBufferLoopUnrolling>(*unroll_strategy);
    pipeline.AddPass<TupleSimplifier>();
    pipeline.AddPass<HloDCE>();
  }
}
absl::Status RunPostFusionPasses(
    HloModule* hlo_module,
    std::function<absl::Status(HloPassPipeline*, const DebugOptions&)>
        add_custom_kernel_replacement_passes) {
  const DebugOptions& opts = hlo_module->config().debug_options();
  HloPassPipeline pipeline("post-fusion optimization");
  pipeline.AddPass<RenameFusions>();
  pipeline.AddPass<AllGatherCombiner>(
      opts.xla_gpu_all_gather_combine_threshold_bytes(),
      256,
      opts.xla_gpu_enable_all_gather_combine_by_dim());
  pipeline.AddPass<AllReduceCombiner>(
      opts.xla_gpu_all_reduce_combine_threshold_bytes(),
      256);
  pipeline.AddPass<ReduceScatterCombiner>(
      opts.xla_gpu_reduce_scatter_combine_threshold_bytes(),
      256,
      opts.xla_gpu_enable_reduce_scatter_combine_by_dim());
  pipeline.AddPass<AllReduceContiguous>();
  TF_RETURN_IF_ERROR(add_custom_kernel_replacement_passes(&pipeline, opts));
  int32_t blueconnect_num_devices_per_host =
      hlo_module->config()
          .debug_options()
          .xla_gpu_all_reduce_blueconnect_num_devices_per_host();
  if (blueconnect_num_devices_per_host > 0) {
    pipeline.AddPass<AllReduceBlueConnect>(blueconnect_num_devices_per_host);
  }
  AddDoubleBufferingPasses(opts, pipeline);
  return pipeline.Run(hlo_module).status();
}
absl::Status RunPostFusionCollectiveOptimizationPasses(HloModule* hlo_module) {
  HloPassPipeline pipeline("post-fusion-collectives optimization");
  AsyncCollectiveCreator::CollectiveCreatorConfig config;
  config.convert_all_reduce = HloPredicateTrue;
  config.convert_collective_broadcast = HloPredicateTrue;
  config.convert_collective_permute = HloPredicateTrue;
  config.convert_all_gather = HloPredicateTrue;
  config.convert_reduce_scatter = HloPredicateTrue;
  config.convert_all_to_all = HloPredicateTrue;
  pipeline.AddPass<AsyncCollectiveCreator>(std::move(config));
  absl::flat_hash_set<DebugOptions::CollectiveOpType> disabled_async_ops;
  for (auto collective_op_type : hlo_module->config()
                                     .debug_options()
                                     .xla_gpu_disable_async_collectives()) {
    disabled_async_ops.insert(
        static_cast<DebugOptions::CollectiveOpType>(collective_op_type));
  }
  auto convert_to_async = [&disabled_async_ops](const HloInstruction* inst) {
    switch (inst->opcode()) {
      case HloOpcode::kAllReduceStart:
        return !disabled_async_ops.contains(DebugOptions::ALLREDUCE);
      case HloOpcode::kCollectivePermuteStart:
        return !disabled_async_ops.contains(DebugOptions::COLLECTIVEPERMUTE);
      case HloOpcode::kAllGatherStart:
        return !disabled_async_ops.contains(DebugOptions::ALLGATHER);
      case HloOpcode::kAsyncStart: {
        auto async_inst = Cast<HloAsyncInstruction>(inst);
        switch (async_inst->async_wrapped_opcode()) {
          case HloOpcode::kCollectiveBroadcast:
            return !disabled_async_ops.contains(
                DebugOptions::COLLECTIVEBROADCAST);
          case HloOpcode::kReduceScatter:
            return !disabled_async_ops.contains(DebugOptions::REDUCESCATTER);
          case HloOpcode::kAllToAll:
            return !disabled_async_ops.contains(DebugOptions::ALLTOALL);
          default:
            return false;
        }
      }
      default:
        return false;
    }
  };
  pipeline.AddPass<AsyncCollectiveAnnotator>(convert_to_async);
  return pipeline.Run(hlo_module).status();
}
absl::Status RunPostFusionSimplificationPasses(
    HloModule* hlo_module,
    const AlgebraicSimplifierOptions& layout_insensitive_algsimp_opts,
    se::GpuComputeCapability gpu_version) {
  HloPassPipeline pipeline("post-fusion-simplification-pipeline optimization");
  AlgebraicSimplifierOptions options = layout_insensitive_algsimp_opts;
  options.set_is_layout_sensitive(true);
  pipeline.AddPass<GpuAlgebraicSimplifier>(options, gpu_version);
  pipeline.AddPass<HloComputationDeduplicator>(
      true);
  if (hlo_module->config()
          .debug_options()
          .xla_gpu_multi_streamed_windowed_einsum()) {
    pipeline.AddPass<StreamAttributeAnnotator>();
    pipeline.AddPass<StreamAttributeAsyncWrapper>();
  }
  return pipeline.Run(hlo_module).status();
}
absl::Status RunPostFusionVerificationPasses(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    const GpuCompiler::CompileOptions& options,
    const Compiler::TargetConfig& gpu_target_config) {
  HloPassPipeline pipeline("post-fusion-verification-pipeline optimization");
  if (hlo_module->config()
          .debug_options()
          .xla_gpu_verify_triton_fusion_numerics()) {
    TF_ASSIGN_OR_RETURN(
        AutotuneConfig autotune_config,
        GetAutotuneConfig(stream_exec, hlo_module->config().debug_options(),
                          options, gpu_target_config));
    pipeline.AddPass<TritonFusionNumericsVerifier>(autotune_config);
  }
  return pipeline.Run(hlo_module).status();
}
absl::Status RunLayoutNormalizationPasses(
    HloModule* hlo_module, const se::GpuComputeCapability& gpu_version) {
  HloPassPipeline layout_normalization_pipeline("layout normalization");
  const DebugOptions& debug_options = hlo_module->config().debug_options();
  AlgebraicSimplifierOptions opts =
      GpuCompiler::GetAlgebraicSimplifierOptions(hlo_module->config());
  opts.set_supports_non_canonical_dots(false);
  opts.set_is_layout_sensitive(true);
  opts.set_enable_conv_operand_swap(false);
  opts.set_minmax_propagate_nan(!debug_options.xla_gpu_enable_fast_min_max());
  opts.set_enable_unconditional_reduce_of_concat_replacement(false);
  layout_normalization_pipeline.AddPass<ReshapeDecomposer>();
  layout_normalization_pipeline.AddPass<HloPassFix<MoveCopyToUsers>>();
  layout_normalization_pipeline.AddPass<LayoutNormalization>(
      &NormalizeLayoutForGpuCustomCalls);
  layout_normalization_pipeline.AddPass<HloPassFix<GpuAlgebraicSimplifier>>(
      opts, gpu_version);
  layout_normalization_pipeline.AddPass<BroadcastCanonicalizer>();
  layout_normalization_pipeline.AddPass<ScatterSimplifier>();
  return layout_normalization_pipeline.Run(hlo_module).status();
}
absl::Status RunAsyncDotPasses(HloModule* hlo_module) {
  HloPassPipeline pipeline("async-wrapper");
  const DebugOptions& debug_options = hlo_module->config().debug_options();
  if (debug_options.xla_gpu_async_dot()) {
    pipeline.AddPass<AsyncWrapper>([](HloInstruction* instruction) {
      if (IsCublasGemm(*instruction)) {
        return true;
      }
      if (instruction->called_computations().size() == 1 &&
          IsTritonFusedComputation(
              *instruction->called_computations().front())) {
        return true;
      }
      return false;
    });
  }
  return pipeline.Run(hlo_module).status();
}
absl::Status RunDynamicSliceFusionPasses(HloModule* hlo_module,
                                         se::Platform::Id platform_id) {
  if (hlo_module->config()
          .debug_options()
          .xla_gpu_enable_dynamic_slice_fusion()) {
    HloPassPipeline pipeline("dynamic-slice");
    TF_ASSIGN_OR_RETURN(se::Platform * platform,
                        se::PlatformManager::PlatformWithId(platform_id));
    pipeline.AddPass<DynamicSliceFusionRewriter>(platform->Name());
    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }
  return absl::OkStatus();
}
}  
absl::Status GpuCompiler::RunCollectiveScheduleLinearizerPasses(
    HloModule* hlo_module, se::StreamExecutor* stream_exec) {
  HloPassPipeline pipeline("collective-schedule-linearizer");
  pipeline.AddPass<CollectivesScheduleLinearizer>(
      [this, stream_exec](const HloModule* module) {
        return RequiresCollectiveScheduleLinearizer(module, stream_exec);
      });
  return pipeline.Run(hlo_module).status();
}
absl::Status GpuCompiler::OptimizeHloModule(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    const CompileOptions& options, const TargetConfig& gpu_target_config) {
  tsl::profiler::TraceMe traceme("GpuCompiler::OptimizeHloModule");
  CheckNotScheduled(hlo_module);
  LogDebugOptions(hlo_module);
  MaybeOwningThreadPool thread_pool = CreateMaybeOwningThreadPool(
      hlo_module->config()
          .debug_options()
          .xla_gpu_force_compilation_parallelism(),
      options.thread_pool,
      tsl::port::MaxParallelism());
  AlgebraicSimplifierOptions layout_insensitive_algsimp_opts =
      LayoutInsensitiveAlgebraicSimplifierOptions(
          hlo_module->config(), gpu_target_config,
          GetAlgebraicSimplifierOptions(hlo_module->config()));
  TF_RETURN_IF_ERROR(RunPreSPMDPartitionerPasses(hlo_module));
  TF_RETURN_IF_ERROR(RunSPMDPasses(hlo_module, gpu_target_config,
                                   layout_insensitive_algsimp_opts));
  TF_RETURN_IF_ERROR(RunOptimizationPasses(hlo_module, gpu_target_config,
                                           layout_insensitive_algsimp_opts));
  se::GpuComputeCapability gpu_version =
      gpu_target_config.device_description.gpu_compute_capability();
  TF_RETURN_IF_ERROR(RunCollectiveOptimizationPasses(
      hlo_module, layout_insensitive_algsimp_opts, gpu_version));
  se::dnn::VersionInfo dnn_version = gpu_target_config.dnn_version_info;
  if (stream_exec != nullptr) {
    gpu_version = GetGpuVersion(stream_exec);
    TF_ASSIGN_OR_RETURN(dnn_version, GetDnnVersionInfo(stream_exec));
  }
  TF_RETURN_IF_ERROR(OptimizeHloConvolutionCanonicalization(
      hlo_module, gpu_version, dnn_version, options.device_allocator,
      gpu_target_config.device_description.runtime_version()));
  TF_RETURN_IF_ERROR(
      RunLayoutAssignmentPasses(hlo_module, gpu_version, dnn_version));
  TF_RETURN_IF_ERROR(RunLayoutNormalizationPasses(hlo_module, gpu_version));
  TF_RETURN_IF_ERROR(OptimizeHloPostLayoutAssignment(
      hlo_module, stream_exec, options, gpu_target_config,
      thread_pool.get_mutable()));
  TF_RETURN_IF_ERROR(RunPostLayoutCollectivePipelinerPasses(hlo_module));
  TF_RETURN_IF_ERROR(RunDynamicSliceFusionPasses(hlo_module, PlatformId()));
  TF_RETURN_IF_ERROR(RunFusionPasses(hlo_module, gpu_target_config,
                                     thread_pool.get_mutable(),
                                     ShapeSizeBytesFunction()));
  TF_RETURN_IF_ERROR(RunPostFusionPasses(
      hlo_module,
      [this](HloPassPipeline* pipeline, const DebugOptions& debug_options) {
        return AddCustomKernelReplacementPasses(pipeline, debug_options);
      }));
  TF_RETURN_IF_ERROR(RunPostFusionCollectiveOptimizationPasses(hlo_module));
  TF_RETURN_IF_ERROR(RunPostFusionSimplificationPasses(
      hlo_module, layout_insensitive_algsimp_opts, gpu_version));
  TF_RETURN_IF_ERROR(RunPostFusionVerificationPasses(
      hlo_module, stream_exec, options, gpu_target_config));
  TF_RETURN_IF_ERROR(
      RunCollectiveScheduleLinearizerPasses(hlo_module, stream_exec));
  TF_RETURN_IF_ERROR(RunAsyncDotPasses(hlo_module));
  return absl::OkStatus();
}  
AlgebraicSimplifierOptions GpuCompiler::GetAlgebraicSimplifierOptions(
    const HloModuleConfig& config) {
  AlgebraicSimplifierOptions opts;
  opts.set_enable_dot_strength_reduction(
      config.debug_options().xla_gpu_enable_dot_strength_reduction());
  return opts;
}
absl::Status GpuCompiler::PrepareHloModuleForIrEmitting(HloModule* hlo_module) {
  return PrepareHloModuleForIrEmittingPipeline(*hlo_module, GetCanShareBuffer())
      .Run(hlo_module)
      .status();
}
namespace {
void AddGemmRewriterPasses(HloPassPipeline& pipeline,
                           const DebugOptions& debug_options,
                           const se::GpuComputeCapability gpu_version,
                           const se::SemanticVersion& toolkit_version) {
  GemmRewriterOptions::BiasMode bias_mode =
      GemmRewriterOptions::BiasMode::kBias;
  if (debug_options.xla_gpu_async_dot()) {
    bias_mode = GemmRewriterOptions::BiasMode::kNoBias;
  }
  pipeline.AddPass<GemmRewriter>(
      gpu_version, toolkit_version,
      GemmRewriterOptions{GemmRewriterOptions::DType::kFp8Only, bias_mode});
  pipeline.AddPass<GemmRewriter>(
      gpu_version, toolkit_version,
      GemmRewriterOptions{GemmRewriterOptions::DType::kNonFp8Only, bias_mode});
}
}  
absl::Status GpuCompiler::OptimizeHloPostLayoutAssignment(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    const CompileOptions& options, const TargetConfig& gpu_target_config,
    tsl::thread::ThreadPool* thread_pool) {
  const DebugOptions& debug_options = hlo_module->config().debug_options();
  const se::GpuComputeCapability gpu_version =
      gpu_target_config.device_description.gpu_compute_capability();
  const AlgebraicSimplifierOptions simplifier_options = [&] {
    AlgebraicSimplifierOptions opts =
        GetAlgebraicSimplifierOptions(hlo_module->config());
    opts.set_supports_non_canonical_dots(false);
    opts.set_is_layout_sensitive(true);
    opts.set_enable_conv_operand_swap(false);
    opts.set_minmax_propagate_nan(!debug_options.xla_gpu_enable_fast_min_max());
    opts.set_enable_unconditional_reduce_of_concat_replacement(false);
    return opts;
  }();
  TF_ASSIGN_OR_RETURN(AutotuneConfig autotune_config,
                      GetAutotuneConfig(stream_exec, debug_options, options,
                                        gpu_target_config));
  const GpuFloatSupport bf16_support(gpu_version, BF16);
  const GpuFloatSupport f8e5m2_support(gpu_version, F8E5M2, F16);
  const GpuFloatSupport f8e4m3_support(gpu_version, F8E4M3, F16);
  const GpuFloatSupport f8e4m3fn_support(gpu_version, F8E4M3FN, F16);
  const FloatSupport f8e4m3b11fnuz_support(F8E4M3B11FNUZ, F16);
  const GpuFloatSupport f8e5m2fnuz_support(gpu_version, F8E5M2FNUZ, F16);
  const GpuFloatSupport f8e4m3fnuz_support(gpu_version, F8E4M3FNUZ, F16);
  const GpuFloatSupport f8e3m4_support(gpu_version, F8E3M4, F16);
  auto add_float_normalization = [&](HloPassPipeline& pipeline) {
    auto& sub_pipeline =
        pipeline.AddPass<HloPassPipeline>("float_normalization");
    sub_pipeline.AddPass<FloatNormalization>(&bf16_support);
    sub_pipeline.AddPass<FloatNormalization>(&f8e5m2_support);
    sub_pipeline.AddPass<FloatNormalization>(&f8e4m3_support);
    sub_pipeline.AddPass<FloatNormalization>(&f8e4m3fn_support);
    sub_pipeline.AddPass<FloatNormalization>(&f8e4m3b11fnuz_support);
    sub_pipeline.AddPass<FloatNormalization>(&f8e5m2fnuz_support);
    sub_pipeline.AddPass<FloatNormalization>(&f8e4m3fnuz_support);
    sub_pipeline.AddPass<FloatNormalization>(&f8e3m4_support);
    if (debug_options.xla_allow_excess_precision()) {
      sub_pipeline.AddPass<SimplifyFPConversions>();
    }
  };
  {
    HloPassPipeline pipeline("hlo normalization");
    pipeline.AddPass<HloPassFix<GpuAlgebraicSimplifier>>(simplifier_options,
                                                         gpu_version);
    pipeline.AddPass<TransposeFolding>(CanFoldTransposeOperandIntoDot,
                                       TransposeFolding::NeverFoldTranspose);
    pipeline.AddPass<ReshapeDecomposer>();
    pipeline.AddPass<ReduceDecomposer>([&](const HloInstruction* r) {
      return IsReductionFromOrToContiguousDimensions(*r);
    });
    if (debug_options.xla_gpu_enable_custom_fusions()) {
      pipeline.AddPass<SimplifyFPConversions>();
      pipeline.AddPass<CustomKernelFusionRewriter>(
          &gpu_target_config.device_description);
      pipeline.AddPass<CustomKernelFusionAutotuner>(autotune_config);
    }
    se::GpuComputeCapability gpu_version =
        gpu_target_config.device_description.gpu_compute_capability();
    pipeline.AddPass<AlgorithmChecker>(gpu_version);
    const auto* cuda_cc = std::get_if<se::CudaComputeCapability>(&gpu_version);
    const auto* rocm_cc = std::get_if<se::RocmComputeCapability>(&gpu_version);
    if (debug_options.xla_gpu_enable_triton_gemm() &&
        (cuda_cc != nullptr &&
         cuda_cc->IsAtLeast(se::CudaComputeCapability::AMPERE))) {
      pipeline.AddPass<GemvRewriter>();
      pipeline.AddPass<GemmFusion>(gpu_version);
    } else if (cuda_cc != nullptr &&
               cuda_cc->major == se::CudaComputeCapability::VOLTA) {
      pipeline.AddPass<SimplifyFPConversions>();
      pipeline.AddPass<CustomKernelFusionRewriter>(
          &gpu_target_config.device_description);
      pipeline.AddPass<CustomKernelFusionAutotuner>(autotune_config);
    }
    AddGemmRewriterPasses(
        pipeline, debug_options, gpu_version,
        gpu_target_config.device_description.runtime_version());
    pipeline.AddPass<GemmBroadcastFoldingRewriter>();
    pipeline.AddPass<LayoutNormalization>(&NormalizeLayoutForGpuCustomCalls);
    pipeline.AddPass<HloPassFix<GpuAlgebraicSimplifier>>(simplifier_options,
                                                         gpu_version);
    pipeline.AddPass<ScatterSimplifier>();
    pipeline.AddPass<BroadcastCanonicalizer>();
    pipeline.AddPass<TransposeDimensionGrouper>();
    pipeline.AddPass<ReductionDegenerateDimRemover>();
    pipeline.AddPass<ReductionLayoutNormalizer>();
    if (debug_options
            .xla_gpu_experimental_enable_triton_softmax_priority_fusion() &&
        ((cuda_cc != nullptr &&
          cuda_cc->IsAtLeast(se::CudaComputeCapability::AMPERE)) ||
         rocm_cc != nullptr)) {
      add_float_normalization(pipeline);
      pipeline.AddPass<HloPassFix<GpuAlgebraicSimplifier>>(simplifier_options,
                                                           gpu_version);
      pipeline.AddPass<HloCSE>(true);
      pipeline.AddPass<HloConstantFolding>();
      pipeline.AddPass<HloDCE>();
      pipeline.AddPass<SoftmaxRewriterTriton>(
          gpu_target_config.device_description, ShapeSizeBytesFunction(),
          true);
    }
    pipeline.AddPass<ReductionDimensionGrouper>();
    bool ignore_small_reduce_dims =
        !debug_options.xla_gpu_enable_priority_fusion();
    pipeline.AddPass<HloPassFix<ReductionSplitter>>(ignore_small_reduce_dims);
    pipeline.AddPass<HloPassFix<TreeReductionRewriter>>(gpu_version);
    pipeline.AddPass<SubByteNormalization>(
        SubByteNormalization::SET_ELEMENT_SIZE);
    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }
  HloPassPipeline pipeline("post-layout_assignment");
  AddHloVerifier(&pipeline, !debug_options.xla_experimental_ignore_channel_id(),
                 HloVerifierOpts{}
                     .MakeLayoutSensitive()
                     .WithInstructionCanChangeLayout(
                         LayoutAssignment::InstructionCanChangeLayout)
                     .VerifyBroadcastDimensionsOrder()
                     .VerifyReshapeIsBitcast(),
                 true);
  add_float_normalization(pipeline);
  TF_RETURN_IF_ERROR(AddGemmFusionAutotuningPasses(
      &pipeline, hlo_module, autotune_config, thread_pool,
      options.key_value_store,
      gpu_target_config.device_description.runtime_version()));
  pipeline.AddPass<CallInliner>();
  AddGemmRewriterPasses(pipeline, debug_options, gpu_version,
                        gpu_target_config.device_description.runtime_version());
  pipeline.AddPass<GemmBroadcastFoldingRewriter>();
  pipeline.AddPass<HostOffloader>(
      static_cast<int64_t>(stream_executor::MemoryType::kHost));
  TF_RETURN_IF_ERROR(
      AddConvAndGemmAutotuningPasses(&pipeline, gpu_version, options,
                                     hlo_module, autotune_config, thread_pool));
  add_float_normalization(pipeline);
  pipeline.AddPass<TupleSimplifier>();
  pipeline.AddPass<HloPassFix<GpuAlgebraicSimplifier>>(simplifier_options,
                                                       gpu_version);
  if (debug_options.xla_allow_excess_precision()) {
    pipeline.AddPass<SimplifyFPConversions>();
  }
  pipeline.AddPass<HloCSE>(true);
  pipeline.AddPass<HostMemoryTransferAsyncifier>(
      static_cast<int64_t>(stream_executor::MemoryType::kHost));
#ifdef NDEBUG
  HloVerifierOpts opts = HloVerifierOpts{}
                             .MakeLayoutSensitive()
                             .WithInstructionCanChangeLayout(
                                 LayoutAssignment::InstructionCanChangeLayout)
                             .VerifyBroadcastDimensionsOrder()
                             .VerifyReshapeIsBitcast();
  opts.verify_unique_channel_ids =
      !debug_options.xla_experimental_ignore_channel_id();
  pipeline.AddPass<HloVerifier>(
      std::make_unique<DefaultVerifierMetadata>(std::move(opts)),
      "end-of-post-layout_assignment");
#endif  
  TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  return absl::OkStatus();
}
 absl::StatusOr<Compiler::TargetConfig> GpuCompiler::GetTargetConfig(
    const Compiler::CompileOptions& options, const DebugOptions& debug_opts,
    se::StreamExecutor* executor) {
  if (options.target_config.has_value()) {
    return *options.target_config;
  }
  if (!debug_opts.xla_gpu_target_config_filename().empty()) {
    std::string gpu_target_config_string;
    TF_RETURN_IF_ERROR(tsl::ReadFileToString(
        tsl::Env::Default(), debug_opts.xla_gpu_target_config_filename(),
        &gpu_target_config_string));
    stream_executor::GpuTargetConfigProto gpu_target_config_proto;
    if (!tsl::protobuf::TextFormat::ParseFromString(gpu_target_config_string,
                                                    &gpu_target_config_proto)) {
      return absl::FailedPreconditionError(
          "Failed to parse GpuTargetConfigProto");
    }
    return Compiler::TargetConfig{gpu_target_config_proto};
  }
  if (executor) {
    Compiler::TargetConfig target_config = Compiler::TargetConfig{executor};
    int64_t device_memory_size =
        target_config.device_description.device_memory_size();
    if (device_memory_size == -1) {
      return absl::FailedPreconditionError(
          "When running on an NVIDIA simulation device, you must use "
          "--xla_gpu_target_config_filename to pass in target information. "
          "The target config from StreamExecutor is inaccurate.");
    }
    return target_config;
  }
  return absl::InternalError(
      "Either GPU has to be attached, or --xla_gpu_target_config_filename "
      "has to be specified to specify the target to compile for.");
}
absl::StatusOr<std::unique_ptr<HloModule>> GpuCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  const DebugOptions debug_opts = module->config().debug_options();
  TF_RETURN_IF_ERROR(LoadAutotuneResultsFromFile(debug_opts));
  bool is_deviceless = options.target_config.has_value() ||
                       !debug_opts.xla_gpu_target_config_filename().empty();
  TF_ASSIGN_OR_RETURN(TargetConfig gpu_target_config,
                      GetTargetConfig(options, debug_opts, stream_exec));
  const std::optional<std::string> unoptimized_fingerprint =
      MaybeUploadUnoptimizedGpuSymbols(module.get(),
                                       gpu_target_config.ToProto());
  XLA_SCOPED_LOGGING_TIMER_IF(
      absl::StrCat("GpuCompiler::RunHloPasses for ", module->name()),
      !options.is_autotuning_compilation);
  uint64_t start_usecs = tsl::Env::Default()->NowMicros();
  tsl::profiler::TraceMe activity(
      [&] { return absl::StrCat("HLO Transforms:", module->name()); },
      tsl::profiler::TraceMeLevel::kInfo);
  TF_RETURN_IF_ERROR(OptimizeHloModule(module.get(),
                                       is_deviceless ? nullptr : stream_exec,
                                       options, gpu_target_config));
  TF_RETURN_IF_ERROR(PrepareHloModuleForIrEmitting(module.get()));
  if (module->config()
          .debug_options()
          .xla_gpu_experimental_enable_fusion_block_level_rewriter()) {
    HloPassPipeline pipeline("fusion-block-level-rewriter-pipeline");
    pipeline.AddPass<FusionBlockLevelRewriter>(
        gpu_target_config.device_description, ShapeSizeBytesFunction());
    TF_RETURN_IF_ERROR(pipeline.Run(module.get()).status());
  }
  uint64_t end_usecs = tsl::Env::Default()->NowMicros();
  RecordHloPassesDuration(end_usecs - start_usecs);
  DumpHloModuleMetadataIfEnabled({module.get()});
  AutotuneResults autotune_results;
  TF_ASSIGN_OR_RETURN(
      AutotuneConfig autotune_config,
      GetAutotuneConfig(stream_exec, debug_opts, options, gpu_target_config));
  if (!is_deviceless) {
    TF_RETURN_IF_ERROR(
        AutotunerUtil::SerializeAutotuneResults(&autotune_results));
    TF_RETURN_IF_ERROR(SerializeAutotuneResultsToFile(debug_opts));
  }
  const std::optional<std::string> optimized_fingerprint =
      MaybeUploadOptimizedGpuSymbols(module.get(), autotune_results);
  if (unoptimized_fingerprint.has_value() &&
      optimized_fingerprint.has_value()) {
    MaybeUploadGpuSymbolMapping(*unoptimized_fingerprint,
                                *optimized_fingerprint);
  }
  if (DumpingEnabledForHloModule(*module)) {
    TF_ASSIGN_OR_RETURN(
        std::string autotune_results,
        AutotunerUtil::SerializeAutotuneResults(true));
    DumpToFileInDirOrStdout(*module, "", "autotune_results.pbtxt",
                            autotune_results);
  }
  return std::move(module);
}
namespace {
absl::Status RunPostSchedulingCopyInsertion(
    HloModule* module,
    const HloDataflowAnalysis::CanShareBuffer& can_share_buffer) {
  constexpr int64_t kRegionBasedLiveRangeAnalysisLimit = -1;
  const int64_t kUseRegionBasedLiveRangeAnalysis =
      module->config()
              .debug_options()
              .xla_gpu_copy_insertion_use_region_analysis()
          ? kRegionBasedLiveRangeAnalysisLimit
          : 0;
  CopyInsertion copy_insertion(can_share_buffer,
                               kUseRegionBasedLiveRangeAnalysis);
  TF_RETURN_IF_ERROR(copy_insertion.RemoveUnnecessaryCopies(module));
  HloSchedule saved_schedule = module->schedule();
  module->clear_schedule();
  TF_RETURN_IF_ERROR(
      copy_insertion.CopyInsertion::AddSpecialCaseCopies(module));
  TF_RETURN_IF_ERROR(HloDCE().Run(module).status());
  TF_RETURN_IF_ERROR(saved_schedule.Update());
  TF_RETURN_IF_ERROR(module->set_schedule(std::move(saved_schedule)));
  return absl::OkStatus();
}
}  
using OutputInfoMap =
    absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>;
static void NullDiagnosticHandler(const llvm::DiagnosticInfo* diag_info,
                                  void* context) {
  std::string error_string;
  llvm::raw_string_ostream string_printer(error_string);
  llvm::DiagnosticPrinterRawOStream diagnostic_printer(string_printer);
  diag_info->print(diagnostic_printer);
  VLOG(5) << error_string;
}
namespace {
std::unique_ptr<llvm::Module> CopyToContext(const llvm::Module& module,
                                            llvm::LLVMContext& context) {
  llvm::SmallString<0> bitcode;
  llvm::raw_svector_ostream bitcode_ostream(bitcode);
  llvm::WriteBitcodeToFile(module, bitcode_ostream);
  llvm::Expected<std::unique_ptr<llvm::Module>> new_module =
      llvm::parseBitcodeFile(
          llvm::MemoryBufferRef(llvm::StringRef(bitcode.data(), bitcode.size()),
                                "split_module"),
          context);
  CHECK(new_module) << "Failed to parse bitcode "
                    << llvm::toString(new_module.takeError());
  return std::move(new_module.get());
}
}  
absl::StatusOr<GpuCompiler::BackendCompileResult>
GpuCompiler::CompileSingleModule(const HloModuleConfig& module_config,
                                 se::GpuComputeCapability gpu_version,
                                 const HloModule* debug_module,
                                 llvm::Module* llvm_module, bool relocatable,
                                 const CompileOptions& options,
                                 std::optional<int> shard_number) {
  {
    XLA_SCOPED_LOGGING_TIMER_IF(
        absl::StrCat(
            "GpuCompiler::RunBackend - Running LLVM verifier for ",
            (debug_module != nullptr ? debug_module->name() : "(unknown)")),
        VLOG_IS_ON(4) && !options.is_autotuning_compilation);
    llvm_module->getContext().setDiagnosticHandlerCallBack(
        NullDiagnosticHandler, nullptr);
    std::string err;
    llvm::raw_string_ostream err_stream(err);
    TF_RET_CHECK(!llvm::verifyModule(*llvm_module, &err_stream))
        << "Invalid LLVM IR before optimizations:\n"
        << err_stream.str()
        << "\nThis probably indicates a bug in the HLO -> LLVM IR "
           "lowering. Rerun with --xla_dump_to to get the IR"
        << (debug_module
                ? absl::StrCat(" and looks for files with name containing: *",
                               FilenameFor(*debug_module, "", ""), "*")
                : ".");
  }
  TF_ASSIGN_OR_RETURN(
      BackendCompileResult result,
      CompileTargetBinary(module_config, llvm_module, gpu_version, relocatable,
                          debug_module, options));
  const bool should_dump = DumpingEnabledForHloModule(
      debug_module ? debug_module->name() : "", module_config.debug_options());
  if (should_dump) {
    if (debug_module) {
      llvm_ir::DumpIrIfEnabled(
          *debug_module, *llvm_module,
          true,
          shard_number.has_value() ? std::to_string(*shard_number) : "");
    } else {
      LOG(ERROR) << "Dumping is not implemented since the file name cannot be "
                    "inferred. Please implement (potentially MLIR) module -> "
                    "filename heuristic.";
    }
  }
  if (user_post_optimization_hook_) {
    user_post_optimization_hook_(*llvm_module);
  }
  if (should_dump) {
    absl::string_view ptx = result.asm_text;
    if (debug_module) {
      DumpToFileInDirOrStdout(*debug_module, "",
                              shard_number.has_value()
                                  ? (std::to_string(*shard_number) + ".ptx")
                                  : "ptx",
                              ptx);
    } else {
      LOG(ERROR) << "Dumping is not implemented since the file name cannot be "
                    "inferred. Please implement (potentially MLIR) module -> "
                    "filename heuristic.";
    }
  }
  return result;
}
namespace {
int CountFunctions(const llvm::Module& module) {
  int num_functions = 0;
  for (const llvm::Function& func : module.functions()) {
    if (!func.isDeclaration() &&
        func.getLinkage() == llvm::GlobalValue::LinkageTypes::ExternalLinkage) {
      ++num_functions;
    }
  }
  return num_functions;
}
std::string SingleFunctionName(const llvm::Module& module) {
  std::string name;
  for (const llvm::Function& func : module.functions()) {
    if (!func.isDeclaration() &&
        func.getLinkage() == llvm::GlobalValue::LinkageTypes::ExternalLinkage) {
      if (name.empty()) {
        name = func.getName().str();
      } else {
        return "";
      }
    }
  }
  return name;
}
}  
absl::StatusOr<GpuCompiler::BackendCompileResult> GpuCompiler::CompileAndLink(
    const HloModuleConfig& module_config,
    CompileModuleResults& compile_module_results,
    se::GpuComputeCapability gpu_version, se::StreamExecutor* stream_exec,
    const CompileOptions& options, const HloModule* debug_module) {
  llvm::Module* llvm_module = &*compile_module_results.llvm_module;
  bool force_module_split =
      module_config.debug_options().xla_llvm_force_inline_before_split();
  if (force_module_split) {
    for (llvm::Function& func : llvm_module->functions()) {
      if (func.getNumUses() > 0 && !func.isDeclaration()) {
        VLOG(4) << absl::StrFormat("Inlining function %s with %d users.\n",
                                   func.getName().str(), func.getNumUses());
        std::vector<llvm::CallInst*> calls_to_inline;
        for (auto* user : func.users()) {
          if (auto* call = llvm::dyn_cast<llvm::CallInst>(user)) {
            calls_to_inline.push_back(call);
          }
        }
        for (auto* call_to_inline : calls_to_inline) {
          llvm::InlineFunctionInfo inline_function_info;
          if (!llvm::InlineFunction(*call_to_inline, inline_function_info)
                   .isSuccess()) {
            return absl::InternalError("Can not inline function " +
                                       func.getName().str());
          };
        }
      }
    }
  }
  llvm::DenseMap<llvm::StringRef, llvm::Constant*> const_initializer_map;
  llvm::Module& module_with_constants =
      (compile_module_results.llvm_module_constants == nullptr)
          ? *llvm_module
          : *compile_module_results.llvm_module_constants;
  for (llvm::GlobalVariable& gv : module_with_constants.globals()) {
    if (gv.hasName() && gv.isConstant() && gv.hasInitializer() &&
        gv.hasExternalLinkage()) {
      llvm::Constant* initializer = gv.getInitializer();
      unsigned int num_elements = 0;
      if (auto* caz =
              llvm::dyn_cast<llvm::ConstantAggregateZero>(initializer)) {
        num_elements = caz->getElementCount().getFixedValue();
      } else if (auto* cds = llvm::dyn_cast<llvm::ConstantDataSequential>(
                     initializer)) {
        num_elements = cds->getNumElements();
      }
      if (num_elements > 0) {
        const_initializer_map[gv.getName()] = initializer;
      }
    }
  }
  llvm_ir::DumpIrIfEnabled(*debug_module, *llvm_module,
                           false, "inlined");
  absl::string_view cache_path =
      module_config.debug_options().xla_gpu_kernel_cache_file();
  const bool use_cache = !cache_path.empty();
  struct NamedModule {
    std::string name;
    std::unique_ptr<llvm::Module> module;
  };
  std::vector<NamedModule> llvm_modules;
  MaybeOwningThreadPool thread_pool = CreateMaybeOwningThreadPool(
      module_config.debug_options()
          .xla_gpu_force_compilation_parallelism(),
      options.thread_pool,
      1);
  int num_modules = CountFunctions(*llvm_module);
  if (thread_pool.get() != nullptr && !use_cache) {
    num_modules = std::max(1, std::min(thread_pool->NumThreads(), num_modules));
  }
  if (compile_module_results.llvm_module_constants != nullptr) {
    llvm_modules.reserve(num_modules + 1);
    llvm_modules.push_back(
        {"", std::move(compile_module_results.llvm_module_constants)});
  } else {
    llvm_modules.reserve(num_modules);
  }
  int single_function_module_count = 0;
  llvm::SplitModule(
      *llvm_module, num_modules,
      [&](std::unique_ptr<llvm::Module> module) {
        for (llvm::GlobalVariable& gv : module->globals()) {
          if (gv.hasName() && gv.isConstant() && !gv.hasInitializer() &&
              const_initializer_map.count(gv.getName()) != 0) {
            gv.setInitializer(const_initializer_map[gv.getName()]);
            gv.setLinkage(llvm::GlobalValue::InternalLinkage);
          }
        }
        const std::string name = SingleFunctionName(*module);
        if (!name.empty()) {
          ++single_function_module_count;
        }
        llvm_modules.push_back({name, std::move(module)});
      },
      true, true);
  VLOG(2) << "Single-function cacheable modules: "
          << single_function_module_count << " / " << llvm_modules.size();
  struct NamedCompileResult {
    std::string name;
    absl::StatusOr<BackendCompileResult> result;
  };
  std::vector<NamedCompileResult> compile_results(llvm_modules.size());
  if (thread_pool.get() != nullptr) {
    tsl::BlockingCounter counter(llvm_modules.size());
    for (int i = 0; i < llvm_modules.size(); ++i) {
      thread_pool.get_mutable()->Schedule(
          [&compile_results, i, &llvm_modules, &counter, this, &module_config,
           &gpu_version, &debug_module, &options] {
            llvm::LLVMContext new_context;
            std::unique_ptr<llvm::Module> new_module =
                CopyToContext(*llvm_modules.at(i).module, new_context);
            compile_results.at(i) = {
                llvm_modules.at(i).name,
                CompileSingleModule(module_config, gpu_version, debug_module,
                                    new_module.get(),
                                    true, options,
                                    i)};
            counter.DecrementCount();
          });
    }
    counter.Wait();
  } else {
    for (int i = 0; i < llvm_modules.size(); ++i) {
      compile_results.at(i) = {
          llvm_modules.at(i).name,
          CompileSingleModule(module_config, gpu_version, debug_module,
                              &*llvm_modules.at(i).module,
                              true, options,
                              i)};
    }
  }
  std::string ptx_snippets;
  std::vector<std::vector<uint8_t>> binaries_to_link;
  binaries_to_link.reserve(compile_results.size());
  std::vector<KernelReuseCache::NamedBinary> binaries_to_cache;
  binaries_to_cache.reserve(single_function_module_count);
  for (const auto& [name, maybe_result] : compile_results) {
    TF_ASSIGN_OR_RETURN(auto result, maybe_result);
    if (result.binary.empty()) {
      continue;
    }
    ptx_snippets += result.asm_text;
    ptx_snippets += "\n";
    binaries_to_link.push_back(result.binary);
    if (!name.empty()) {
      binaries_to_cache.push_back({name, result.binary});
    }
  }
  if (use_cache) {
    std::string resolved_path;
    if (!tsl::io::ResolveTestPrefixes(cache_path, resolved_path)) {
      return FailedPrecondition("File path can not be resolved: %s",
                                cache_path);
    }
    const CompilationCacheProto& current_cache =
        compile_module_results.kernel_compilation_cache;
    const bool cache_file_exists =
        tsl::Env::Default()->FileExists(resolved_path).ok();
    if (cache_file_exists) {
      int loaded_kernel_count = 0;
      for (const auto& [name, entry] : current_cache.entries()) {
        if (llvm_module->getFunction(name) != nullptr) {
          VLOG(5) << "Using the just compiled kernel for " << name;
          TF_RET_CHECK(entry.binary().empty())
              << name
              << " is a just compiled kernel and is not expected to have a "
                 "binary yet.";
          continue;
        }
        const uint8_t* binary =
            reinterpret_cast<const uint8_t*>(entry.binary().data());
        binaries_to_link.push_back(
            std::vector<uint8_t>(binary, binary + entry.binary().size()));
        VLOG(5) << "Using " << name << " from cache: " << entry.binary().size();
        ++loaded_kernel_count;
      }
      VLOG(2) << "Using " << loaded_kernel_count << " / "
              << current_cache.entries_size() << " cached kernels.";
    }
    if (!binaries_to_cache.empty()) {
      TF_RETURN_IF_ERROR(
          UpdateDiskKernelCache(resolved_path, cache_file_exists,
                                current_cache, binaries_to_cache));
    }
  }
  auto maybe_backend_result =
      LinkModules(gpu_version, stream_exec, std::move(binaries_to_link),
                  module_config.debug_options());
  if (!maybe_backend_result.ok()) {
    LOG(ERROR) << "The CUDA linking API did not work. Please use XLA_FLAGS="
                  "--xla_gpu_enable_llvm_module_compilation_parallelism=false "
                  "to bypass it, but expect to get longer compilation time due "
                  "to the lack of multi-threading. Original error: "
               << maybe_backend_result.status();
    return maybe_backend_result.status();
  }
  VLOG(4) << "Binary size after linking [B]: " << maybe_backend_result->size();
  compile_module_results.kernel_compilation_cache.Clear();
  return BackendCompileResult{ptx_snippets, std::move(*maybe_backend_result)};
}
absl::StatusOr<GpuCompiler::CompileResultWithMetadata>
GpuCompiler::CompileToBackendResult(
    HloModule* module, llvm::LLVMContext* llvm_context,
    se::StreamExecutor* executor, const CompileOptions& options,
    const se::DeviceDescription& gpu_device_info) {
  tsl::profiler::TraceMe traceme("GpuCompiler::CompileToBackendResult");
  TF_RETURN_IF_ERROR(RunPreSchedulingPasses(module, executor));
  TF_ASSIGN_OR_RETURN(
      ScheduleMetadata schedule_metadata,
      ScheduleGpuModule(module, pointer_size_, gpu_device_info));
  TF_RETURN_IF_ERROR(RunPostSchedulingPipelines(
      module, schedule_metadata.scheduler_mem_limit, gpu_device_info));
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      se::PlatformManager::PlatformWithId(PlatformId()));
  bool can_use_link_modules = (executor != nullptr);
  if (can_use_link_modules) {
    TF_ASSIGN_OR_RETURN(can_use_link_modules,
                        CanUseLinkModules(module->config()));
  }
  const bool split_modules =
      can_use_link_modules &&
      module->config()
          .debug_options()
          .xla_gpu_enable_llvm_module_compilation_parallelism();
  const bool use_cache =
      split_modules &&
      !module->config().debug_options().xla_gpu_kernel_cache_file().empty();
  TF_ASSIGN_OR_RETURN(
      CompileModuleResults compile_module_results,
      CompileModuleToLlvmIr(module, llvm_context, target_triple_, data_layout_,
                            platform->Name(), platform->id(), gpu_device_info,
                            GetCanShareBuffer(), BufferSizeBytesFunction(),
                            use_cache));
  if (user_pre_optimization_hook_) {
    user_pre_optimization_hook_(*compile_module_results.llvm_module);
    if (compile_module_results.llvm_module_constants != nullptr) {
      user_pre_optimization_hook_(
          *compile_module_results.llvm_module_constants);
    }
  }
  llvm_ir::DumpIrIfEnabled(*module, *compile_module_results.llvm_module,
                           false);
  if (compile_module_results.llvm_module_constants != nullptr) {
    llvm_ir::DumpIrIfEnabled(*module,
                             *compile_module_results.llvm_module_constants,
                             false, "constants");
  }
  BackendCompileResult backend_result;
  if (split_modules) {
    TF_ASSIGN_OR_RETURN(backend_result,
                        CompileAndLink(module->config(), compile_module_results,
                                       gpu_device_info.gpu_compute_capability(),
                                       executor, options, module));
  } else {
    CHECK(compile_module_results.llvm_module_constants == nullptr);
    TF_ASSIGN_OR_RETURN(
        backend_result,
        CompileSingleModule(module->config(),
                            gpu_device_info.gpu_compute_capability(), module,
                            &*compile_module_results.llvm_module,
                            false, options,
                            std::nullopt));
  }
  RecordXlaDeviceBinarySize(backend_result.binary.size());
  if (DumpingEnabledForHloModule(*module)) {
    DumpToFileInDirOrStdout(
        *module, "", "thunk_sequence.txt",
        compile_module_results.executable->ToString(0));
  }
  return CompileResultWithMetadata{std::move(backend_result),
                                   std::move(compile_module_results)};
}
absl::StatusOr<std::unique_ptr<Executable>> GpuCompiler::RunBackend(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  tsl::profiler::ScopedAnnotation backend_annotation{[&] {
    return absl::StrFormat("XlaCompileBackend:#module=%s,program_id=%d#",
                           module->name(), module->unique_id());
  }};
  BinaryMap dnn_compiled_graphs;
  if (stream_exec) {
    TF_RETURN_IF_ERROR(RunCudnnCompilerPasses(module.get(), stream_exec,
                                              &dnn_compiled_graphs));
  }
  const DebugOptions& debug_opts = module->config().debug_options();
  TF_ASSIGN_OR_RETURN(TargetConfig gpu_target_config,
                      GetTargetConfig(options, debug_opts, stream_exec));
  if (DumpingEnabledForHloModule(*module)) {
    std::string textproto;
    tsl::protobuf::TextFormat::PrintToString(gpu_target_config.ToProto(),
                                             &textproto);
    DumpToFileInDirOrStdout(*module, "", "gpu_target_config.pbtxt", textproto);
  }
  if (!options.is_autotuning_compilation) {
    VLOG(1) << "Starting to compile HLO module " << module->name();
  }
  XLA_SCOPED_LOGGING_TIMER_IF(
      absl::StrCat("GpuCompiler::RunBackend for ", module->name()),
      !options.is_autotuning_compilation);
  std::string slow_compilation_msg =
      absl::StrCat("Compiling module ", module->name());
  auto slow_compile_alarm = SlowCompilationAlarm(slow_compilation_msg);
  if (options.is_autotuning_compilation) {
    if (module->config().debug_options().xla_embed_ir_in_executable()) {
      LOG(WARNING) << "Doing autotuning compilations with "
                      "xla_embed_ir_in_executable wastes memory!";
    }
  }
  llvm::LLVMContext llvm_context;
  const se::DeviceDescription& gpu_device_info =
      gpu_target_config.device_description;
  if (module->config().hlo_profiling_enabled() || VLOG_IS_ON(1)) {
    HloCostAnalysis::Options cost_analysis_options{ShapeSizeBytesFunction()};
    cost_analysis_options.set_bytes_per_second(
        gpu_device_info.memory_bandwidth());
    GpuHloCostAnalysis cost_analysis(cost_analysis_options, gpu_device_info);
    TF_RETURN_IF_ERROR(module->entry_computation()->Accept(&cost_analysis));
    if (!options.is_autotuning_compilation) {
      VLOG(1) << "HLO memory read+written: "
              << tsl::strings::HumanReadableNumBytes(
                     cost_analysis.bytes_accessed());
    }
    if (module->config().hlo_profiling_enabled()) {
      LOG(ERROR) << "--xla_hlo_profile for GPU is unsupported.";
    }
  }
  TF_ASSIGN_OR_RETURN(
      CompileResultWithMetadata res,
      CompileToBackendResult(module.get(), &llvm_context, stream_exec, options,
                             gpu_device_info));
  if (DumpingEnabledForHloModule(*module)) {
    DumpToFileInDirOrStdout(
        *module, "", "thunk_sequence.txt",
        res.compile_module_results.executable->ToString(0));
  }
  bool embed_ir_in_executable =
      module->config().debug_options().xla_embed_ir_in_executable();
  int64_t debug_buffer_assignment_show_max =
      module->config().debug_options().xla_debug_buffer_assignment_show_max();
  tsl::profiler::ScopedAnnotation annotation([&] {
    return absl::StrFormat("XlaCreateGpuExecutable:#module=%s#",
                           module->name());
  });
  TF_ASSIGN_OR_RETURN(
      auto gpu_executable,
      GpuExecutable::Create(GpuExecutable::Params{
          (options.is_autotuning_compilation &&
                        !res.backend_result.binary.empty())
              ? std::string()
              : std::move(res.backend_result.asm_text),
          std::move(res.backend_result.binary),
          std::move(dnn_compiled_graphs),
          gpu_device_info.gpu_compute_capability(),
          std::move(res.compile_module_results.executable),
          std::move(res.compile_module_results.constants),
          std::move(res.compile_module_results.output_info),
          std::move(res.compile_module_results.module_name),
          std::move(res.compile_module_results.output_shape),
          (res.compile_module_results.use_original_allocations
               ? std::optional<std::vector<BufferAllocation>>()
               : std::move(res.compile_module_results.allocations)),
          std::move(res.compile_module_results.buffer_assignment),
          debug_buffer_assignment_show_max,
          options.is_autotuning_compilation
              ? std::unique_ptr<HloModule>()
              : std::move(module),
          !options.is_autotuning_compilation}));
  if (embed_ir_in_executable) {
    std::string ir_module_string_before_opt =
        llvm_ir::DumpToString(res.compile_module_results.llvm_module.get());
    gpu_executable->set_ir_module_string(ir_module_string_before_opt);
    DCHECK_NE("", ir_module_string_before_opt);
  }
  IncrementCompiledProgramsCount();
  if (!options.is_autotuning_compilation && gpu_executable->has_module()) {
    auto hlo_proto = std::make_unique<HloProto>();
    *hlo_proto->mutable_buffer_assignment() =
        gpu_executable->buffer_assignment()->ToProto();
    gpu_executable->set_hlo_proto(std::move(hlo_proto));
    gpu_executable->set_debug_info(
        gpu_executable->buffer_assignment()->GetStats().ToString());
  }
  return static_cast<std::unique_ptr<Executable>>(std::move(gpu_executable));
}
absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
GpuCompiler::CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                                const AotCompilationOptions& options) {
  CHECK_EQ(options.PlatformId(), PlatformId());
  std::vector<std::unique_ptr<HloModule>> modules =
      module_group->ConsumeModules();
  std::vector<std::unique_ptr<HloModule>> optimized_modules;
  optimized_modules.reserve(modules.size());
  for (std::unique_ptr<HloModule>& module : modules) {
    if (!module->has_schedule()) {
      tsl::profiler::ScopedAnnotation annotation{[&] {
        return absl::StrFormat("XlaCompile:#module=%s,program_id=%d#",
                               module->name(), module->unique_id());
      }};
      CompileOptions compile_options;
      compile_options.device_allocator = options.device_allocator();
      compile_options.target_config = options.target_config();
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<HloModule> optimized_module,
          RunHloPasses(std::move(module), options.executor(), compile_options));
      optimized_modules.push_back(std::move(optimized_module));
    } else {
      optimized_modules.push_back(std::move(module));
    }
  }
  modules = std::move(optimized_modules);
  std::vector<std::unique_ptr<AotCompilationResult>> results;
  const std::optional<Compiler::TargetConfig>& target_config =
      options.target_config();
  CHECK(target_config.has_value() || options.executor() != nullptr);
  const se::DeviceDescription& gpu_device_info =
      target_config.has_value() ? target_config->device_description
                                : options.executor()->GetDeviceDescription();
  for (const std::unique_ptr<HloModule>& module : modules) {
    llvm::LLVMContext llvm_context;
    TF_ASSIGN_OR_RETURN(
        CompileResultWithMetadata res,
        CompileToBackendResult(module.get(), &llvm_context, options.executor(),
                               {options.device_allocator()}, gpu_device_info));
    TF_ASSIGN_OR_RETURN(
        results.emplace_back(),
        GpuThunkAotCompilationResult::FromModule(
            module.get(), res.compile_module_results.buffer_assignment.get(),
            res.backend_result.asm_text, res.backend_result.binary,
            res.backend_result.dnn_compiled_graphs));
  }
  return std::move(results);
}
HloCostAnalysis::ShapeSizeFunction GpuCompiler::ShapeSizeBytesFunction() const {
  return [pointer_size = pointer_size_](const Shape& shape) {
    return GetSizeOfShape(shape, pointer_size);
  };
}
absl::StatusOr<std::unique_ptr<AotCompilationResult>> GpuCompiler::Export(
    Executable* executable) const {
  auto* gpu_executable = tensorflow::down_cast<GpuExecutable*>(executable);
  if (!gpu_executable) return Internal("GpuExecutable is null");
  return GpuThunkAotCompilationResult::FromModule(
      &gpu_executable->module(), gpu_executable->buffer_assignment(),
      gpu_executable->text(), gpu_executable->binary(),
      gpu_executable->dnn_compiled_graphs());
}
absl::Status GpuCompiler::RunPreSchedulingPasses(
    HloModule* module, se::StreamExecutor* stream_exec) {
  HloPassPipeline pipeline("pre-scheduling-passes");
  pipeline.AddPass<FusionWrapper>();
  return pipeline.Run(module).status();
}
HloCostAnalysis::Options CreateHloAnalysisOpts(
    const HloModule& module, const se::DeviceDescription& gpu_device_info,
    ShapeSizeFn shape_size_fn) {
  HloCostAnalysis::Options hlo_cost_analysis_options;
  hlo_cost_analysis_options.shape_size = shape_size_fn;
  std::optional<HloRematerialization::HostMemoryOffloadConfig>
      offloading_config = std::nullopt;
  if (module.config().debug_options().xla_gpu_enable_host_memory_offloading()) {
    constexpr float kGiga = 1e+9;
    constexpr float kFma = 2;
    float flops_per_sec = gpu_device_info.core_count() *
                          gpu_device_info.fpus_per_core() *
                          gpu_device_info.clock_rate_ghz() * kGiga * kFma;
    int64_t host_memory_space_color =
        static_cast<int64_t>(se::MemoryType::kHost);
    hlo_cost_analysis_options.set_flops_per_second(flops_per_sec);
    hlo_cost_analysis_options.set_transcendentals_per_second(flops_per_sec);
    offloading_config =
        std::make_optional<HloRematerialization::HostMemoryOffloadConfig>(
            host_memory_space_color,
            gpu_device_info.memory_bandwidth(),
            gpu_device_info.memory_bandwidth());
  }
  return hlo_cost_analysis_options;
}
HloRematerialization::Options CreateRematOpts(
    const HloModule& module, const se::DeviceDescription& gpu_device_info,
    HloCostAnalysis& hlo_cost_analysis, int64_t scheduler_mem_limit) {
  bool enable_offloading =
      module.config().debug_options().xla_gpu_enable_host_memory_offloading();
  std::optional<HloRematerialization::HostMemoryOffloadConfig>
      offloading_config = std::nullopt;
  if (enable_offloading) {
    int64_t host_memory_space_color =
        static_cast<int64_t>(se::MemoryType::kHost);
    offloading_config =
        std::make_optional<HloRematerialization::HostMemoryOffloadConfig>(
            host_memory_space_color,
            gpu_device_info.memory_bandwidth(),
            gpu_device_info.memory_bandwidth());
  }
  HloRematerialization::RematerializationModeConfig
      rematerialization_mode_config(true, true,
                                    enable_offloading);
  HloRematerialization::Options options(
      hlo_cost_analysis, rematerialization_mode_config,
      scheduler_mem_limit,
      1, 1,
      0, nullptr,
      offloading_config);
  return options;
}
absl::Status GpuCompiler::RunPostSchedulingPipelines(
    HloModule* module, int64_t scheduler_mem_limit,
    const se::DeviceDescription& gpu_device_info) const {
  TF_RETURN_IF_ERROR(
      RunPostSchedulingCopyInsertion(module, GetCanShareBuffer()));
  HloPassPipeline main_pipeline("post-scheduling-passes");
  HloPredicate is_nop =
      HloPredicateIsOp<HloOpcode::kParameter, HloOpcode::kConstant,
                       HloOpcode::kBitcast, HloOpcode::kGetTupleElement>;
  {
    HloPassPipeline& pipeline =
        main_pipeline.AddPass<HloPassPipeline>("async-to-sync-converter");
    if (module->config()
            .debug_options()
            .xla_gpu_enable_pipelined_collectives() ||
        module->config().debug_options().xla_gpu_enable_pipelined_p2p()) {
      pipeline.AddPass<PipelinedP2PRewriter>();
    }
    pipeline.AddPass<GpuConvertAsyncCollectivesToSync>(is_nop);
  }
  HloRematerialization::RematerializationSizes sizes;
  HloCostAnalysis::Options hlo_cost_analysis_opts =
      CreateHloAnalysisOpts(*module, gpu_device_info, ShapeSizeBytesFunction());
  HloCostAnalysis hlo_cost_analysis(hlo_cost_analysis_opts);
  HloRematerialization::Options remat_opts = CreateRematOpts(
      *module, gpu_device_info, hlo_cost_analysis, scheduler_mem_limit);
  {
    HloPassPipeline& pipeline =
        main_pipeline.AddPass<HloPassPipeline>("remat-pipeline");
    pipeline.AddPass<HloRematerialization>(remat_opts, sizes);
    pipeline.AddPass<StreamAttributeAnnotator>();
    pipeline.AddPass<OptimizationBarrierExpander>();
  }
  {
    HloPassPipeline& pipeline =
        main_pipeline.AddPass<HloPassPipeline>("fusion-wrapper");
    pipeline.AddPass<FusionWrapper>();
  }
  {
    HloPassPipeline& pipeline =
        main_pipeline.AddPass<HloPassPipeline>("command-buffer-scheduling");
    pipeline.AddPass<CommandBufferScheduling>(gpu_device_info);
    pipeline.AddPass<SanitizeConstantNames>();
  }
  if (module->config().debug_options().xla_gpu_enable_pgle_accuracy_checker()) {
    AddHloVerifier(
        &main_pipeline,
        module->config().debug_options().xla_experimental_ignore_channel_id(),
        HloVerifierOpts{}.VerifyInstructionNameUnchanged());
  }
  return main_pipeline.Run(module).status();
}
absl::Status GpuCompiler::LoadAutotuneResultsFromFile(
    const DebugOptions& debug_options) {
  if (absl::string_view file_path =
          debug_options.xla_gpu_load_autotune_results_from();
      !file_path.empty()) {
    static absl::once_flag once;
    absl::Status status = absl::OkStatus();
    absl::call_once(once, [&file_path, &status] {
      status = AutotunerUtil::LoadAutotuneResultsFromFile(file_path);
    });
    TF_RETURN_IF_ERROR(status);
  }
  return absl::OkStatus();
}
absl::Status GpuCompiler::SerializeAutotuneResultsToFile(
    const DebugOptions& debug_options) {
  if (absl::string_view file_path =
          debug_options.xla_gpu_dump_autotune_results_to();
      !file_path.empty()) {
    TF_RETURN_IF_ERROR(
        AutotunerUtil::SerializeAutotuneResultsToFile(file_path));
  }
  return absl::OkStatus();
}
absl::StatusOr<std::unique_ptr<AotCompilationResult>>
GpuCompiler::LoadAotCompilationResult(
    const std::string& serialized_aot_result) {
  return LoadAotCompilationResultStatic(serialized_aot_result);
}
absl::StatusOr<std::unique_ptr<AotCompilationResult>>
GpuCompiler::LoadAotCompilationResultStatic(
    const std::string& serialized_aot_result) {
  return GpuThunkAotCompilationResult::FromString(serialized_aot_result);
}
}  
}  