#include "xla/service/gpu/autotuning/gemm_algorithm_picker.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/autotuning/autotuner_compile_util.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/buffer_comparator.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/variant_visitor.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/gpu/redzone_allocator.h"
#include "xla/tsl/util/proto/proto_utils.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/scoped_annotation.h"
namespace xla {
namespace gpu {
namespace {
using se::gpu::BlasLt;
absl::StatusOr<BlasLt::Epilogue> AsBlasLtEpilogue(
    GemmBackendConfig_Epilogue epilogue) {
  switch (epilogue) {
    case GemmBackendConfig::DEFAULT:
      return BlasLt::Epilogue::kDefault;
    case GemmBackendConfig::RELU:
      return BlasLt::Epilogue::kReLU;
    case GemmBackendConfig::GELU:
      return BlasLt::Epilogue::kGELU;
    case GemmBackendConfig::GELU_AUX:
      return BlasLt::Epilogue::kGELUWithAux;
    case GemmBackendConfig::BIAS:
      return BlasLt::Epilogue::kBias;
    case GemmBackendConfig::BIAS_RELU:
      return BlasLt::Epilogue::kBiasThenReLU;
    case GemmBackendConfig::BIAS_GELU:
      return BlasLt::Epilogue::kBiasThenGELU;
    case GemmBackendConfig::BIAS_GELU_AUX:
      return BlasLt::Epilogue::kBiasThenGELUWithAux;
    default:
      return Internal("Unsupported Epilogue.");
  }
}
class GemmAutotuner {
  const AutotuneConfig& autotune_config_;
  RedzoneBuffers rz_buffers_;
  se::Stream* stream_ = nullptr;
  bool deterministic_ops_ = false;
  size_t solutions_limit_ = 0;
  size_t num_algorithms_left_ = 0;
 public:
  explicit GemmAutotuner(const AutotuneConfig& autotune_config)
      : autotune_config_(autotune_config) {}
  const AutotuneConfig& config() const { return autotune_config_; }
  size_t num_algorithms_left() const { return num_algorithms_left_; }
  absl::StatusOr<AutotuneResult> operator()(const HloInstruction* gemm,
                                            const AutotuneCacheKey& key) {
    num_algorithms_left_ = 0;
    if (autotune_config_.IsDeviceless()) {
      return AutotuneResult{};
    }
    VLOG(3) << "Starting autotune of GemmThunk " << gemm->ToString();
    TF_ASSIGN_OR_RETURN(stream_, autotune_config_.GetStream());
    const DebugOptions& debug_options =
        gemm->GetModule()->config().debug_options();
    deterministic_ops_ = RequireDeterminism(gemm->GetModule()->config());
    solutions_limit_ = debug_options.xla_gpu_autotune_max_solutions();
    TF_ASSIGN_OR_RETURN(auto gemm_config, GemmConfig::For(gemm));
    absl::MutexLock gpu_lock(&GetGpuMutex(stream_->parent()));
    TF_ASSIGN_OR_RETURN(rz_buffers_, RedzoneBuffers::FromInstruction(
                                         *gemm, autotune_config_, debug_options,
                                         RedzoneBuffers::kAllInputsAllOutputs));
    return IsCublasLtMatmul(*gemm) || IsCublasLtMatmulF8(*gemm)
               ? TuneGpuBlasLt(gemm, gemm_config)
               : TuneGpuBlas(gemm, gemm_config);
  }
 private:
  se::DeviceMemoryBase LhsBuffer() { return rz_buffers_.input_buffers().at(0); }
  se::DeviceMemoryBase RhsBuffer() { return rz_buffers_.input_buffers().at(1); }
  se::DeviceMemoryBase OutputBuffer() {
    return rz_buffers_.output_buffers().at(0);
  }
  const Shape& GetOutputShape(const HloInstruction* gemm) {
    return gemm->shape().IsTuple() ? gemm->shape().tuple_shapes(0)
                                   : gemm->shape();
  }
  absl::StatusOr<AutotuneResult> TuneGpuBlasLt(const HloInstruction* gemm,
                                               const GemmConfig& gemm_config) {
    auto workspace_buffer =
        rz_buffers_.output_buffers().at(gemm->shape().tuple_shapes_size() - 1);
    GpuBackendConfig gpu_config =
        gemm->backend_config<GpuBackendConfig>().value();
    const GemmBackendConfig& backend_config = gpu_config.gemm_backend_config();
    bool has_matrix_bias = gemm_config.beta != 0.;
    TF_ASSIGN_OR_RETURN(
        bool has_vector_bias,
        gpublas_lt::EpilogueAddsVectorBias(backend_config.epilogue()));
    TF_ASSIGN_OR_RETURN(
        bool has_aux_output,
        gpublas_lt::EpilogueHasAuxiliaryOutput(backend_config.epilogue()));
    TF_ASSIGN_OR_RETURN(auto epilogue,
                        AsBlasLtEpilogue(backend_config.epilogue()));
    se::DeviceMemoryBase a_scale_buffer, b_scale_buffer, c_scale_buffer,
        d_scale_buffer, d_amax_buffer, bias_buffer, aux_buffer;
    if (has_vector_bias) {
      bias_buffer = rz_buffers_.input_buffers().at(has_matrix_bias ? 3 : 2);
    }
    if (has_aux_output) {
      aux_buffer = rz_buffers_.output_buffers().at(1);
    }
    TF_ASSIGN_OR_RETURN(auto plan,
                        BlasLt::GetMatmulPlan(stream_, gemm_config, epilogue));
    TF_ASSIGN_OR_RETURN(
        auto algorithms,
        plan->GetAlgorithms( 128,
                             workspace_buffer.size()));
    auto tuned_func = [&](const BlasLt::MatmulAlgorithm& algorithm)
        -> absl::StatusOr<se::blas::ProfileResult> {
      TF_RETURN_IF_ERROR(plan->ExecuteOnStream(
          stream_, LhsBuffer(), RhsBuffer(), OutputBuffer(), OutputBuffer(),
          bias_buffer, aux_buffer, a_scale_buffer, b_scale_buffer,
          c_scale_buffer, d_scale_buffer, d_amax_buffer, algorithm,
          workspace_buffer));
      se::blas::ProfileResult profile_result;
      profile_result.set_warmup_run_executed(true);
      TF_RETURN_IF_ERROR(plan->ExecuteOnStream(
          stream_, LhsBuffer(), RhsBuffer(), OutputBuffer(), OutputBuffer(),
          bias_buffer, aux_buffer, a_scale_buffer, b_scale_buffer,
          c_scale_buffer, d_scale_buffer, d_amax_buffer, algorithm,
          workspace_buffer, &profile_result));
      return std::move(profile_result);
    };
    return GetBestAlgorithm<BlasLt::MatmulAlgorithm>(
        gemm, algorithms, gemm_config.beta,  true,
        tuned_func);
  }
  absl::StatusOr<AutotuneResult> TuneGpuBlas(const HloInstruction* gemm,
                                             const GemmConfig& gemm_config) {
    auto workspace_buffer = rz_buffers_.output_buffers().at(1);
    std::vector<se::blas::AlgorithmType> algorithms;
    TF_ASSIGN_OR_RETURN(GemmConfig::DescriptorsTuple desc,
                        gemm_config.GetMatrixDescriptors(
                            LhsBuffer(), RhsBuffer(), OutputBuffer()));
    auto blas = stream_->parent()->AsBlas();
    if (blas == nullptr) {
      return absl::InternalError("No BLAS support for stream");
    }
    blas->GetBlasGemmAlgorithms(stream_, desc.lhs, desc.rhs, &desc.output,
                                &gemm_config.alpha, &gemm_config.beta,
                                &algorithms);
    auto tuned_func = [&](const se::blas::AlgorithmType& algorithm)
        -> absl::StatusOr<se::blas::ProfileResult> {
      static_cast<void>(RunGemm(gemm_config, LhsBuffer(), RhsBuffer(),
                                OutputBuffer(), workspace_buffer,
                                deterministic_ops_, stream_, algorithm));
      se::blas::ProfileResult profile_result;
      profile_result.set_warmup_run_executed(true);
      TF_RETURN_IF_ERROR(RunGemm(gemm_config, LhsBuffer(), RhsBuffer(),
                                 OutputBuffer(), workspace_buffer,
                                 deterministic_ops_, stream_, algorithm,
                                 &profile_result));
      return std::move(profile_result);
    };
    return GetBestAlgorithm<se::blas::AlgorithmType>(
        gemm, algorithms, gemm_config.beta,  false,
        tuned_func);
  }
  template <typename AlgoT, typename TunedFunc>
  absl::StatusOr<AutotuneResult> GetBestAlgorithm(
      const HloInstruction* gemm, absl::Span<const AlgoT> algorithms,
      double beta, bool return_algo_index, TunedFunc&& run_benchmark) {
    static_assert(std::is_invocable_r_v<absl::StatusOr<se::blas::ProfileResult>,
                                        TunedFunc, const AlgoT&>,
                  "Tuned function has incorrect prototype!");
    if (!stream_->parent()->SynchronizeAllActivity()) {
      return Internal("Failed to synchronize GPU for autotuning.");
    }
    tsl::profiler::ScopedAnnotation annotation([&] {
      return absl::StrFormat("XlaAutotunerMeasurement:#hlo_op=%s#",
                             gemm->name());
    });
    auto& hlo_module_config = gemm->GetModule()->mutable_config();
    const auto& output_shape = GetOutputShape(gemm);
    se::DeviceMemoryBase reference_buffer;
    if (autotune_config_.should_check_correctness()) {
      TF_ASSIGN_OR_RETURN(reference_buffer,
                          rz_buffers_.RedzoneAllocator().AllocateBytes(
                              ShapeUtil::ByteSizeOf(output_shape)));
    }
    BufferComparator comparator(
        output_shape,
        hlo_module_config.debug_options().xla_gpu_autotune_gemm_rtol(),
         !autotune_config_.should_skip_wrong_results());
    std::vector<AutotuneResult> results;
    results.reserve(algorithms.size());
    std::optional<int64_t> reference_algorithm;
    auto num = algorithms.size();
    if (solutions_limit_ > 0) num = std::min(num, solutions_limit_);
    for (size_t i = 0; i < num; i++) {
      const AlgoT& algorithm = algorithms[i];
      if (autotune_config_.should_reinit_output_buffer() && beta != 0) {
        int64_t rng_state = 0;
        InitializeBuffer(stream_, output_shape.element_type(), &rng_state,
                         OutputBuffer());
      }
      TF_ASSIGN_OR_RETURN(auto profile_result, run_benchmark(algorithm));
      AutotuneResult& result = results.emplace_back();
      result.mutable_gemm()->set_algorithm(profile_result.algorithm());
      if (!profile_result.is_valid()) {  
        result.mutable_failure()->set_kind(AutotuneResult::DISQUALIFIED);
        continue;
      }
      VLOG(2) << "gemm algorithm " << profile_result.algorithm() << " took "
              << profile_result.elapsed_time_in_ms() << "ms";
      *result.mutable_run_time() = tsl::proto_utils::ToDurationProto(
          absl::Milliseconds(profile_result.elapsed_time_in_ms()));
      if (!autotune_config_.should_check_correctness()) {
        num_algorithms_left_++;
        continue;
      }
      TF_ASSIGN_OR_RETURN(
          se::RedzoneAllocator::RedzoneCheckStatus rz_check_status,
          rz_buffers_.RedzoneAllocator().CheckRedzones());
      if (!rz_check_status.ok()) {
        result.mutable_failure()->set_kind(AutotuneResult::REDZONE_MODIFIED);
        *result.mutable_failure()->mutable_msg() =
            rz_check_status.RedzoneFailureMsg();
        LOG(ERROR) << "Detected out-of-bounds write in gemm buffer";
        CHECK(!autotune_config_.should_crash_on_check_failure());
        continue;
      }
      num_algorithms_left_++;
      if (!reference_algorithm) {
        TF_RETURN_IF_ERROR(stream_->Memcpy(&reference_buffer, OutputBuffer(),
                                           OutputBuffer().size()));
        reference_algorithm = profile_result.algorithm();
        continue;
      }
      TF_ASSIGN_OR_RETURN(
          bool outputs_match,
          comparator.CompareEqual(stream_, OutputBuffer(),
                                  reference_buffer));
      if (!outputs_match) {
        LOG(ERROR) << "Results mismatch between different GEMM algorithms. "
                   << "This is likely a bug/unexpected loss of precision.";
        CHECK(!autotune_config_.should_crash_on_check_failure());
        auto kind = AutotuneResult::WRONG_RESULT;
        if (autotune_config_.should_skip_wrong_results()) {
          kind = AutotuneResult::DISQUALIFIED;
          num_algorithms_left_--;  
        }
        result.mutable_failure()->set_kind(kind);
        result.mutable_failure()->mutable_reference_gemm()->set_algorithm(
            *reference_algorithm);
      }
    }  
    absl::StatusOr<AutotuneResult> best =
        PickBestResult(results, gemm->ToString(), hlo_module_config);
    if (best.ok()) {
      if (!return_algo_index) return best;
      for (size_t i = 0; i < results.size(); ++i) {
        if (best->gemm().algorithm() == results[i].gemm().algorithm()) {
          best->mutable_gemm()->set_algorithm(i);
          return best;
        }
      }
      return Internal("unknown best algorithm");
    }
    LOG(WARNING) << "Failed to find best cuBLAS algorithm, GEMM performance "
                    "might be suboptimal: "
                 << best.status();
    return AutotuneResult{};
  }  
};  
absl::StatusOr<bool> RunOnInstruction(HloInstruction* gemm,
                                      GemmAutotuner& autotuner) {
  VLOG(3) << "Loading the autotune result of GemmThunk " << gemm->ToString();
  GpuBackendConfig gpu_config =
      gemm->backend_config<GpuBackendConfig>().value();
  GemmBackendConfig& backend_config = *gpu_config.mutable_gemm_backend_config();
  if (backend_config.alpha_real() == 0.0 &&
      backend_config.alpha_imag() == 0.0 && backend_config.beta() == 0.0) {
    VLOG(3) << "Skip degenerate gemm instruction auto tuning";
    return false;
  }
  const AutotuneConfig& config = autotuner.config();
  AutotuneCacheKey key(config.GetModelStr(), *gemm);
  TF_ASSIGN_OR_RETURN(AutotuneResult algorithm,
                      AutotunerUtil::Autotune(
                          gemm, config, [&] { return autotuner(gemm, key); }));
  auto old_algorithm = backend_config.selected_algorithm();
  bool update_algorithm =
      IsCublasLtMatmulF8(*gemm) ||
      std::visit(VariantVisitor{[](const se::CudaComputeCapability& cc) {
                                  return !cc.IsAtLeast(
                                      se::CudaComputeCapability::AMPERE);
                                },
                                [](const se::RocmComputeCapability&) {
                                  return true;  
                                }},
                 config.GetGpuComputeCapability());
  if (update_algorithm) {
    int64_t new_algorithm{};
    if (algorithm.has_gemm()) {
      new_algorithm = algorithm.gemm().algorithm();
    } else {
      new_algorithm = se::blas::kDefaultAlgorithm;
    }
    if (new_algorithm == old_algorithm &&
        backend_config.has_selected_algorithm()) {
      return false;
    }
    backend_config.set_selected_algorithm(new_algorithm);
    TF_RETURN_IF_ERROR(gemm->set_backend_config(gpu_config));
    return true;  
  }
  return false;  
}
absl::StatusOr<bool> RunOnComputation(HloComputation* computation,
                                      GemmAutotuner& autotuner,
                                      size_t* num_algorithms_left) {
  bool changed = false;
  for (HloInstruction* instr : computation->instructions()) {
    if (IsCublasGemm(*instr)) {
      TF_ASSIGN_OR_RETURN(bool result, RunOnInstruction(instr, autotuner));
      *num_algorithms_left =
          std::max(*num_algorithms_left, autotuner.num_algorithms_left());
      changed |= result;
    }
  }
  return changed;
}
}  
absl::StatusOr<bool> GemmAlgorithmPicker::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_SCOPED_LOGGING_TIMER(
      absl::StrCat("GemmAlgorithmPicker for ", module->name()));
  num_algorithms_left_ = 0;
  if (module->config().debug_options().xla_gpu_autotune_level() == 0) {
    VLOG(2) << "GEMM auto-tuning disabled, GemmAlgorithmPicker returning early";
    return false;
  }
  GemmAutotuner autotuner(config_);
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation, autotuner,
                                                      &num_algorithms_left_));
    changed |= result;
  }
  return changed;
}
}  
}  