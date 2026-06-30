#include "xla/service/gpu/transforms/gemm_rewriter.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/permutation_util.h"
#include "xla/primitive_util.h"
#include "xla/service/algorithm_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/ml_dtypes.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace gpu {
namespace {
namespace m = match;
absl::Status SetName(HloModule *module, HloInstruction *gemm) {
  if (IsCublasLtMatmul(*gemm)) {
    module->SetAndUniquifyInstrName(gemm, "cublas-lt-matmul");
    return absl::OkStatus();
  }
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                      gemm->backend_config<GpuBackendConfig>());
  const GemmBackendConfig &config = gpu_config.gemm_backend_config();
  const DotDimensionNumbers &dot_dims = config.dot_dimension_numbers();
  bool is_batch_dot = !dot_dims.lhs_batch_dimensions().empty() ||
                      !dot_dims.rhs_batch_dimensions().empty();
  module->SetAndUniquifyInstrName(
      gemm, is_batch_dot ? "cublas-batch-gemm" : "cublas-gemm");
  return absl::OkStatus();
}
bool SupportsEpilogueFusion(PrimitiveType type) {
  switch (type) {
    case F8E4M3FN:
    case F8E5M2:
    case F16:
    case BF16:
    case F32:
    case F64:
      return true;
    default:
      return false;
  }
}
bool IsF8Type(const HloInstruction *instr) {
  return primitive_util::IsF8Type(instr->shape().element_type());
}
Shape PadShapeToMultipleOf16(const Shape old_shape,
                             const absl::Span<const int64_t> batch_dims) {
  Shape padded_shape = old_shape;
  for (int i = 0; i < old_shape.rank(); ++i) {
    if (!absl::c_linear_search(batch_dims, i)) {
      int64_t padded_dimension =
          RoundUpTo<int64_t>(old_shape.dimensions(i), 16);
      padded_shape.set_dimensions(i, padded_dimension);
    }
  }
  return padded_shape;
}
HloInstruction *PadOperandToTargetShape(const Shape &target,
                                        HloInstruction *x) {
  if (ShapeUtil::Equal(target, x->shape()) ||
      !ShapeUtil::SameElementType(x->shape(), target)) {
    return x;
  }
  PaddingConfig padding_config;
  for (int i = 0; i < x->shape().rank(); ++i) {
    auto dimension = padding_config.add_dimensions();
    dimension->set_edge_padding_low(0);
    dimension->set_edge_padding_high(target.dimensions(i) -
                                     x->shape().dimensions(i));
    dimension->set_interior_padding(0);
  }
  HloInstruction *zero = x->AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::Zero(x->shape().element_type())));
  return x->AddInstruction(
      HloInstruction::CreatePad(target, x, zero, padding_config));
}
HloInstruction *PadOperandToMultipleOf16(absl::Span<const int64_t> batch_dims,
                                         HloInstruction *x) {
  Shape padded_shape = PadShapeToMultipleOf16(x->shape(), batch_dims);
  return PadOperandToTargetShape(padded_shape, x);
}
absl::StatusOr<HloInstruction *> InvertAndConvertScalar(HloInstruction *scalar,
                                                        bool invert) {
  DCHECK(ShapeUtil::IsScalar(scalar->shape()));
  if (invert) {
    Literal one_literal = LiteralUtil::One(scalar->shape().element_type());
    HloInstruction *one = scalar->parent()->AddInstruction(
        HloInstruction::CreateConstant(one_literal.Clone()));
    TF_ASSIGN_OR_RETURN(scalar, MakeBinaryHlo(HloOpcode::kDivide, one, scalar,
                                              &scalar->metadata()));
  }
  if (scalar->shape().element_type() != F32) {
    scalar = MakeConvertToHlo(scalar, F32, &scalar->metadata());
  }
  return scalar;
}
using InstrPath = std::vector<std::pair<HloInstruction *, int>>;
std::optional<InstrPath> FindF8SubgraphRecursive(
    HloInstruction *instr, absl::flat_hash_set<int> &visited_instrs) {
  if (!visited_instrs.emplace(instr->unique_id()).second) {
    return std::nullopt;
  }
  if (IsF8Type(instr)) {
    return InstrPath{{instr, -1}};
  }
  if (instr->operand_count() == 1 || instr->opcode() == HloOpcode::kDivide ||
      instr->opcode() == HloOpcode::kDynamicSlice ||
      instr->opcode() == HloOpcode::kPad) {
    std::optional<InstrPath> subgraph =
        FindF8SubgraphRecursive(instr->mutable_operand(0), visited_instrs);
    if (subgraph) {
      subgraph->emplace_back(std::make_pair(instr, 0));
    }
    return subgraph;
  } else if (instr->opcode() == HloOpcode::kMultiply ||
             instr->opcode() == HloOpcode::kSelect) {
    for (int k = 0; k < 2; ++k) {
      int operand_idx = k + (instr->opcode() == HloOpcode::kSelect);
      std::optional<InstrPath> subgraph = FindF8SubgraphRecursive(
          instr->mutable_operand(operand_idx), visited_instrs);
      if (subgraph) {
        subgraph->emplace_back(std::make_pair(instr, operand_idx));
        return subgraph;
      }
    }
  }
  return std::nullopt;
}
struct MatchedFp8Param {
  HloInstruction *fp8_input = nullptr;
  HloInstruction *scale = nullptr;
  bool mult_scale = false;
  InstrPath commutative_ops;
};
std::optional<MatchedFp8Param> MatchFp8Param(HloInstruction *instr) {
  absl::flat_hash_set<int> visited_instrs;
  std::optional<InstrPath> maybe_subgraph =
      FindF8SubgraphRecursive(instr, visited_instrs);
  if (!maybe_subgraph) {
    return std::nullopt;
  }
  InstrPath &subgraph = maybe_subgraph.value();
  MatchedFp8Param param;
  if (subgraph.size() == 1) {
    CHECK(IsF8Type(subgraph[0].first));
    param.fp8_input = subgraph[0].first;
    return param;
  }
  int num_dequant_ops;
  if (subgraph.size() > 2 &&
      Match(subgraph[2].first,
            m::MultiplyAnyOrder(m::Convert(m::Op(&param.fp8_input)),
                                m::Broadcast(m::Op(&param.scale))))) {
    param.mult_scale = true;
    num_dequant_ops = 2;
  } else if (subgraph.size() > 2 &&
             Match(subgraph[2].first,
                   m::Divide(m::Convert(m::Op(&param.fp8_input)),
                             m::Broadcast(m::Op(&param.scale))))) {
    param.mult_scale = false;
    num_dequant_ops = 2;
  } else if (subgraph.size() > 1 &&
             Match(subgraph[1].first, m::Convert(m::Op(&param.fp8_input)))) {
    param.scale = nullptr;
    num_dequant_ops = 1;
  } else {
    VLOG(1) << "Possible intended FP8 GEMM operating on "
            << instr->ToShortString() << " not rewritten into FP8 Custom Call.";
    return std::nullopt;
  }
  auto preserves_element_type = [](const HloInstruction *instr) -> bool {
    return ShapeUtil::SameElementType(instr->shape(),
                                      instr->operand(0)->shape());
  };
  auto use_spmd_partitioning = [](const HloInstruction *instr) -> bool {
    return instr->GetModule()->config().use_spmd_partitioning();
  };
  int start = 1 + num_dequant_ops;
  for (int i = start; i < subgraph.size(); ++i) {
    if (!Match(
            subgraph[i].first,
            m::AnyOf<HloInstruction>(
                m::Bitcast().WithPredicate(preserves_element_type),
                m::Broadcast(), m::Copy(), m::DynamicSlice(), m::Pad(),
                m::Reshape(), m::Select(), m::Slice(), m::Transpose(),
                m::AllGather().WithPredicate(use_spmd_partitioning),
                m::AllToAll().WithPredicate(use_spmd_partitioning),
                m::CollectivePermute().WithPredicate(use_spmd_partitioning)))) {
      VLOG(1) << "Possible intended FP8 GEMM operating on "
              << instr->ToShortString()
              << " not rewritten into FP8 Custom Call.";
      return std::nullopt;
    }
    if (Match(subgraph[i].first, m::Select()) &&
        !Match(subgraph[i].first->operand(subgraph[i].second == 2 ? 1 : 2),
               m::Broadcast(m::ConstantScalar(0)))) {
      VLOG(1) << "Possible intended FP8 GEMM operating on "
              << instr->ToShortString()
              << " not rewritten into FP8 Custom Call. Select requires a zero "
                 "operand to be exchanged with dequantization.";
      return std::nullopt;
    }
  }
  param.commutative_ops = {subgraph.begin() + start, subgraph.end()};
  return param;
}
HloInstruction *TransposeMatrix(HloInstruction *instr, int64_t contracting_dim,
                                absl::Span<const int64_t> batch_dims) {
  auto input_shape = instr->shape();
  std::vector<int64_t> permutation(input_shape.dimensions_size(), -1);
  for (int64_t batch_dim : batch_dims) {
    permutation[batch_dim] = batch_dim;
  }
  int non_contracting_dim;
  for (int i = 0; i < input_shape.dimensions_size(); ++i) {
    if (permutation[i] == -1 && contracting_dim != i) {
      non_contracting_dim = i;
    }
  }
  if (Layout::Equal()(input_shape.layout(),
                      LayoutUtil::GetDefaultLayoutForShape(input_shape))) {
    permutation[non_contracting_dim] = contracting_dim;
    permutation[contracting_dim] = non_contracting_dim;
    Shape new_shape = ShapeUtil::PermuteDimensions(permutation, input_shape);
    *new_shape.mutable_layout() = input_shape.layout();
    return instr->AddInstruction(
        HloInstruction::CreateTranspose(new_shape, instr, permutation));
  }
  Shape normalized_input_shape =
      ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
          input_shape);
  auto a0 = MakeBitcastHlo(instr, normalized_input_shape);
  std::vector<int64_t> layout_permuation(
      input_shape.layout().minor_to_major().begin(),
      input_shape.layout().minor_to_major().end());
  absl::c_reverse(layout_permuation);
  auto inv_perm = InversePermutation(layout_permuation);
  int new_contracting_dim = inv_perm[contracting_dim];
  int new_non_contracting_dim = inv_perm[non_contracting_dim];
  absl::c_iota(permutation, 0);
  std::swap(permutation[new_contracting_dim],
            permutation[new_non_contracting_dim]);
  Shape transpose_shape =
      ShapeUtil::PermuteDimensions(permutation, a0->shape());
  *transpose_shape.mutable_layout() = a0->shape().layout();
  HloInstruction *normalized_transpose = instr->AddInstruction(
      HloInstruction::CreateTranspose(transpose_shape, a0, permutation));
  Shape final_shape = ShapeUtil::PermuteDimensions(inv_perm, transpose_shape);
  *final_shape.mutable_layout() = input_shape.layout();
  return MakeBitcastHlo(normalized_transpose, final_shape);
}
HloInstruction *MaybeConstantFoldBias(HloInstruction *bias) {
  constexpr int kMaxMaterializeBiasBytes = 8 * 1024 * 1024;
  auto is_nonscalar = [](const HloInstruction *instr) {
    return !ShapeUtil::IsEffectiveScalar(instr->shape());
  };
  auto broadcast_of_nonscalar =
      m::Broadcast(m::Constant().WithPredicate(is_nonscalar));
  if (ShapeUtil::ByteSizeOf(bias->shape()) <= kMaxMaterializeBiasBytes &&
      (Match(bias, broadcast_of_nonscalar) ||
       Match(bias, m::Reshape(broadcast_of_nonscalar)) ||
       Match(bias, m::Transpose(broadcast_of_nonscalar)) ||
       Match(bias, m::Bitcast(broadcast_of_nonscalar)))) {
    HloEvaluator evaluator(0);
    Literal result;
    if (evaluator.TryEvaluate(
            bias, &result,
            true)) {
      return bias->parent()->AddInstruction(
          HloInstruction::CreateConstant(std::move(result)));
    }
  }
  return bias;
}
auto Gemm(HloInstruction **instr) {
  return m::CustomCall(instr, {kGemmCallTarget});
}
auto CublasLtMatmul(HloInstruction **instr) {
  return m::CustomCall(instr, {kCublasLtMatmulCallTarget});
}
auto CublasLtMatmulF8(HloInstruction **instr) {
  return m::CustomCall(instr, {kCublasLtMatmulF8CallTarget});
}
auto CublasLtMatmulMaybeF8(HloInstruction **instr) {
  return m::CustomCall(
      instr, {kCublasLtMatmulCallTarget, kCublasLtMatmulF8CallTarget});
}
auto GemmOrCublasLtMatmul(HloInstruction **instr) {
  return m::CustomCall(instr, {kGemmCallTarget, kCublasLtMatmulCallTarget});
}
auto GemmOrCublasLtMatmulMaybeF8(HloInstruction **instr) {
  return m::CustomCall(instr, {kGemmCallTarget, kCublasLtMatmulCallTarget,
                               kCublasLtMatmulF8CallTarget});
}
auto BcastConstScalar(HloInstruction **instr, double value) {
  return m::Broadcast(instr, m::ConstantScalar(value));
}
auto BcastConstScalar(double value) { return BcastConstScalar(nullptr, value); }
auto BcastConstScalarNear(double value) {
  return m::Broadcast(m::ConstantScalar().WithPredicate(
      [expected = value](const HloInstruction *instr) {
        std::optional<double> actual =
            xla::Cast<const HloConstantInstruction>(instr)
                ->literal()
                .GetAsDouble({});
        if (!actual.has_value()) return false;
        double epsilon;
        switch (instr->shape().element_type()) {
          case F16:
            epsilon = 128 * std::numeric_limits<Eigen::half>::epsilon();
            break;
          case BF16:
            epsilon = 128 * std::numeric_limits<bfloat16>::epsilon();
            break;
          case F32:
            epsilon = 128 * std::numeric_limits<float>::epsilon();
            break;
          case F64:
            epsilon = 128 * std::numeric_limits<double>::epsilon();
            break;
          default:
            return false;
        }
        return abs(*actual - expected) < (abs(*actual + expected) * epsilon);
      }));
}
template <typename Pattern>
auto OptionalSlice(HloInstruction **optional_slice, Pattern pattern) {
  return m::AnyOf<HloInstruction>(m::Slice(optional_slice, pattern),
                                  std::move(pattern));
}
template <typename Pattern>
auto OptionalConvert(HloInstruction **optional_convert, Pattern pattern) {
  return m::AnyOf<HloInstruction>(m::Convert(optional_convert, pattern),
                                  std::move(pattern));
}
template <typename Pattern>
auto OptionalBitcast(HloInstruction **optional_bitcast, Pattern pattern) {
  return m::AnyOf<HloInstruction>(m::Bitcast(optional_bitcast, pattern),
                                  std::move(pattern));
}
class GemmRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit GemmRewriterVisitor(const se::GpuComputeCapability &gpu_version,
                               se::SemanticVersion toolkit_version,
                               const GemmRewriterOptions options)
      : gpu_version_(gpu_version),
        toolkit_version_(toolkit_version),
        options_(options) {}
  absl::Status HandleDot(HloInstruction *instr) override {
    if (!IsMatrixMultiplication(*instr) &&
        !IsMatrixVectorMultiplication(*instr)) {
      return absl::OkStatus();
    }
    if (Cast<HloDotInstruction>(instr)->sparse_operands()) {
      return absl::OkStatus();
    }
    int64_t gemm_rewrite_size_threshold =
        instr->GetModule()
            ->config()
            .debug_options()
            .xla_gpu_gemm_rewrite_size_threshold();
    TF_ASSIGN_OR_RETURN(bool is_matmul_tiny,
                        IsMatrixMultiplicationTooSmallForRewriting(
                            *instr, gemm_rewrite_size_threshold));
    if (is_matmul_tiny && IsDotSupportedByClassicalEmitters(*instr)) {
      return absl::OkStatus();
    }
    CHECK(!instr->IsRank2Transpose());
    if (instr->operand(0)->IsRank2Transpose() ||
        instr->operand(1)->IsRank2Transpose()) {
      return absl::OkStatus();
    }
    TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                        instr->backend_config<GpuBackendConfig>());
    GemmBackendConfig &gemm_backend_config =
        *gpu_backend_config.mutable_gemm_backend_config();
    gemm_backend_config.set_alpha_real(1.0);
    gemm_backend_config.set_alpha_imag(0.0);
    gemm_backend_config.set_beta(0.0);
    *gemm_backend_config.mutable_dot_dimension_numbers() =
        instr->dot_dimension_numbers();
    *gemm_backend_config.mutable_precision_config() = instr->precision_config();
    HloInstruction *lhs = instr->mutable_operand(0);
    HloInstruction *rhs = instr->mutable_operand(1);
    auto attributes = instr->frontend_attributes().map();
    gemm_backend_config.set_grad_x(attributes["grad_x"] == "true");
    gemm_backend_config.set_grad_y(attributes["grad_y"] == "true");
    int64_t lhs_batch_dims_size =
        instr->dot_dimension_numbers().lhs_batch_dimensions_size();
    bool is_lhs_vector =
        lhs->shape().dimensions_size() == lhs_batch_dims_size + 1;
    bool is_rhs_vector =
        rhs->shape().dimensions_size() == lhs_batch_dims_size + 1;
    int64_t lhs_stride =
        is_lhs_vector ? lhs->shape().dimensions(lhs_batch_dims_size)
                      : lhs->shape().dimensions(lhs_batch_dims_size) *
                            lhs->shape().dimensions(lhs_batch_dims_size + 1);
    int64_t rhs_stride =
        is_rhs_vector ? rhs->shape().dimensions(lhs_batch_dims_size)
                      : rhs->shape().dimensions(lhs_batch_dims_size) *
                            rhs->shape().dimensions(lhs_batch_dims_size + 1);
    gemm_backend_config.set_lhs_stride(lhs_stride);
    gemm_backend_config.set_rhs_stride(rhs_stride);
    switch (options_.dtype) {
      case GemmRewriterOptions::DType::kFp8Only: {
        TF_ASSIGN_OR_RETURN(
            bool supported_by_cublaslt,
            GemmIsSupportedByCublasLt(*instr, gemm_backend_config));
        std::optional<MatchedFp8Param> a, b;
        if (supported_by_cublaslt && instr->opcode() == HloOpcode::kDot &&
            (a = MatchFp8Param(
                 const_cast<HloInstruction *>(instr->operand(0)))) &&
            (b = MatchFp8Param(
                 const_cast<HloInstruction *>(instr->operand(1))))) {
          if (IsRocm(gpu_version_) &&
              toolkit_version_ < stream_executor::SemanticVersion{6, 2, 0} &&
              instr->shape().element_type() != F16 &&
              instr->shape().element_type() != F32) {
            TF_ASSIGN_OR_RETURN(
                instr, TurnF8DotWithUnsupportedOutputTypeIntoF32(instr));
          }
          TF_ASSIGN_OR_RETURN(bool created_call,
                              CreateF8CustomCall(instr, gpu_backend_config,
                                                 a.value(), b.value()));
          if (created_call) {
            return absl::OkStatus();
          }
        }
        if (IsF8Type(instr->operand(0))) {
          TF_ASSIGN_OR_RETURN(instr, TurnF8DotIntoF16Dot(instr));
        }
        break;
      }
      case GemmRewriterOptions::DType::kNonFp8Only: {
        TF_ASSIGN_OR_RETURN(
            absl::string_view gemm_custom_call_target,
            GetNonFp8GemmCustomCallTarget(*instr, gemm_backend_config));
        const Shape &output_shape = instr->shape();
        HloInstruction *gemm_call =
            instr->AddInstruction(HloInstruction::CreateCustomCall(
                output_shape,
                {instr->mutable_operand(0), instr->mutable_operand(1)},
                gemm_custom_call_target));
        TF_RETURN_IF_ERROR(gemm_call->set_backend_config(gpu_backend_config));
        TF_RETURN_IF_ERROR(ReplaceInstruction(instr, gemm_call));
      } break;
    };
    return absl::OkStatus();
  }
  absl::Status HandleMultiply(HloInstruction *instr) override {
    HloInstruction *alpha, *existing_gemm;
    if (Match(instr,
              m::MultiplyAnyOrder(
                  GemmOrCublasLtMatmulMaybeF8(&existing_gemm).WithOneUser(),
                  m::Broadcast(m::ConstantScalar(&alpha)).WithOneUser()))) {
      TF_ASSIGN_OR_RETURN(auto gpu_config,
                          existing_gemm->backend_config<GpuBackendConfig>());
      GemmBackendConfig &config = *gpu_config.mutable_gemm_backend_config();
      if (existing_gemm->shape().element_type() == S32) {
        return absl::OkStatus();
      }
      if (config.beta() == 0.0 && existing_gemm->user_count() == 1) {
        complex128 prev_alpha = {config.alpha_real(), config.alpha_imag()};
        complex128 new_alpha =
            *alpha->literal().GetAsComplex128({}) * prev_alpha;
        config.set_alpha_real(new_alpha.real());
        config.set_alpha_imag(new_alpha.imag());
        TF_RETURN_IF_ERROR(existing_gemm->set_backend_config(gpu_config));
        return ReplaceInstruction(instr, existing_gemm);
      }
    }
    HloInstruction *d_scale;
    if (Match(instr, m::MultiplyAnyOrder(
                         CublasLtMatmulF8(&existing_gemm).WithOneUser(),
                         m::Broadcast(m::Op(&d_scale)).WithOneUser()))) {
      return F8ScaleD(instr, existing_gemm, d_scale);
    }
    HloInstruction *cdf, *slice_or_bitcast = nullptr;
    if (Match(instr, m::MultiplyAnyOrder(
                         m::AnyOf<HloInstruction>(
                             m::Slice(&slice_or_bitcast,
                                      CublasLtMatmulMaybeF8(&existing_gemm)),
                             m::Bitcast(&slice_or_bitcast,
                                        CublasLtMatmulMaybeF8(&existing_gemm)),
                             CublasLtMatmulMaybeF8(&existing_gemm)),
                         m::Op(&cdf).WithOneUser())) &&
        Match(cdf,
              m::MultiplyAnyOrder(
                  BcastConstScalar(0.5),
                  m::AddAnyOrder(
                      BcastConstScalar(1.0),
                      m::Tanh(
                          m::MultiplyAnyOrder(
                              BcastConstScalarNear(sqrt(M_2_PI)),
                              m::AddAnyOrder(
                                  m::Op().Is(slice_or_bitcast ? slice_or_bitcast
                                                              : existing_gemm),
                                  m::MultiplyAnyOrder(
                                      BcastConstScalarNear(0.044715),
                                      m::MultiplyAnyOrder(
                                          m::Op().Is(slice_or_bitcast
                                                         ? slice_or_bitcast
                                                         : existing_gemm),
                                          m::MultiplyAnyOrder(
                                              m::Op().Is(slice_or_bitcast
                                                             ? slice_or_bitcast
                                                             : existing_gemm),
                                              m::Op().Is(slice_or_bitcast
                                                             ? slice_or_bitcast
                                                             : existing_gemm))
                                              .WithOneUser())
                                          .WithOneUser())
                                      .WithOneUser())
                                  .WithOneUser())
                              .WithOneUser())
                          .WithOneUser())))) {
      return FuseGeluActivation(instr, existing_gemm, slice_or_bitcast);
    }
    return absl::OkStatus();
  }
  absl::Status HandleDivide(HloInstruction *instr) override {
    HloInstruction *existing_gemm, *d_scale;
    if (Match(instr, m::Divide(CublasLtMatmulF8(&existing_gemm).WithOneUser(),
                               m::Broadcast(m::Op(&d_scale)).WithOneUser()))) {
      return F8ScaleD(instr, existing_gemm, d_scale);
    }
    return absl::OkStatus();
  }
  absl::Status HandleAdd(HloInstruction *instr) override {
    if (options_.bias_mode == GemmRewriterOptions::BiasMode::kNoBias) {
      return absl::OkStatus();
    }
    HloInstruction *bias, *existing_gemm = nullptr;
    HloInstruction *optional_slice = nullptr;
    HloInstruction *optional_convert = nullptr;
    HloInstruction *optional_bitcast = nullptr;
    if (Match(instr,
              m::AddAnyOrder(
                  OptionalBitcast(
                      &optional_bitcast,
                      OptionalSlice(
                          &optional_slice,
                          CublasLtMatmulMaybeF8(&existing_gemm).WithOneUser())
                          .WithOneUser())
                      .WithOneUser(),
                  m::Broadcast(&bias,
                               OptionalConvert(&optional_convert, m::Op()))))) {
      TF_ASSIGN_OR_RETURN(
          bool was_fused,
          FuseVectorBiasAdd(instr, bias, existing_gemm, optional_slice,
                            optional_convert, optional_bitcast));
      if (was_fused) {
        return absl::OkStatus();
      }
    }
    if (Match(
            instr,
            m::AddAnyOrder(
                m::Bitcast(CublasLtMatmulMaybeF8(&existing_gemm).WithOneUser())
                    .WithOneUser(),
                m::Broadcast(&bias, m::Op()).WithOneUser()))) {
      TF_ASSIGN_OR_RETURN(
          HloInstruction * new_add,
          MakeBinaryHlo(HloOpcode::kAdd, existing_gemm,
                        MakeBitcastHlo(bias, existing_gemm->shape())));
      TF_RETURN_IF_ERROR(
          ReplaceInstruction(instr, MakeBitcastHlo(new_add, instr->shape())));
      instr = new_add;
    }
    auto is_not_broadcast = [](const HloInstruction *instr) {
      return instr->opcode() != HloOpcode::kBroadcast;
    };
    if (Match(instr,
              m::AddAnyOrder(
                  m::Bitcast(
                      GemmOrCublasLtMatmulMaybeF8(&existing_gemm).WithOneUser())
                      .WithOneUser(),
                  m::Op(&bias).WithPredicate(is_not_broadcast)))) {
      HloInstruction *new_bitcast =
          MakeBitcastHlo(bias, existing_gemm->shape(), &bias->metadata());
      TF_ASSIGN_OR_RETURN(HloInstruction * new_add,
                          MakeBinaryHlo(HloOpcode::kAdd, existing_gemm,
                                        new_bitcast, &bias->metadata()));
      TF_RETURN_IF_ERROR(
          ReplaceInstruction(instr, MakeBitcastHlo(new_add, instr->shape())));
      instr = new_add;
    }
    if (Match(instr,
              m::AddAnyOrder(
                  m::AnyOf<HloInstruction>(
                      GemmOrCublasLtMatmul(&existing_gemm).WithOneUser(),
                      m::Convert(
                          GemmOrCublasLtMatmul(&existing_gemm).WithOneUser())
                          .WithOneUser()),
                  m::Op(&bias).WithPredicate(is_not_broadcast)))) {
      TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                          existing_gemm->backend_config<GpuBackendConfig>());
      const GemmBackendConfig &gemm_backend_config =
          gpu_backend_config.gemm_backend_config();
      TF_ASSIGN_OR_RETURN(
          bool types_are_supported,
          IsLegacyCublasMatmul(*existing_gemm)
              ? TypesAreSupportedByLegacyCublas(*existing_gemm,
                                                gemm_backend_config, instr)
              : TypesAreSupportedByCublasLt(*existing_gemm, gemm_backend_config,
                                            instr));
      bool has_no_consumer =
          instr->shape().element_type() ==
              existing_gemm->shape().element_type() ||
          instr->user_count() == 0 ||
          (instr->user_count() == 1 &&
           instr->users()[0]->opcode() == HloOpcode::kTuple &&
           instr->users()[0]->user_count() == 0);
      if (types_are_supported && has_no_consumer) {
        return FuseMatrixBiasAdd(instr, bias, existing_gemm);
      }
    }
    HloInstruction *optional_bitcast_matrix = nullptr;
    HloInstruction *optional_slice_matrix = nullptr;
    if (Match(instr,
              m::AddAnyOrder(
                  OptionalBitcast(
                      &optional_bitcast_matrix,
                      OptionalSlice(&optional_slice_matrix,
                                    GemmOrCublasLtMatmulMaybeF8(&existing_gemm)
                                        .WithOneUser()))
                      .WithOneUser(),
                  m::Op(&bias).WithPredicate(is_not_broadcast)))) {
      if (!IsF8Type(bias)) {
        return FuseMatrixBiasAdd(instr, bias, existing_gemm,
                                 optional_bitcast_matrix,
                                 optional_slice_matrix);
      }
    }
    return absl::OkStatus();
  }
  absl::Status HandleMaximum(HloInstruction *instr) override {
    HloInstruction *existing_gemm, *zeros;
    HloInstruction *optional_slice_or_bitcast = nullptr;
    if (Match(instr,
              m::MaximumAnyOrder(
                  m::AnyOf<HloInstruction>(
                      m::Slice(
                          &optional_slice_or_bitcast,
                          CublasLtMatmulMaybeF8(&existing_gemm).WithOneUser()),
                      m::Bitcast(
                          &optional_slice_or_bitcast,
                          CublasLtMatmulMaybeF8(&existing_gemm).WithOneUser()),
                      CublasLtMatmulMaybeF8(&existing_gemm))
                      .WithOneUser(),
                  m::Broadcast(&zeros, m::ConstantScalar(0))))) {
      TF_RETURN_IF_ERROR(FuseReluActivation(instr, zeros, existing_gemm,
                                            optional_slice_or_bitcast));
    }
    return absl::OkStatus();
  }
  absl::Status HandleConvert(HloInstruction *instr) override {
    HloInstruction *clamp_lower, *clamp_upper, *existing_gemm,
        *d_scale = nullptr, *binary = nullptr;
    if (Match(instr,
              m::Convert(
                  m::Clamp(
                      m::Broadcast(m::ConstantScalar(&clamp_lower)),
                      m::AnyOf<HloInstruction>(
                          CublasLtMatmulF8(&existing_gemm),
                          m::Divide(&binary, CublasLtMatmulF8(&existing_gemm),
                                    m::Broadcast(m::Op(&d_scale))),
                          m::MultiplyAnyOrder(&binary,
                                              CublasLtMatmulF8(&existing_gemm),
                                              m::Broadcast(m::Op(&d_scale)))),
                      m::Broadcast(m::ConstantScalar(&clamp_upper)))
                      .WithOneUser()))) {
      return F8ConvertD(
          instr, existing_gemm, d_scale, clamp_lower, clamp_upper,
          (binary && binary->opcode() == HloOpcode::kMultiply));
    }
    return absl::OkStatus();
  }
  static bool IsCuda(const se::GpuComputeCapability &gpu_version) {
    return std::holds_alternative<se::CudaComputeCapability>(gpu_version);
  }
  static absl::StatusOr<se::CudaComputeCapability> GetCudaComputeCapability(
      const se::GpuComputeCapability &gpu_version) {
    auto *cuda_cc = std::get_if<se::CudaComputeCapability>(&gpu_version);
    if (cuda_cc == nullptr) {
      return absl::InvalidArgumentError("Compute Capability is not CUDA.");
    }
    return *cuda_cc;
  }
  static bool IsRocm(const se::GpuComputeCapability &gpu_version) {
    return std::holds_alternative<se::RocmComputeCapability>(gpu_version);
  }
  static absl::StatusOr<se::RocmComputeCapability> GetRocmComputeCapability(
      const se::GpuComputeCapability &gpu_version) {
    auto rocm_cc = std::get_if<se::RocmComputeCapability>(&gpu_version);
    if (rocm_cc == nullptr) {
      return absl::InvalidArgumentError("Compute Capability is not ROCm.");
    }
    return *rocm_cc;
  }
  absl::StatusOr<bool> CreateF8CustomCall(HloInstruction *instr,
                                          GpuBackendConfig &gpu_backend_config,
                                          MatchedFp8Param a,
                                          MatchedFp8Param b) {
    GemmBackendConfig &gemm_backend_config =
        *gpu_backend_config.mutable_gemm_backend_config();
    if (IsCuda(gpu_version_)) {
      TF_ASSIGN_OR_RETURN(auto cuda_compute_capability,
                          GetCudaComputeCapability(gpu_version_));
      if (!cuda_compute_capability.IsAtLeast(8, 9)) {
        VLOG(1) << "FP8 Custom Calls require Ada, Hopper, or later "
                   "architectures. Got: "
                << cuda_compute_capability.ToString()
                << " and toolkit version: " << toolkit_version_;
        return false;
      }
      if (toolkit_version_ < stream_executor::SemanticVersion{12, 0, 0}) {
        VLOG(1) << "FP8 Custom Calls require CUDA 12.0 or newer.";
        return false;
      }
    }
    if (IsRocm(gpu_version_)) {
      TF_ASSIGN_OR_RETURN(auto rocm_compute_capability,
                          GetRocmComputeCapability(gpu_version_));
      if (!rocm_compute_capability.has_fp8_support()) {
        VLOG(1) << "FP8 Custom Calls require MI300, or later architectures.";
        return false;
      }
      if (toolkit_version_ < stream_executor::SemanticVersion{6, 0, 0}) {
        VLOG(1) << "FP8 Custom Calls require ROCm 6.0 or newer.";
        return false;
      }
    }
    PrimitiveType a_type = a.fp8_input->shape().element_type();
    PrimitiveType b_type = b.fp8_input->shape().element_type();
    if (IsCuda(gpu_version_)) {
      if (a_type == F8E5M2 && b_type == F8E5M2) {
        VLOG(1)
            << "Failed to rewrite " << instr->ToShortString()
            << " into FP8 Custom Call. The element type of one of the operands "
               "must be F8E4M3FN.";
        return false;
      }
      if ((a_type != F8E5M2 && a_type != F8E4M3FN) ||
          (b_type != F8E5M2 && b_type != F8E4M3FN)) {
        VLOG(1) << "Failed to rewrite " << instr->ToShortString()
                << " into FP8 Custom Call. The input types must be F8E5M2 or "
                   "F8E4M3FN, but got "
                << PrimitiveType_Name(a_type) << " and "
                << PrimitiveType_Name(b_type);
        return false;
      }
    }
    if (IsRocm(gpu_version_)) {
      if (a_type == F8E5M2FNUZ && b_type == F8E5M2FNUZ) {
        VLOG(1)
            << "Failed to rewrite " << instr->ToShortString()
            << " into FP8 Custom Call. The element type of one of the operands "
               "must be F8E4M3FNUZ.";
        return false;
      }
      if ((a_type != F8E5M2FNUZ && a_type != F8E4M3FNUZ) ||
          (b_type != F8E5M2FNUZ && b_type != F8E4M3FNUZ)) {
        VLOG(1)
            << "Failed to rewrite " << instr->ToShortString()
            << " into FP8 Custom Call. The input types must be F8E5M2FNUZ or "
               "F8E4M3FNUZ, but got "
            << PrimitiveType_Name(a_type) << " and "
            << PrimitiveType_Name(b_type);
        return false;
      }
    }
    absl::Span<const int64_t> a_batch_dims =
        gemm_backend_config.dot_dimension_numbers().lhs_batch_dimensions();
    absl::Span<const int64_t> b_batch_dims =
        gemm_backend_config.dot_dimension_numbers().rhs_batch_dimensions();
    const size_t num_batch_dims = a_batch_dims.size();
    std::array<bool, 2> mult_scale{a.mult_scale, b.mult_scale};
    std::array<HloInstruction *, 2> scales{a.scale, b.scale}, inv_scales,
        scales_f32;
    HloInstruction *one_constant = nullptr;
    auto one = [&one_constant, instr]() -> HloInstruction * {
      if (!one_constant) {
        one_constant = instr->AddInstruction(
            HloInstruction::CreateConstant(LiteralUtil::One(F32)));
      }
      return one_constant;
    };
    for (int i = 0; i < scales.size(); ++i) {
      if (scales[i]) {
        if (!ShapeUtil::IsScalar(scales[i]->shape())) {
          VLOG(1) << "Failed to rewrite " << instr->ToShortString()
                  << " into FP8 Custom Call. The scaling factors must be "
                     "scalars.";
          return false;
        }
        if (!mult_scale[i]) {
          inv_scales[i] = instr->AddInstruction(HloInstruction::CreateBinary(
              scales[i]->shape(), HloOpcode::kDivide, one(), scales[i]));
        }
        scales_f32[i] = mult_scale[i] ? scales[i] : inv_scales[i];
        if (scales_f32[i]->shape().element_type() != F32) {
          scales_f32[i] = instr->AddInstruction(HloInstruction::CreateConvert(
              ShapeUtil::MakeScalarShape(F32), scales_f32[i]));
        }
      } else {
        scales_f32[i] = one();
      }
    }
    PrimitiveType d_type = instr->shape().element_type();
    bool supported_d_type = (d_type == BF16 || d_type == F16 || d_type == F32);
    if (IsCuda(gpu_version_) && (d_type == F8E4M3FN || d_type == F8E5M2)) {
      supported_d_type = true;
    }
    if (IsRocm(gpu_version_) &&
        toolkit_version_ >= stream_executor::SemanticVersion{6, 2, 0} &&
        (d_type == F8E4M3FNUZ || d_type == F8E5M2FNUZ)) {
      supported_d_type = true;
    }
    if (!supported_d_type) {
      VLOG(1) << "Failed to rewrite " << instr->ToShortString()
              << " into FP8 Custom Call. Output element type must be "
              << (IsCuda(gpu_version_) ? "F8E4M3FN, F8E5M2, BF16, F16 or F32. "
                  : toolkit_version_ >=
                          stream_executor::SemanticVersion{6, 2, 0}
                      ? "F8E4M3FNUZ, F8E5M2FNUZ, BF16, F16 or F32. "
                      : "BF16, F16 or F32. ")
              << "Actual element type is " << PrimitiveType_Name(d_type);
      return false;
    }
    absl::Span<const int64_t> a_contracting_dims =
        gemm_backend_config.dot_dimension_numbers()
            .lhs_contracting_dimensions();
    absl::Span<const int64_t> b_contracting_dims =
        gemm_backend_config.dot_dimension_numbers()
            .rhs_contracting_dimensions();
    if (a_contracting_dims.size() != 1 || b_contracting_dims.size() != 1) {
      VLOG(1) << "Failed to rewrite " << instr->ToShortString()
              << " into FP8 Custom Call. A and B must have one contracting "
                 "dimension.";
      return false;
    }
    for (const MatchedFp8Param &param : {a, b}) {
      const HloInstruction *input = param.commutative_ops.empty()
                                        ? param.fp8_input
                                        : param.commutative_ops.back().first;
      if (input->shape().rank() != num_batch_dims + 2) {
        VLOG(1) << "Failed to rewrite " << instr->ToShortString()
                << "into FP8 Custom Call. Inputs must have exactly one "
                   "contracting and one non-contracting dimension.";
        return false;
      }
    }
    auto shift_ops = [&instr](HloInstruction *&x, InstrPath &x_ops) -> void {
      for (std::pair<HloInstruction *, int> op : x_ops) {
        std::vector<HloInstruction *> operands = {x};
        if (op.first->opcode() == HloOpcode::kDynamicSlice) {
          for (int i = 1; i < op.first->operand_count(); ++i) {
            operands.emplace_back(op.first->mutable_operand(i));
          }
        }
        if (op.first->opcode() == HloOpcode::kPad) {
          HloInstruction *convert =
              instr->AddInstruction(HloInstruction::CreateConvert(
                  ShapeUtil::ChangeElementType(op.first->operand(1)->shape(),
                                               x->shape().element_type()),
                  op.first->mutable_operand(1)));
          operands.emplace_back(convert);
        }
        if (op.first->opcode() == HloOpcode::kSelect) {
          operands.emplace(operands.begin(), op.first->mutable_operand(0));
          int operand_idx = op.second == 2 ? 1 : 2;
          HloInstruction *convert =
              instr->AddInstruction(HloInstruction::CreateConvert(
                  ShapeUtil::ChangeElementType(
                      op.first->operand(operand_idx)->shape(),
                      x->shape().element_type()),
                  op.first->mutable_operand(operand_idx)));
          operands.emplace(operands.begin() + operand_idx, convert);
        }
        x = instr->AddInstruction(op.first->CloneWithNewOperands(
            ShapeUtil::MakeShapeWithDenseLayout(
                x->shape().element_type(), op.first->shape().dimensions(),
                op.first->shape().layout().minor_to_major()),
            operands));
      }
      return;
    };
    shift_ops(a.fp8_input, a.commutative_ops);
    shift_ops(b.fp8_input, b.commutative_ops);
    TF_ASSIGN_OR_RETURN(GemmConfig gemm_config,
                        GemmConfig::For(instr, gemm_backend_config));
    DotDimensionNumbers *dim_nums =
        gemm_backend_config.mutable_dot_dimension_numbers();
    if (gemm_config.lhs_layout.order == MatrixLayout::Order::kColumnMajor) {
      CHECK(a_contracting_dims[0] == num_batch_dims ||
            a_contracting_dims[0] == num_batch_dims + 1);
      if (a_contracting_dims[0] == num_batch_dims) {
        dim_nums->set_lhs_contracting_dimensions(0, num_batch_dims + 1);
      } else {
        dim_nums->set_lhs_contracting_dimensions(0, num_batch_dims);
      }
      a.fp8_input =
          TransposeMatrix(a.fp8_input, a_contracting_dims[0], a_batch_dims);
    }
    if (gemm_config.rhs_layout.order == MatrixLayout::Order::kRowMajor) {
      CHECK(b_contracting_dims[0] == num_batch_dims ||
            b_contracting_dims[0] == num_batch_dims + 1);
      if (b_contracting_dims[0] == num_batch_dims) {
        dim_nums->set_rhs_contracting_dimensions(0, num_batch_dims + 1);
      } else {
        dim_nums->set_rhs_contracting_dimensions(0, num_batch_dims);
      }
      b.fp8_input =
          TransposeMatrix(b.fp8_input, b_contracting_dims[0], b_batch_dims);
    }
    a.fp8_input = PadOperandToMultipleOf16(a_batch_dims, a.fp8_input);
    b.fp8_input = PadOperandToMultipleOf16(b_batch_dims, b.fp8_input);
    std::vector<int64_t> out_batch_dims(num_batch_dims);
    std::iota(out_batch_dims.begin(), out_batch_dims.end(), 0);
    Shape new_output_shape =
        PadShapeToMultipleOf16(instr->shape(), out_batch_dims);
    std::vector<HloInstruction *> operands_list = {
        a.fp8_input, b.fp8_input, scales_f32[0], scales_f32[1]};
    HloInstruction *new_custom_call =
        instr->AddInstruction(HloInstruction::CreateCustomCall(
            ShapeUtil::MakeShapeWithDenseLayout(
                instr->shape().element_type(), new_output_shape.dimensions(),
                instr->shape().layout().minor_to_major()),
            operands_list, kCublasLtMatmulF8CallTarget));
    TF_RETURN_IF_ERROR(new_custom_call->set_backend_config(gpu_backend_config));
    TF_RETURN_IF_ERROR(SetName(instr->GetModule(), new_custom_call));
    HloInstruction *slice = nullptr;
    if (new_output_shape.dimensions() != instr->shape().dimensions()) {
      std::vector<int64_t> start_indices(instr->shape().rank(), 0);
      std::vector<int64_t> strides(instr->shape().rank(), 1);
      slice = instr->AddInstruction(HloInstruction::CreateSlice(
          instr->shape(), new_custom_call, start_indices,
          instr->shape().dimensions(), strides));
    }
    TF_RETURN_IF_ERROR(
        ReplaceInstruction(instr, slice ? slice : new_custom_call));
    VLOG(1) << instr->ToString() << " rewritten into FP8 Custom Call.";
    return true;
  }
  absl::Status F8ScaleD(HloInstruction *instr, HloInstruction *existing_gemm,
                        HloInstruction *d_scale) {
    if (!ShapeUtil::IsScalar(d_scale->shape())) {
      return absl::OkStatus();
    }
    if (!existing_gemm->operand(2)->IsConstant() ||
        existing_gemm->operand(2)->literal().GetAsDouble({}) != 1.) {
      return absl::OkStatus();
    }
    TF_ASSIGN_OR_RETURN(auto gpu_backend_config,
                        existing_gemm->backend_config<GpuBackendConfig>());
    const GemmBackendConfig &config = gpu_backend_config.gemm_backend_config();
    if ((config.epilogue() != GemmBackendConfig::DEFAULT &&
         config.epilogue() != GemmBackendConfig::RELU) ||
        config.beta() != 0.) {
      return absl::OkStatus();
    }
    TF_ASSIGN_OR_RETURN(
        d_scale,
        InvertAndConvertScalar(d_scale, instr->opcode() == HloOpcode::kDivide));
    TF_RETURN_IF_ERROR(existing_gemm->ReplaceOperandWith(2, d_scale));
    TF_RETURN_IF_ERROR(ReplaceInstruction(instr, existing_gemm));
    VLOG(1) << "Scaling of FP8 GEMM fused into Custom Call.";
    return absl::OkStatus();
  }
  absl::Status F8ConvertD(HloInstruction *instr, HloInstruction *existing_gemm,
                          HloInstruction *d_scale, HloInstruction *clamp_lower,
                          HloInstruction *clamp_upper,
                          bool mult_scale = false) {
    if (instr->shape().element_type() == F8E4M3FN) {
      if (!clamp_lower->literal().IsAllFloat(static_cast<float>(
              std::numeric_limits<tsl::float8_e4m3fn>::lowest())) ||
          !clamp_upper->literal().IsAllFloat(static_cast<float>(
              std::numeric_limits<tsl::float8_e4m3fn>::max()))) {
        return absl::OkStatus();
      }
    } else if (instr->shape().element_type() == F8E5M2) {
      if (!clamp_lower->literal().IsAllFloat(static_cast<float>(
              std::numeric_limits<tsl::float8_e5m2>::lowest())) ||
          !clamp_upper->literal().IsAllFloat(static_cast<float>(
              std::numeric_limits<tsl::float8_e5m2>::max()))) {
        return absl::OkStatus();
      }
    } else {
      return absl::OkStatus();
    }
    if (d_scale && !ShapeUtil::IsScalar(d_scale->shape())) {
      return absl::OkStatus();
    }
    const std::vector<HloInstruction *> gemm_users = existing_gemm->users();
    HloInstruction *reduce_damax = nullptr;
    if (gemm_users.size() == 2) {
      TF_ASSIGN_OR_RETURN(auto gpu_config,
                          existing_gemm->backend_config<GpuBackendConfig>());
      const GemmBackendConfig &config = gpu_config.gemm_backend_config();
      for (int i = 0; i < gemm_users.size(); ++i) {
        HloInstruction *maybe_reduce = nullptr;
        if (gemm_users[i]->opcode() == HloOpcode::kAbs) {
          if (gemm_users[i]->users().size() != 1) continue;
          maybe_reduce = gemm_users[i]->users()[0];
        } else {
          if (config.epilogue() != GemmBackendConfig::BIAS_RELU &&
              config.epilogue() != GemmBackendConfig::RELU)
            continue;
          maybe_reduce = gemm_users[i];
        }
        if (maybe_reduce->opcode() == HloOpcode::kReduce &&
            maybe_reduce->operands().size() == 2 &&
            maybe_reduce->operand(1)->opcode() == HloOpcode::kConstant &&
            ShapeUtil::IsScalar(maybe_reduce->operand(1)->shape())) {
          HloInstruction *reduce = maybe_reduce;
          HloComputation *reduce_comp = reduce->to_apply();
          HloInstruction *reduce_comp_root = reduce_comp->root_instruction();
          if (reduce->operand(1)->literal().GetAsDouble({}) <= 0. &&
              reduce_comp_root->opcode() == HloOpcode::kMaximum &&
              reduce_comp_root->operand(0)->opcode() == HloOpcode::kParameter &&
              reduce_comp_root->operand(1)->opcode() == HloOpcode::kParameter) {
            reduce_damax = reduce;
          }
        }
      }
      if (!reduce_damax) {
        return absl::OkStatus();
      }
    } else if (gemm_users.size() > 2) {
      return absl::OkStatus();
    }
    TF_ASSIGN_OR_RETURN(auto gpu_backend_config,
                        existing_gemm->backend_config<GpuBackendConfig>());
    const GemmBackendConfig &gemm_backend_config =
        gpu_backend_config.gemm_backend_config();
    if (gemm_backend_config.beta() != 0.0) {
      if (existing_gemm->operand(2)->shape().element_type() != BF16 &&
          existing_gemm->operand(2)->shape().element_type() != F16) {
        VLOG(1) << "The scaling and conversion of the result of "
                << existing_gemm->ToShortString()
                << " is not fused into the FP8 Custom Call because it "
                   "conflicts with the existing fusion of the addition of a "
                   "matrix bias with element type other than BF16 or F16.";
        return absl::OkStatus();
      } else {
        xla::Cast<HloCustomCallInstruction>(existing_gemm)
            ->set_output_to_operand_aliasing({});
      }
    }
    if (d_scale) {
      TF_ASSIGN_OR_RETURN(d_scale,
                          InvertAndConvertScalar(d_scale, !mult_scale));
    } else {
      d_scale = instr->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::One(F32)));
    }
    existing_gemm->AppendOperand(d_scale);
    if (reduce_damax) {
      return F8AddDAmax(instr, existing_gemm, reduce_damax);
    }
    std::unique_ptr<HloInstruction> new_gemm =
        existing_gemm->CloneWithNewShape(instr->shape());
    TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(instr, std::move(new_gemm)));
    VLOG(1) << "Conversion" << (reduce_damax ? " and amax calculation" : "")
            << " fused into FP8 GEMM.";
    return absl::OkStatus();
  }
  absl::Status F8AddDAmax(HloInstruction *instr, HloInstruction *existing_gemm,
                          HloInstruction *reduce_damax) {
    Shape damax_shape = ShapeUtil::MakeScalarShape(F32);
    Shape tuple_shape =
        ShapeUtil::MakeTupleShape({instr->shape(), damax_shape});
    HloInstruction *gemm_and_damax =
        instr->AddInstruction(existing_gemm->CloneWithNewShape(tuple_shape));
    TF_ASSIGN_OR_RETURN(auto gpu_config,
                        gemm_and_damax->backend_config<GpuBackendConfig>());
    GemmBackendConfig &config = *gpu_config.mutable_gemm_backend_config();
    config.set_damax_output(true);
    TF_RETURN_IF_ERROR(gemm_and_damax->set_backend_config(gpu_config));
    HloInstruction *d =
        instr->AddInstruction(HloInstruction::CreateGetTupleElement(
            instr->shape(), gemm_and_damax, 0));
    HloInstruction *damax = instr->AddInstruction(
        HloInstruction::CreateGetTupleElement(damax_shape, gemm_and_damax, 1));
    HloInstruction *damax_converted = instr->AddInstruction(
        HloInstruction::CreateConvert(reduce_damax->shape(), damax));
    TF_RETURN_IF_ERROR(ReplaceInstruction(reduce_damax, damax_converted));
    TF_RETURN_IF_ERROR(ReplaceInstruction(instr, d));
    return absl::OkStatus();
  }
  absl::Status FuseMatrixBiasAdd(HloInstruction *instr, HloInstruction *bias,
                                 const HloInstruction *gemm,
                                 HloInstruction *bitcast = nullptr,
                                 HloInstruction *slice = nullptr) {
    TF_RET_CHECK(Shape::Equal().IgnoreElementType()(bias->shape(),
                                                    bitcast ? bitcast->shape()
                                                    : slice ? slice->shape()
                                                            : gemm->shape()));
    if (gemm->shape().element_type() == S32) {
      return absl::OkStatus();
    }
    if (slice) {
      int slice_op_dim = slice->operand(0)->shape().rank();
      if (slice->slice_starts() != std::vector<int64_t>(slice_op_dim, 0) ||
          slice->slice_strides() != std::vector<int64_t>(slice_op_dim, 1)) {
        return absl::OkStatus();
      }
    }
    bool can_overwrite_bias = [bias]() {
      if (bias->user_count() > 1) {
        return false;
      }
      if (bias->opcode() != HloOpcode::kParameter) {
        return true;
      }
      if (!bias->parent()->IsEntryComputation()) {
        return false;
      }
      const auto &in_out_alias_config =
          bias->GetModule()->input_output_alias_config();
      return in_out_alias_config.ParameterHasAlias(bias->parameter_number(),
                                                   {});
    }();
    bool want_to_fuse_bias = IsCublasLtMatmulF8(*gemm) ||
                             IsCublasLtMatmul(*gemm) || can_overwrite_bias;
    auto gpu_config = gemm->backend_config<GpuBackendConfig>().value();
    GemmBackendConfig &config = *gpu_config.mutable_gemm_backend_config();
    bool supported_epilogue =
        ((config.epilogue() == GemmBackendConfig::DEFAULT) ||
         (config.epilogue() == GemmBackendConfig::BIAS));
    if ((config.beta() != 0) || !want_to_fuse_bias ||
        (gemm->user_count() != 1) || !supported_epilogue) {
      return absl::OkStatus();
    }
    config.set_beta(1.0);
    std::vector<HloInstruction *> operands(gemm->operands().begin(),
                                           gemm->operands().end());
    HloInstruction *maybe_constant_folded_bias = MaybeConstantFoldBias(bias);
    if (bitcast) {
      maybe_constant_folded_bias =
          instr->AddInstruction(HloInstruction::CreateBitcast(
              slice->shape(), maybe_constant_folded_bias));
    }
    maybe_constant_folded_bias =
        PadOperandToTargetShape(gemm->shape(), maybe_constant_folded_bias);
    operands.insert(operands.begin() + 2, maybe_constant_folded_bias);
    std::unique_ptr<HloInstruction> fused_op =
        gemm->CloneWithNewOperands(gemm->shape(), operands);
    fused_op->mutable_shape()->set_element_type(bias->shape().element_type());
    TF_RETURN_IF_ERROR(fused_op->set_backend_config(gpu_config));
    if (IsLegacyCublasMatmul(*fused_op) || can_overwrite_bias) {
      xla::Cast<HloCustomCallInstruction>(fused_op.get())
          ->set_output_to_operand_aliasing({{{}, {2, {}}}});
    }
    TF_RETURN_IF_ERROR(SetName(instr->GetModule(), fused_op.get()));
    if (slice) {
      fused_op = slice->CloneWithNewOperands(
          slice->shape(),
          {slice->parent()->AddInstruction(std::move(fused_op))});
    }
    if (bitcast) {
      fused_op = bitcast->CloneWithNewOperands(
          bitcast->shape(),
          {bitcast->parent()->AddInstruction(std::move(fused_op))});
    }
    return ReplaceWithNewInstruction(instr, std::move(fused_op));
  }
  absl::StatusOr<bool> FuseVectorBiasAdd(HloInstruction *instr,
                                         HloInstruction *broadcast,
                                         HloInstruction *gemm,
                                         HloInstruction *slice = nullptr,
                                         HloInstruction *convert = nullptr,
                                         HloInstruction *bitcast = nullptr) {
    if (!bitcast) {
      TF_RET_CHECK(ShapeUtil::Compatible(
          broadcast->shape(), (slice ? slice->shape() : gemm->shape())));
    }
    if (!SupportsEpilogueFusion(gemm->shape().element_type())) {
      return false;
    }
    HloInstruction *bias = broadcast->mutable_operand(0);
    TF_ASSIGN_OR_RETURN(auto gpu_config,
                        gemm->backend_config<GpuBackendConfig>());
    GemmBackendConfig &config = *gpu_config.mutable_gemm_backend_config();
    const DotDimensionNumbers &dot_dims = config.dot_dimension_numbers();
    size_t num_col_dims = gemm->operand(1)->shape().rank() -
                          dot_dims.rhs_batch_dimensions_size() -
                          dot_dims.rhs_contracting_dimensions_size();
    if ((gemm->user_count() != 1) ||
        (config.epilogue() != GemmBackendConfig::DEFAULT) ||
        (bias->shape().rank() != num_col_dims)) {
      return false;
    }
    absl::Span<const int64_t> broadcast_dims = broadcast->dimensions();
    for (size_t i = 0; i < num_col_dims; ++i) {
      int64_t dim =
          (bitcast ? bitcast : gemm)->shape().layout().minor_to_major(i);
      auto it = absl::c_find(broadcast_dims, dim);
      if (it == broadcast_dims.end()) {
        return false;
      }
      int64_t vector_dim = it - broadcast_dims.begin();
      if (bias->shape().layout().minor_to_major(i) != vector_dim) {
        return false;
      }
    }
    std::vector<HloInstruction *> operands(gemm->operands().begin(),
                                           gemm->operands().end());
    if (gemm->custom_call_target() == kCublasLtMatmulF8CallTarget &&
        config.beta() != 0.0) {
      return true;
    }
    if (gemm->custom_call_target() == kCublasLtMatmulF8CallTarget &&
        bias->shape().element_type() == F32) {
      if (convert == nullptr) {
        return false;
      }
      HloInstruction *bias_f16_or_bf16 = convert->mutable_operand(0);
      auto compatible_bias_type = [](const PrimitiveType bias_type,
                                     const PrimitiveType output_type) {
        if (bias_type == BF16) {
          return output_type == F8E4M3FN || output_type == F8E5M2 ||
                 output_type == F32 || output_type == BF16;
        } else if (bias_type == F16) {
          return output_type == F16 || output_type == F8E4M3FN ||
                 output_type == F8E5M2;
        }
        return false;
      };
      if (compatible_bias_type(bias_f16_or_bf16->shape().element_type(),
                               gemm->shape().element_type())) {
        bias = bias_f16_or_bf16;
      } else {
        VLOG(1) << "Epilogue fusion of FP32 vector bias into FP8 GEMM is "
                   "currently not supported. See the cublasLT support matrix.";
        return false;
      }
    }
    if (gemm->custom_call_target() == kCublasLtMatmulF8CallTarget && bitcast) {
      bias = PadOperandToMultipleOf16(
          config.dot_dimension_numbers().rhs_batch_dimensions(), bias);
    }
    operands.push_back(bias);
    config.set_epilogue(GemmBackendConfig::BIAS);
    std::unique_ptr<HloInstruction> result =
        gemm->CloneWithNewOperands(gemm->shape(), operands);
    TF_RETURN_IF_ERROR(result->set_backend_config(gpu_config));
    TF_RETURN_IF_ERROR(SetName(result->GetModule(), result.get()));
    if (slice) {
      result = slice->CloneWithNewOperands(
          slice->shape(), {slice->parent()->AddInstruction(std::move(result))});
    }
    if (bitcast) {
      result = bitcast->CloneWithNewOperands(
          bitcast->shape(),
          {bitcast->parent()->AddInstruction(std::move(result))});
    }
    TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(instr, std::move(result)));
    return true;
  }
  absl::Status FuseReluActivation(HloInstruction *instr,
                                  HloInstruction *broadcast,
                                  HloInstruction *gemm,
                                  HloInstruction *slice_or_bitcast = nullptr) {
    TF_RET_CHECK(ShapeUtil::Compatible(
        broadcast->shape(),
        (slice_or_bitcast ? slice_or_bitcast->shape() : gemm->shape())));
    if (!SupportsEpilogueFusion(gemm->shape().element_type())) {
      return absl::OkStatus();
    }
    if (gemm->user_count() != 1) {
      return absl::OkStatus();
    }
    TF_ASSIGN_OR_RETURN(auto gpu_config,
                        gemm->backend_config<GpuBackendConfig>());
    GemmBackendConfig &config = *gpu_config.mutable_gemm_backend_config();
    if (config.epilogue() == GemmBackendConfig::DEFAULT) {
      config.set_epilogue(GemmBackendConfig::RELU);
    } else if (config.epilogue() == GemmBackendConfig::BIAS) {
      config.set_epilogue(GemmBackendConfig::BIAS_RELU);
    } else {
      return absl::OkStatus();
    }
    std::unique_ptr<HloInstruction> result = gemm->Clone();
    TF_RETURN_IF_ERROR(result->set_backend_config(gpu_config));
    TF_RETURN_IF_ERROR(SetName(result->GetModule(), result.get()));
    if (slice_or_bitcast) {
      result = slice_or_bitcast->CloneWithNewOperands(
          slice_or_bitcast->shape(),
          {slice_or_bitcast->parent()->AddInstruction(std::move(result))});
    }
    return ReplaceWithNewInstruction(instr, std::move(result));
  }
  absl::Status FuseGeluActivation(HloInstruction *multiply,
                                  HloInstruction *gemm,
                                  HloInstruction *slice_or_bitcast = nullptr) {
    if (!SupportsEpilogueFusion(gemm->shape().element_type())) {
      return absl::OkStatus();
    }
    if (IsCuda(gpu_version_) &&
        toolkit_version_ < stream_executor::SemanticVersion{12, 4, 0} &&
        IsCublasLtMatmulF8(*gemm)) {
      return absl::OkStatus();
    }
    bool has_aux = gemm->user_count() > 4;
    TF_ASSIGN_OR_RETURN(auto gpu_config,
                        gemm->backend_config<GpuBackendConfig>());
    GemmBackendConfig &config = *gpu_config.mutable_gemm_backend_config();
    if (config.epilogue() == GemmBackendConfig::DEFAULT) {
      config.set_epilogue(has_aux ? GemmBackendConfig::GELU_AUX
                                  : GemmBackendConfig::GELU);
    } else if (config.epilogue() == GemmBackendConfig::BIAS) {
      config.set_epilogue(has_aux ? GemmBackendConfig::BIAS_GELU_AUX
                                  : GemmBackendConfig::BIAS_GELU);
    } else {
      return absl::OkStatus();
    }
    std::unique_ptr<HloInstruction> output = gemm->CloneWithNewShape(
        has_aux ? ShapeUtil::MakeTupleShape({gemm->shape(), gemm->shape()})
                : gemm->shape());
    TF_RETURN_IF_ERROR(output->set_backend_config(gpu_config));
    TF_RETURN_IF_ERROR(SetName(multiply->GetModule(), output.get()));
    if (slice_or_bitcast) {
      output = slice_or_bitcast->CloneWithNewOperands(
          slice_or_bitcast->shape(),
          {gemm->parent()->AddInstruction(std::move(output))});
    }
    if (has_aux) {
      HloInstruction *tuple_output =
          gemm->parent()->AddInstruction(std::move(output));
      TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
          gemm, HloInstruction::CreateGetTupleElement(tuple_output, 1)));
      output = HloInstruction::CreateGetTupleElement(tuple_output, 0);
    }
    return ReplaceWithNewInstruction(multiply, std::move(output));
  }
 private:
  se::GpuComputeCapability gpu_version_;
  stream_executor::SemanticVersion toolkit_version_;
  GemmRewriterOptions options_;
  absl::StatusOr<absl::string_view> GetNonFp8GemmCustomCallTarget(
      const HloInstruction &instr,
      const GemmBackendConfig &gemm_backend_config) const {
    if (!instr.GetModule()
             ->config()
             .debug_options()
             .xla_gpu_enable_cublaslt()) {
      return absl::string_view(kGemmCallTarget);
    }
    const HloInstruction *lhs = instr.operand(0);
    const HloInstruction *rhs = instr.operand(1);
    if (lhs->shape().element_type() == S8 ||
        rhs->shape().element_type() == S8) {
      return absl::string_view(kGemmCallTarget);
    }
    TF_ASSIGN_OR_RETURN(bool gemm_is_supported_by_cublas_lt,
                        GemmIsSupportedByCublasLt(instr, gemm_backend_config));
    if (gemm_is_supported_by_cublas_lt) {
      return absl::string_view(kCublasLtMatmulCallTarget);
    }
    return absl::string_view(kGemmCallTarget);
  }
  absl::StatusOr<bool> TypesAreSupportedByLegacyCublas(
      const HloInstruction &instr, const GemmBackendConfig &gemm_backend_config,
      const HloInstruction *bias = nullptr) const {
    const PrimitiveType a_dtype = instr.operand(0)->shape().element_type();
    const PrimitiveType b_dtype = instr.operand(1)->shape().element_type();
    const PrimitiveType output_type =
        bias ? bias->shape().element_type() : instr.shape().element_type();
    const std::array<PrimitiveType, 12> supported_type = {
        PrimitiveType::S8,  PrimitiveType::F16, PrimitiveType::BF16,
        PrimitiveType::F32, PrimitiveType::S32, PrimitiveType::F64,
        PrimitiveType::C64, PrimitiveType::C128};
    if (!absl::c_linear_search(supported_type, output_type)) return false;
    TF_ASSIGN_OR_RETURN(const se::blas::DataType output_dtype,
                        se::gpu::AsBlasDataType(output_type));
    TF_ASSIGN_OR_RETURN(
        const se::blas::ComputationType compute_type,
        se::gpu::GetBlasComputationType(
            instr.precision_config().algorithm(), a_dtype, output_type,
            stream_executor::blas::kDefaultComputePrecision));
    se::blas::DataType scale_type =
        se::gpu::GetScaleType(output_dtype, compute_type);
    using se::blas::ComputationType;
    using se::blas::DataType;
    const std::array<
        std::tuple<ComputationType, DataType ,
                   PrimitiveType , PrimitiveType ,
                   DataType >,
        32>
        supported_type_combinations = {{
            {ComputationType::kF16, DataType::kHalf, PrimitiveType::F16,
             PrimitiveType::F16, DataType::kHalf},
            {ComputationType::kI32, DataType::kInt32, PrimitiveType::S8,
             PrimitiveType::S8, DataType::kInt32},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::BF16,
             PrimitiveType::BF16, DataType::kBF16},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::F16,
             PrimitiveType::F16, DataType::kHalf},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::S8,
             PrimitiveType::S8, DataType::kFloat},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::BF16,
             PrimitiveType::BF16, DataType::kFloat},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::F16,
             PrimitiveType::F16, DataType::kFloat},
            {ComputationType::kF32, DataType::kFloat, PrimitiveType::F32,
             PrimitiveType::F32, DataType::kFloat},
            {ComputationType::kF32, DataType::kComplexFloat, PrimitiveType::C64,
             PrimitiveType::C64, DataType::kComplexFloat},
            {ComputationType::kF16AsF32, DataType::kFloat, PrimitiveType::F32,
             PrimitiveType::F32, DataType::kFloat},
            {ComputationType::kF16AsF32, DataType::kComplexFloat,
             PrimitiveType::C64, PrimitiveType::C64, DataType::kComplexFloat},
            {ComputationType::kBF16AsF32, DataType::kFloat, PrimitiveType::F32,
             PrimitiveType::F32, DataType::kFloat},
            {ComputationType::kBF16AsF32, DataType::kComplexFloat,
             PrimitiveType::C64, PrimitiveType::C64, DataType::kComplexFloat},
            {ComputationType::kTF32AsF32, DataType::kFloat, PrimitiveType::F32,
             PrimitiveType::F32, DataType::kFloat},
            {ComputationType::kTF32AsF32, DataType::kComplexFloat,
             PrimitiveType::C64, PrimitiveType::C64, DataType::kComplexFloat},
            {ComputationType::kF64, DataType::kDouble, PrimitiveType::F64,
             PrimitiveType::F64, DataType::kDouble},
            {ComputationType::kF64, DataType::kComplexDouble,
             PrimitiveType::C128, PrimitiveType::C128,
             DataType::kComplexDouble},
        }};
    return absl::c_linear_search(
        supported_type_combinations,
        std::make_tuple(compute_type, scale_type, a_dtype, b_dtype,
                        output_dtype));
  }
  absl::StatusOr<bool> TypesAreSupportedByCublasLt(
      const HloInstruction &instr, const GemmBackendConfig &backend_config,
      const HloInstruction *bias = nullptr) const {
    const PrimitiveType a_dtype = instr.operand(0)->shape().element_type();
    const PrimitiveType b_dtype = instr.operand(1)->shape().element_type();
    const PrimitiveType output_type =
        bias ? bias->shape().element_type() : instr.shape().element_type();
    const std::array<PrimitiveType, 12> supported_type = {
        PrimitiveType::F8E5M2FNUZ, PrimitiveType::F8E4M3FNUZ,
        PrimitiveType::F8E5M2,     PrimitiveType::F8E4M3FN,
        PrimitiveType::S8,         PrimitiveType::F16,
        PrimitiveType::BF16,       PrimitiveType::F32,
        PrimitiveType::S32,        PrimitiveType::F64,
        PrimitiveType::C64,        PrimitiveType::C128};
    if (!absl::c_linear_search(supported_type, output_type)) return false;
    TF_ASSIGN_OR_RETURN(const se::blas::DataType output_dtype,
                        se::gpu::AsBlasDataType(output_type));
    const int max_precision = *absl::c_max_element(
        backend_config.precision_config().operand_precision());
    const PrecisionConfig::Algorithm algorithm =
        backend_config.precision_config().algorithm();
    if (!algorithm_util::IsSupportedByCublasOrCublasLt(algorithm, gpu_version_))
      return false;
    TF_ASSIGN_OR_RETURN(
        const se::blas::ComputationType compute_type,
        se::gpu::GetBlasComputationType(
            algorithm, a_dtype, instr.shape().element_type(), max_precision));
    se::blas::DataType scale_type =
        se::gpu::GetScaleType(output_dtype, compute_type);
    using se::blas::ComputationType;
    using se::blas::DataType;
    using TypeCombinations = std::initializer_list<std::tuple<
        ComputationType, DataType , PrimitiveType ,
        PrimitiveType , DataType >>;
    const TypeCombinations supported_cublas_type_combinations = {
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
         PrimitiveType::F8E4M3FN, DataType::kBF16},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
         PrimitiveType::F8E4M3FN, DataType::kF8E4M3FN},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
         PrimitiveType::F8E4M3FN, DataType::kHalf},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
         PrimitiveType::F8E4M3FN, DataType::kFloat},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
         PrimitiveType::F8E5M2, DataType::kBF16},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
         PrimitiveType::F8E5M2, DataType::kF8E4M3FN},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
         PrimitiveType::F8E5M2, DataType::kF8E5M2},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
         PrimitiveType::F8E5M2, DataType::kHalf},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FN,
         PrimitiveType::F8E5M2, DataType::kFloat},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E5M2,
         PrimitiveType::F8E4M3FN, DataType::kBF16},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E5M2,
         PrimitiveType::F8E4M3FN, DataType::kF8E4M3FN},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E5M2,
         PrimitiveType::F8E4M3FN, DataType::kF8E5M2},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E5M2,
         PrimitiveType::F8E4M3FN, DataType::kHalf},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E5M2,
         PrimitiveType::F8E4M3FN, DataType::kFloat},
        {ComputationType::kF32, DataType::kComplexFloat, PrimitiveType::C64,
         PrimitiveType::C64, DataType::kComplexFloat},
        {ComputationType::kF16AsF32, DataType::kFloat, PrimitiveType::F32,
         PrimitiveType::F32, DataType::kFloat},
        {ComputationType::kF16AsF32, DataType::kComplexFloat,
         PrimitiveType::C64, PrimitiveType::C64, DataType::kComplexFloat},
        {ComputationType::kBF16AsF32, DataType::kFloat, PrimitiveType::F32,
         PrimitiveType::F32, DataType::kFloat},
        {ComputationType::kBF16AsF32, DataType::kComplexFloat,
         PrimitiveType::C64, PrimitiveType::C64, DataType::kComplexFloat},
        {ComputationType::kTF32AsF32, DataType::kFloat, PrimitiveType::F32,
         PrimitiveType::F32, DataType::kFloat},
        {ComputationType::kTF32AsF32, DataType::kComplexFloat,
         PrimitiveType::C64, PrimitiveType::C64, DataType::kComplexFloat},
        {ComputationType::kF64, DataType::kDouble, PrimitiveType::F64,
         PrimitiveType::F64, DataType::kDouble},
        {ComputationType::kF64, DataType::kComplexDouble, PrimitiveType::C128,
         PrimitiveType::C128, DataType::kComplexDouble},
    };
    if (IsCuda(gpu_version_) &&
        absl::c_linear_search(supported_cublas_type_combinations,
                              std::tuple{compute_type, scale_type, a_dtype,
                                         b_dtype, output_dtype})) {
      return true;
    }
    const TypeCombinations supported_hipblas_type_combinations = {
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FNUZ,
         PrimitiveType::F8E4M3FNUZ, DataType::kBF16},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FNUZ,
         PrimitiveType::F8E4M3FNUZ, DataType::kF8E4M3FNUZ},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FNUZ,
         PrimitiveType::F8E4M3FNUZ, DataType::kHalf},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FNUZ,
         PrimitiveType::F8E4M3FNUZ, DataType::kFloat},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FNUZ,
         PrimitiveType::F8E5M2FNUZ, DataType::kBF16},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FNUZ,
         PrimitiveType::F8E5M2FNUZ, DataType::kF8E4M3FNUZ},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FNUZ,
         PrimitiveType::F8E5M2FNUZ, DataType::kF8E5M2FNUZ},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FNUZ,
         PrimitiveType::F8E5M2FNUZ, DataType::kHalf},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E4M3FNUZ,
         PrimitiveType::F8E5M2FNUZ, DataType::kFloat},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E5M2FNUZ,
         PrimitiveType::F8E4M3FNUZ, DataType::kBF16},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E5M2FNUZ,
         PrimitiveType::F8E4M3FNUZ, DataType::kF8E4M3FNUZ},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E5M2FNUZ,
         PrimitiveType::F8E4M3FNUZ, DataType::kF8E5M2FNUZ},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E5M2FNUZ,
         PrimitiveType::F8E4M3FNUZ, DataType::kHalf},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F8E5M2FNUZ,
         PrimitiveType::F8E4M3FNUZ, DataType::kFloat},
    };
    if (IsRocm(gpu_version_) &&
        absl::c_linear_search(supported_hipblas_type_combinations,
                              std::tuple{compute_type, scale_type, a_dtype,
                                         b_dtype, output_dtype})) {
      return true;
    }
    const TypeCombinations supported_type_combinations = {
        {ComputationType::kF16, DataType::kHalf, PrimitiveType::F16,
         PrimitiveType::F16, DataType::kHalf},
        {ComputationType::kI32, DataType::kInt32, PrimitiveType::S8,
         PrimitiveType::S8, DataType::kInt32},
        {ComputationType::kI32, DataType::kFloat, PrimitiveType::S8,
         PrimitiveType::S8, DataType::kInt8},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::BF16,
         PrimitiveType::BF16, DataType::kBF16},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F16,
         PrimitiveType::F16, DataType::kHalf},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::S8,
         PrimitiveType::S8, DataType::kFloat},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::BF16,
         PrimitiveType::BF16, DataType::kFloat},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F16,
         PrimitiveType::F16, DataType::kFloat},
        {ComputationType::kF32, DataType::kFloat, PrimitiveType::F32,
         PrimitiveType::F32, DataType::kFloat},
    };
    return absl::c_linear_search(
        supported_type_combinations,
        std::make_tuple(compute_type, scale_type, a_dtype, b_dtype,
                        output_dtype));
  }
  absl::StatusOr<bool> GemmIsSupportedByCublasLt(
      const HloInstruction &instr,
      const GemmBackendConfig &gemm_backend_config) const {
    const HloInstruction *lhs = instr.operand(0);
    const Shape &output_shape = instr.shape();
    TF_ASSIGN_OR_RETURN(
        bool types_are_supported_by_cublas_lt,
        TypesAreSupportedByCublasLt(instr, gemm_backend_config));
    if (!types_are_supported_by_cublas_lt) {
      return false;
    }
    constexpr int64_t kMaxBatchCount = 65535;
    const auto &batch_dimensions =
        gemm_backend_config.dot_dimension_numbers().lhs_batch_dimensions();
    int batch_count = (batch_dimensions.empty() ? 0 : 1);
    for (auto batch_dimension : batch_dimensions) {
      batch_count *= lhs->shape().dimensions(batch_dimension);
    }
    if (batch_count > kMaxBatchCount) {
      return false;
    }
    if (auto isrocm = std::get_if<se::RocmComputeCapability>(&gpu_version_);
        isrocm) {
      if (!isrocm->has_hipblaslt()) {
        return false;
      }
    }
    constexpr int kMaxDimensionSize{4194240};
    if (output_shape.element_type() != C64) {
      return true;
    }
    if (std::holds_alternative<se::CudaComputeCapability>(gpu_version_)) {
      if (std::get<se::CudaComputeCapability>(gpu_version_).IsAtLeastAmpere()) {
        return true;
      }
    }
    TF_ASSIGN_OR_RETURN(GemmConfig gemm_config,
                        GemmConfig::For(&instr, gemm_backend_config));
    return gemm_config.rhs_layout.num_cols <= kMaxDimensionSize;
  }
  absl::StatusOr<HloInstruction *> TurnF8DotWithUnsupportedOutputTypeIntoF32(
      HloInstruction *instr) {
    Shape output_f32_shape = instr->shape();
    output_f32_shape.set_element_type(F32);
    HloInstruction *f32_dot =
        instr->AddInstruction(instr->CloneWithNewShape(output_f32_shape));
    HloInstruction *convert = instr->AddInstruction(
        HloInstruction::CreateConvert(instr->shape(), f32_dot));
    TF_RETURN_IF_ERROR(ReplaceInstruction(instr, convert));
    return f32_dot;
  }
  absl::StatusOr<HloInstruction *> TurnF8DotIntoF16Dot(HloInstruction *instr) {
    DCHECK(IsF8Type(instr->operand(0)));
    DCHECK(IsF8Type(instr->operand(1)));
    PrimitiveType conv_type =
        instr->shape().element_type() == BF16 ? BF16 : F16;
    for (int i = 0; i < 2; ++i) {
      Shape operand_f16_shape = instr->operand(i)->shape();
      operand_f16_shape.set_element_type(conv_type);
      HloInstruction *convert =
          instr->AddInstruction(HloInstruction::CreateConvert(
              operand_f16_shape, instr->mutable_operand(i)));
      TF_RETURN_IF_ERROR(instr->ReplaceOperandWith(i, convert));
    }
    if (IsF8Type(instr)) {
      Shape output_f16_shape = instr->shape();
      output_f16_shape.set_element_type(F16);
      HloInstruction *f16_dot =
          instr->AddInstruction(instr->CloneWithNewShape(output_f16_shape));
      HloInstruction *convert_to_f8 = instr->AddInstruction(
          HloInstruction::CreateConvert(instr->shape(), f16_dot));
      TF_RETURN_IF_ERROR(ReplaceInstruction(instr, convert_to_f8));
      return f16_dot;
    } else {
      return instr;
    }
  }
};
class GemmWorkspaceRewriteVisitor : public DfsHloRewriteVisitor {
 public:
  explicit GemmWorkspaceRewriteVisitor(
      const se::GpuComputeCapability &gpu_version)
      : gpu_version_(gpu_version) {}
  absl::Status HandleCustomCall(HloInstruction *instr) override {
    bool has_aux_output = false;
    if (instr->custom_call_target() == kCublasLtMatmulCallTarget ||
        instr->custom_call_target() == kCublasLtMatmulF8CallTarget) {
      TF_ASSIGN_OR_RETURN(const auto gpu_config,
                          instr->backend_config<xla::gpu::GpuBackendConfig>());
      const xla::gpu::GemmBackendConfig &config =
          gpu_config.gemm_backend_config();
      xla::gpu::GemmBackendConfig_Epilogue epilogue = config.epilogue();
      TF_ASSIGN_OR_RETURN(
          has_aux_output,
          xla::gpu::gpublas_lt::EpilogueHasAuxiliaryOutput(epilogue));
      if (!((instr->shape().IsTuple() &&
             instr->shape().tuple_shapes_size() ==
                 has_aux_output + config.damax_output() + 1) ||
            instr->shape().IsArray())) {
        return absl::OkStatus();
      }
    } else if (instr->custom_call_target() != kGemmCallTarget ||
               !instr->shape().IsArray()) {
      return absl::OkStatus();
    }
    auto *cuda_cc = std::get_if<se::CudaComputeCapability>(&gpu_version_);
    int64_t workspace = cuda_cc == nullptr ? GemmConfig::kDefaultWorkspace
                        : cuda_cc->IsAtLeastHopper()
                            ? GemmConfig::kHopperWorkspace
                            : GemmConfig::kDefaultWorkspace;
    if (instr->custom_call_target() == kGemmCallTarget) {
      int64_t operands_byte_size = 0;
      for (auto &operand : instr->operands()) {
        operands_byte_size += ShapeUtil::ByteSizeOf(operand->shape());
      }
      workspace = std::min(workspace, operands_byte_size);
    }
    std::vector<Shape> output_shapes = instr->shape().IsArray()
                                           ? std::vector<Shape>{instr->shape()}
                                           : instr->shape().tuple_shapes();
    output_shapes.emplace_back(ShapeUtil::MakeShape(S8, {workspace}));
    Shape output_shape = ShapeUtil::MakeTupleShape(output_shapes);
    HloInstruction *new_call = instr->AddInstruction(
        instr->CloneWithNewOperands(output_shape, instr->operands()));
    auto *custom_call = xla::Cast<HloCustomCallInstruction>(new_call);
    if (!custom_call->output_to_operand_aliasing().empty()) {
      custom_call->set_output_to_operand_aliasing({{{0}, {2, {}}}});
    }
    if (instr->shape().IsTuple()) {
      for (auto user : instr->users()) {
        auto user_get_tuple =
            dynamic_cast<HloGetTupleElementInstruction *>(user);
        TF_RET_CHECK(user_get_tuple);
        HloInstruction *get_output =
            instr->AddInstruction(HloInstruction::CreateGetTupleElement(
                new_call, user_get_tuple->tuple_index()));
        TF_RETURN_IF_ERROR(ReplaceInstruction(user_get_tuple, get_output));
      }
      return absl::OkStatus();
    } else {
      HloInstruction *get_output = instr->AddInstruction(
          HloInstruction::CreateGetTupleElement(new_call, 0));
      return ReplaceInstruction(instr, get_output);
    }
  }
 private:
  se::GpuComputeCapability gpu_version_;
};
absl::StatusOr<bool> RunOnComputation(HloComputation *computation,
                                      se::GpuComputeCapability gpu_version,
                                      se::SemanticVersion toolkit_version,
                                      GemmRewriterOptions options) {
  GemmRewriterVisitor visitor(gpu_version, toolkit_version, options);
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  GemmWorkspaceRewriteVisitor workspace_visitor(gpu_version);
  TF_RETURN_IF_ERROR(computation->Accept(&workspace_visitor));
  return visitor.changed();
}
}  
GemmRewriter::GemmRewriter(se::GpuComputeCapability gpu_version,
                           se::SemanticVersion toolkit_version,
                           GemmRewriterOptions options)
    : gpu_version_(gpu_version),
      toolkit_version_(toolkit_version),
      options_(options) {}
absl::StatusOr<bool> GemmRewriter::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  bool changed = false;
  for (HloComputation *computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result,
                        RunOnComputation(computation, gpu_version_,
                                         toolkit_version_, options_));
    changed |= result;
  }
  return changed;
}
}  
}  