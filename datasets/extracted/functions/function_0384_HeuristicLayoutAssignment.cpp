#include "xla/service/gpu/transforms/layout_assignment.h"
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/host_memory_offload_annotations.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/tsl/util/env_var.h"
#include "xla/util.h"
#include "xla/window_util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace gpu {
using se::dnn::DataLayout;
using se::dnn::FilterLayout;
static std::tuple<DataLayout, FilterLayout, DataLayout>
HeuristicLayoutAssignment(const HloInstruction* instr,
                          const se::GpuComputeCapability& gpu_version,
                          const se::dnn::VersionInfo& dnn_version) {
  constexpr auto kAllNCHW =
      std::make_tuple(DataLayout::kBatchDepthYX, FilterLayout::kOutputInputYX,
                      DataLayout::kBatchDepthYX);
  constexpr auto kAllNCHW_VECT_C =
      std::make_tuple(DataLayout::kBatchDepthYX4, FilterLayout::kOutputInputYX4,
                      DataLayout::kBatchDepthYX4);
  constexpr auto kAllNHWC =
      std::make_tuple(DataLayout::kBatchYXDepth, FilterLayout::kOutputYXInput,
                      DataLayout::kBatchYXDepth);
  const ConvolutionDimensionNumbers& dnums =
      instr->convolution_dimension_numbers();
  Shape input_shape = instr->operand(0)->shape();
  PrimitiveType input_ty = instr->operand(0)->shape().element_type();
  if (primitive_util::IsIntegralType(input_ty)) {
    if (input_ty == S8 && dnums.input_spatial_dimensions_size() == 2 &&
        input_shape.dimensions_size() == 5) {
      VLOG(2) << "Using NCHW_VECT_C for int8_t conv " << instr->ToString();
      return kAllNCHW_VECT_C;
    }
    VLOG(2) << "Using NHWC for int8_t conv " << instr->ToString();
    return kAllNHWC;
  }
  if (primitive_util::IsF8Type(input_ty)) {
    VLOG(2) << "Using NHWC for FP8 conv " << instr->ToString();
    return kAllNHWC;
  }
  const DebugOptions& debug_options =
      instr->GetModule()->config().debug_options();
  if (debug_options.xla_gpu_force_conv_nchw()) {
    VLOG(2) << "Overriding layout to NCHW for " << instr->ToString();
    return kAllNCHW;
  }
  if (debug_options.xla_gpu_force_conv_nhwc()) {
    VLOG(2) << "Overriding layout to NHWC for " << instr->ToString();
    return kAllNHWC;
  }
  const auto* rocm_compute_capability =
      std::get_if<se::RocmComputeCapability>(&gpu_version);
  if (rocm_compute_capability && input_ty == F16) return kAllNHWC;
  const bool isFloat16 = (input_ty == F16) || (input_ty == BF16);
  if (std::holds_alternative<se::CudaComputeCapability>(gpu_version)) {
    const auto* cuda_compute_capability =
        std::get_if<se::CudaComputeCapability>(&gpu_version);
    bool is_volta =
        cuda_compute_capability &&
        cuda_compute_capability->IsAtLeast(se::CudaComputeCapability::VOLTA);
    if (!isFloat16 || !is_volta ||
        instr->shape().tuple_shapes(0).dimensions_size() != 4) {
      return kAllNCHW;
    }
    if (std::make_tuple(dnn_version.major_version(),
                        dnn_version.minor_version()) <= std::make_tuple(7, 3) &&
        instr->custom_call_target() == kCudnnConvBackwardInputCallTarget &&
        window_util::HasStride(instr->window())) {
      return kAllNCHW;
    }
  } else if (std::holds_alternative<se::RocmComputeCapability>(gpu_version)) {
    bool is_enabled = false;
    TF_CHECK_OK(tsl::ReadBoolFromEnvVar("TF_USE_ROCM_NHWC",
                                        false, &is_enabled));
    auto rocm_compute_capability =
        std::get<se::RocmComputeCapability>(gpu_version);
    if (!isFloat16 || (!rocm_compute_capability.has_nhwc_layout_support()) ||
        instr->shape().tuple_shapes(0).dimensions_size() != 4 || !is_enabled) {
      return kAllNCHW;
    }
  }
  VLOG(2) << "Using heuristic to figure out layouts for " << instr->ToString();
  return kAllNHWC;
}
absl::Status GpuLayoutAssignment::AddBackendConstraintsToDnnConvCustomCall(
    HloCustomCallInstruction* instr, LayoutConstraints* constraints) {
  Shape lhs_shape = instr->operand(0)->shape();
  Shape rhs_shape = instr->operand(1)->shape();
  Shape result_shape = instr->shape().tuple_shapes(0);
  Shape* input_shape;
  Shape* filter_shape;
  Shape* output_shape;
  TF_ASSIGN_OR_RETURN(auto kind, GetCudnnConvKind(instr));
  switch (kind) {
    case CudnnConvKind::kForward:
    case CudnnConvKind::kForwardActivation:
    case CudnnConvKind::kForwardGraph:
      input_shape = &lhs_shape;
      filter_shape = &rhs_shape;
      output_shape = &result_shape;
      break;
    case CudnnConvKind::kBackwardInput:
      input_shape = &result_shape;
      filter_shape = &rhs_shape;
      output_shape = &lhs_shape;
      break;
    case CudnnConvKind::kBackwardFilter:
      input_shape = &lhs_shape;
      filter_shape = &result_shape;
      output_shape = &rhs_shape;
      break;
  }
  {
    DataLayout input;
    FilterLayout filter;
    DataLayout output;
    std::tie(input, filter, output) =
        HeuristicLayoutAssignment(instr, gpu_version_, dnn_version_);
    TF_ASSIGN_OR_RETURN(
        std::tie(*input_shape->mutable_layout(),
                 *filter_shape->mutable_layout(),
                 *output_shape->mutable_layout()),
        StreamExecutorConvLayoutsToXlaLayouts(
            instr->convolution_dimension_numbers(), input, filter, output));
  }
  TF_ASSIGN_OR_RETURN(
      const LogicalBuffer* call_result_buf,
      points_to_analysis_->GetBufferDefinedAt(instr, {0}));
  TF_RETURN_IF_ERROR(SetOperandLayout(lhs_shape, instr, 0));
  TF_RETURN_IF_ERROR(SetOperandLayout(rhs_shape, instr, 1));
  TF_RETURN_IF_ERROR(SetBufferLayout(result_shape.layout(), *call_result_buf));
  if (kind == CudnnConvKind::kForwardActivation &&
      instr->operand_count() == 4) {
    TF_RETURN_IF_ERROR(SetOperandLayout(*output_shape, instr, 3));
  }
  if (kind == CudnnConvKind::kForwardGraph) {
    for (int k = 2; k < instr->operand_count(); ++k) {
      if (!ShapeUtil::IsScalar(instr->operand(k)->shape())) {
        TF_RETURN_IF_ERROR(SetOperandLayout(*output_shape, instr, k));
      }
    }
  }
  if (instr->operand_count() > 2 && kind != CudnnConvKind::kForwardActivation &&
      kind != CudnnConvKind::kForwardGraph) {
    return Internal(
        "Invalid convolution. Conv has a side input, but kind is not fused "
        "conv forward or graph conv foward: %s",
        instr->ToString());
  }
  return absl::OkStatus();
}
namespace {
void SetFortranLayout(Shape* shape) {
  LayoutUtil::SetToDefaultLayout(shape);
  int n = shape->mutable_layout()->minor_to_major_size();
  CHECK_GE(n, 2);
  std::swap(shape->mutable_layout()->mutable_minor_to_major()->at(0),
            shape->mutable_layout()->mutable_minor_to_major()->at(1));
}
bool DotCanSupportShapeWithLayout(const HloInstruction* dot,
                                  const Shape& shape) {
  const DotDimensionNumbers& dot_dims = dot->dot_dimension_numbers();
  return MatrixLayout::For(shape, dot_dims.lhs_batch_dimensions().size(),
                           dot->operand(0)->shape().rank() -
                               dot_dims.lhs_contracting_dimensions().size() -
                               dot_dims.lhs_batch_dimensions().size(),
                           dot_dims.rhs_batch_dimensions().size(),
                           dot->operand(1)->shape().rank() -
                               dot_dims.rhs_contracting_dimensions().size() -
                               dot_dims.rhs_batch_dimensions().size())
      .ok();
}
}  
absl::Status GpuLayoutAssignment::AddBackendConstraints(
    LayoutConstraints* constraints) {
  auto post_order = constraints->computation()->MakeInstructionPostOrder();
  for (auto iterator = post_order.rbegin(); iterator != post_order.rend();
       ++iterator) {
    HloInstruction* instruction = *iterator;
    if (IsCustomCallToDnnConvolution(*instruction)) {
      TF_RETURN_IF_ERROR(AddBackendConstraintsToDnnConvCustomCall(
          Cast<HloCustomCallInstruction>(instruction), constraints));
    }
    CHECK(!IsCublasGemm(*instruction))
        << "Gemm rewriting should run after layout assignment";
    if (instruction->opcode() == HloOpcode::kDot) {
      const Shape& output_shape = instruction->shape();
      const Shape& lhs_shape = instruction->operand(0)->shape();
      const Shape& rhs_shape = instruction->operand(1)->shape();
      const DotDimensionNumbers& dot_dims =
          instruction->dot_dimension_numbers();
      absl::Span<const int64_t> lhs_batch_dims =
          dot_dims.lhs_batch_dimensions();
      absl::Span<const int64_t> lhs_contracting_dims =
          dot_dims.lhs_contracting_dimensions();
      TF_ASSIGN_OR_RETURN(std::vector<int64_t> lhs_non_contracting_dims,
                          GetNonContractingDims(lhs_shape, lhs_batch_dims,
                                                lhs_contracting_dims));
      absl::Span<const int64_t> rhs_batch_dims =
          dot_dims.rhs_batch_dimensions();
      absl::Span<const int64_t> rhs_contracting_dims =
          dot_dims.rhs_contracting_dimensions();
      TF_ASSIGN_OR_RETURN(std::vector<int64_t> rhs_non_contracting_dims,
                          GetNonContractingDims(rhs_shape, rhs_batch_dims,
                                                rhs_contracting_dims));
      const DebugOptions& debug_options =
          instruction->GetModule()->config().debug_options();
      bool is_bf16_to_bf16 =
          (output_shape.element_type() == PrimitiveType::BF16 &&
           lhs_shape.element_type() == PrimitiveType::BF16 &&
           rhs_shape.element_type() == PrimitiveType::BF16);
      bool is_s8_to_s32 = (output_shape.element_type() == PrimitiveType::S32 &&
                           lhs_shape.element_type() == PrimitiveType::S8 &&
                           rhs_shape.element_type() == PrimitiveType::S8 &&
                           output_shape.dimensions_size() == 2 &&
                           lhs_shape.dimensions_size() == 2 &&
                           rhs_shape.dimensions_size() == 2);
      bool is_fp8_to_fp8 =
          (lhs_shape.element_type() == PrimitiveType::F8E4M3FN &&
           rhs_shape.element_type() == PrimitiveType::F8E4M3FN);
      if (is_s8_to_s32 || is_fp8_to_fp8 ||
          (is_bf16_to_bf16 &&
           debug_options.xla_gpu_ensure_minor_dot_contraction_dims())) {
        TF_RETURN_IF_ERROR(SetOperandMajorToMinorLayout(
            instruction, 0,
            {lhs_batch_dims, lhs_non_contracting_dims, lhs_contracting_dims}));
        TF_RETURN_IF_ERROR(SetOperandMajorToMinorLayout(
            instruction, 1,
            {rhs_batch_dims, rhs_non_contracting_dims, rhs_contracting_dims}));
        TF_RETURN_IF_ERROR(SetDotLayout(instruction, constraints));
      } else {
        if (!lhs_batch_dims.empty() || lhs_contracting_dims.size() > 1 ||
            lhs_non_contracting_dims.size() > 1) {
          TF_RETURN_IF_ERROR(SetDotOperandLayout(instruction, 0, lhs_batch_dims,
                                                 lhs_contracting_dims,
                                                 lhs_non_contracting_dims));
        }
        if (!rhs_batch_dims.empty() || rhs_non_contracting_dims.size() > 1 ||
            rhs_contracting_dims.size() > 1) {
          TF_RETURN_IF_ERROR(SetDotOperandLayout(instruction, 1, rhs_batch_dims,
                                                 rhs_contracting_dims,
                                                 rhs_non_contracting_dims));
        }
        if (!lhs_batch_dims.empty() || lhs_non_contracting_dims.size() > 1 ||
            rhs_non_contracting_dims.size() > 1) {
          TF_RETURN_IF_ERROR(SetDotLayout(instruction, constraints));
        }
      }
    } else if (instruction->opcode() == HloOpcode::kTranspose) {
      const HloInstruction* operand = instruction->operand(0);
      if ((operand->opcode() != HloOpcode::kDot) ||
          (operand->user_count() > 1)) {
        continue;
      }
      Shape shape = operand->shape();
      *shape.mutable_layout() =
          LayoutUtil::MakeLayoutFromMajorToMinor(instruction->dimensions());
      if (DotCanSupportShapeWithLayout(operand, shape)) {
        TF_RETURN_IF_ERROR(
            SetOperandLayout(shape, instruction, 0));
      }
    } else if (instruction->opcode() == HloOpcode::kFft) {
      Shape op0_shape = instruction->operand(0)->shape();
      LayoutUtil::SetToDefaultLayout(&op0_shape);
      Shape output_shape = instruction->shape();
      LayoutUtil::SetToDefaultLayout(&output_shape);
      TF_RETURN_IF_ERROR(SetOperandLayout(op0_shape, instruction, 0));
      TF_RETURN_IF_ERROR(SetInstructionLayout(output_shape, instruction));
    } else if (instruction->opcode() == HloOpcode::kSort &&
               instruction->operand(0)->shape().rank() > 1) {
      Shape keys_shape = instruction->operand(0)->shape();
      Layout keys_layout =
          LayoutUtil::GetDefaultLayoutForRank(keys_shape.rank());
      for (int64_t i = 0; i < instruction->operand_count(); ++i) {
        Shape shape = instruction->operand(i)->shape();
        *shape.mutable_layout() = keys_layout;
        TF_RETURN_IF_ERROR(SetOperandLayout(shape, instruction, i));
        const LogicalBuffer* output_buffer;
        if (instruction->shape().IsArray()) {
          TF_ASSIGN_OR_RETURN(
              output_buffer,
              points_to_analysis_->GetBufferDefinedAt(instruction, {}));
        } else {
          TF_ASSIGN_OR_RETURN(
              output_buffer,
              points_to_analysis_->GetBufferDefinedAt(instruction, {i}));
        }
        TF_RETURN_IF_ERROR(SetBufferLayout(keys_layout, *output_buffer));
      }
    } else if (IsCustomCallToTopK(*instruction)) {
      Layout default_layout = LayoutUtil::GetDefaultLayoutForRank(
          instruction->operand(0)->shape().rank());
      TF_ASSIGN_OR_RETURN(
          auto values_buffer,
          points_to_analysis_->GetBufferDefinedAt(instruction, {0}));
      TF_RETURN_IF_ERROR(SetBufferLayout(default_layout, *values_buffer));
      TF_ASSIGN_OR_RETURN(
          auto indices_buffer,
          points_to_analysis_->GetBufferDefinedAt(instruction, {1}));
      TF_RETURN_IF_ERROR(SetBufferLayout(default_layout, *indices_buffer));
    } else if (instruction->opcode() == HloOpcode::kTriangularSolve) {
      Shape op0_shape = instruction->operand(0)->shape();
      Shape op1_shape = instruction->operand(1)->shape();
      Shape output_shape = instruction->shape();
      SetFortranLayout(&op0_shape);
      SetFortranLayout(&op1_shape);
      SetFortranLayout(&output_shape);
      TF_RETURN_IF_ERROR(SetOperandLayout(op0_shape, instruction, 0));
      TF_RETURN_IF_ERROR(SetOperandLayout(op1_shape, instruction, 1));
      TF_RETURN_IF_ERROR(SetInstructionLayout(output_shape, instruction));
    } else if (instruction->opcode() == HloOpcode::kReduceScatter) {
      auto ars = Cast<HloReduceScatterInstruction>(instruction);
      TF_RETURN_IF_ERROR(SetInstructionLayout(
          ShapeUtil::MoveDimToMajor(ars->shape(), ars->scatter_dimension()),
          ars));
    } else if (instruction->opcode() == HloOpcode::kAllGather) {
      auto ag = Cast<HloAllGatherInstruction>(instruction);
      TF_RETURN_IF_ERROR(SetInstructionLayout(
          ShapeUtil::MoveDimToMajor(ag->shape(), ag->all_gather_dimension()),
          ag));
    } else if (instruction->opcode() == HloOpcode::kAllToAll &&
               instruction->shape().IsArray()) {
      auto* all_to_all = Cast<HloAllToAllInstruction>(instruction);
      TF_RETURN_IF_ERROR(SetInstructionLayout(
          ShapeUtil::MoveDimToMajor(all_to_all->shape(),
                                    *all_to_all->split_dimension()),
          all_to_all));
    } else if (instruction->opcode() == HloOpcode::kSend) {
      Shape s = instruction->operand(0)->shape();
      LayoutUtil::SetToDefaultLayout(&s);
      TF_RETURN_IF_ERROR(SetInstructionLayout(s, instruction->operand(0)));
      TF_RETURN_IF_ERROR(
          SetArrayOperandLayout(s.layout(), instruction->operand(0), 0));
    } else if (instruction->opcode() == HloOpcode::kRecv) {
      Shape s = instruction->shape();
      ShapeUtil::ForEachMutableSubshape(
          &s, [&](Shape* subshape, const ShapeIndex& index) {
            LayoutUtil::SetToDefaultLayout(subshape);
          });
      TF_RETURN_IF_ERROR(SetInstructionLayout(s, instruction));
    }
  }
  return absl::OkStatus();
}
absl::Status GpuLayoutAssignment::SetDotOperandLayout(
    const HloInstruction* instruction, int64_t operand,
    absl::Span<const int64_t> batch_dims, absl::Span<const int64_t> row_dims,
    absl::Span<const int64_t> col_dims) {
  Shape shape = instruction->operand(operand)->shape();
  if (shape.has_layout() &&
      MatrixLayout::For(shape, batch_dims, row_dims, col_dims).ok())
    return SetOperandLayout(shape, instruction, operand);
  LayoutUtil::SetToDefaultLayout(&shape);
  if (MatrixLayout::For(shape, batch_dims, row_dims, col_dims).ok())
    return SetOperandLayout(shape, instruction, operand);
  return SetOperandMajorToMinorLayout(
      instruction, operand,
      {batch_dims, row_dims, col_dims});
}
absl::Status GpuLayoutAssignment::SetOperandMajorToMinorLayout(
    const HloInstruction* instruction, int64_t operand,
    std::initializer_list<absl::Span<const int64_t>> dim_groups) {
  size_t size = 0;
  for (auto group : dim_groups) size += group.size();
  std::vector<int64_t> major_to_minor;
  major_to_minor.reserve(size);
  for (const auto& group : dim_groups) {
    major_to_minor.insert(major_to_minor.end(), group.begin(), group.end());
  }
  Shape shape = instruction->operand(operand)->shape();
  *shape.mutable_layout() =
      LayoutUtil::MakeLayoutFromMajorToMinor(major_to_minor);
  return SetOperandLayout(shape, instruction, operand);
}
absl::Status GpuLayoutAssignment::SetDotLayout(
    const HloInstruction* instruction, LayoutConstraints* constraints) {
  for (const HloInstruction* user : instruction->users()) {
    for (int64_t i = 0; i < user->operand_count(); ++i) {
      if (user->operand(i) != instruction) {
        continue;
      }
      const ShapeLayout* constraint = constraints->OperandLayout(user, i);
      if ((constraint != nullptr) &&
          DotCanSupportShapeWithLayout(instruction, constraint->shape())) {
        return SetInstructionLayout(constraint->shape(), instruction);
      }
    }
  }
  return SetInstructionLayout(
      LayoutUtil::GetWithDefaultLayout(instruction->shape()), instruction);
}
bool GpuLayoutAssignment::PropagateReductionLayoutToOperand(
    const HloInstruction* user) {
  int64_t reduction_size = 1;
  for (int64_t reduction_dim : user->dimensions()) {
    reduction_size *= user->operand(0)->shape().dimensions(reduction_dim);
  }
  int64_t kept_dimension_size = ShapeUtil::ElementsIn(user->shape());
  return IsUnnestedReductionFasterThanElemental(
      {true, {1, kept_dimension_size, reduction_size}});
}
bool GpuLayoutAssignment::InstructionCanChangeLayoutInstance(
    const HloInstruction* instruction) {
  const HloCustomCallInstruction* custom_call =
      DynCast<HloCustomCallInstruction>(instruction);
  if (custom_call != nullptr &&
      (custom_call->custom_call_target() ==
           host_memory_offload_annotations::kMoveToHostCustomCallTarget ||
       custom_call->custom_call_target() ==
           host_memory_offload_annotations::kMoveToDeviceCustomCallTarget ||
       custom_call->custom_call_target() == kTopKCustomCallTarget)) {
    return false;
  }
  return LayoutAssignment::InstructionCanChangeLayoutInstance(instruction);
}
}  
}  