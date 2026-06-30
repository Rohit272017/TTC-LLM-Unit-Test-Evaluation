#include "xla/service/gpu/transforms/cudnn_vectorize_convolutions.h"
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/cudnn_support_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace gpu {
namespace {
static std::vector<HloCustomCallInstruction*> GetRelevantConvs(
    HloComputation* comp) {
  std::vector<HloCustomCallInstruction*> convs;
  for (HloInstruction* instr : comp->instructions()) {
    if (instr->opcode() != HloOpcode::kCustomCall ||
        (instr->custom_call_target() != kCudnnConvForwardCallTarget &&
         instr->custom_call_target() !=
             kCudnnConvBiasActivationForwardCallTarget) ||
        instr->operand_count() < 2) {
      continue;
    }
    PrimitiveType input_ty = instr->operand(0)->shape().element_type();
    PrimitiveType output_ty = instr->shape().tuple_shapes(0).element_type();
    if (input_ty == output_ty && (input_ty == S8 || input_ty == U8)) {
      convs.push_back(Cast<HloCustomCallInstruction>(instr));
    }
  }
  return convs;
}
static absl::StatusOr<HloComputation*> BuilderToHloComputation(
    XlaBuilder& b, XlaOp root, HloComputation* sibling_computation) {
  TF_ASSIGN_OR_RETURN(XlaComputation comp, b.Build(root));
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape, comp.GetProgramShape());
  HloModuleConfig config(program_shape);
  TF_ASSIGN_OR_RETURN(auto new_module,
                      HloModule::CreateFromProto(comp.proto(), config));
  HloModule* dest_module = sibling_computation->parent();
  HloCloneContext context(dest_module);
  return dest_module->DeepCloneComputation(new_module->entry_computation(),
                                           &context);
}
static XlaOp SplitAtDim(XlaOp instr, int64_t dim, int64_t vect_size) {
  XlaBuilder& b = *instr.builder();
  Shape shape = b.GetShape(instr).value();
  DimensionVector new_dims(shape.dimensions().begin(),
                           shape.dimensions().end());
  CHECK_EQ(new_dims[dim] % vect_size, 0);
  new_dims[dim] /= vect_size;
  new_dims.insert(new_dims.begin() + dim + 1, vect_size);
  return Reshape(instr, new_dims);
}
static Shape SplitShapeAtDim(Shape shape, int64_t dim, int64_t vect_size) {
  DimensionVector new_dims(shape.dimensions().begin(),
                           shape.dimensions().end());
  CHECK_EQ(new_dims[dim] % vect_size, 0);
  new_dims[dim] /= vect_size;
  new_dims.insert(new_dims.begin() + dim + 1, vect_size);
  return ShapeUtil::MakeShape(shape.element_type(), new_dims);
}
static XlaOp MoveDim(XlaOp instr, int64_t src, int64_t dst) {
  XlaBuilder& b = *instr.builder();
  int64_t rank = b.GetShape(instr)->dimensions_size();
  DimensionVector idxs(rank);
  absl::c_iota(idxs, 0);
  if (src < dst) {
    idxs.insert(idxs.begin() + dst, src);
    idxs.erase(idxs.begin() + src);
  } else {
    idxs.erase(idxs.begin() + src);
    idxs.insert(idxs.begin() + dst, src);
  }
  return Transpose(instr, idxs);
}
static XlaOp RevectorizeInstr(XlaOp instr, int64_t dim, int64_t vect_dim,
                              int64_t vect_size) {
  XlaBuilder& b = *instr.builder();
  Shape shape = b.GetShape(instr).value();
  auto size = [&](int64_t d) { return shape.dimensions(d); };
  CHECK_LE(size(vect_dim), vect_size);
  CHECK_EQ(vect_size % size(vect_dim), 0);
  int64_t split_factor = vect_size / size(vect_dim);
  CHECK_EQ(size(dim) % split_factor, 0);
  instr = SplitAtDim(instr, dim, split_factor);
  if (vect_dim > dim) {
    vect_dim++;
  }
  instr = MoveDim(instr, dim + 1, vect_dim);
  if (vect_dim > dim) {
    vect_dim--;
  }
  return Collapse(instr, {vect_dim, vect_dim + 1});
}
static XlaOp UnrevectorizeInstr(XlaOp instr, int64_t dim, int64_t vect_dim,
                                int64_t orig_vect_size) {
  XlaBuilder& b = *instr.builder();
  Shape shape = b.GetShape(instr).value();
  auto size = [&](int64_t d) { return shape.dimensions(d); };
  CHECK_GE(size(vect_dim), orig_vect_size);
  CHECK_EQ(size(vect_dim) % orig_vect_size, 0);
  instr = SplitAtDim(instr, vect_dim, orig_vect_size);
  if (dim > vect_dim) {
    dim++;
  }
  instr = MoveDim(instr, vect_dim, dim + 1);
  if (dim > vect_dim) {
    dim--;
  }
  return Collapse(instr, {dim, dim + 1});
}
static ConvolutionDimensionNumbers VectorizeDnums(
    ConvolutionDimensionNumbers dnums, bool reordered_filter) {
  int64_t input_vect_dim = dnums.input_feature_dimension();
  if (dnums.input_batch_dimension() > input_vect_dim) {
    dnums.set_input_batch_dimension(dnums.input_batch_dimension() + 1);
  }
  for (int64_t& d : *dnums.mutable_input_spatial_dimensions()) {
    if (d > input_vect_dim) {
      ++d;
    }
  }
  if (!reordered_filter) {
    int64_t kernel_vect_dim = dnums.kernel_input_feature_dimension();
    if (dnums.kernel_output_feature_dimension() > kernel_vect_dim) {
      dnums.set_kernel_output_feature_dimension(
          dnums.kernel_output_feature_dimension() + 1);
    }
    for (int64_t& d : *dnums.mutable_kernel_spatial_dimensions()) {
      if (d > kernel_vect_dim) {
        ++d;
      }
    }
  }
  int64_t output_vect_dim = dnums.output_feature_dimension();
  if (dnums.output_batch_dimension() > output_vect_dim) {
    dnums.set_output_batch_dimension(dnums.output_batch_dimension() + 1);
  }
  for (int64_t& d : *dnums.mutable_output_spatial_dimensions()) {
    if (d > output_vect_dim) {
      ++d;
    }
  }
  return dnums;
}
absl::Status ReorderInt8NchwVect(HloCustomCallInstruction* conv,
                                 XlaOp* operands) {
  bool has_bias = conv->operand_count() > 2;
  VLOG(1) << "Reordering filter" << (has_bias ? " and bias" : "")
          << " (replacement for cudnnReorderFilterAndBias)";
  auto builder = operands->builder();
  ConvolutionDimensionNumbers dnums = conv->convolution_dimension_numbers();
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                      conv->backend_config<GpuBackendConfig>());
  CudnnConvBackendConfig& config =
      *gpu_config.mutable_cudnn_conv_backend_config();
  config.set_reordered_int8_nchw_vect(true);
  TF_RETURN_IF_ERROR(conv->set_backend_config(gpu_config));
  TF_ASSIGN_OR_RETURN(Shape filter_shape, builder->GetShape(operands[1]));
  TF_ASSIGN_OR_RETURN(auto reorder, CudnnInferTransposeForFilterReordering(
                                        filter_shape, dnums));
  XlaOp reshape = Reshape(reorder.transpose_shape, operands[1]);
  XlaOp transpose = Transpose(reshape, reorder.permutation);
  operands[1] = Reshape(reorder.result_shape, transpose);
  dnums.set_kernel_output_feature_dimension(0);
  dnums.set_kernel_input_feature_dimension(1);
  dnums.set_kernel_spatial_dimensions(0, 2);
  dnums.set_kernel_spatial_dimensions(1, 3);
  conv->set_convolution_dimension_numbers(dnums);
  if (has_bias) {
    TF_ASSIGN_OR_RETURN(Shape bias_shape, builder->GetShape(operands[2]));
    TF_ASSIGN_OR_RETURN(reorder,
                        CudnnInferTransposeForBiasReordering(bias_shape));
    reshape = Reshape(reorder.transpose_shape, operands[2]);
    transpose = Transpose(reshape, reorder.permutation);
    operands[2] = Reshape(reorder.result_shape, transpose);
  }
  return absl::OkStatus();
}
static absl::StatusOr<bool> TryRevectorizeConv(
    const se::CudaComputeCapability& compute_capability,
    const se::dnn::VersionInfo& cudnn_version, HloCustomCallInstruction* conv,
    int vect_size) {
  const Shape& input_shape = conv->operand(0)->shape();
  const Shape& kernel_shape = conv->operand(1)->shape();
  const Shape& output_shape = conv->shape().tuple_shapes(0);
  const ConvolutionDimensionNumbers* dnums =
      &conv->convolution_dimension_numbers();
  std::optional<int64_t> input_vect_dim;
  std::optional<int64_t> kernel_vect_dim;
  std::optional<int64_t> output_vect_dim;
  std::tie(input_vect_dim, kernel_vect_dim, output_vect_dim) =
      FindVectorizedFeatureDims(*dnums, input_shape, kernel_shape,
                                output_shape);
  if (!input_vect_dim.has_value() || !kernel_vect_dim.has_value() ||
      !output_vect_dim.has_value()) {
    return false;
  }
  int64_t input_feat_size =
      input_shape.dimensions(dnums->input_feature_dimension());
  int64_t output_feat_size =
      output_shape.dimensions(dnums->output_feature_dimension());
  int64_t input_vect_size = input_shape.dimensions(*input_vect_dim);
  int64_t output_vect_size = output_shape.dimensions(*output_vect_dim);
  if (vect_size % input_vect_size != 0 || vect_size % output_vect_size != 0 ||
      input_feat_size % (vect_size / input_vect_size) != 0 ||
      output_feat_size % (vect_size / output_vect_size) != 0) {
    return false;
  }
  if (primitive_util::IsIntegralType(input_shape.element_type())) {
    TF_ASSIGN_OR_RETURN(bool supported_target_vectorization,
                        CudnnSupportsOptimizedIntegerConvolution(
                            compute_capability, *conv, vect_size));
    if (!supported_target_vectorization) {
      VLOG(3) << "Skipping re-vectorization of conv to vector size: "
              << vect_size << ": " << conv->ToString();
      return false;
    }
  }
  VLOG(1) << "Re-vectorizing conv channels from "
          << input_shape.dimensions(*input_vect_dim) << " to " << vect_size
          << ": " << conv->ToString();
  XlaBuilder b(absl::StrCat(conv->name(), ".revectorized"));
  b.SetOpMetadata(conv->metadata());
  XlaOp filter = Parameter(&b, 1, conv->operand(1)->shape(), "filter");
  absl::InlinedVector<XlaOp, 4> new_operands = {
      RevectorizeInstr(Parameter(&b, 0, conv->operand(0)->shape(), "input"),
                       dnums->input_feature_dimension(), *input_vect_dim,
                       vect_size),
      RevectorizeInstr(filter, dnums->kernel_input_feature_dimension(),
                       *kernel_vect_dim, vect_size),
  };
  if (conv->operand_count() > 2) {
    new_operands.push_back(Parameter(&b, 2, conv->operand(2)->shape(), "bias"));
  }
  if (conv->operand_count() > 3) {
    new_operands.push_back(RevectorizeInstr(
        Parameter(&b, 3, conv->operand(3)->shape(), "side_input"),
        dnums->input_feature_dimension(), *input_vect_dim, vect_size));
  }
  if (conv->operand_count() > 4) {
    return InvalidArgument(
        "Don't understand a conv with more than 4 arguments: %s",
        conv->ToString());
  }
  const auto& debug_options = conv->GetModule()->config().debug_options();
  bool use_reordering =
      input_shape.element_type() == xla::S8 && vect_size == 32 &&
      debug_options.xla_gpu_enable_cudnn_int8x32_convolution_reordering() &&
      cudnn_version >= se::dnn::VersionInfo{8, 3, 0};
  if (use_reordering) {
    int64_t kernel_vect_size = kernel_shape.dimensions(*kernel_vect_dim);
    if (kernel_vect_size == 4 || kernel_vect_size == 32) {
      new_operands[1] = filter;
    }
    TF_RETURN_IF_ERROR(ReorderInt8NchwVect(conv, new_operands.data()));
    dnums = &conv->convolution_dimension_numbers();
  }
  DimensionVector new_output_dims(output_shape.dimensions().begin(),
                                  output_shape.dimensions().end());
  new_output_dims[dnums->output_feature_dimension()] /=
      (vect_size / output_vect_size);
  new_output_dims[*output_vect_dim] = vect_size;
  XlaOp new_conv = CustomCallWithConvDnums(
      &b, conv->custom_call_target(), new_operands,
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeShape(output_shape.element_type(), new_output_dims),
           ShapeUtil::MakeShape(U8, {0})}),
      {},
      conv->raw_backend_config_string(), false,
      {}, nullptr,
      conv->window(),
      *dnums);
  XlaOp new_conv_result = GetTupleElement(new_conv, 0);
  XlaOp new_conv_scratch = GetTupleElement(new_conv, 1);
  XlaOp new_conv_result_unrevectorized = UnrevectorizeInstr(
      new_conv_result, dnums->output_feature_dimension(), *output_vect_dim,
      output_shape.dimensions(*output_vect_dim));
  TF_ASSIGN_OR_RETURN(
      HloComputation * new_conv_comp,
      BuilderToHloComputation(
          b, Tuple(&b, {new_conv_result_unrevectorized, new_conv_scratch}),
          conv->parent()));
  auto new_conv_comp_instrs = new_conv_comp->instructions();
  auto new_conv_it =
      absl::c_find_if(new_conv_comp_instrs, [](HloInstruction* instr) {
        return instr->opcode() == HloOpcode::kCustomCall;
      });
  if (new_conv_it != new_conv_comp_instrs.end()) {
    new_conv_comp->parent()->SetAndUniquifyInstrName(*new_conv_it,
                                                     conv->name());
  }
  VLOG(1) << "Re-vectorized conv to " << new_conv_comp->ToString();
  TF_RETURN_IF_ERROR(conv->parent()->ReplaceWithNewInstruction(
      conv, HloInstruction::CreateCall(conv->shape(), conv->operands(),
                                       new_conv_comp)));
  return true;
}
static absl::StatusOr<bool> TryVectorizeConv(
    const se::CudaComputeCapability& compute_capability,
    const se::dnn::VersionInfo& cudnn_version, HloCustomCallInstruction* conv,
    int64_t vect_size) {
  const Shape& input_shape = conv->operand(0)->shape();
  const Shape& output_shape = conv->shape().tuple_shapes(0);
  const ConvolutionDimensionNumbers* dnums =
      &conv->convolution_dimension_numbers();
  int64_t in_channels =
      input_shape.dimensions(dnums->input_feature_dimension());
  int64_t out_channels =
      output_shape.dimensions(dnums->output_feature_dimension());
  if (in_channels % vect_size != 0 || out_channels % vect_size != 0) {
    return false;
  }
  if (input_shape.dimensions_size() >
      2 + dnums->input_spatial_dimensions_size()) {
    return false;
  }
  if (primitive_util::IsIntegralType(input_shape.element_type())) {
    TF_ASSIGN_OR_RETURN(bool supported_target_vectorization,
                        CudnnSupportsOptimizedIntegerConvolution(
                            compute_capability, *conv, vect_size));
    if (!supported_target_vectorization) {
      VLOG(3) << "Skipping vectorization of conv to vector size: " << vect_size
              << ": " << conv->ToString();
      return false;
    }
  }
  VLOG(1) << "Vectorizing conv channels by " << vect_size << ": "
          << conv->ToString();
  XlaBuilder b(absl::StrCat(conv->name(), ".revectorized"));
  b.SetOpMetadata(conv->metadata());
  XlaOp filter = Parameter(&b, 1, conv->operand(1)->shape(), "filter");
  absl::InlinedVector<XlaOp, 4> new_operands = {
      SplitAtDim(Parameter(&b, 0, conv->operand(0)->shape(), "input"),
                 dnums->input_feature_dimension(), vect_size),
      SplitAtDim(filter, dnums->kernel_input_feature_dimension(), vect_size),
  };
  if (conv->operand_count() > 2) {
    new_operands.push_back(Parameter(&b, 2, conv->operand(2)->shape(), "bias"));
  }
  if (conv->operand_count() > 3) {
    new_operands.push_back(
        SplitAtDim(Parameter(&b, 3, conv->operand(3)->shape(), "side_input"),
                   dnums->output_feature_dimension(), vect_size));
  }
  if (conv->operand_count() > 4) {
    return InvalidArgument(
        "Don't understand a conv with more than 4 arguments: %s",
        conv->ToString());
  }
  const auto& debug_options = conv->GetModule()->config().debug_options();
  bool use_reordering =
      input_shape.element_type() == xla::S8 && vect_size == 32 &&
      debug_options.xla_gpu_enable_cudnn_int8x32_convolution_reordering() &&
      cudnn_version >= se::dnn::VersionInfo{8, 3, 0};
  if (use_reordering) {
    new_operands[1] = filter;
    TF_RETURN_IF_ERROR(ReorderInt8NchwVect(conv, new_operands.data()));
    dnums = &conv->convolution_dimension_numbers();
  }
  Shape new_output_shape = SplitShapeAtDim(
      output_shape, dnums->output_feature_dimension(), vect_size);
  XlaOp new_conv = CustomCallWithConvDnums(
      &b, conv->custom_call_target(), new_operands,
      ShapeUtil::MakeTupleShape(
          {new_output_shape, ShapeUtil::MakeShape(U8, {0})}),
      {},
      conv->raw_backend_config_string(), false,
      {}, nullptr,
      conv->window(),
      VectorizeDnums(*dnums, use_reordering));
  XlaOp new_conv_result = GetTupleElement(new_conv, 0);
  XlaOp new_conv_scratch = GetTupleElement(new_conv, 1);
  XlaOp conv_result_collapsed =
      Collapse(new_conv_result, {dnums->output_feature_dimension(),
                                 dnums->output_feature_dimension() + 1});
  TF_ASSIGN_OR_RETURN(
      HloComputation * new_conv_comp,
      BuilderToHloComputation(
          b, Tuple(&b, {conv_result_collapsed, new_conv_scratch}),
          conv->parent()));
  VLOG(1) << "Vectorized conv to: " << new_conv_comp->ToString();
  TF_RETURN_IF_ERROR(conv->parent()->ReplaceWithNewInstruction(
      conv, HloInstruction::CreateCall(conv->shape(), conv->operands(),
                                       new_conv_comp)));
  return true;
}
}  
absl::StatusOr<bool> CudnnVectorizeConvolutions::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloCustomCallInstruction* conv : GetRelevantConvs(comp)) {
      bool local_changed = false;
      if (compute_capability_.IsAtLeast(7, 5)) {
        TF_ASSIGN_OR_RETURN(
            local_changed,
            TryRevectorizeConv(compute_capability_, cudnn_version_, conv, 32));
        if (!local_changed) {
          TF_ASSIGN_OR_RETURN(
              local_changed,
              TryVectorizeConv(compute_capability_, cudnn_version_, conv, 32));
        }
      }
      if (!local_changed) {
        TF_ASSIGN_OR_RETURN(
            local_changed,
            TryVectorizeConv(compute_capability_, cudnn_version_, conv, 4));
      }
      changed |= local_changed;
    }
  }
  return changed;
}
}  
}  