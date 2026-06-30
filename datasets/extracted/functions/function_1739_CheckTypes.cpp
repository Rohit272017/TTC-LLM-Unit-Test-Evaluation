#include "xla/service/gpu/transforms/conv_rewriter.h"
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/permutation_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/window_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace gpu {
namespace {
absl::Status CheckTypes(HloInstruction* conv,
                        const se::GpuComputeCapability cc) {
  auto valid_shape = [conv, &cc](const Shape& shape) -> absl::Status {
    PrimitiveType type = shape.element_type();
    if (!primitive_util::IsFloatingPointType(type) &&
        !primitive_util::IsIntegralType(type)) {
      return Unimplemented(
          "Convolutions must have floating-point or integral operands/outputs, "
          "but got convolution with type %s: %s",
          primitive_util::LowercasePrimitiveTypeName(type), conv->ToString());
    }
    if (primitive_util::IsF8Type(type)) {
      if (type != F8E4M3FN && type != F8E5M2) {
        return Unimplemented(
            "The only FP8 types supported in convolutions are f8e5m2 and "
            "f8e4m3, "
            "but got convolution with FP8 type %s: %s",
            primitive_util::LowercasePrimitiveTypeName(type), conv->ToString());
      }
      if (!std::holds_alternative<se::CudaComputeCapability>(cc)) {
        return Unimplemented(
            "FP8 convolutions are only supported on CUDA GPUs, but got "
            "FP8 convolution on ROCm GPU: %s",
            conv->ToString());
      } else if (!std::get<se::CudaComputeCapability>(cc).IsAtLeastHopper()) {
        return Unimplemented(
            "FP8 convolutions are only supported on CUDA GPUs with compute "
            "capability at least 9.0, but got "
            "FP8 convolution on GPU with compute capability %s: %s",
            std::get<se::CudaComputeCapability>(cc).ToString(),
            conv->ToString());
      }
    }
    return absl::OkStatus();
  };
  TF_RETURN_IF_ERROR(valid_shape(conv->shape()));
  TF_RETURN_IF_ERROR(valid_shape(conv->operand(0)->shape()));
  TF_RETURN_IF_ERROR(valid_shape(conv->operand(1)->shape()));
  return absl::OkStatus();
}
using ConvolutionMatch = std::optional<
    std::tuple<Window, ConvolutionDimensionNumbers, HloInstruction*>>;
bool MaybeConv1dToConv2d(HloInstruction* conv) {
  if (conv->window().dimensions().size() != 2) {
    return false;
  }
  if (conv->operand(1)->opcode() != HloOpcode::kReshape) {
    return false;
  }
  auto filter = conv->operand(1);
  std::optional<ShapeUtil::ShapeEqualityDescriptor> reshape_degenerate =
      filter->ReshapeMerelyInsertsOrDeletes1SizedDimensions();
  if (reshape_degenerate.has_value() &&
      reshape_degenerate->deleted_dimensions.empty() &&
      reshape_degenerate->inserted_dimensions.size() == 1) {
    const auto& dnums = conv->convolution_dimension_numbers();
    for (auto dim : dnums.kernel_spatial_dimensions()) {
      if (dim == reshape_degenerate->inserted_dimensions[0]) {
        return true;
      }
    }
  }
  return false;
}
bool CanImplementAsGpuForwardConv(HloInstruction* conv) {
  const ConvolutionDimensionNumbers& dnums =
      conv->convolution_dimension_numbers();
  if (dnums.input_spatial_dimensions_size() > 3) {
    return false;
  }
  if (ShapeUtil::IsZeroElementArray(conv->operand(0)->shape()) ||
      ShapeUtil::IsZeroElementArray(conv->operand(1)->shape())) {
    return false;
  }
  if (dnums.input_spatial_dimensions_size() == 2
          ? !window_util::AllOrNoneReversed(conv->window())
          : window_util::HasWindowReversal(conv->window())) {
    return false;
  }
  return true;
}
ConvolutionMatch MatchBackwardFilter(HloInstruction* conv) {
  VLOG(2) << "Trying to match convolution backward filter.";
  if (conv->feature_group_count() > 1) {
    VLOG(1) << conv->ToString()
            << " is a forward convolution. All grouped backward filters are "
               "mapped to batch grouped convolutions in tf2xla bridge. Hence "
               "backward filter "
               "convolutions cannot have feature groups greater than 1 at this "
               "point. No need to fold to backward filter.";
    return std::nullopt;
  }
  CHECK_EQ(HloOpcode::kConvolution, conv->opcode());
  const ConvolutionDimensionNumbers& conv_dnums =
      conv->convolution_dimension_numbers();
  auto input_batch_dim = conv_dnums.input_batch_dimension();
  auto input_feature_dim = conv_dnums.input_feature_dimension();
  auto input_spatial_dims = conv_dnums.input_spatial_dimensions();
  auto kernel_input_feature_dim = conv_dnums.kernel_input_feature_dimension();
  auto kernel_output_feature_dim = conv_dnums.kernel_output_feature_dimension();
  auto kernel_spatial_dims = conv_dnums.kernel_spatial_dimensions();
  auto output_batch_dim = conv_dnums.output_batch_dimension();
  auto output_feature_dim = conv_dnums.output_feature_dimension();
  auto output_spatial_dims = conv_dnums.output_spatial_dimensions();
  for (const WindowDimension& window_dim : conv->window().dimensions()) {
    if (window_dim.stride() != 1) {
      VLOG(1) << "Forward convolution's window "
              << conv->window().ShortDebugString()
              << " should have stride of 1.";
      return std::nullopt;
    }
    if (window_dim.base_dilation() != 1) {
      VLOG(1) << "Forward convolution's window "
              << conv->window().ShortDebugString()
              << " should have no base (LHS) dilation.";
      return std::nullopt;
    }
    if (window_dim.padding_low() < 0) {
      VLOG(1) << "Padding low should be non-negative.";
      return std::nullopt;
    }
    if (window_dim.window_reversal()) {
      VLOG(1) << "Window reversal field not supported";
      return std::nullopt;
    }
  }
  int small_kernel_dimension_num = 0;
  for (int i = 0; i < kernel_spatial_dims.size(); ++i) {
    if (conv->operand(1)->shape().dimensions(kernel_spatial_dims[i]) <=
        conv->shape().dimensions(output_spatial_dims[i])) {
      small_kernel_dimension_num += 1;
    }
  }
  if ((kernel_spatial_dims.empty() || small_kernel_dimension_num > 1 ||
       (!MaybeConv1dToConv2d(conv) && small_kernel_dimension_num == 1)) &&
      !window_util::HasWindowDilation(conv->window())) {
    VLOG(1) << conv->ToString()
            << " is a regular forward convolution. No need "
               "to fold it to a backward filter convolution....";
    return std::nullopt;
  }
  Window backward_conv_window;
  for (int i = 0; i < input_spatial_dims.size(); ++i) {
    WindowDimension* dim = backward_conv_window.add_dimensions();
    int64_t filter_size = conv->shape().dimensions(output_spatial_dims[i]);
    dim->set_size(filter_size);
    dim->set_stride(conv->window().dimensions(i).window_dilation());
    dim->set_padding_low(conv->window().dimensions(i).padding_low());
    dim->set_base_dilation(1);
    dim->set_window_dilation(1);
    int64_t input_size =
        conv->operand(0)->shape().dimensions(input_spatial_dims[i]);
    int64_t output_size = conv->window().dimensions(i).size();
    int64_t padded_input_size = filter_size + (output_size - 1) * dim->stride();
    int64_t min_padding_high =
        padded_input_size - input_size - dim->padding_low();
    int64_t max_padding_high = min_padding_high + dim->stride() - 1;
    CHECK_GE(dim->padding_low(), 0);
    if (dim->padding_low() >= min_padding_high &&
        dim->padding_low() <= max_padding_high) {
      dim->set_padding_high(dim->padding_low());
    } else {
      if (dim->padding_low() < min_padding_high) {
        dim->set_padding_high(min_padding_high);
      } else {
        dim->set_padding_high(max_padding_high);
      }
    }
    if (dim->padding_high() < 0) {
      LOG(WARNING)
          << "Fusing this pattern to backward filter convolution would cause "
             "negative padding ("
          << dim->padding_high()
          << ") on right/bottom of the weight gradients, which is not "
             "supported by GpuConvPaddingLegalization (b/32744257). "
             "Falling back to "
             "unfused convolution for instruction: "
          << conv->ToString();
      return std::nullopt;
    }
  }
  ConvolutionDimensionNumbers backward_conv_dnums;
  backward_conv_dnums.set_input_batch_dimension(input_feature_dim);
  backward_conv_dnums.set_input_feature_dimension(input_batch_dim);
  for (int i = 0; i < input_spatial_dims.size(); ++i) {
    backward_conv_dnums.add_input_spatial_dimensions(input_spatial_dims[i]);
  }
  backward_conv_dnums.set_output_batch_dimension(kernel_input_feature_dim);
  backward_conv_dnums.set_output_feature_dimension(kernel_output_feature_dim);
  for (int i = 0; i < kernel_spatial_dims.size(); ++i) {
    backward_conv_dnums.add_output_spatial_dimensions(kernel_spatial_dims[i]);
  }
  backward_conv_dnums.set_kernel_input_feature_dimension(output_batch_dim);
  backward_conv_dnums.set_kernel_output_feature_dimension(output_feature_dim);
  for (int i = 0; i < output_spatial_dims.size(); ++i) {
    backward_conv_dnums.add_kernel_spatial_dimensions(output_spatial_dims[i]);
  }
  HloInstruction* lhs = conv->mutable_operand(0);
  return std::make_tuple(backward_conv_window, backward_conv_dnums, lhs);
}
ConvolutionMatch MatchBackwardInput(HloInstruction* conv) {
  VLOG(2) << "Trying to match convolution backward input.";
  if (conv->feature_group_count() > 1) {
    return std::nullopt;
  }
  CHECK_EQ(HloOpcode::kConvolution, conv->opcode());
  HloInstruction* reverse_filter = conv->mutable_operand(1);
  ConvolutionDimensionNumbers dnums = conv->convolution_dimension_numbers();
  auto kernel_out_feature_dim = dnums.kernel_output_feature_dimension();
  auto kernel_out_features =
      reverse_filter->shape().dimensions(kernel_out_feature_dim);
  if (conv->feature_group_count() > 1 &&
      kernel_out_features == conv->feature_group_count()) {
    return std::nullopt;
  }
  bool is_reversed_filter =
      reverse_filter->opcode() == HloOpcode::kReverse &&
      absl::c_is_permutation(dnums.kernel_spatial_dimensions(),
                             reverse_filter->dimensions());
  bool is_reversed_conv1d_filter =
      MaybeConv1dToConv2d(conv) &&
      reverse_filter->operand(0)->opcode() == HloOpcode::kReverse;
  bool is_1x1_filter =
      absl::c_all_of(conv->window().dimensions(),
                     [](const WindowDimension& d) { return d.size() == 1; });
  if (!is_reversed_filter && !is_reversed_conv1d_filter &&
      !(window_util::HasBaseDilation(conv->window()) &&
        (reverse_filter->IsConstant() || is_1x1_filter))) {
    VLOG(1) << "Can't match to backwards convolution. Either filter is not "
               "kReverse, or it's not a base-dilated conv with a 1x1 or "
               "constant filter.";
    return std::nullopt;
  }
  for (const WindowDimension& window_dim : conv->window().dimensions()) {
    if (window_dim.stride() != 1) {
      VLOG(1) << "Forward convolution's window "
              << conv->window().ShortDebugString()
              << " should have stride of 1.";
      return std::nullopt;
    }
    if (window_dim.window_dilation() != 1) {
      VLOG(1) << "Forward convolution's window "
              << conv->window().ShortDebugString()
              << " should have no window dilation.";
      return std::nullopt;
    }
    if (window_dim.window_reversal()) {
      VLOG(1) << "Window reversal field not supported";
      return std::nullopt;
    }
  }
  const auto& input_spatial_dims = dnums.input_spatial_dimensions();
  const auto& output_spatial_dims = dnums.output_spatial_dimensions();
  CHECK_EQ(conv->window().dimensions().size(), input_spatial_dims.size());
  CHECK_EQ(output_spatial_dims.size(), input_spatial_dims.size());
  const Window& old_window = conv->window();
  Window new_window = old_window;
  for (size_t i = 0; i < input_spatial_dims.size(); ++i) {
    auto dim = new_window.mutable_dimensions(i);
    dim->set_stride(old_window.dimensions(i).base_dilation());
    dim->set_base_dilation(1);
    auto kernel_size = old_window.dimensions(i).size();
    auto backward_padding_low =
        kernel_size - 1 - old_window.dimensions(i).padding_low();
    if (backward_padding_low < 0) {
      LOG(WARNING)
          << "The low padding of the backward convolution would be negative ("
          << backward_padding_low
          << "), which isn't supported by GpuConvPaddingLegalization "
             "for now (b/32744257).";
      return std::nullopt;
    }
    dim->set_padding_low(backward_padding_low);
    auto unpadded_input_size = conv->shape().dimensions(output_spatial_dims[i]);
    auto output_size =
        conv->operand(0)->shape().dimensions(input_spatial_dims[i]);
    auto padded_input_size = kernel_size + dim->stride() * (output_size - 1);
    auto total_pad_size = padded_input_size - unpadded_input_size;
    auto min_padding_high = total_pad_size - backward_padding_low;
    auto max_padding_high = min_padding_high + dim->stride() - 1;
    if (backward_padding_low >= min_padding_high &&
        backward_padding_low <= max_padding_high) {
      dim->set_padding_high(backward_padding_low);
    } else {
      if (backward_padding_low < min_padding_high) {
        dim->set_padding_high(min_padding_high);
      } else {
        dim->set_padding_high(max_padding_high);
      }
    }
    if (dim->padding_high() < 0) {
      LOG(WARNING) << "Fusing this pattern to backward convolution would cause "
                      "negative padding ("
                   << dim->padding_high()
                   << ") on right/bottom of the activations, which is not "
                      "supported by GpuConvPaddingLegalization (b/32744257). "
                      "Falling back to unfused convolution for instruction: "
                   << conv->ToString();
      return std::nullopt;
    }
  }
  auto conv_dnums = conv->convolution_dimension_numbers();
  dnums.set_kernel_input_feature_dimension(
      conv_dnums.kernel_output_feature_dimension());
  dnums.set_kernel_output_feature_dimension(
      conv_dnums.kernel_input_feature_dimension());
  for (int i = 0; i < input_spatial_dims.size(); ++i) {
    dnums.set_input_spatial_dimensions(i,
                                       conv_dnums.output_spatial_dimensions(i));
    dnums.set_output_spatial_dimensions(i,
                                        conv_dnums.input_spatial_dimensions(i));
  }
  dnums.set_input_feature_dimension(conv_dnums.output_feature_dimension());
  dnums.set_input_batch_dimension(conv_dnums.output_batch_dimension());
  dnums.set_output_feature_dimension(conv_dnums.input_feature_dimension());
  dnums.set_output_batch_dimension(conv_dnums.input_batch_dimension());
  if (reverse_filter->opcode() != HloOpcode::kReverse &&
      reverse_filter->IsConstant()) {
    HloComputation* c = conv->parent();
    reverse_filter = c->AddInstruction(
        HloInstruction::CreateReverse(reverse_filter->shape(), reverse_filter,
                                      dnums.kernel_spatial_dimensions()));
    reverse_filter = c->AddInstruction(
        HloInstruction::CreateReverse(reverse_filter->shape(), reverse_filter,
                                      dnums.kernel_spatial_dimensions()));
    TF_CHECK_OK(conv->ReplaceOperandWith(1, reverse_filter));
  }
  HloInstruction* rhs = reverse_filter;
  if (rhs->opcode() == HloOpcode::kReverse) {
    rhs = rhs->mutable_operand(0);
  } else if (is_reversed_conv1d_filter) {
    auto src = rhs->mutable_operand(0)->mutable_operand(0);
    rhs = conv->parent()->AddInstruction(
        HloInstruction::CreateReshape(rhs->shape(), src));
  }
  if (conv->feature_group_count() == 1) {
    return std::make_tuple(new_window, dnums, rhs);
  }
  int64_t input_feature_dimension = dnums.kernel_input_feature_dimension();
  int64_t output_feature_dimension = dnums.kernel_output_feature_dimension();
  if (std::abs(input_feature_dimension - output_feature_dimension) != 1) {
    return std::nullopt;
  }
  int64_t input_features = rhs->shape().dimensions(input_feature_dimension);
  int64_t output_features = rhs->shape().dimensions(output_feature_dimension);
  std::vector<int64_t> reshape_dims = SpanToVector(rhs->shape().dimensions());
  auto num_groups = conv->feature_group_count();
  CHECK_EQ(input_features % num_groups, 0)
      << "Input feature count should be an exact multiple of feature group "
         "count";
  reshape_dims[input_feature_dimension] =
      reshape_dims[input_feature_dimension] / num_groups;
  reshape_dims.insert(reshape_dims.begin() + input_feature_dimension,
                      num_groups);
  HloComputation* c = conv->parent();
  rhs = c->AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(rhs->shape().element_type(), reshape_dims), rhs));
  std::vector<int64_t> transpose_dims(rhs->shape().dimensions_size());
  std::iota(transpose_dims.begin(), transpose_dims.end(), 0);
  transpose_dims.erase(transpose_dims.begin() + input_feature_dimension);
  transpose_dims.insert(transpose_dims.begin() + output_feature_dimension,
                        input_feature_dimension);
  std::vector<int64_t> transpose_reshape_dims =
      SpanToVector(rhs->shape().dimensions());
  transpose_reshape_dims.erase(transpose_reshape_dims.begin() +
                               input_feature_dimension);
  transpose_reshape_dims.insert(
      transpose_reshape_dims.begin() + output_feature_dimension, num_groups);
  rhs = c->AddInstruction(HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(rhs->shape().element_type(), transpose_reshape_dims),
      rhs, transpose_dims));
  Shape new_shape = rhs->shape();
  new_shape.DeleteDimension(output_feature_dimension);
  new_shape.set_dimensions(output_feature_dimension,
                           output_features * num_groups);
  rhs = c->AddInstruction(HloInstruction::CreateReshape(new_shape, rhs));
  return std::make_tuple(new_window, dnums, rhs);
}
HloInstruction* CreateGpuConv(absl::string_view call_target, const Shape& shape,
                              HloInstruction* lhs, HloInstruction* rhs,
                              const Window& window,
                              const ConvolutionDimensionNumbers& dnums,
                              int64_t feature_group_count,
                              const PrecisionConfig& precision_config,
                              const OpMetadata& metadata) {
  HloComputation* computation = lhs->parent();
  Shape call_shape =
      ShapeUtil::MakeTupleShape({shape, ShapeUtil::MakeShape(U8, {0})});
  HloInstruction* custom_call = computation->AddInstruction(
      HloInstruction::CreateCustomCall(call_shape, {lhs, rhs}, call_target));
  custom_call->set_window(window);
  custom_call->set_convolution_dimension_numbers(dnums);
  custom_call->set_feature_group_count(feature_group_count);
  *custom_call->mutable_precision_config() = precision_config;
  custom_call->set_metadata(metadata);
  std::optional<std::string> name;
  if (call_target == kCudnnConvForwardCallTarget) {
    name = "cudnn-conv";
  } else if (call_target == kCudnnConvBackwardInputCallTarget) {
    name = "cudnn-conv-bw-input";
  } else if (call_target == kCudnnConvBackwardFilterCallTarget) {
    name = "cudnn-conv-bw-filter";
  } else if (call_target == kCudnnConvBiasActivationForwardCallTarget) {
    name = "cudnn-conv-bias-activation";
  }
  if (name.has_value()) {
    computation->parent()->SetAndUniquifyInstrName(custom_call, *name);
  }
  return custom_call;
}
HloInstruction* ConvertBatchGroupedToFeatureGroupedConvolution(
    HloInstruction* conv) {
  CHECK_EQ(conv->feature_group_count(), 1);
  int64_t num_groups = conv->batch_group_count();
  auto dim_numbers = conv->convolution_dimension_numbers();
  auto lhs = conv->mutable_operand(0);
  auto rhs = conv->mutable_operand(1);
  int64_t input_batch_dimension = dim_numbers.input_batch_dimension();
  Shape output_shape = conv->shape();
  int64_t input_feature_dimension = dim_numbers.input_feature_dimension();
  int64_t input_feature = lhs->shape().dimensions(input_feature_dimension);
  HloComputation* computation = lhs->parent();
  auto add = [&](std::unique_ptr<HloInstruction> inst) {
    return computation->AddInstruction(std::move(inst));
  };
  std::vector<int64_t> reshape_dims = SpanToVector(lhs->shape().dimensions());
  reshape_dims[input_batch_dimension] =
      reshape_dims[input_batch_dimension] / num_groups;
  reshape_dims.insert(reshape_dims.begin() + input_batch_dimension, num_groups);
  lhs = add(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(lhs->shape().element_type(), reshape_dims), lhs));
  std::vector<int64_t> transpose_dims(lhs->shape().dimensions_size());
  std::iota(transpose_dims.begin(), transpose_dims.end(), 0);
  transpose_dims.erase(transpose_dims.begin() + input_batch_dimension);
  transpose_dims.insert(transpose_dims.begin() + input_feature_dimension,
                        input_batch_dimension);
  std::vector<int64_t> transpose_reshape_dims =
      ComposePermutations(lhs->shape().dimensions(), transpose_dims);
  lhs = add(HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(lhs->shape().element_type(), transpose_reshape_dims),
      lhs, transpose_dims));
  Shape new_shape = lhs->shape();
  new_shape.DeleteDimension(input_feature_dimension);
  new_shape.set_dimensions(input_feature_dimension, input_feature * num_groups);
  lhs = add(HloInstruction::CreateReshape(new_shape, lhs));
  std::vector<HloInstruction*> new_operands = {lhs, rhs};
  auto new_conv = conv->CloneWithNewOperands(output_shape, new_operands);
  new_conv->set_feature_group_count(num_groups);
  new_conv->set_batch_group_count(1);
  new_conv->set_convolution_dimension_numbers(dim_numbers);
  return computation->AddInstruction(std::move(new_conv));
}
CudnnConvBackendConfig GetDefaultBackendConfig() {
  CudnnConvBackendConfig config;
  config.set_conv_result_scale(1);
  return config;
}
static absl::StatusOr<HloInstruction*> CreateCustomCallHelper(
    HloInstruction* conv, const se::GpuComputeCapability& cc) {
  TF_RETURN_IF_ERROR(CheckTypes(conv, cc));
  if (ConvolutionMatch m = MatchBackwardInput(conv)) {
    auto& [window, dnums, rhs] = *m;
    return CreateGpuConv(kCudnnConvBackwardInputCallTarget, conv->shape(),
                         conv->mutable_operand(0), rhs, window, dnums,
                         conv->feature_group_count(), conv->precision_config(),
                         conv->metadata());
  }
  if (ConvolutionMatch m = MatchBackwardFilter(conv)) {
    auto& [window, dnums, lhs] = *m;
    return CreateGpuConv(kCudnnConvBackwardFilterCallTarget, conv->shape(), lhs,
                         conv->mutable_operand(1), window, dnums,
                         conv->batch_group_count(), conv->precision_config(),
                         conv->metadata());
  }
  if (CanImplementAsGpuForwardConv(conv)) {
    if (conv->batch_group_count() > 1) {
      conv = ConvertBatchGroupedToFeatureGroupedConvolution(conv);
    }
    return CreateGpuConv(kCudnnConvForwardCallTarget, conv->shape(),
                         conv->mutable_operand(0), conv->mutable_operand(1),
                         conv->window(), conv->convolution_dimension_numbers(),
                         conv->feature_group_count(), conv->precision_config(),
                         conv->metadata());
  }
  return nullptr;
}
absl::StatusOr<bool> RunOnInstruction(HloInstruction* conv,
                                      const se::GpuComputeCapability& cc) {
  CHECK_EQ(conv->opcode(), HloOpcode::kConvolution);
  TF_ASSIGN_OR_RETURN(HloInstruction * custom_call,
                      CreateCustomCallHelper(conv, cc));
  if (custom_call == nullptr) {
    return false;
  }
  GpuBackendConfig gpu_backend_config;
  *gpu_backend_config.mutable_cudnn_conv_backend_config() =
      GetDefaultBackendConfig();
  TF_RETURN_IF_ERROR(custom_call->set_backend_config(gpu_backend_config));
  VLOG(1) << "Replacing convolution " << conv->ToString() << " with "
          << custom_call->ToString();
  TF_RETURN_IF_ERROR(conv->parent()->ReplaceWithNewInstruction(
      conv,
      HloInstruction::CreateGetTupleElement(conv->shape(), custom_call, 0)));
  return true;
}
absl::StatusOr<bool> RunOnComputation(HloComputation* computation,
                                      const se::GpuComputeCapability& cc) {
  std::vector<HloInstruction*> convs;
  for (auto* hlo : computation->instructions()) {
    if (hlo->opcode() == HloOpcode::kConvolution) {
      convs.push_back(hlo);
    }
  }
  bool changed = false;
  for (HloInstruction* conv : convs) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnInstruction(conv, cc));
    changed |= result;
  }
  return changed;
}
}  
absl::StatusOr<bool> ConvRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(2, "ConvRewriter::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result,
                        RunOnComputation(computation, compute_capability_));
    changed |= result;
  }
  XLA_VLOG_LINES(2, "ConvRewriter::Run(), after:\n" + module->ToString());
  return changed;
}
 bool ConvRewriter::ConvIsLowerable(HloInstruction* conv) {
  return CanImplementAsGpuForwardConv(conv) || MatchBackwardFilter(conv) ||
         MatchBackwardInput(conv);
}
}  
}  