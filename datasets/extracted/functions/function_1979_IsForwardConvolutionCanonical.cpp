#include "xla/service/gpu/transforms/conv_padding_legalization.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/window_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace gpu {
namespace {
bool IsForwardConvolutionCanonical(const HloInstruction& conv) {
  CHECK(conv.custom_call_target() == kCudnnConvForwardCallTarget ||
        conv.custom_call_target() ==
            kCudnnConvBiasActivationForwardCallTarget ||
        conv.custom_call_target() == kCudnnConvForwardGraphCallTarget);
  return window_util::HasSymmetricPadding(conv.window()) &&
         !window_util::HasNegativePadding(conv.window()) &&
         !window_util::HasDilation(conv.window());
}
HloInstruction* MaybePaddedAndSlicedInput(
    Window* conv_window, const ConvolutionDimensionNumbers& conv_dnums,
    HloInstruction* input) {
  HloComputation* computation = input->parent();
  if (!window_util::HasSymmetricPadding(*conv_window) ||
      window_util::HasBaseDilation(*conv_window)) {
    PaddingConfig padding_config =
        MakeNoPaddingConfig(input->shape().dimensions_size());
    for (size_t i = 0; i < conv_dnums.input_spatial_dimensions().size(); ++i) {
      int64_t dim = conv_dnums.input_spatial_dimensions(i);
      if (conv_window->dimensions(i).padding_low() > 0) {
        padding_config.mutable_dimensions(dim)->set_edge_padding_low(
            conv_window->dimensions(i).padding_low());
        conv_window->mutable_dimensions(i)->set_padding_low(0);
      }
      if (conv_window->dimensions(i).padding_high() > 0) {
        padding_config.mutable_dimensions(dim)->set_edge_padding_high(
            conv_window->dimensions(i).padding_high());
        conv_window->mutable_dimensions(i)->set_padding_high(0);
      }
      if (conv_window->dimensions(i).base_dilation() != 1) {
        padding_config.mutable_dimensions(dim)->set_interior_padding(
            conv_window->dimensions(i).base_dilation() - 1);
        conv_window->mutable_dimensions(i)->set_base_dilation(1);
      }
    }
    PrimitiveType element_type = input->shape().element_type();
    HloInstruction* padding = computation->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::Zero(element_type)));
    input =
        MakePadHlo(input, padding, padding_config, &input->metadata()).value();
  }
  if (window_util::HasNegativePadding(*conv_window)) {
    std::vector<int64_t> start_indices(input->shape().dimensions_size(), 0);
    std::vector<int64_t> limit_indices(input->shape().dimensions().begin(),
                                       input->shape().dimensions().end());
    std::vector<int64_t> strides(input->shape().dimensions_size(), 1);
    for (size_t i = 0; i < conv_dnums.input_spatial_dimensions().size(); ++i) {
      int64_t dim = conv_dnums.input_spatial_dimensions(i);
      if (conv_window->dimensions(i).padding_low() < 0) {
        start_indices[dim] += -conv_window->dimensions(i).padding_low();
        conv_window->mutable_dimensions(i)->set_padding_low(0);
      }
      if (conv_window->dimensions(i).padding_high() < 0) {
        limit_indices[dim] -= -conv_window->dimensions(i).padding_high();
        conv_window->mutable_dimensions(i)->set_padding_high(0);
      }
    }
    input = MakeSliceHlo(input, start_indices, limit_indices, strides).value();
  }
  return input;
}
HloInstruction* MaybePaddedKernel(const Window& conv_window,
                                  const ConvolutionDimensionNumbers& conv_dnums,
                                  HloInstruction* kernel) {
  if (!window_util::HasWindowDilation(conv_window)) {
    return kernel;
  }
  PaddingConfig padding_config;
  padding_config.mutable_dimensions()->Reserve(
      kernel->shape().dimensions_size());
  for (size_t i = 0; i < kernel->shape().dimensions_size(); ++i) {
    padding_config.add_dimensions();
  }
  for (size_t i = 0; i < conv_dnums.kernel_spatial_dimensions().size(); ++i) {
    int64_t dim = conv_dnums.kernel_spatial_dimensions(i);
    padding_config.mutable_dimensions(dim)->set_interior_padding(
        conv_window.dimensions(i).window_dilation() - 1);
  }
  HloComputation* computation = kernel->parent();
  PrimitiveType element_type = kernel->shape().element_type();
  HloInstruction* padding = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(element_type)));
  return MakePadHlo(kernel, padding, padding_config, &kernel->metadata())
      .value();
}
}  
bool ConvPaddingLegalization::CanonicalizeForwardConvolution(
    HloInstruction* conv) {
  if (IsForwardConvolutionCanonical(*conv)) {
    return false;
  }
  Window new_conv_window = conv->window();
  HloInstruction* new_input = MaybePaddedAndSlicedInput(
      &new_conv_window, conv->convolution_dimension_numbers(),
      conv->mutable_operand(0));
  HloInstruction* new_kernel =
      MaybePaddedKernel(new_conv_window, conv->convolution_dimension_numbers(),
                        conv->mutable_operand(1));
  for (size_t i = 0; i < new_conv_window.dimensions_size(); ++i) {
    WindowDimension* dim = new_conv_window.mutable_dimensions(i);
    dim->set_size(new_kernel->shape().dimensions(
        conv->convolution_dimension_numbers().kernel_spatial_dimensions(i)));
    dim->set_window_dilation(1);
  }
  VLOG(1) << "Canonicalizing forward conv";
  std::vector<HloInstruction*> operands(conv->operands().begin(),
                                        conv->operands().end());
  operands[0] = new_input;
  operands[1] = new_kernel;
  auto new_conv = conv->parent()->AddInstruction(
      conv->CloneWithNewOperands(conv->shape(), operands));
  new_conv->set_window(new_conv_window);
  VLOG(1) << "Replacing:\n  " << conv->ToString() << "\nwith:\n  "
          << new_conv->ToString();
  TF_CHECK_OK(conv->parent()->ReplaceInstruction(conv, new_conv));
  return true;
}
namespace {
void IncreasePaddingLowBy(int64_t delta, WindowDimension* window_dim) {
  window_dim->set_padding_low(window_dim->padding_low() + delta);
}
void IncreasePaddingHighBy(int64_t delta, WindowDimension* window_dim) {
  window_dim->set_padding_high(window_dim->padding_high() + delta);
}
}  
bool ConvPaddingLegalization::CanonicalizeBackwardFilterConvolution(
    HloInstruction* backward_conv) {
  CHECK_EQ(backward_conv->custom_call_target(),
           kCudnnConvBackwardFilterCallTarget);
  if (window_util::HasSymmetricPadding(backward_conv->window())) {
    return false;
  }
  HloInstruction* input = backward_conv->mutable_operand(0);
  Window new_backward_conv_window = backward_conv->window();
  PaddingConfig input_padding_config =
      MakeNoPaddingConfig(input->shape().rank());
  ConvolutionDimensionNumbers backward_conv_dnums =
      backward_conv->convolution_dimension_numbers();
  for (size_t i = 0; i < backward_conv->window().dimensions_size(); ++i) {
    int64_t padding_low = backward_conv->window().dimensions(i).padding_low();
    int64_t padding_high = backward_conv->window().dimensions(i).padding_high();
    if (padding_low < 0 || padding_high < 0) {
      return false;
    }
    int64_t new_conv_padding = std::min(padding_low, padding_high);
    int64_t dim = backward_conv_dnums.input_spatial_dimensions(i);
    input_padding_config.mutable_dimensions(dim)->set_edge_padding_low(
        padding_low - new_conv_padding);
    input_padding_config.mutable_dimensions(dim)->set_edge_padding_high(
        padding_high - new_conv_padding);
    auto* new_dim = new_backward_conv_window.mutable_dimensions(i);
    new_dim->set_padding_low(new_conv_padding);
    new_dim->set_padding_high(new_conv_padding);
  }
  HloComputation* computation = backward_conv->parent();
  HloInstruction* output = backward_conv->mutable_operand(1);
  HloInstruction* padding =
      computation->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(input->shape().element_type())));
  HloInstruction* padded_input =
      MakePadHlo(input, padding, input_padding_config).value();
  HloInstruction* new_backward_conv =
      computation->AddInstruction(backward_conv->CloneWithNewOperands(
          backward_conv->shape(), {padded_input, output}));
  new_backward_conv->set_window(new_backward_conv_window);
  VLOG(1) << "Canonicalizing backward filter conv";
  VLOG(1) << "Replacing:\n  " << backward_conv->ToString() << "\nwith:\n  "
          << new_backward_conv->ToString();
  TF_CHECK_OK(
      computation->ReplaceInstruction(backward_conv, new_backward_conv));
  return true;
}
bool ConvPaddingLegalization::CanonicalizeBackwardInputConvolution(
    HloInstruction* backward_conv) {
  if (window_util::HasSymmetricPadding(backward_conv->window())) {
    return false;
  }
  Window new_backward_conv_window = backward_conv->window();
  ConvolutionDimensionNumbers backward_conv_dnums =
      backward_conv->convolution_dimension_numbers();
  Shape backward_conv_shape = backward_conv->shape().tuple_shapes(0);
  Shape new_backward_conv_shape = backward_conv_shape;
  for (size_t i = 0; i < backward_conv->window().dimensions_size(); ++i) {
    int64_t padding_low = backward_conv->window().dimensions(i).padding_low();
    int64_t padding_high = backward_conv->window().dimensions(i).padding_high();
    if (padding_low < 0 || padding_high < 0) {
      return false;
    }
    if (padding_low > padding_high) {
      IncreasePaddingLowBy(padding_high - padding_low,
                           new_backward_conv_window.mutable_dimensions(i));
    } else if (padding_low < padding_high) {
      IncreasePaddingHighBy(padding_low - padding_high,
                            new_backward_conv_window.mutable_dimensions(i));
    }
    int64_t dim = backward_conv_dnums.input_spatial_dimensions(i);
    new_backward_conv_shape.set_dimensions(
        dim, new_backward_conv_shape.dimensions(dim) +
                 std::abs(padding_low - padding_high));
  }
  HloComputation* computation = backward_conv->parent();
  HloInstruction* output = backward_conv->mutable_operand(0);
  HloInstruction* filter = backward_conv->mutable_operand(1);
  HloInstruction* new_backward_conv_call =
      computation->AddInstruction(backward_conv->CloneWithNewOperands(
          ShapeUtil::MakeTupleShape(
              {new_backward_conv_shape, ShapeUtil::MakeShape(U8, {0})}),
          {output, filter}));
  new_backward_conv_call->set_window(new_backward_conv_window);
  HloInstruction* new_backward_conv =
      computation->AddInstruction(HloInstruction::CreateGetTupleElement(
          new_backward_conv_shape, new_backward_conv_call, 0));
  HloInstruction* new_backward_conv_scratch =
      computation->AddInstruction(HloInstruction::CreateGetTupleElement(
          new_backward_conv_call->shape().tuple_shapes(1),
          new_backward_conv_call, 1));
  std::vector<int64_t> start_indices(
      new_backward_conv->shape().dimensions_size(), 0LL);
  std::vector<int64_t> limit_indices(
      new_backward_conv->shape().dimensions().begin(),
      new_backward_conv->shape().dimensions().end());
  std::vector<int64_t> strides(new_backward_conv->shape().dimensions_size(),
                               1LL);
  for (size_t i = 0; i < backward_conv->window().dimensions_size(); ++i) {
    int64_t padding_low = backward_conv->window().dimensions(i).padding_low();
    int64_t padding_high = backward_conv->window().dimensions(i).padding_high();
    int64_t dim = backward_conv_dnums.input_spatial_dimensions(i);
    if (padding_low > padding_high) {
      start_indices[dim] += padding_low - padding_high;
    } else if (padding_low < padding_high) {
      limit_indices[dim] -= padding_high - padding_low;
    }
  }
  Shape slice_shape =
      ShapeInference::InferSliceShape(new_backward_conv->shape(), start_indices,
                                      limit_indices, strides)
          .value();
  CHECK(ShapeUtil::Compatible(slice_shape, backward_conv_shape))
      << ShapeUtil::HumanString(slice_shape) << " vs "
      << ShapeUtil::HumanString(backward_conv_shape);
  HloInstruction* slice = computation->AddInstruction(
      HloInstruction::CreateSlice(backward_conv_shape, new_backward_conv,
                                  start_indices, limit_indices, strides));
  HloInstruction* new_tuple = computation->AddInstruction(
      HloInstruction::CreateTuple({slice, new_backward_conv_scratch}));
  VLOG(1) << "Canonicalizing backward input conv";
  VLOG(1) << "Replacing:\n  " << backward_conv->ToString() << "\nwith:\n  "
          << new_tuple->ToString();
  TF_CHECK_OK(computation->ReplaceInstruction(backward_conv, new_tuple));
  return true;
}
absl::StatusOr<bool> ConvPaddingLegalization::RunOnComputation(
    HloComputation* computation) {
  bool changed = false;
  std::vector<HloCustomCallInstruction*> convs;
  for (auto* instr : computation->instructions()) {
    if (IsCustomCallToDnnConvolution(*instr)) {
      convs.push_back(Cast<HloCustomCallInstruction>(instr));
    }
  }
  for (HloCustomCallInstruction* instruction : convs) {
    TF_ASSIGN_OR_RETURN(auto kind, GetCudnnConvKind(instruction));
    changed |= [&] {
      switch (kind) {
        case CudnnConvKind::kForward:
        case CudnnConvKind::kForwardActivation:
        case CudnnConvKind::kForwardGraph:
          return CanonicalizeForwardConvolution(instruction);
        case CudnnConvKind::kBackwardInput:
          return CanonicalizeBackwardInputConvolution(instruction);
        case CudnnConvKind::kBackwardFilter:
          return CanonicalizeBackwardFilterConvolution(instruction);
      }
    }();
  }
  return changed;
}
absl::StatusOr<bool> ConvPaddingLegalization::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  return changed;
}
}  
}  