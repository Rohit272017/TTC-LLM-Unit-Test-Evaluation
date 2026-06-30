#include "xla/service/convolution_group_converter.h"
#include <algorithm>
#include <memory>
#include <vector>
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
namespace xla {
namespace {
class ConvolutionVisitor : public DfsHloVisitorWithDefault {
 public:
  absl::Status DefaultAction(HloInstruction* ) override {
    return absl::OkStatus();
  }
  absl::Status HandleConvolution(HloInstruction* convolution) override;
  absl::Status HandleBatchGroupCount(HloInstruction* convolution);
  static bool Run(HloComputation* computation,
                  std::function<bool(HloInstruction*)> should_expand,
                  std::function<bool(HloInstruction*)> is_cost_viable,
                  bool convert_batch_groups_only, bool filter_expansion);
  const bool changed() const { return changed_; }
  ~ConvolutionVisitor() override = default;
 private:
  explicit ConvolutionVisitor(
      HloComputation* computation,
      std::function<bool(HloInstruction*)> should_expand,
      std::function<bool(HloInstruction*)> is_cost_viable,
      bool convert_batch_groups_only, bool filter_expansion)
      : computation_(computation),
        filter_expansion_(filter_expansion),
        convert_batch_groups_only_(convert_batch_groups_only),
        should_expand_(should_expand),
        is_cost_viable_(is_cost_viable) {}
  HloComputation* computation_;
  bool changed_ = false;
  bool filter_expansion_;
  bool convert_batch_groups_only_;
  std::function<bool(HloInstruction*)> should_expand_;
  std::function<bool(HloInstruction*)> is_cost_viable_;
};
bool ConvolutionVisitor::Run(
    HloComputation* computation,
    std::function<bool(HloInstruction*)> should_expand,
    std::function<bool(HloInstruction*)> is_cost_viable,
    bool convert_batch_groups_only, bool filter_expansion) {
  ConvolutionVisitor visitor(computation, should_expand, is_cost_viable,
                             convert_batch_groups_only, filter_expansion);
  TF_CHECK_OK(computation->Accept(&visitor));
  return visitor.changed_;
}
Shape ExpandedFilterShape(const Shape& shape, int64_t group_count,
                          int64_t input_feature_dim) {
  int64_t num_dims = shape.dimensions_size();
  CHECK_GE(num_dims, 2);
  Shape expanded_shape = shape;
  expanded_shape.set_dimensions(
      input_feature_dim, shape.dimensions(input_feature_dim) * group_count);
  return expanded_shape;
}
std::vector<int32_t> GetMaskIds(int64_t group_size, int64_t group_count) {
  std::vector<int32_t> values;
  values.reserve(group_count * group_size);
  for (int i = 0; i < group_count; ++i) {
    for (int j = 0; j < group_size; ++j) {
      values.push_back(i);
    }
  }
  return values;
}
HloInstruction* GetExpandedFilterMask(
    const Shape& filter_shape, int64_t kernel_input_feature_dim,
    int64_t kernel_output_feature_dim, int64_t group_count,
    const std::function<HloInstruction*(std::unique_ptr<HloInstruction>)>&
        add_instruction) {
  Shape expanded_filter_shape =
      ExpandedFilterShape(filter_shape, group_count, kernel_input_feature_dim);
  Shape mask_shape =
      ShapeUtil::MakeShape(S32, expanded_filter_shape.dimensions());
  int64_t output_feature = filter_shape.dimensions(kernel_output_feature_dim);
  int64_t group_size = filter_shape.dimensions(kernel_input_feature_dim);
  const std::vector<int32_t> input_feature_filter_mask =
      GetMaskIds(group_size, group_count);
  const std::vector<int32_t> output_feature_filter_mask =
      GetMaskIds(output_feature / group_count, group_count);
  auto mask1 = add_instruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<int32_t>(input_feature_filter_mask)));
  auto broadcasted_mask1 = add_instruction(HloInstruction::CreateBroadcast(
      mask_shape, mask1, {kernel_input_feature_dim}));
  auto mask2 = add_instruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<int32_t>(output_feature_filter_mask)));
  auto broadcasted_mask2 = add_instruction(HloInstruction::CreateBroadcast(
      mask_shape, mask2, {kernel_output_feature_dim}));
  Shape predicate_shape =
      ShapeUtil::MakeShape(PRED, expanded_filter_shape.dimensions());
  return add_instruction(HloInstruction::CreateCompare(
      predicate_shape, broadcasted_mask1, broadcasted_mask2,
      ComparisonDirection::kEq));
}
absl::Status ConvolutionVisitor::HandleBatchGroupCount(
    HloInstruction* convolution) {
  auto dim_numbers = convolution->convolution_dimension_numbers();
  auto activation = convolution->mutable_operand(0);
  auto filter = convolution->mutable_operand(1);
  int64_t batch_group_count = convolution->batch_group_count();
  if (batch_group_count == 1 ||
      (should_expand_ && !should_expand_(convolution))) {
    return absl::OkStatus();
  }
  VLOG(2) << "Dealing with batch_group_count " << batch_group_count
          << " for convolution " << convolution->ToString() << "\n";
  auto add = [&](std::unique_ptr<HloInstruction> inst) {
    return computation_->AddInstruction(std::move(inst));
  };
  int64_t input_batch_dimension = dim_numbers.input_batch_dimension();
  const int64_t input_feature_dimension = dim_numbers.input_feature_dimension();
  int64_t output_batch_dimension = dim_numbers.output_batch_dimension();
  int64_t output_feature_dimension = dim_numbers.output_feature_dimension();
  const int64_t kernel_input_feature_dimension =
      dim_numbers.kernel_input_feature_dimension();
  const int64_t kernel_output_feature_dimension =
      dim_numbers.kernel_output_feature_dimension();
  const int64_t input_batch =
      activation->shape().dimensions(input_batch_dimension);
  const int64_t output_feature =
      filter->shape().dimensions(kernel_output_feature_dimension);
  if (output_feature != batch_group_count || input_batch != batch_group_count) {
    std::vector<int64_t> input_sizes(activation->shape().dimensions().begin(),
                                     activation->shape().dimensions().end());
    input_sizes[input_batch_dimension] /= batch_group_count;
    input_sizes.insert(input_sizes.begin() + input_batch_dimension,
                       batch_group_count);
    activation = MakeReshapeHlo(input_sizes, activation).value();
    for (auto& d : *dim_numbers.mutable_input_spatial_dimensions()) {
      if (d > input_batch_dimension) {
        ++d;
      }
    }
    dim_numbers.add_input_spatial_dimensions(input_batch_dimension);
    dim_numbers.set_input_batch_dimension(input_batch_dimension + 1);
    if (input_feature_dimension > input_batch_dimension) {
      dim_numbers.set_input_feature_dimension(input_feature_dimension + 1);
    }
    std::vector<int64_t> kernel_sizes(filter->shape().dimensions().begin(),
                                      filter->shape().dimensions().end());
    kernel_sizes[kernel_output_feature_dimension] /= batch_group_count;
    kernel_sizes.insert(kernel_sizes.begin() + kernel_output_feature_dimension,
                        batch_group_count);
    filter = MakeReshapeHlo(kernel_sizes, filter).value();
    for (auto& d : *dim_numbers.mutable_kernel_spatial_dimensions()) {
      if (d > kernel_output_feature_dimension) {
        ++d;
      }
    }
    dim_numbers.add_kernel_spatial_dimensions(kernel_output_feature_dimension);
    dim_numbers.set_kernel_output_feature_dimension(
        kernel_output_feature_dimension + 1);
    if (kernel_input_feature_dimension > kernel_output_feature_dimension) {
      dim_numbers.set_kernel_input_feature_dimension(
          kernel_input_feature_dimension + 1);
    }
    for (auto& d : *dim_numbers.mutable_output_spatial_dimensions()) {
      if (d > output_feature_dimension) {
        ++d;
      }
    }
    dim_numbers.add_output_spatial_dimensions(output_feature_dimension);
    dim_numbers.set_output_feature_dimension(output_feature_dimension + 1);
    if (output_batch_dimension > output_feature_dimension) {
      dim_numbers.set_output_batch_dimension(output_batch_dimension + 1);
    }
    Window window = convolution->window();
    auto window_dim = window.add_dimensions();
    window_dim->set_base_dilation(batch_group_count);
    window_dim->set_size(batch_group_count);
    window_dim->set_stride(batch_group_count - 1);
    window_dim->set_padding_low(0);
    window_dim->set_padding_high(0);
    window_dim->set_window_reversal(false);
    window_dim->set_window_dilation(1);
    HloInstruction* new_convolution =
        MakeConvolveHlo(
            activation, filter, convolution->feature_group_count(),
            1, window, dim_numbers,
            convolution->precision_config(),
            convolution->shape().element_type())
            .value();
    convolution->SetupDerivedInstruction(new_convolution);
    TF_CHECK_OK(computation_->ReplaceInstruction(
        convolution,
        MakeReshapeHlo(convolution->shape(), new_convolution).value()));
    changed_ = true;
    return absl::OkStatus();
  }
  VLOG(2) << "is_cost_viable_ " << is_cost_viable_(convolution);
  const bool cost_too_high = !is_cost_viable_(convolution);
  if (cost_too_high || filter_expansion_) {
    HloInstruction* filter_mask =
        GetExpandedFilterMask(convolution->shape(), output_batch_dimension,
                              output_feature_dimension, batch_group_count, add);
    auto expanded_filter_shape = ExpandedFilterShape(
        convolution->shape(), batch_group_count, output_batch_dimension);
    VLOG(2) << "output_batch_dimension " << output_batch_dimension;
    VLOG(2) << "New output shape of convolution "
            << expanded_filter_shape.ToString();
    auto new_convolution = add(HloInstruction::CreateConvolve(
        expanded_filter_shape, activation, filter,
        1, 1,
        convolution->window(), dim_numbers, convolution->precision_config()));
    VLOG(2) << "Expanded convolution " << new_convolution->ToString();
    auto zero = add(HloInstruction::CreateConstant(
        LiteralUtil::Zero(expanded_filter_shape.element_type())));
    auto zero_filter =
        add(HloInstruction::CreateBroadcast(expanded_filter_shape, zero, {}));
    auto new_filter = add(HloInstruction::CreateTernary(
        expanded_filter_shape, HloOpcode::kSelect, filter_mask, new_convolution,
        zero_filter));
    PrimitiveType reduce_type = new_filter->shape().element_type();
    auto reduce_window_shape = new_convolution->shape();
    reduce_window_shape.set_dimensions(output_batch_dimension, 1);
    if (primitive_util::BitWidth(reduce_type) < primitive_util::BitWidth(F32)) {
      reduce_type = F32;
      reduce_window_shape.set_element_type(F32);
      Shape convert_shape = new_filter->shape();
      convert_shape.set_element_type(F32);
      new_filter =
          add(HloInstruction::CreateConvert(convert_shape, new_filter));
    }
    auto zero_literal = LiteralUtil::Zero(reduce_type);
    auto zero_scalar =
        add(HloInstruction::CreateConstant(std::move(zero_literal)));
    auto reduce_function = [&]() -> HloComputation* {
      HloComputation::Builder b("add_computation");
      Shape shape = ShapeUtil::MakeShape(reduce_type, {});
      auto lhs =
          b.AddInstruction(HloInstruction::CreateParameter(0, shape, "lhs"));
      auto rhs =
          b.AddInstruction(HloInstruction::CreateParameter(1, shape, "rhs"));
      auto scalar_op = b.AddInstruction(
          HloInstruction::CreateBinary(shape, HloOpcode::kAdd, lhs, rhs));
      return computation_->parent()->AddEmbeddedComputation(b.Build(scalar_op));
    };
    Window window;
    for (int64_t i = 0; i < new_convolution->shape().dimensions_size(); ++i) {
      auto* dim = window.add_dimensions();
      dim->set_padding_low(0);
      dim->set_padding_high(0);
      dim->set_window_dilation(1);
      dim->set_base_dilation(1);
      if (i == output_batch_dimension) {
        dim->set_stride(batch_group_count);
        dim->set_size(batch_group_count);
      } else {
        dim->set_stride(1);
        dim->set_size(1);
      }
    }
    auto reduce_window = add(HloInstruction::CreateReduceWindow(
        reduce_window_shape, new_filter, zero_scalar, window,
        reduce_function()));
    Shape convert_back_shape = reduce_window->shape();
    convert_back_shape.set_element_type(activation->shape().element_type());
    auto reduce_window_converted =
        HloInstruction::CreateConvert(convert_back_shape, reduce_window);
    TF_CHECK_OK(computation_->ReplaceWithNewInstruction(
        convolution, std::move(reduce_window_converted)));
    changed_ = true;
  }
  return absl::OkStatus();
}
absl::Status ConvolutionVisitor::HandleConvolution(
    HloInstruction* convolution) {
  if (convert_batch_groups_only_) {
    return HandleBatchGroupCount(convolution);
  }
  auto add = [&](std::unique_ptr<HloInstruction> inst) {
    return computation_->AddInstruction(std::move(inst));
  };
  int64_t group_count = convolution->feature_group_count();
  if (group_count == 1 || (should_expand_ && !should_expand_(convolution))) {
    return absl::OkStatus();
  }
  changed_ = true;
  ConvolutionDimensionNumbers dim_numbers =
      convolution->convolution_dimension_numbers();
  auto filter = convolution->mutable_operand(1);
  int64_t kernel_input_feature_dim =
      dim_numbers.kernel_input_feature_dimension();
  int64_t group_size = filter->shape().dimensions(kernel_input_feature_dim);
  int64_t kernel_output_feature_dim =
      dim_numbers.kernel_output_feature_dimension();
  auto expanded_filter_shape = ExpandedFilterShape(filter->shape(), group_count,
                                                   kernel_input_feature_dim);
  HloInstruction* filter_mask =
      GetExpandedFilterMask(filter->shape(), kernel_input_feature_dim,
                            kernel_output_feature_dim, group_count, add);
  HloInstruction* expanded_filter;
  if (group_size == 1) {
    bool depthwise_separable =
        (group_count == filter->shape().dimensions(kernel_output_feature_dim));
    if (!filter_expansion_ && depthwise_separable) {
      changed_ = false;
      return absl::OkStatus();
    }
    VLOG(2) << "is_cost_viable_ " << is_cost_viable_(convolution);
    if (!is_cost_viable_(convolution) || filter_expansion_) {
      Shape reshaped_filter_shape =
          ShapeUtil::DeleteDimension(kernel_input_feature_dim, filter->shape());
      auto reshaped_filter =
          add(HloInstruction::CreateReshape(reshaped_filter_shape, filter));
      std::vector<int64_t> broadcast_dims;
      for (int64_t i = 0; i < filter->shape().dimensions_size(); ++i) {
        if (i == kernel_input_feature_dim) {
          continue;
        }
        broadcast_dims.push_back(i);
      }
      expanded_filter = add(HloInstruction::CreateBroadcast(
          expanded_filter_shape, reshaped_filter, broadcast_dims));
      auto zero = add(HloInstruction::CreateConstant(
          LiteralUtil::Zero(expanded_filter_shape.element_type())));
      auto zero_filter =
          add(HloInstruction::CreateBroadcast(expanded_filter_shape, zero, {}));
      auto new_filter = add(HloInstruction::CreateTernary(
          expanded_filter_shape, HloOpcode::kSelect, filter_mask,
          expanded_filter, zero_filter));
      auto new_convolution = HloInstruction::CreateConvolve(
          convolution->shape(), convolution->mutable_operand(0), new_filter,
          1, 1,
          convolution->window(), dim_numbers, convolution->precision_config());
      return computation_->ReplaceWithNewInstruction(
          convolution, std::move(new_convolution));
    }
    std::vector<int64_t> new_filter_dimension;
    new_filter_dimension.reserve(filter->shape().rank() + 1);
    const int64_t depthwise_multiplier =
        filter->shape().dimensions(kernel_output_feature_dim) / group_count;
    for (int64_t i = 0; i < filter->shape().rank(); ++i) {
      if (i == kernel_output_feature_dim) {
        new_filter_dimension.push_back(group_count);
        new_filter_dimension.push_back(depthwise_multiplier);
      } else {
        new_filter_dimension.push_back(filter->shape().dimensions(i));
      }
    }
    if (kernel_input_feature_dim > kernel_output_feature_dim) {
      dim_numbers.set_kernel_input_feature_dimension(kernel_input_feature_dim +
                                                     1);
    }
    for (auto& dim : *dim_numbers.mutable_kernel_spatial_dimensions()) {
      if (dim > kernel_output_feature_dim) {
        ++dim;
      }
    }
    dim_numbers.add_kernel_spatial_dimensions(kernel_output_feature_dim + 1);
    HloInstruction* new_filter =
        computation_->AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(filter->shape().element_type(),
                                 new_filter_dimension),
            filter));
    auto new_activation_shape = convolution->operand(0)->shape();
    dim_numbers.add_input_spatial_dimensions(new_activation_shape.rank());
    ShapeUtil::AppendMajorDimension(1, &new_activation_shape);
    HloInstruction* new_activation =
        computation_->AddInstruction(HloInstruction::CreateReshape(
            new_activation_shape, convolution->mutable_operand(0)));
    auto new_window = convolution->window();
    auto new_dim = new_window.add_dimensions();
    new_dim->set_size(depthwise_multiplier);
    new_dim->set_window_reversal(true);
    new_dim->set_padding_low(depthwise_multiplier - 1);
    new_dim->set_padding_high(depthwise_multiplier - 1);
    new_dim->set_stride(1);
    new_dim->set_window_dilation(1);
    new_dim->set_base_dilation(1);
    std::vector<int64_t> new_output_dimension;
    new_output_dimension.reserve(convolution->shape().rank() + 1);
    for (int64_t i = 0; i < convolution->shape().rank(); ++i) {
      if (i == dim_numbers.output_feature_dimension()) {
        new_output_dimension.push_back(group_count);
        new_output_dimension.push_back(depthwise_multiplier);
      } else {
        new_output_dimension.push_back(convolution->shape().dimensions(i));
      }
    }
    if (dim_numbers.output_batch_dimension() >
        dim_numbers.output_feature_dimension()) {
      dim_numbers.set_output_batch_dimension(
          dim_numbers.output_batch_dimension() + 1);
    }
    for (auto& dim : *dim_numbers.mutable_output_spatial_dimensions()) {
      if (dim > dim_numbers.output_feature_dimension()) {
        ++dim;
      }
    }
    dim_numbers.add_output_spatial_dimensions(
        dim_numbers.output_feature_dimension() + 1);
    auto new_convolution_output_shape = ShapeUtil::MakeShape(
        convolution->shape().element_type(), new_output_dimension);
    HloInstruction* new_convolution =
        computation_->AddInstruction(HloInstruction::CreateConvolve(
            new_convolution_output_shape, new_activation, new_filter,
            group_count, 1,
            new_window, dim_numbers, convolution->precision_config()));
    return computation_->ReplaceWithNewInstruction(
        convolution,
        HloInstruction::CreateReshape(convolution->shape(), new_convolution));
  }
  HloInstruction* activation = convolution->mutable_operand(0);
  std::vector<int64_t> input_sizes(activation->shape().dimensions().begin(),
                                   activation->shape().dimensions().end());
  const int64_t input_feature_dimension = dim_numbers.input_feature_dimension();
  input_sizes[input_feature_dimension] /= group_count;
  input_sizes.insert(input_sizes.begin() + input_feature_dimension,
                     group_count);
  activation = MakeReshapeHlo(input_sizes, activation).value();
  for (auto& d : *dim_numbers.mutable_input_spatial_dimensions()) {
    if (d > input_feature_dimension) {
      ++d;
    }
  }
  dim_numbers.add_input_spatial_dimensions(input_feature_dimension);
  dim_numbers.set_input_feature_dimension(input_feature_dimension + 1);
  if (dim_numbers.input_batch_dimension() > input_feature_dimension) {
    dim_numbers.set_input_batch_dimension(dim_numbers.input_batch_dimension() +
                                          1);
  }
  std::vector<int64_t> kernel_sizes(filter->shape().dimensions().begin(),
                                    filter->shape().dimensions().end());
  const int64_t kernel_output_feature_dimension =
      dim_numbers.kernel_output_feature_dimension();
  kernel_sizes[kernel_output_feature_dimension] /= group_count;
  kernel_sizes.insert(kernel_sizes.begin() + kernel_output_feature_dimension,
                      group_count);
  filter = MakeReshapeHlo(kernel_sizes, filter).value();
  for (auto& d : *dim_numbers.mutable_kernel_spatial_dimensions()) {
    if (d > kernel_output_feature_dimension) {
      ++d;
    }
  }
  dim_numbers.add_kernel_spatial_dimensions(kernel_output_feature_dimension);
  dim_numbers.set_kernel_output_feature_dimension(
      kernel_output_feature_dimension + 1);
  if (dim_numbers.kernel_input_feature_dimension() >
      kernel_output_feature_dimension) {
    dim_numbers.set_kernel_input_feature_dimension(
        dim_numbers.kernel_input_feature_dimension() + 1);
  }
  const int64_t output_feature_dimension =
      dim_numbers.output_feature_dimension();
  for (auto& d : *dim_numbers.mutable_output_spatial_dimensions()) {
    if (d > output_feature_dimension) {
      ++d;
    }
  }
  dim_numbers.add_output_spatial_dimensions(output_feature_dimension);
  dim_numbers.set_output_feature_dimension(output_feature_dimension + 1);
  if (dim_numbers.output_batch_dimension() > output_feature_dimension) {
    dim_numbers.set_output_batch_dimension(
        dim_numbers.output_batch_dimension() + 1);
  }
  Window window = convolution->window();
  auto window_dim = window.add_dimensions();
  window_dim->set_base_dilation(group_count);
  window_dim->set_size(group_count);
  window_dim->set_stride(group_count - 1);
  window_dim->set_padding_low(0);
  window_dim->set_padding_high(0);
  window_dim->set_window_reversal(false);
  window_dim->set_window_dilation(1);
  HloInstruction* new_convolution =
      MakeConvolveHlo(
          activation, filter, 1,
          1, window, dim_numbers,
          convolution->precision_config(),
          convolution->shape().element_type())
          .value();
  convolution->SetupDerivedInstruction(new_convolution);
  changed_ = true;
  return computation_->ReplaceInstruction(
      convolution,
      MakeReshapeHlo(convolution->shape(), new_convolution).value());
}
}  
absl::StatusOr<bool> ConvolutionGroupConverter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      2, "ConvolutionGroupConverter::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    if (ConvolutionVisitor::Run(comp, should_expand_, is_cost_viable_,
                                convert_batch_groups_only_,
                                filter_expansion_)) {
      changed = true;
    }
  }
  XLA_VLOG_LINES(
      2, "ConvolutionGroupConverter::Run(), after:\n" + module->ToString());
  return changed;
}
}  