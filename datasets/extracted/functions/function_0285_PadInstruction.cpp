#include "xla/service/gpu/transforms/cudnn_pad_for_convolutions.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "absl/functional/bind_front.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/cudnn_support_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace gpu {
static HloInstruction* PadInstruction(HloInstruction* instr,
                                      const Shape& new_shape) {
  HloComputation* comp = instr->parent();
  const Shape& shape = instr->shape();
  PaddingConfig pad_config = MakeNoPaddingConfig(shape.rank());
  bool added_padding = false;
  for (int64_t dim = 0; dim < shape.rank(); ++dim) {
    if (shape.dimensions(dim) == new_shape.dimensions(dim)) {
      continue;
    }
    CHECK_GT(new_shape.dimensions(dim), shape.dimensions(dim));
    pad_config.mutable_dimensions(dim)->set_edge_padding_high(
        new_shape.dimensions(dim) - shape.dimensions(dim));
    added_padding = true;
  }
  if (!added_padding) {
    return instr;
  }
  auto* zero = comp->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(shape.element_type())));
  return comp->AddInstruction(
      HloInstruction::CreatePad(new_shape, instr, zero, pad_config),
      &instr->metadata());
}
static absl::Status PadConv(HloCustomCallInstruction* conv,
                            absl::Span<const Shape> new_input_shapes,
                            const Shape& new_result_shape) {
  CHECK_EQ(0, conv->shape().tuple_shapes(1).dimensions(0))
      << "conv must use 0 scratch bytes, i.e. this pass must be run "
         "before CudnnConvAlgorithmPicker.";
  std::vector<HloInstruction*> new_operands;
  new_operands.reserve(conv->operand_count());
  for (int i = 0; i < conv->operand_count(); ++i) {
    new_operands.push_back(
        PadInstruction(conv->mutable_operand(i), new_input_shapes[i]));
  }
  const Shape& result_shape = conv->shape().tuple_shapes(0);
  bool changed = false;
  for (int i = 0; i < conv->operand_count(); ++i) {
    changed |= (new_operands[i] != conv->mutable_operand(i));
  }
  CHECK(changed) << "We should have had to pad at least one input operand.";
  auto add = [&](std::unique_ptr<HloInstruction> new_instr) {
    return conv->parent()->AddInstruction(std::move(new_instr));
  };
  Shape new_conv_shape = ShapeUtil::MakeTupleShape(
      {new_result_shape, ShapeUtil::MakeShape(U8, {0})});
  auto* new_conv =
      add(conv->CloneWithNewOperands(new_conv_shape, new_operands));
  new_conv->SetAndSanitizeName(conv->name());
  VLOG(2) << "Padded features of " << conv->ToString() << ", replaced with "
          << new_conv->ToString();
  if (!ShapeUtil::Equal(result_shape, new_result_shape)) {
    std::vector<int64_t> start_indices(result_shape.dimensions_size(), 0);
    std::vector<int64_t> end_indices(result_shape.dimensions().begin(),
                                     result_shape.dimensions().end());
    std::vector<int64_t> strides(result_shape.dimensions_size(), 1);
    auto* new_conv_result = add(
        HloInstruction::CreateGetTupleElement(new_result_shape, new_conv, 0));
    auto* empty_temp_buffer =
        add(HloInstruction::CreateConstant(LiteralUtil::CreateR1<uint8_t>({})));
    auto* sliced_result = add(HloInstruction::CreateSlice(
        result_shape, new_conv_result, start_indices, end_indices, strides));
    new_conv =
        add(HloInstruction::CreateTuple({sliced_result, empty_temp_buffer}));
  }
  return conv->parent()->ReplaceInstruction(conv, new_conv);
}
static std::vector<HloCustomCallInstruction*> GetRelevantConvs(
    HloComputation* comp) {
  std::vector<HloCustomCallInstruction*> convs;
  for (HloInstruction* instr : comp->instructions()) {
    if (IsCustomCallToDnnConvolution(*instr)) {
      convs.push_back(Cast<HloCustomCallInstruction>(instr));
    }
  }
  return convs;
}
static absl::StatusOr<bool> ResolveAndPad(
    HloCustomCallInstruction* conv,
    std::function<absl::StatusOr<bool>(HloCustomCallInstruction* conv,
                                       std::vector<Shape>* new_input_shapes,
                                       Shape* new_result_shape)>
        resolve_pad_shapes) {
  std::vector<Shape> new_input_shapes;
  Shape new_result_shape;
  TF_ASSIGN_OR_RETURN(bool result, resolve_pad_shapes(conv, &new_input_shapes,
                                                      &new_result_shape));
  if (result) {
    TF_RETURN_IF_ERROR(PadConv(conv, new_input_shapes, new_result_shape));
    return true;
  }
  return false;
}
static absl::StatusOr<bool> TryResolvePaddedShapesForTensorCore(
    HloCustomCallInstruction* conv, std::vector<Shape>* new_input_shapes_ptr,
    Shape* new_result_shape_ptr) {
  TF_ASSIGN_OR_RETURN(auto kind, GetCudnnConvKind(conv));
  const auto& dnums = conv->convolution_dimension_numbers();
  auto* lhs = conv->mutable_operand(0);
  auto* rhs = conv->mutable_operand(1);
  const Shape& result_shape = conv->shape().tuple_shapes(0);
  if (result_shape.element_type() != PrimitiveType::F16) {
    return false;
  }
  if (conv->feature_group_count() > 1 || conv->batch_group_count() > 1) {
    VLOG(2) << "Do not pad grouped convolution.";
    return false;
  }
  if (kind == CudnnConvKind::kForwardActivation) {
    return false;
  }
  Shape new_lhs_shape = lhs->shape();
  Shape new_rhs_shape = rhs->shape();
  Shape& new_result_shape = *new_result_shape_ptr;
  new_result_shape = conv->shape().tuple_shapes(0);
  Shape* new_input_shape;
  Shape* new_filter_shape;
  Shape* new_output_shape;
  std::tie(new_input_shape, new_filter_shape, new_output_shape) = [&] {
    switch (kind) {
      case CudnnConvKind::kForward:
      case CudnnConvKind::kForwardActivation:
      case CudnnConvKind::kForwardGraph:
        return std::make_tuple(&new_lhs_shape, &new_rhs_shape,
                               &new_result_shape);
      case CudnnConvKind::kBackwardInput:
        return std::make_tuple(&new_result_shape, &new_rhs_shape,
                               &new_lhs_shape);
      case CudnnConvKind::kBackwardFilter:
        return std::make_tuple(&new_lhs_shape, &new_result_shape,
                               &new_rhs_shape);
    }
  }();
  auto input_features =
      new_input_shape->dimensions(dnums.input_feature_dimension());
  auto output_features =
      new_output_shape->dimensions(dnums.output_feature_dimension());
  if (input_features == 3 && (output_features == 32 || output_features == 64)) {
    new_input_shape->set_dimensions(dnums.input_feature_dimension(), 4);
    new_filter_shape->set_dimensions(dnums.kernel_input_feature_dimension(), 4);
  } else {
    auto pad_dim = [](Shape* s, int64_t dim) {
      s->set_dimensions(dim, RoundUpTo<int64_t>(s->dimensions(dim), 8));
    };
    pad_dim(new_input_shape, dnums.input_feature_dimension());
    pad_dim(new_filter_shape, dnums.kernel_input_feature_dimension());
    pad_dim(new_filter_shape, dnums.kernel_output_feature_dimension());
    pad_dim(new_output_shape, dnums.output_feature_dimension());
    static constexpr double kMaxBytesTouchedBound = 1.35;
    auto check_size_increase = [&](const Shape& old_shape,
                                   const Shape& new_shape) {
      int64_t old_bytes = ShapeUtil::ByteSizeOf(old_shape);
      int64_t new_bytes = ShapeUtil::ByteSizeOf(new_shape);
      if (new_bytes <= old_bytes * kMaxBytesTouchedBound) {
        return true;
      }
      VLOG(3)
          << "Not padding convolution; doing so would change input / result "
             "shape from "
          << ShapeUtil::HumanString(old_shape) << " to "
          << ShapeUtil::HumanString(new_shape) << ", a size increase of "
          << new_bytes / static_cast<double>(old_bytes) << "x > "
          << kMaxBytesTouchedBound << "x: " << conv->ToString();
      return false;
    };
    if (!check_size_increase(lhs->shape(), new_lhs_shape) ||
        !check_size_increase(rhs->shape(), new_rhs_shape) ||
        !check_size_increase(result_shape, new_result_shape)) {
      return false;
    }
  }
  if (ShapeUtil::Equal(lhs->shape(), new_lhs_shape) &&
      ShapeUtil::Equal(rhs->shape(), new_rhs_shape)) {
    VLOG(3) << "No need to pad features of " << conv->ToString();
    return false;
  }
  new_input_shapes_ptr->push_back(new_lhs_shape);
  new_input_shapes_ptr->push_back(new_rhs_shape);
  return true;
}
absl::StatusOr<bool> TryResolvePaddedShapesForIntegerConvolution(
    int pad_to, const se::CudaComputeCapability& compute_capability,
    HloCustomCallInstruction* conv, std::vector<Shape>* new_input_shapes_ptr,
    Shape* new_result_shape_ptr) {
  TF_ASSIGN_OR_RETURN(auto kind, GetCudnnConvKind(conv));
  const Shape& input_shape = conv->operand(0)->shape();
  const Shape& kernel_shape = conv->operand(1)->shape();
  const Shape& result_shape = conv->shape().tuple_shapes(0);
  if (!primitive_util::IsIntegralType(input_shape.element_type())) {
    return false;
  }
  if (kind != CudnnConvKind::kForward &&
      kind != CudnnConvKind::kForwardActivation) {
    return false;
  }
  const auto& dnums = conv->convolution_dimension_numbers();
  std::vector<Shape>& new_input_shapes = *new_input_shapes_ptr;
  for (auto operand : conv->operands()) {
    new_input_shapes.push_back(operand->shape());
  }
  Shape& new_result_shape = *new_result_shape_ptr;
  new_result_shape = conv->shape().tuple_shapes(0);
  std::optional<int64_t> input_vect_dim;
  std::optional<int64_t> kernel_vect_dim;
  std::optional<int64_t> result_vect_dim;
  std::tie(input_vect_dim, kernel_vect_dim, result_vect_dim) =
      FindVectorizedFeatureDims(dnums, input_shape, kernel_shape, result_shape);
  int64_t input_vect_size =
      input_vect_dim.has_value() ? input_shape.dimensions(*input_vect_dim) : 1;
  int64_t kernel_vect_size = kernel_vect_dim.has_value()
                                 ? kernel_shape.dimensions(*kernel_vect_dim)
                                 : 1;
  int64_t result_vect_size = result_vect_dim.has_value()
                                 ? result_shape.dimensions(*result_vect_dim)
                                 : 1;
  if (pad_to % input_vect_size != 0 || pad_to % kernel_vect_size != 0 ||
      pad_to % result_vect_size != 0) {
    return false;
  }
  TF_ASSIGN_OR_RETURN(bool cudnn_supports,
                      CudnnSupportsOptimizedIntegerConvolution(
                          compute_capability, *conv, pad_to));
  if (!cudnn_supports) {
    return false;
  }
  {
    auto pad_dim = [&](Shape* s, int64_t dim, int64_t cur_vect_size) {
      CHECK_EQ(pad_to % cur_vect_size, 0);
      s->set_dimensions(
          dim, RoundUpTo<int64_t>(s->dimensions(dim), pad_to / cur_vect_size));
    };
    switch (kind) {
      case CudnnConvKind::kForward:
        CHECK_EQ(new_input_shapes.size(), 2);
        pad_dim(new_input_shapes.data(), dnums.input_feature_dimension(),
                input_vect_size);
        pad_dim(&new_input_shapes[1], dnums.kernel_input_feature_dimension(),
                kernel_vect_size);
        pad_dim(&new_input_shapes[1], dnums.kernel_output_feature_dimension(),
                1);
        pad_dim(&new_result_shape, dnums.output_feature_dimension(),
                result_vect_size);
        break;
      case CudnnConvKind::kForwardActivation:
        CHECK(new_input_shapes.size() == 3 || new_input_shapes.size() == 4);
        pad_dim(new_input_shapes.data(), dnums.input_feature_dimension(),
                input_vect_size);
        pad_dim(&new_input_shapes[1], dnums.kernel_input_feature_dimension(),
                kernel_vect_size);
        pad_dim(&new_input_shapes[1], dnums.kernel_output_feature_dimension(),
                1);
        pad_dim(&new_input_shapes[2], 0, 1);
        if (new_input_shapes.size() == 4) {
          pad_dim(&new_input_shapes[3], dnums.output_feature_dimension(),
                  result_vect_size);
        }
        pad_dim(&new_result_shape, dnums.output_feature_dimension(),
                result_vect_size);
        break;
      default:
        CHECK(false);
    }
    static constexpr double kMaxBytesTouchedBound = 2;
    auto check_size_increase = [&](const Shape& old_shape,
                                   const Shape& new_shape) {
      int64_t old_bytes = ShapeUtil::ByteSizeOf(old_shape);
      int64_t new_bytes = ShapeUtil::ByteSizeOf(new_shape);
      if (new_bytes < old_bytes * kMaxBytesTouchedBound) {
        return true;
      }
      VLOG(3)
          << "Not padding convolution; doing so would change input / result "
             "shape from "
          << ShapeUtil::HumanString(old_shape) << " to "
          << ShapeUtil::HumanString(new_shape) << ", a size increase of "
          << new_bytes / static_cast<double>(old_bytes)
          << "x >= " << kMaxBytesTouchedBound << "x: " << conv->ToString();
      return false;
    };
    if (!check_size_increase(conv->operand(0)->shape(), new_input_shapes[0]) ||
        !check_size_increase(result_shape, new_result_shape)) {
      return false;
    }
  }
  bool changed = false;
  for (int64_t i = 0; i < conv->operand_count(); ++i) {
    changed |=
        !ShapeUtil::Equal(conv->operand(i)->shape(), new_input_shapes[i]);
  }
  if (!changed) {
    VLOG(3) << "No need to pad features of " << conv->ToString();
  }
  return changed;
}
absl::StatusOr<bool> CudnnPadForConvolutions::Run(
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
            ResolveAndPad(conv, absl::bind_front(
                                    TryResolvePaddedShapesForIntegerConvolution,
                                    32, compute_capability_)));
      }
      if (!local_changed) {
        TF_ASSIGN_OR_RETURN(
            local_changed,
            ResolveAndPad(conv, absl::bind_front(
                                    TryResolvePaddedShapesForIntegerConvolution,
                                    4, compute_capability_)));
      }
      changed |= local_changed;
    }
    if (compute_capability_.IsAtLeast(se::CudaComputeCapability::VOLTA)) {
      for (HloCustomCallInstruction* conv : GetRelevantConvs(comp)) {
        TF_ASSIGN_OR_RETURN(
            bool local_changed,
            ResolveAndPad(conv, TryResolvePaddedShapesForTensorCore));
        changed |= local_changed;
      }
    }
  }
  return changed;
}
}  
}  