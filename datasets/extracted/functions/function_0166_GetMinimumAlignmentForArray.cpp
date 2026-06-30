#include "xla/service/cpu/ir_emission_utils.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout_util.h"
#include "xla/service/cpu/cpu_runtime.h"
#include "xla/shape_util.h"
#include "xla/window_util.h"
namespace xla {
namespace cpu {
int64_t GetMinimumAlignmentForArray(
    const Shape& shape, const TargetMachineFeatures& target_machine_features) {
  CHECK(LayoutUtil::IsDenseArray(shape));
  int64_t allocation_size_bytes =
      ShapeUtil::ElementsIn(shape) *
      ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  return target_machine_features.minimum_alignment_for_allocation(
      allocation_size_bytes);
}
bool PotentiallyImplementedAsEigenConvolution(
    const HloInstruction& convolution,
    const TargetMachineFeatures& target_machine_features) {
  const Shape& input_shape = convolution.operand(0)->shape();
  const Shape& kernel_shape = convolution.operand(1)->shape();
  const Shape& output_shape = convolution.shape();
  auto is_aligned = [&](const Shape& shape) {
    return GetMinimumAlignmentForArray(shape, target_machine_features) >=
           TargetMachineFeatures::kEigenExpectedTensorAlignment;
  };
  if (!is_aligned(input_shape) || !is_aligned(kernel_shape) ||
      !is_aligned(output_shape)) {
    return false;
  }
  if (ShapeUtil::IsZeroElementArray(input_shape) ||
      ShapeUtil::IsZeroElementArray(kernel_shape)) {
    return false;
  }
  CHECK(
      ShapeUtil::SameElementTypeIgnoringFpPrecision(input_shape, kernel_shape));
  PrimitiveType primitive_type = input_shape.element_type();
  if (primitive_type != F16 && primitive_type != F32) {
    return false;
  }
  if (window_util::HasWindowReversal(convolution.window())) {
    return false;
  }
  const ConvolutionDimensionNumbers& dnums =
      convolution.convolution_dimension_numbers();
  const int64_t num_spatial_dims = dnums.output_spatial_dimensions_size();
  if (num_spatial_dims < 1 || num_spatial_dims > 3) {
    return false;
  }
  for (int64_t i = 0; i < num_spatial_dims; ++i) {
    if (dnums.input_spatial_dimensions(i) != i + 1) {
      return false;
    }
    if (dnums.kernel_spatial_dimensions(i) != i) {
      return false;
    }
    if (dnums.output_spatial_dimensions(i) != i + 1) {
      return false;
    }
  }
  return dnums.input_batch_dimension() == 0 &&
         dnums.input_feature_dimension() == input_shape.dimensions_size() - 1 &&
         dnums.output_batch_dimension() == 0 &&
         dnums.output_feature_dimension() ==
             output_shape.dimensions_size() - 1 &&
         dnums.kernel_input_feature_dimension() ==
             kernel_shape.dimensions_size() - 2 &&
         dnums.kernel_output_feature_dimension() ==
             kernel_shape.dimensions_size() - 1;
}
}  
}  