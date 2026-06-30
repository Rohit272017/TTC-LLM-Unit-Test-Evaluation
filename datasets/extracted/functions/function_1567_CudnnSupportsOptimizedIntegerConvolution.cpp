#include "xla/service/gpu/cudnn_support_utils.h"
#include <cstdint>
#include <vector>
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/window_util.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
namespace xla {
namespace gpu {
absl::StatusOr<bool> CudnnSupportsOptimizedIntegerConvolution(
    const se::CudaComputeCapability& compute_capability,
    HloCustomCallInstruction& conv, int vector_size) {
  TF_ASSIGN_OR_RETURN(auto kind, GetCudnnConvKind(&conv));
  const Shape& input_shape = conv.operand(0)->shape();
  const Shape& kernel_shape = conv.operand(1)->shape();
  const Shape& result_shape = conv.shape().tuple_shapes(0);
  const auto& dnums = conv.convolution_dimension_numbers();
  if (vector_size != 4 && vector_size != 32) {
    VLOG(3) << "Unsupported vector size for integer convolution: "
            << vector_size;
    return false;
  }
  if ((vector_size == 32 && !compute_capability.IsAtLeast(7, 5)) ||
      !compute_capability.IsAtLeast(6, 1)) {
    VLOG(3) << "Compute capability " << compute_capability.ToString()
            << " is not sufficent for int8x" << vector_size
            << " vectorization.";
    return false;
  }
  if (kind != CudnnConvKind::kForward &&
      kind != CudnnConvKind::kForwardActivation) {
    VLOG(3) << "Convolution kind is not forward or foward-activation: "
            << conv.ToString();
    return false;
  }
  if (!primitive_util::IsIntegralType(input_shape.element_type()) ||
      !primitive_util::IsIntegralType(kernel_shape.element_type())) {
    VLOG(3) << "Convolution does not accept integer inputs/weights: "
            << conv.ToString();
    return false;
  }
  if (dnums.input_spatial_dimensions().size() != 2 ||
      dnums.kernel_spatial_dimensions().size() != 2 ||
      dnums.output_spatial_dimensions().size() != 2) {
    VLOG(3) << "Convolution is not 2D: " << conv.ToString();
    return false;
  }
  if (vector_size == 32 &&
      !primitive_util::IsIntegralType(result_shape.element_type())) {
    VLOG(3) << "int8x32 convolutions only support integer output: "
            << conv.ToString();
    return false;
  }
  if (vector_size == 32) {
    int64_t W = input_shape.dimensions(dnums.input_spatial_dimensions()[0]);
    int64_t H = input_shape.dimensions(dnums.input_spatial_dimensions()[1]);
    int64_t R = kernel_shape.dimensions(dnums.kernel_spatial_dimensions()[0]);
    int64_t S = kernel_shape.dimensions(dnums.kernel_spatial_dimensions()[1]);
    const int64_t dilationW = conv.window().dimensions()[0].base_dilation();
    const int64_t dilationH = conv.window().dimensions()[1].base_dilation();
    if ((W <= (R - 1) * dilationW) || (H <= (S - 1) * dilationH)) {
      VLOG(3) << "Conv spatial filter/input dimensions are too small for "
                 "vecotrized int8x32 convolution: "
              << conv.ToString();
      return false;
    }
  }
  if (window_util::HasDilation(conv.window())) {
    VLOG(3) << "Vectorized integer convolutions do not support dilation: "
            << conv.ToString();
    return false;
  }
  return true;
}
absl::StatusOr<CudnnReorderTransposeConfig>
CudnnInferTransposeForFilterReordering(
    const Shape& shape, const ConvolutionDimensionNumbers& dimension_numbers) {
  if (shape.rank() != 4 && shape.rank() != 5) {
    return Internal("Filter shape has unexpected rank.");
  }
  const int64_t dO = dimension_numbers.kernel_output_feature_dimension();
  const int64_t dI = dimension_numbers.kernel_input_feature_dimension();
  const int64_t dH = dimension_numbers.kernel_spatial_dimensions().at(0);
  const int64_t dW = dimension_numbers.kernel_spatial_dimensions().at(1);
  bool revectorize = shape.rank() == 5;
  const int64_t dZ = revectorize ? 10 - dO - dI - dH - dW : -1;
  const int64_t vsize = revectorize ? shape.dimensions(dZ) : 1;
  if (shape.dimensions(dO) % 32 != 0 ||
      shape.dimensions(dI) % (32 / vsize) != 0 ||
      (revectorize && vsize != 4 && vsize != 32)) {
    return Internal("Filter shape is not vectorizable.");
  }
  std::vector<int64_t> output = {
      shape.dimensions(dO), shape.dimensions(dI) / (32 / vsize),
      shape.dimensions(dH), shape.dimensions(dW), 32};
  Shape output_shape = ShapeUtil::MakeShape(shape.element_type(), output);
  auto calc_index = [&](int dim) {
    bool split_v = vsize == 32;
    return (revectorize
                ? (dI < dim ? 2 - split_v : 0) + (dZ < dim ? 1 + split_v : 0)
                : (dI < dim ? 3 : 0)) +
           (dO < dim ? 3 : 0) + (dH < dim) + (dW < dim);
  };
  int idx_O = calc_index(dO);
  int idx_I = calc_index(dI);
  int idx_H = calc_index(dH);
  int idx_W = calc_index(dW);
  int idx_Y = vsize == 32 ? calc_index(dZ) : idx_I + 1;
  int idx_Z = vsize == 4 ? calc_index(dZ) : vsize == 32 ? idx_Y + 1 : idx_I + 2;
  std::vector<int64_t> dims(8);
  dims[idx_O] = shape.dimensions(dO) / 8;
  dims[idx_O + 1] = 4;
  dims[idx_O + 2] = 2;
  dims[idx_I] = shape.dimensions(dI) / (32 / vsize);
  dims[idx_Y] = 8;
  dims[idx_Z] = 4;
  dims[idx_H] = shape.dimensions(dH);
  dims[idx_W] = shape.dimensions(dW);
  Shape split_shape = ShapeUtil::MakeShape(shape.element_type(), dims);
  std::vector<int64_t> permutation = {idx_I,     idx_H, idx_W,     idx_O,
                                      idx_O + 2, idx_Y, idx_O + 1, idx_Z};
  return CudnnReorderTransposeConfig{split_shape, output_shape, permutation};
}
absl::StatusOr<CudnnReorderTransposeConfig>
CudnnInferTransposeForBiasReordering(const Shape& shape) {
  if (shape.rank() != 1) {
    return Internal("Bias shape has unexpected rank.");
  }
  if (shape.dimensions(0) % 32 != 0) {
    return Internal("Bias shape is not vectorizable.");
  }
  std::vector<int64_t> dims = {shape.dimensions(0) / 32, 4, 2, 4};
  Shape split_shape = ShapeUtil::MakeShape(shape.element_type(), dims);
  std::vector<int64_t> permutation = {0, 2, 1, 3};
  return CudnnReorderTransposeConfig{split_shape, shape, permutation};
}
bool IsWorkspaceAllocationRoot(const HloInstruction& root) {
  return root.IsRoot() && root.opcode() == HloOpcode::kTuple &&
         root.operand_count() == 2 &&
         root.operand(1)->IsCustomCall(kWorkspaceAllocationCustomCallTarget) &&
         root.operand(1)->operand_count() == 0;
}
}  
}  