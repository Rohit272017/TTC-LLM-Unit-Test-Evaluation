#define EIGEN_USE_THREADS
#include <math.h>
#include "unsupported/Eigen/CXX11/Tensor"  
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/errors.h"
namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
void CalculateUsedRange(const Tensor& input, qint32* used_min_quantized,
                        qint32* used_max_quantized) {
  auto input_array = input.flat<qint32>();
  Eigen::Tensor<qint32, 0, Eigen::RowMajor> min = input_array.minimum();
  Eigen::Tensor<qint32, 0, Eigen::RowMajor> max = input_array.maximum();
  *used_min_quantized = min();
  *used_max_quantized = max();
}
class RequantizationRangeOp : public OpKernel {
 public:
  explicit RequantizationRangeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    OP_REQUIRES(ctx, ctx->input(1).NumElements() > 0,
                errors::InvalidArgument("Input min must not be empty."));
    OP_REQUIRES(ctx, ctx->input(2).NumElements() > 0,
                errors::InvalidArgument("Input max must not be empty."));
    const float input_min_float = ctx->input(1).flat<float>()(0);
    const float input_max_float = ctx->input(2).flat<float>()(0);
    Tensor* output_min = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output_min));
    Tensor* output_max = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({}), &output_max));
    qint32 used_min_quantized;
    qint32 used_max_quantized;
    CalculateUsedRange(input, &used_min_quantized, &used_max_quantized);
    const float used_min_float = std::min(
        0.0f,
        QuantizedToFloat(used_min_quantized, input_min_float, input_max_float));
    const float used_max_float =
        QuantizedToFloat(used_max_quantized, input_min_float, input_max_float);
    output_min->flat<float>().setConstant(used_min_float);
    output_max->flat<float>().setConstant(used_max_float);
  }
};
REGISTER_KERNEL_BUILDER(Name("RequantizationRange")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint32>("Tinput"),
                        RequantizationRangeOp);
}  