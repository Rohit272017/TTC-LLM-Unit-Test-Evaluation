#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/reshape_op.h"
namespace tensorflow {
class QuantizedReshapeOp : public ReshapeOp {
 public:
  explicit QuantizedReshapeOp(OpKernelConstruction* c) : ReshapeOp(c) {}
  void Compute(OpKernelContext* ctx) override {
    ReshapeOp::Compute(ctx);
    if (!ctx->status().ok()) {
      return;
    }
    const auto& input_min_float_tensor = ctx->input(2);
    const auto& input_min_float_shape = input_min_float_tensor.shape();
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(input_min_float_shape) ||
                    (TensorShapeUtils::IsVector(input_min_float_shape) &&
                     (input_min_float_shape.dim_size(0) == 1)),
                errors::InvalidArgument(
                    "input_min must be a scalar or a vector of 1 element"));
    const float input_min_float = input_min_float_tensor.flat<float>()(0);
    const auto& input_max_float_tensor = ctx->input(3);
    const auto& input_max_float_shape = input_max_float_tensor.shape();
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(input_max_float_shape) ||
                    (TensorShapeUtils::IsVector(input_max_float_shape) &&
                     (input_max_float_shape.dim_size(0) == 1)),
                errors::InvalidArgument(
                    "input_max must be a scalar or a vector of 1 element"));
    const float input_max_float = input_max_float_tensor.flat<float>()(0);
    Tensor* output_min = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({}), &output_min));
    output_min->flat<float>()(0) = input_min_float;
    Tensor* output_max = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({}), &output_max));
    output_max->flat<float>()(0) = input_max_float;
  }
};
#define REGISTER_CPU_KERNEL(type)                         \
  REGISTER_KERNEL_BUILDER(Name("QuantizedReshape")        \
                              .Device(DEVICE_CPU)         \
                              .HostMemory("shape")        \
                              .TypeConstraint<type>("T"), \
                          QuantizedReshapeOp)
REGISTER_CPU_KERNEL(::tensorflow::quint8);
REGISTER_CPU_KERNEL(::tensorflow::qint32);
#undef REGISTER_CPU_KERNEL
}  