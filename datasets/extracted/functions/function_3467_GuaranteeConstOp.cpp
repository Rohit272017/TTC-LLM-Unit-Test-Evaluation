#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
namespace tensorflow {
namespace {
class GuaranteeConstOp : public OpKernel {
 public:
  explicit GuaranteeConstOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    const DataType input_dtype = ctx->input_dtype(0);
    OP_REQUIRES(ctx, input_dtype != DT_RESOURCE,
                errors::InvalidArgument(
                    "Input tensor cannot be a resource variable handle."));
    const Tensor& input_tensor = ctx->input(0);
    Tensor* output = nullptr;
    if (!ctx->forward_input_to_output_with_shape(0, 0, input_tensor.shape(),
                                                 &output)) {
      ctx->set_output(0, input_tensor);
    }
  }
  bool IsExpensive() override { return false; }
};
REGISTER_KERNEL_BUILDER(Name("GuaranteeConst").Device(DEVICE_CPU),
                        GuaranteeConstOp);
}  
}  