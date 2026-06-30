#include <vector>
#include "tensorflow/compiler/tf2xla/lib/broadcast.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/xla_builder.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
namespace tensorflow {
namespace {
class BroadcastToOp : public XlaOpKernel {
 public:
  explicit BroadcastToOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {}
  void Compile(XlaOpKernelContext* context) override {
    TensorShape output_shape;
    OP_REQUIRES_OK(context,
                   context->ConstantInputAsShape(
                       1, &output_shape, xla::ValueInferenceMode::kUpperBound));
    auto output_status_or =
        BroadcastTo(context->Input(0), output_shape.dim_sizes());
    OP_REQUIRES_OK(context, output_status_or.status());
    auto output = output_status_or.value();
    std::vector<bool> dynamic_dims;
    OP_REQUIRES_OK(
        context, context->ResolveInputDynamismIntoPredVector(1, &dynamic_dims));
    for (int64_t dim = 0; dim < dynamic_dims.size(); ++dim) {
      if (dynamic_dims[dim]) {
        output = xla::SetDimensionSize(
            output,
            xla::Reshape(xla::Slice(context->Input(1), {dim}, {dim + 1}, {1}),
                         {}),
            dim);
      }
    }
    context->SetOutput(0, output);
  }
};
REGISTER_XLA_OP(Name("BroadcastTo").CompileTimeConstantInput("shape"),
                BroadcastToOp);
}  
}  