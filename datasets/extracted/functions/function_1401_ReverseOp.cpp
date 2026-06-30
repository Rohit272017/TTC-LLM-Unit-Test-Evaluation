#include <vector>
#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"
namespace tensorflow {
namespace {
class ReverseOp : public XlaOpKernel {
 public:
  explicit ReverseOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape x_shape = ctx->InputShape(0);
    const TensorShape revd_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(revd_shape),
                errors::InvalidArgument("axes must be a vector, not shape ",
                                        revd_shape.DebugString()));
    OP_REQUIRES(ctx, revd_shape.num_elements() == x_shape.dims(),
                errors::InvalidArgument("axes ", revd_shape.DebugString(),
                                        " must have same number of elements as"
                                        " than input tensor has dimensions ",
                                        x_shape.DebugString(), "."));
    if (revd_shape.num_elements() == 0) {
      ctx->SetOutput(0, ctx->Input(0));
      return;
    }
    xla::Literal lax;
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(1, &lax));
    std::vector<int64_t> dimensions;
    for (int d = 0; d < x_shape.dims(); ++d) {
      if (lax.Get<bool>({d})) {
        dimensions.push_back(d);
      }
    }
    ctx->SetOutput(0, xla::Rev(ctx->Input(0), dimensions));
  }
};
REGISTER_XLA_OP(Name("Reverse").CompileTimeConstantInput("dims"), ReverseOp);
class ReverseV2Op : public XlaOpKernel {
 public:
  explicit ReverseV2Op(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape x_shape = ctx->InputShape(0);
    const TensorShape axes_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(axes_shape),
                errors::InvalidArgument("axes must be a vector, not shape ",
                                        axes_shape.DebugString()));
    OP_REQUIRES(ctx, axes_shape.num_elements() <= x_shape.dims(),
                errors::InvalidArgument("axes ", axes_shape.DebugString(),
                                        " can not have more elements"
                                        " than input tensor has dimensions ",
                                        x_shape.DebugString(), "."));
    if (axes_shape.num_elements() == 0) {
      ctx->SetOutput(0, ctx->Input(0));
      return;
    }
    std::vector<int64_t> axes;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(1, &axes));
    absl::InlinedVector<bool, 8> witnessed_axes(x_shape.dims(), false);
    for (int d = 0; d < axes.size(); ++d) {
      OP_REQUIRES(
          ctx, (-x_shape.dims() <= axes[d]) && (axes[d] < x_shape.dims()),
          errors::InvalidArgument(axes[d], " is out of range [-",
                                  x_shape.dims(), ", ", x_shape.dims(), ")."));
      if (axes[d] < 0) {
        axes[d] += x_shape.dims();
      }
      OP_REQUIRES(ctx, !witnessed_axes[axes[d]],
                  errors::InvalidArgument("canonicalized axis ", axes[d],
                                          " was repeated."));
      witnessed_axes[axes[d]] = true;
    }
    ctx->SetOutput(0, xla::Rev(ctx->Input(0), axes));
  }
};
REGISTER_XLA_OP(Name("ReverseV2").CompileTimeConstantInput("axis"),
                ReverseV2Op);
}  
}  