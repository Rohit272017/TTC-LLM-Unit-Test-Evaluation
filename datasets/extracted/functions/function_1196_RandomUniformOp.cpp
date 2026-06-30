#include <vector>
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "tensorflow/compiler/tf2xla/lib/broadcast.h"
#include "tensorflow/compiler/tf2xla/lib/random.h"
#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/dynamic_shaped_ops.h"
#include "xla/hlo/builder/value_inference.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tsl/platform/statusor.h"
namespace tensorflow {
namespace {
class RandomUniformOp : public XlaOpKernel {
 public:
  explicit RandomUniformOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(
                            0, &shape, xla::ValueInferenceMode::kUpperBound));
    const DataType dtype = output_type(0);
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));
    xla::XlaBuilder* b = ctx->builder();
    LOG_FIRST_N(WARNING, 1)
        << "Warning: Using tf.random.uniform with XLA compilation will ignore "
           "seeds; consider using tf.random.stateless_uniform instead if "
           "reproducible behavior is desired. "
        << name();
    xla::XlaOp result = xla::RngUniform(XlaHelpers::Zero(b, dtype),
                                        XlaHelpers::One(b, dtype), xla_shape);
    auto result_status_or =
        SetAllDimensionSizes(&ctx->value_inference(), result, ctx->Input(0));
    OP_REQUIRES_OK(ctx, result_status_or.status());
    result = result_status_or.value();
    ctx->SetOutput(0, result);
  }
 private:
  RandomUniformOp(const RandomUniformOp&) = delete;
  void operator=(const RandomUniformOp&) = delete;
};
REGISTER_XLA_OP(Name("RandomUniform").CompileTimeConstantInput("shape"),
                RandomUniformOp);
REGISTER_XLA_OP(Name("RandomShuffle"), MlirXlaOpKernel);
class RandomUniformIntOp : public XlaOpKernel {
 public:
  explicit RandomUniformIntOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx,
                   TensorShapeToXLAShape(input_type(1), shape, &xla_shape));
    const TensorShape minval_shape = ctx->InputShape(1);
    const TensorShape maxval_shape = ctx->InputShape(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(minval_shape),
                errors::InvalidArgument("minval must be 0-D, got shape ",
                                        minval_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(maxval_shape),
                errors::InvalidArgument("maxval must be 0-D, got shape ",
                                        maxval_shape.DebugString()));
    auto minval = ctx->Input(1);
    auto maxval = ctx->Input(2);
    LOG_FIRST_N(WARNING, 1)
        << "Warning: Using tf.random.uniform with XLA compilation will ignore "
           "seeds; consider using tf.random.stateless_uniform instead if "
           "reproducible behavior is desired. "
        << name();
    ctx->SetOutput(0, xla::RngUniform(minval, maxval, xla_shape));
  }
 private:
  RandomUniformIntOp(const RandomUniformIntOp&) = delete;
  void operator=(const RandomUniformIntOp&) = delete;
};
REGISTER_XLA_OP(Name("RandomUniformInt").CompileTimeConstantInput("shape"),
                RandomUniformIntOp);
class RandomStandardNormalOp : public XlaOpKernel {
 public:
  explicit RandomStandardNormalOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    const DataType dtype = output_type(0);
    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(
                            0, &shape, xla::ValueInferenceMode::kUpperBound));
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));
    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp result = xla::RngNormal(XlaHelpers::Zero(b, dtype),
                                       XlaHelpers::One(b, dtype), xla_shape);
    auto result_status_or =
        SetAllDimensionSizes(&ctx->value_inference(), result, ctx->Input(0));
    OP_REQUIRES_OK(ctx, result_status_or.status());
    result = result_status_or.value();
    ctx->SetOutput(0, result);
  }
 private:
  RandomStandardNormalOp(const RandomStandardNormalOp&) = delete;
  void operator=(const RandomStandardNormalOp&) = delete;
};
REGISTER_XLA_OP(Name("RandomStandardNormal").CompileTimeConstantInput("shape"),
                RandomStandardNormalOp);
class TruncatedNormalOp : public XlaOpKernel {
 public:
  explicit TruncatedNormalOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    const DataType dtype = output_type(0);
    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));
    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp one = xla::One(b, xla_shape.element_type());
    xla::XlaOp min_positive =
        xla::MinPositiveNormalValue(b, xla_shape.element_type());
    LOG_FIRST_N(WARNING, 1)
        << "Warning: Using tf.random.truncated_normal with XLA "
           "compilation will ignore seeds; consider using "
           "tf.random.stateless_truncated_normal instead if "
           "reproducible behavior is desired. "
        << name();
    auto uniform = xla::RngUniform(min_positive, one, xla_shape);
    ctx->SetOutput(0, TruncatedNormal(uniform));
  }
};
REGISTER_XLA_OP(Name("TruncatedNormal")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_FLOAT, DT_DOUBLE}),
                TruncatedNormalOp);
static absl::StatusOr<xla::XlaOp> BroadcastParameters(
    xla::XlaOp params, TensorShape& output_shape) {
  int rank = output_shape.dims();
  std::vector<int64_t> bcast_shape;
  for (int i = 1; i < rank; ++i) {
    bcast_shape.push_back(output_shape.dim_size(i));
  }
  bcast_shape.push_back(output_shape.dim_size(0));
  TF_ASSIGN_OR_RETURN(xla::XlaOp bcast_params,
                      BroadcastTo(params, bcast_shape));
  std::vector<int64_t> permutation;
  permutation.push_back(rank - 1);
  for (int i = 0; i < rank - 1; ++i) {
    permutation.push_back(i);
  }
  return xla::Transpose(bcast_params, permutation);
}
class ParameterizedTruncatedNormalOp : public XlaOpKernel {
 public:
  explicit ParameterizedTruncatedNormalOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    const DataType dtype = output_type(0);
    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));
    OP_REQUIRES(ctx, xla_shape.rank() >= 1,
                errors::InvalidArgument(
                    "shape parameter must have rank >= 1, received (",
                    xla::ShapeUtil::HumanString(xla_shape), ")"));
    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp one = xla::One(b, xla_shape.element_type());
    xla::XlaOp min_positive =
        xla::MinPositiveNormalValue(b, xla_shape.element_type());
    LOG_FIRST_N(WARNING, 1)
        << "Warning: Using tf.random.truncated_normal with XLA "
           "compilation will ignore seeds; consider using "
           "tf.random.stateless_truncated_normal instead if "
           "reproducible behavior is desired. "
        << name();
    xla::XlaOp uniform = xla::RngUniform(min_positive, one, xla_shape);
    auto result = b->ReportErrorOrReturn([&]() -> absl::StatusOr<xla::XlaOp> {
      TF_ASSIGN_OR_RETURN(xla::XlaOp means,
                          BroadcastParameters(ctx->Input(1), shape));
      TF_ASSIGN_OR_RETURN(xla::XlaOp stddevs,
                          BroadcastParameters(ctx->Input(2), shape));
      TF_ASSIGN_OR_RETURN(xla::XlaOp minvals,
                          BroadcastParameters(ctx->Input(3), shape));
      TF_ASSIGN_OR_RETURN(xla::XlaOp maxvals,
                          BroadcastParameters(ctx->Input(4), shape));
      return ParameterizedTruncatedNormal(uniform, means, stddevs, minvals,
                                          maxvals);
    });
    ctx->SetOutput(0, result);
  }
};
REGISTER_XLA_OP(Name("ParameterizedTruncatedNormal")
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_FLOAT, DT_DOUBLE}),
                ParameterizedTruncatedNormalOp);
}  
}  