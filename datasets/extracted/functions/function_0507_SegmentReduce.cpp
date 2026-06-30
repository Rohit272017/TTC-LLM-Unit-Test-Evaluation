#include <vector>
#include "tensorflow/compiler/tf2xla/lib/scatter.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/value_inference.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
namespace tensorflow {
namespace {
class SegmentReduce : public XlaOpKernel {
 public:
  explicit SegmentReduce(OpKernelConstruction* ctx, bool indices_are_sorted)
      : XlaOpKernel(ctx), indices_are_sorted_(indices_are_sorted) {
    DataType dtype;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype));
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(dtype, &type_));
  }
  virtual xla::XlaOp InitialValue(xla::XlaBuilder* builder) = 0;
  virtual xla::XlaOp Combine(xla::XlaOp a, xla::XlaOp b) = 0;
  void Compile(XlaOpKernelContext* ctx) override {
    auto data = ctx->Input(0);
    TensorShape data_shape = ctx->InputShape(0);
    auto indices = ctx->Input(1);
    TensorShape indices_shape = ctx->InputShape(1);
    int64_t num_segments;
    OP_REQUIRES_OK(ctx,
                   ctx->ConstantInputAsIntScalar(
                       2, &num_segments, xla::ValueInferenceMode::kUpperBound));
    OP_REQUIRES(ctx, data_shape.dims() >= indices_shape.dims(),
                errors::InvalidArgument(type_string(),
                                        " requires that indices' rank be"
                                        " less than or equal to data's rank."));
    for (int d = 0; d < indices_shape.dims(); ++d) {
      OP_REQUIRES(
          ctx, (data_shape.dim_size(d) == indices_shape.dim_size(d)),
          errors::InvalidArgument(type_string(),
                                  " requires indices shape to be prefix"
                                  " of data_shape, but dimension ",
                                  d, " differs ", data_shape.dim_size(d),
                                  " vs. ", indices_shape.dim_size(d)));
    }
    xla::XlaBuilder* builder = ctx->builder();
    TensorShape buffer_shape = data_shape;
    buffer_shape.RemoveDimRange(0, indices_shape.dims());
    buffer_shape.InsertDim(0, num_segments);
    auto buffer =
        xla::Broadcast(InitialValue(builder), buffer_shape.dim_sizes());
    std::vector<xla::XlaOp> buffer_dims;
    std::vector<bool> buffer_dims_are_dynamic;
    bool num_segments_is_dynamic;
    OP_REQUIRES_OK(
        ctx, ctx->ResolveInputDynamismIntoPred(2, &num_segments_is_dynamic));
    buffer_dims.insert(buffer_dims.begin(), ctx->Input(2));
    buffer_dims_are_dynamic.insert(buffer_dims_are_dynamic.begin(),
                                   num_segments_is_dynamic);
    for (int64_t i = indices_shape.dims(); i < data_shape.dims(); ++i) {
      buffer_dims.push_back(xla::GetDimensionSize(data, i));
      buffer_dims_are_dynamic.push_back(
          ctx->InputXlaShape(0)->is_dynamic_dimension(i));
    }
    for (int64_t i = 0; i < buffer_dims.size(); ++i) {
      if (buffer_dims_are_dynamic[i]) {
        buffer = xla::SetDimensionSize(buffer, buffer_dims[i], i);
      }
    }
    auto combiner = [this](xla::XlaOp a, xla::XlaOp b,
                           xla::XlaBuilder* builder) { return Combine(a, b); };
    auto result = XlaScatter(buffer, data, indices,
                             false, indices_are_sorted_,
                             combiner, builder);
    OP_REQUIRES_OK(ctx, result.status());
    ctx->SetOutput(0, result.value());
  }
 protected:
  xla::PrimitiveType type_;
  bool indices_are_sorted_;
};
template <bool indices_are_sorted>
class SegmentSum : public SegmentReduce {
 public:
  explicit SegmentSum(OpKernelConstruction* ctx)
      : SegmentReduce(ctx, indices_are_sorted) {}
  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
    return xla::Zero(builder, type_);
  };
  xla::XlaOp Combine(xla::XlaOp a, xla::XlaOp b) override { return a + b; };
};
REGISTER_XLA_OP(Name("SegmentSumV2").CompileTimeConstantInput("num_segments"),
                SegmentSum<true>);
REGISTER_XLA_OP(
    Name("UnsortedSegmentSum").CompileTimeConstantInput("num_segments"),
    SegmentSum<false>);
template <bool indices_are_sorted>
class SegmentProd : public SegmentReduce {
 public:
  explicit SegmentProd(OpKernelConstruction* ctx)
      : SegmentReduce(ctx, indices_are_sorted) {}
  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
    return xla::One(builder, type_);
  };
  xla::XlaOp Combine(xla::XlaOp a, xla::XlaOp b) override { return a * b; };
};
REGISTER_XLA_OP(
    Name("UnsortedSegmentProd").CompileTimeConstantInput("num_segments"),
    SegmentProd<false>);
REGISTER_XLA_OP(Name("SegmentProdV2").CompileTimeConstantInput("num_segments"),
                SegmentProd<true>);
template <bool indices_are_sorted>
class SegmentMin : public SegmentReduce {
 public:
  explicit SegmentMin(OpKernelConstruction* ctx)
      : SegmentReduce(ctx, indices_are_sorted) {}
  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
    return xla::MaxFiniteValue(builder, type_);
  };
  xla::XlaOp Combine(xla::XlaOp a, xla::XlaOp b) override {
    return xla::Min(a, b);
  };
};
REGISTER_XLA_OP(
    Name("UnsortedSegmentMin").CompileTimeConstantInput("num_segments"),
    SegmentMin<false>);
REGISTER_XLA_OP(Name("SegmentMinV2").CompileTimeConstantInput("num_segments"),
                SegmentMin<true>);
template <bool indices_are_sorted>
class SegmentMax : public SegmentReduce {
 public:
  explicit SegmentMax(OpKernelConstruction* ctx)
      : SegmentReduce(ctx, indices_are_sorted) {}
  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
    return xla::MinFiniteValue(builder, type_);
  };
  xla::XlaOp Combine(xla::XlaOp a, xla::XlaOp b) override {
    return xla::Max(a, b);
  };
};
REGISTER_XLA_OP(
    Name("UnsortedSegmentMax").CompileTimeConstantInput("num_segments"),
    SegmentMax<false>);
REGISTER_XLA_OP(Name("SegmentMaxV2").CompileTimeConstantInput("num_segments"),
                SegmentMax<true>);
}  
}  