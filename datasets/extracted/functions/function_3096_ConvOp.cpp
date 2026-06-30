#include <cstdint>
#include <vector>
#include "tensorflow/compiler/tf2xla/kernels/conv_op_helpers.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/matrix.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal_util.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
namespace tensorflow {
namespace {
class ConvOp : public XlaOpKernel {
 public:
  explicit ConvOp(OpKernelConstruction* ctx, int num_spatial_dims,
                  bool depthwise)
      : XlaOpKernel(ctx) {
    absl::StatusOr<ConvOpAttrs> attrs =
        ConvOpAttrs::Create(num_spatial_dims, depthwise, ctx);
    OP_REQUIRES_OK(ctx, attrs.status());
    attrs_ = attrs.value();
  }
  void Compile(XlaOpKernelContext* ctx) override {
    absl::StatusOr<xla::XlaOp> conv = MakeXlaForwardConvOp(
        ctx->op_kernel().type_string(), ctx->Input(0), ctx->Input(1), attrs_);
    OP_REQUIRES_OK(ctx, conv.status());
    ctx->SetOutput(0, conv.value());
  }
 protected:
  ConvOpAttrs attrs_;
 private:
  ConvOp(const ConvOp&) = delete;
  void operator=(const ConvOp&) = delete;
};
class ConvNDOp : public XlaOpKernel {
 public:
  explicit ConvNDOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    absl::StatusOr<ConvNDOpAttrs> attrs = ConvNDOpAttrs::Create(ctx);
    OP_REQUIRES_OK(ctx, attrs.status());
    attrs_ = attrs.value();
  }
  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES_VALUE(xla::Shape input_shape, ctx, ctx->InputXlaShape(0));
    int num_spatial_dims = input_shape.rank() - 1 - attrs_.batch_dims;
    OP_REQUIRES_OK(ctx,
                   CheckValidPadding(attrs_.padding, attrs_.explicit_paddings,
                                     num_spatial_dims + 2,
                                     attrs_.data_format));
    ConvOpAttrs forward_attrs;
    forward_attrs.depthwise = false;
    forward_attrs.num_spatial_dims = num_spatial_dims;
    forward_attrs.dilations = attrs_.dilations.empty()
                                  ? std::vector<int32>(num_spatial_dims + 2, 1)
                                  : attrs_.dilations;
    forward_attrs.strides = attrs_.strides;
    forward_attrs.padding = attrs_.padding;
    forward_attrs.explicit_paddings = attrs_.explicit_paddings;
    forward_attrs.data_format = attrs_.data_format;
    xla::XlaOp input = ctx->Input(0);
    xla::XlaOp filter = ctx->Input(1);
    if (attrs_.batch_dims == 0) {
      xla::Shape expanded_input_shape(input_shape);
      for (int i = 0; i < expanded_input_shape.rank() - 1; ++i) {
        expanded_input_shape.set_dimensions(i + 1, input_shape.dimensions(i));
      }
      expanded_input_shape.set_dimensions(0, 1);
      input = xla::Reshape(input, expanded_input_shape.dimensions());
    } else if (attrs_.batch_dims > 1) {
      std::vector<int64_t> to_collapse(attrs_.batch_dims);
      for (int i = 0; i < attrs_.batch_dims; ++i) {
        to_collapse[i] = i;
      }
      input = xla::Collapse(input, to_collapse);
    }
    absl::StatusOr<xla::XlaOp> forward = MakeXlaForwardConvOp(
        ctx->op_kernel().type_string(), input, filter, forward_attrs);
    OP_REQUIRES_OK(ctx, forward.status());
    xla::XlaOp out = forward.value();
    auto* builder = out.builder();
    OP_REQUIRES_VALUE(xla::Shape out_shape, ctx, builder->GetShape(out));
    if (attrs_.batch_dims == 0) {
      xla::Shape no_batch_shape(out_shape);
      no_batch_shape.DeleteDimension(0);
      out = xla::Reshape(out, no_batch_shape.dimensions());
    } else if (attrs_.batch_dims > 1) {
      xla::Shape expanded_out_shape(input_shape);
      for (int i = attrs_.batch_dims; i < input_shape.rank(); ++i) {
        expanded_out_shape.set_dimensions(
            i, out_shape.dimensions(i - (attrs_.batch_dims - 1)));
      }
      out = xla::Reshape(out, expanded_out_shape.dimensions());
    }
    ctx->SetOutput(0, out);
  }
 protected:
  ConvNDOpAttrs attrs_;
};
REGISTER_XLA_CONV_OP(Name("Conv"), ConvNDOp);
class Conv2DOp : public ConvOp {
 public:
  explicit Conv2DOp(OpKernelConstruction* ctx)
      : ConvOp(ctx, 2, false) {}
};
REGISTER_XLA_CONV_OP(Name("Conv2D"), Conv2DOp);
class Conv3DOp : public ConvOp {
 public:
  explicit Conv3DOp(OpKernelConstruction* ctx)
      : ConvOp(ctx, 3, false) {}
};
REGISTER_XLA_CONV_OP(Name("Conv3D"), Conv3DOp);
class DepthwiseConv2DOp : public ConvOp {
 public:
  explicit DepthwiseConv2DOp(OpKernelConstruction* ctx)
      : ConvOp(ctx, 2, true) {}
};
REGISTER_XLA_CONV_OP(Name("DepthwiseConv2dNative"), DepthwiseConv2DOp);
class ConvBackpropInputOp : public XlaOpKernel {
 public:
  explicit ConvBackpropInputOp(OpKernelConstruction* ctx, int num_spatial_dims,
                               bool depthwise)
      : XlaOpKernel(ctx) {
    absl::StatusOr<ConvOpAttrs> attrs =
        ConvOpAttrs::Create(num_spatial_dims, depthwise, ctx);
    OP_REQUIRES_OK(ctx, attrs.status());
    attrs_ = attrs.value();
  }
  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape input_tensor_shape;
    OP_REQUIRES_OK(
        ctx, ctx->ConstantInputAsShape(0, &input_tensor_shape,
                                       xla::ValueInferenceMode::kUpperBound));
    xla::Shape input_shape =
        TensorShapeToXLAShape(ctx->input_xla_type(1), input_tensor_shape);
    OP_REQUIRES(ctx, input_shape.rank() == attrs_.num_spatial_dims + 2,
                errors::InvalidArgument(
                    "The rank of the specified input shape must be "
                    "num_spatial_dims + 2. Expected ",
                    attrs_.num_spatial_dims + 2, " got ", input_shape.rank()));
    xla::XlaOp input_sizes = ctx->Input(0);
    absl::StatusOr<xla::XlaOp> in_backprop = MakeXlaBackpropInputConvOp(
        ctx->op_kernel().type_string(), input_shape, ctx->Input(1),
        ctx->Input(2), attrs_, &input_sizes);
    OP_REQUIRES_OK(ctx, in_backprop.status());
    ctx->SetOutput(0, in_backprop.value());
  }
 protected:
  ConvOpAttrs attrs_;
 private:
  ConvBackpropInputOp(const ConvBackpropInputOp&) = delete;
  void operator=(const ConvBackpropInputOp&) = delete;
};
class Conv2DBackpropInputOp : public ConvBackpropInputOp {
 public:
  explicit Conv2DBackpropInputOp(OpKernelConstruction* ctx)
      : ConvBackpropInputOp(ctx, 2, false) {}
};
REGISTER_XLA_CONV_OP(
    Name("Conv2DBackpropInput").CompileTimeConstantInput("input_sizes"),
    Conv2DBackpropInputOp);
class Conv3DBackpropInputOp : public ConvBackpropInputOp {
 public:
  explicit Conv3DBackpropInputOp(OpKernelConstruction* ctx)
      : ConvBackpropInputOp(ctx, 3, false) {}
};
REGISTER_XLA_CONV_OP(
    Name("Conv3DBackpropInputV2").CompileTimeConstantInput("input_sizes"),
    Conv3DBackpropInputOp);
class DepthwiseConv2DBackpropInputOp : public ConvBackpropInputOp {
 public:
  explicit DepthwiseConv2DBackpropInputOp(OpKernelConstruction* ctx)
      : ConvBackpropInputOp(ctx, 2, true) {}
};
REGISTER_XLA_CONV_OP(Name("DepthwiseConv2dNativeBackpropInput")
                         .CompileTimeConstantInput("input_sizes"),
                     DepthwiseConv2DBackpropInputOp);
class ConvBackpropFilterOp : public XlaOpKernel {
 public:
  explicit ConvBackpropFilterOp(OpKernelConstruction* ctx, int num_spatial_dims,
                                bool depthwise)
      : XlaOpKernel(ctx) {
    absl::StatusOr<ConvOpAttrs> attrs =
        ConvOpAttrs::Create(num_spatial_dims, depthwise, ctx);
    OP_REQUIRES_OK(ctx, attrs.status());
    attrs_ = attrs.value();
  }
  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape filter_tensor_shape;
    OP_REQUIRES_OK(
        ctx, ctx->ConstantInputAsShape(1, &filter_tensor_shape,
                                       xla::ValueInferenceMode::kUpperBound));
    xla::Shape filter_shape =
        TensorShapeToXLAShape(ctx->input_xla_type(0), filter_tensor_shape);
    absl::StatusOr<xla::XlaOp> filter_backprop = MakeXlaBackpropFilterConvOp(
        ctx->op_kernel().type_string(), ctx->Input(0), filter_shape,
        ctx->Input(2), attrs_);
    OP_REQUIRES_OK(ctx, filter_backprop.status());
    ctx->SetOutput(0, filter_backprop.value());
  }
 protected:
  ConvOpAttrs attrs_;
 private:
  ConvBackpropFilterOp(const ConvBackpropFilterOp&) = delete;
  void operator=(const ConvBackpropFilterOp&) = delete;
};
class Conv2DBackpropFilterOp : public ConvBackpropFilterOp {
 public:
  explicit Conv2DBackpropFilterOp(OpKernelConstruction* ctx)
      : ConvBackpropFilterOp(ctx, 2, false) {
  }
};
REGISTER_XLA_CONV_OP(
    Name("Conv2DBackpropFilter").CompileTimeConstantInput("filter_sizes"),
    Conv2DBackpropFilterOp);
class Conv3DBackpropFilterOp : public ConvBackpropFilterOp {
 public:
  explicit Conv3DBackpropFilterOp(OpKernelConstruction* ctx)
      : ConvBackpropFilterOp(ctx, 3, false) {
  }
};
REGISTER_XLA_CONV_OP(
    Name("Conv3DBackpropFilterV2").CompileTimeConstantInput("filter_sizes"),
    Conv3DBackpropFilterOp);
class DepthwiseConv2DBackpropFilterOp : public ConvBackpropFilterOp {
 public:
  explicit DepthwiseConv2DBackpropFilterOp(OpKernelConstruction* ctx)
      : ConvBackpropFilterOp(ctx, 2, true) {}
};
REGISTER_XLA_CONV_OP(Name("DepthwiseConv2dNativeBackpropFilter")
                         .CompileTimeConstantInput("filter_sizes"),
                     DepthwiseConv2DBackpropFilterOp);
}  
}  