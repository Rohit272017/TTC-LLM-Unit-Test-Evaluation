#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/xla_builder.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"
namespace tensorflow {
namespace {
template <typename DstT, typename SrcT>
DstT CastTo(SrcT src) {
  return static_cast<DstT>(src);
}
template <typename DstT,
          typename std::enable_if<std::is_same<DstT, Eigen::half>::value ||
                                  std::is_same<DstT, bfloat16>::value>::type* =
              nullptr>
DstT CastTo(int32_t src) {
  return absl::bit_cast<DstT>(static_cast<uint16>(src));
}
xla::XlaOp GetScalarConst(const TensorProto& proto, xla::XlaBuilder* b) {
  if (!proto.tensor_content().empty()) return xla::XlaOp();
  TensorShape shape(proto.tensor_shape());
  if (shape.num_elements() > 1) {
    switch (proto.dtype()) {
#define HANDLE_SPLAT(DTYPE, field_name, xla_type)                             \
  case DTYPE:                                                                 \
    if (proto.field_name##_val_size() == 0) {                                 \
      return xla::ConstantR0(b, CastTo<xla_type>(0));                         \
    } else if (proto.field_name##_val_size() == 1) {                          \
      return xla::ConstantR0(b, CastTo<xla_type>(proto.field_name##_val(0))); \
    }                                                                         \
    break;
      HANDLE_SPLAT(DT_BOOL, bool, bool);
      HANDLE_SPLAT(DT_INT8, int, int8_t);
      HANDLE_SPLAT(DT_INT16, int, int16_t);
      HANDLE_SPLAT(DT_INT32, int, int32_t);
      HANDLE_SPLAT(DT_INT64, int64, int64_t);
      HANDLE_SPLAT(DT_UINT8, int, uint8_t);
      HANDLE_SPLAT(DT_UINT16, int, uint16_t);
      HANDLE_SPLAT(DT_UINT32, uint32, uint32_t);
      HANDLE_SPLAT(DT_UINT64, uint64, uint64_t);
      HANDLE_SPLAT(DT_FLOAT, float, float);
      HANDLE_SPLAT(DT_DOUBLE, double, double);
      HANDLE_SPLAT(DT_BFLOAT16, half, bfloat16);
      HANDLE_SPLAT(DT_HALF, half, Eigen::half);
#undef HANDLE_SPLAT
#define HANDLE_COMPLEX_SPLAT(DTYPE, field_name, xla_type)                     \
  case DTYPE:                                                                 \
    if (proto.field_name##_val_size() == 2) {                                 \
      return xla::ConstantR0<xla_type>(                                       \
          b, xla_type(proto.field_name##_val(0), proto.field_name##_val(1))); \
    }                                                                         \
    break;
      HANDLE_COMPLEX_SPLAT(DT_COMPLEX64, scomplex, xla::complex64);
      HANDLE_COMPLEX_SPLAT(DT_COMPLEX128, dcomplex, xla::complex128);
#undef HANDLE_COMPLEXSPLAT
      default:
        break;
    }
  }
  return xla::XlaOp();
}
class ConstOp : public XlaOpKernel {
 public:
  explicit ConstOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    const TensorProto* proto = nullptr;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("value", &proto));
    proto_ = *proto;
    OP_REQUIRES(
        ctx, ctx->output_type(0) == proto_.dtype(),
        errors::InvalidArgument("Type mismatch between value (",
                                DataTypeString(proto_.dtype()), ") and dtype (",
                                DataTypeString(ctx->output_type(0)), ")"));
    OP_REQUIRES_OK(ctx, TensorShape::IsValidShape(proto_.tensor_shape()));
  }
  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    TensorShape shape(proto_.tensor_shape());
    if (shape.num_elements() > 1) {
      xla::XlaOp value = GetScalarConst(proto_, b);
      if (value.valid()) {
        ctx->SetOutput(0, xla::Broadcast(value, shape.dim_sizes()));
        return;
      }
    }
    Tensor tensor(proto_.dtype());
    OP_REQUIRES(ctx, tensor.FromProto(cpu_allocator(), proto_),
                errors::InvalidArgument("Cannot parse tensor from proto: ",
                                        proto_.DebugString()));
    ctx->SetConstantOutput(0, tensor);
  }
 private:
  TensorProto proto_;
  ConstOp(const ConstOp&) = delete;
  void operator=(const ConstOp&) = delete;
};
REGISTER_XLA_OP(Name("Const").CompilationOnly(), ConstOp);
}  
}  