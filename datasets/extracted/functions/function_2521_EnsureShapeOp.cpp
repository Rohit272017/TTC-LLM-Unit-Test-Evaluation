#include "tensorflow/core/kernels/shape_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/register_types.h"
namespace tensorflow {
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int32>("out_type"),
                        ShapeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int64_t>("out_type"),
                        ShapeOp<int64_t>);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNEL(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("Shape")                            \
                              .Device(DEVICE_GPU)                  \
                              .HostMemory("output")                \
                              .TypeConstraint<int32>("out_type")   \
                              .TypeConstraint<type>("T"),          \
                          ShapeOp<int32>);                         \
  REGISTER_KERNEL_BUILDER(Name("Shape")                            \
                              .Device(DEVICE_GPU)                  \
                              .HostMemory("output")                \
                              .TypeConstraint<int64_t>("out_type") \
                              .TypeConstraint<type>("T"),          \
                          ShapeOp<int64_t>);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_bool(REGISTER_GPU_KERNEL);
TF_CALL_variant(REGISTER_GPU_KERNEL);
TF_CALL_tstring(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type"),
                        ShapeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64_t>("out_type"),
                        ShapeOp<int64_t>);
#endif  
#define REGISTER_DEFAULT_KERNEL(type)                              \
  REGISTER_KERNEL_BUILDER(Name("Shape")                            \
                              .Device(DEVICE_DEFAULT)              \
                              .HostMemory("output")                \
                              .TypeConstraint<int32>("out_type")   \
                              .TypeConstraint<type>("T"),          \
                          ShapeOp<int32>);                         \
  REGISTER_KERNEL_BUILDER(Name("Shape")                            \
                              .Device(DEVICE_DEFAULT)              \
                              .HostMemory("output")                \
                              .TypeConstraint<int64_t>("out_type") \
                              .TypeConstraint<type>("T"),          \
                          ShapeOp<int64_t>);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_DEFAULT_KERNEL);
TF_CALL_bool(REGISTER_DEFAULT_KERNEL);
TF_CALL_variant(REGISTER_DEFAULT_KERNEL);
#undef REGISTER_DEFAULT_KERNEL
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type"),
                        ShapeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64_t>("out_type"),
                        ShapeOp<int64_t>);
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int32>("out_type"),
                        ShapeNOp<int32>);
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int64_t>("out_type"),
                        ShapeNOp<int64_t>);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNEL(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("ShapeN")                           \
                              .Device(DEVICE_GPU)                  \
                              .HostMemory("output")                \
                              .TypeConstraint<int32>("out_type")   \
                              .TypeConstraint<type>("T"),          \
                          ShapeNOp<int32>);                        \
  REGISTER_KERNEL_BUILDER(Name("ShapeN")                           \
                              .Device(DEVICE_GPU)                  \
                              .HostMemory("output")                \
                              .TypeConstraint<int64_t>("out_type") \
                              .TypeConstraint<type>("T"),          \
                          ShapeNOp<int64_t>)
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_bool(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type"),
                        ShapeNOp<int32>);
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64_t>("out_type"),
                        ShapeNOp<int64_t>);
#endif  
#define REGISTER_DEFAULT_KERNEL(type)                              \
  REGISTER_KERNEL_BUILDER(Name("ShapeN")                           \
                              .Device(DEVICE_DEFAULT)              \
                              .HostMemory("output")                \
                              .TypeConstraint<int32>("out_type")   \
                              .TypeConstraint<type>("T"),          \
                          ShapeNOp<int32>);                        \
  REGISTER_KERNEL_BUILDER(Name("ShapeN")                           \
                              .Device(DEVICE_DEFAULT)              \
                              .HostMemory("output")                \
                              .TypeConstraint<int64_t>("out_type") \
                              .TypeConstraint<type>("T"),          \
                          ShapeNOp<int64_t>)
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_DEFAULT_KERNEL);
TF_CALL_bool(REGISTER_DEFAULT_KERNEL);
#undef REGISTER_DEFAULT_KERNEL
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type"),
                        ShapeNOp<int32>);
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64_t>("out_type"),
                        ShapeNOp<int64_t>);
REGISTER_KERNEL_BUILDER(Name("Rank").Device(DEVICE_CPU).HostMemory("output"),
                        RankOp);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNEL(type)                        \
  REGISTER_KERNEL_BUILDER(Name("Rank")                   \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("output"),     \
                          RankOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_variant(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL
REGISTER_KERNEL_BUILDER(Name("Rank")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        RankOp);
REGISTER_KERNEL_BUILDER(Name("Rank")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<bool>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        RankOp);
#endif  
#define REGISTER_DEFAULT_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("Rank")                   \
                              .Device(DEVICE_DEFAULT)    \
                              .TypeConstraint<type>("T") \
                              .HostMemory("output"),     \
                          RankOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_DEFAULT_KERNEL);
TF_CALL_variant(REGISTER_DEFAULT_KERNEL);
#undef REGISTER_DEFAULT_KERNEL
REGISTER_KERNEL_BUILDER(Name("Rank")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        RankOp);
REGISTER_KERNEL_BUILDER(Name("Rank")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<bool>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        RankOp);
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int32>("out_type"),
                        SizeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int64_t>("out_type"),
                        SizeOp<int64_t>);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNEL(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("Size")                             \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int32>("out_type")   \
                              .HostMemory("output"),               \
                          SizeOp<int32>);                          \
  REGISTER_KERNEL_BUILDER(Name("Size")                             \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int64_t>("out_type") \
                              .HostMemory("output"),               \
                          SizeOp<int64_t>);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_bool(REGISTER_GPU_KERNEL);
TF_CALL_variant(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SizeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64_t>("out_type")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SizeOp<int64_t>);
#endif  
#define REGISTER_DEFAULT_KERNEL(type)                              \
  REGISTER_KERNEL_BUILDER(Name("Size")                             \
                              .Device(DEVICE_DEFAULT)              \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int32>("out_type")   \
                              .HostMemory("output"),               \
                          SizeOp<int32>);                          \
  REGISTER_KERNEL_BUILDER(Name("Size")                             \
                              .Device(DEVICE_DEFAULT)              \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int64_t>("out_type") \
                              .HostMemory("output"),               \
                          SizeOp<int64_t>);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_DEFAULT_KERNEL);
TF_CALL_bool(REGISTER_DEFAULT_KERNEL);
TF_CALL_variant(REGISTER_DEFAULT_KERNEL);
#undef REGISTER_DEFAULT_KERNEL
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SizeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64_t>("out_type")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SizeOp<int64_t>);
REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(DEVICE_CPU)
                            .HostMemory("dim")
                            .TypeConstraint<int32>("Tdim"),
                        ExpandDimsOp<int32>);
REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(DEVICE_CPU)
                            .HostMemory("dim")
                            .TypeConstraint<int64_t>("Tdim"),
                        ExpandDimsOp<int64_t>);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNEL(type)                              \
  REGISTER_KERNEL_BUILDER(Name("ExpandDims")                   \
                              .Device(DEVICE_GPU)              \
                              .TypeConstraint<type>("T")       \
                              .TypeConstraint<int32>("Tdim")   \
                              .HostMemory("dim"),              \
                          ExpandDimsOp<int32>);                \
  REGISTER_KERNEL_BUILDER(Name("ExpandDims")                   \
                              .Device(DEVICE_GPU)              \
                              .TypeConstraint<type>("T")       \
                              .TypeConstraint<int64_t>("Tdim") \
                              .HostMemory("dim"),              \
                          ExpandDimsOp<int64_t>);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_bool(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL
REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Tdim")
                            .HostMemory("input")
                            .HostMemory("dim")
                            .HostMemory("output"),
                        ExpandDimsOp<int32>);
REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64_t>("Tdim")
                            .HostMemory("input")
                            .HostMemory("dim")
                            .HostMemory("output"),
                        ExpandDimsOp<int64_t>);
#endif  
#define REGISTER_DEFAULT_KERNEL(type)                          \
  REGISTER_KERNEL_BUILDER(Name("ExpandDims")                   \
                              .Device(DEVICE_DEFAULT)          \
                              .TypeConstraint<type>("T")       \
                              .TypeConstraint<int32>("Tdim")   \
                              .HostMemory("dim"),              \
                          ExpandDimsOp<int32>);                \
  REGISTER_KERNEL_BUILDER(Name("ExpandDims")                   \
                              .Device(DEVICE_DEFAULT)          \
                              .TypeConstraint<type>("T")       \
                              .TypeConstraint<int64_t>("Tdim") \
                              .HostMemory("dim"),              \
                          ExpandDimsOp<int64_t>);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_DEFAULT_KERNEL);
TF_CALL_bool(REGISTER_DEFAULT_KERNEL);
#undef REGISTER_DEFAULT_KERNEL
REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Tdim")
                            .HostMemory("input")
                            .HostMemory("dim")
                            .HostMemory("output"),
                        ExpandDimsOp<int32>);
REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64_t>("Tdim")
                            .HostMemory("input")
                            .HostMemory("dim")
                            .HostMemory("output"),
                        ExpandDimsOp<int64_t>);
REGISTER_KERNEL_BUILDER(Name("Squeeze").Device(DEVICE_CPU), SqueezeOp);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_KERNEL(type)                                   \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Squeeze").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SqueezeOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_bool(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL
REGISTER_KERNEL_BUILDER(Name("Squeeze")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SqueezeOp);
#endif  
#define REGISTER_DEFAULT_KERNEL(type)                                   \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("Squeeze").Device(DEVICE_DEFAULT).TypeConstraint<type>("T"), \
      SqueezeOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_DEFAULT_KERNEL);
TF_CALL_bool(REGISTER_DEFAULT_KERNEL);
#undef REGISTER_DEFAULT_KERNEL
REGISTER_KERNEL_BUILDER(Name("Squeeze")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SqueezeOp);
class EnsureShapeOp : public OpKernel {
 public:
  explicit EnsureShapeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &expected_shape_));
  }
  void Compute(OpKernelContext* ctx) override {
    TensorShape shape;
    OP_REQUIRES_OK(ctx, shape_op_helpers::GetShape(ctx, 0, &shape));
    if (!expected_shape_.IsCompatibleWith(shape)) {
      ctx->SetStatus(errors::InvalidArgument(
          "Shape of tensor ", this->def().input(0), " ", shape.DebugString(),
          " is not compatible with expected shape ",
          expected_shape_.DebugString(), "."));
    }
    if (IsRefType(ctx->input_dtype(0))) {
      ctx->forward_ref_input_to_ref_output(0, 0);
    } else {
      ctx->set_output(0, ctx->input(0));
    }
  }
  bool IsExpensive() override { return false; }
 private:
  PartialTensorShape expected_shape_;
};
REGISTER_KERNEL_BUILDER(Name("EnsureShape").Device(DEVICE_CPU), EnsureShapeOp);
#define REGISTER_DEVICE_KERNEL(type)                                        \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("EnsureShape").Device(DEVICE_DEFAULT).TypeConstraint<type>("T"), \
      EnsureShapeOp)
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_DEVICE_KERNEL);
REGISTER_DEVICE_KERNEL(Variant);
#undef REGISTER_DEVICE_KERNEL
#define REGISTER_DEVICE_HOST_KERNEL(type)                 \
  REGISTER_KERNEL_BUILDER(Name("EnsureShape")             \
                              .Device(DEVICE_DEFAULT)     \
                              .HostMemory("input")        \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          EnsureShapeOp)
REGISTER_DEVICE_HOST_KERNEL(int32);
REGISTER_DEVICE_HOST_KERNEL(bool);
REGISTER_DEVICE_HOST_KERNEL(tstring);
REGISTER_DEVICE_HOST_KERNEL(ResourceHandle);
#undef REGISTER_DEVICE_HOST_KERNEL
}  