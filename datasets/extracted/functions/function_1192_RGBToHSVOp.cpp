#define EIGEN_USE_THREADS
#include "tensorflow/core/kernels/image/colorspace_op.h"
#include <algorithm>
#include <cmath>
#include "unsupported/Eigen/CXX11/Tensor"  
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
template <typename Device, typename T>
class RGBToHSVOp : public OpKernel {
 public:
  explicit RGBToHSVOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() >= 1,
                errors::InvalidArgument("input must be at least 1D",
                                        input.shape().DebugString()));
    auto channels = input.dim_size(input.dims() - 1);
    OP_REQUIRES(context, channels == 3,
                errors::FailedPrecondition(
                    "input must have 3 channels but input only has ", channels,
                    " channels."));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    typename TTypes<T, 2>::ConstTensor input_data = input.flat_inner_dims<T>();
    typename TTypes<T, 2>::Tensor output_data = output->flat_inner_dims<T>();
    Tensor trange;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<T>::value,
                                        TensorShape({input_data.dimension(0)}),
                                        &trange));
    typename TTypes<T, 1>::Tensor range(trange.tensor<T, 1>());
    functor::RGBToHSV<Device, T>()(context->eigen_device<Device>(), input_data,
                                   range, output_data);
  }
};
template <typename Device, typename T>
class HSVToRGBOp : public OpKernel {
 public:
  explicit HSVToRGBOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() >= 1,
                errors::InvalidArgument("input must be at least 1D",
                                        input.shape().DebugString()));
    auto channels = input.dim_size(input.dims() - 1);
    OP_REQUIRES(context, channels == 3,
                errors::FailedPrecondition(
                    "input must have 3 channels but input only has ", channels,
                    " channels."));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    typename TTypes<T, 2>::ConstTensor input_data = input.flat_inner_dims<T>();
    typename TTypes<T, 2>::Tensor output_data = output->flat_inner_dims<T>();
    functor::HSVToRGB<Device, T>()(context->eigen_device<Device>(), input_data,
                                   output_data);
  }
};
#define REGISTER_CPU(T)                                           \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("RGBToHSV").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      RGBToHSVOp<CPUDevice, T>);                                  \
  template class RGBToHSVOp<CPUDevice, T>;                        \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("HSVToRGB").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      HSVToRGBOp<CPUDevice, T>);                                  \
  template class HSVToRGBOp<CPUDevice, T>;
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);
TF_CALL_half(REGISTER_CPU);
TF_CALL_bfloat16(REGISTER_CPU);
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
namespace functor {
#define DECLARE_GPU(T)                                               \
  template <>                                                        \
  void RGBToHSV<GPUDevice, T>::operator()(                           \
      const GPUDevice& d, TTypes<T, 2>::ConstTensor input_data,      \
      TTypes<T, 1>::Tensor range, TTypes<T, 2>::Tensor output_data); \
  extern template struct RGBToHSV<GPUDevice, T>;                     \
  template <>                                                        \
  void HSVToRGB<GPUDevice, T>::operator()(                           \
      const GPUDevice& d, TTypes<T, 2>::ConstTensor input_data,      \
      TTypes<T, 2>::Tensor output_data);                             \
  extern template struct HSVToRGB<GPUDevice, T>;
TF_CALL_float(DECLARE_GPU);
TF_CALL_double(DECLARE_GPU);
}  
#define REGISTER_GPU(T)                                           \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("RGBToHSV").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      RGBToHSVOp<GPUDevice, T>);                                  \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("HSVToRGB").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      HSVToRGBOp<GPUDevice, T>);
TF_CALL_float(REGISTER_GPU);
TF_CALL_double(REGISTER_GPU);
#endif
}  