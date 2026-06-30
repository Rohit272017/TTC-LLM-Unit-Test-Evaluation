#define EIGEN_USE_THREADS
#include "tensorflow/core/kernels/image/adjust_contrast_op.h"
#include <memory>
#include "unsupported/Eigen/CXX11/Tensor"  
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/determinism.h"
namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
template <typename Device, typename T>
class AdjustContrastOp : public OpKernel {
 public:
  explicit AdjustContrastOp(OpKernelConstruction* context)
      : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& factor = context->input(1);
    const Tensor& min_value = context->input(2);
    const Tensor& max_value = context->input(3);
    OP_REQUIRES(context, input.dims() >= 3,
                errors::InvalidArgument("input must be at least 3-D, got shape",
                                        input.shape().DebugString()));
    const int64_t height = input.dim_size(input.dims() - 3);
    const int64_t width = input.dim_size(input.dims() - 2);
    const int64_t channels = input.dim_size(input.dims() - 1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(factor.shape()),
                errors::InvalidArgument("contrast_factor must be scalar: ",
                                        factor.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(min_value.shape()),
                errors::InvalidArgument("min_value must be scalar: ",
                                        min_value.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(max_value.shape()),
                errors::InvalidArgument("max_value must be scalar: ",
                                        max_value.shape().DebugString()));
    if (std::is_same<Device, GPUDevice>::value) {
      OP_REQUIRES(
          context, !OpDeterminismRequired(),
          errors::Unimplemented(
              "A deterministic GPU implementation of AdjustContrast is not"
              " currently available."));
    }
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    Tensor mean_values;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value,
                                                   TensorShape(input.shape()),
                                                   &mean_values));
    if (input.NumElements() > 0) {
      const int64_t batch = input.NumElements() / (height * width * channels);
      const int64_t shape[4] = {batch, height, width, channels};
      functor::AdjustContrast<Device, T>()(
          context->eigen_device<Device>(), input.shaped<T, 4>(shape),
          factor.scalar<float>(), min_value.scalar<float>(),
          max_value.scalar<float>(), mean_values.shaped<float, 4>(shape),
          output->shaped<float, 4>(shape));
    }
  }
};
#define REGISTER_KERNEL(T)                                              \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("AdjustContrast").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      AdjustContrastOp<CPUDevice, T>);
REGISTER_KERNEL(uint8);
REGISTER_KERNEL(int8);
REGISTER_KERNEL(int16);
REGISTER_KERNEL(int32);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
#undef REGISTER_KERNEL
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
namespace functor {
#define DECLARE_GPU_SPEC(T)                                         \
  template <>                                                       \
  void AdjustContrast<GPUDevice, T>::operator()(                    \
      const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input, \
      typename TTypes<float>::ConstScalar contrast_factor,          \
      typename TTypes<float>::ConstScalar min_value,                \
      typename TTypes<float>::ConstScalar max_value,                \
      typename TTypes<float, 4>::Tensor mean_values,                \
      typename TTypes<float, 4>::Tensor output);                    \
  extern template struct AdjustContrast<GPUDevice, T>;
DECLARE_GPU_SPEC(uint8);
DECLARE_GPU_SPEC(int8);
DECLARE_GPU_SPEC(int16);
DECLARE_GPU_SPEC(int32);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  
#define REGISTER_GPU_KERNEL(T)                                          \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("AdjustContrast").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      AdjustContrastOp<GPUDevice, T>);
REGISTER_GPU_KERNEL(uint8);
REGISTER_GPU_KERNEL(int8);
REGISTER_GPU_KERNEL(int16);
REGISTER_GPU_KERNEL(int32);
REGISTER_GPU_KERNEL(float);
REGISTER_GPU_KERNEL(double);
#undef REGISTER_GPU_KERNEL
#endif  
class AdjustContrastOpV2Base : public OpKernel {
 protected:
  explicit AdjustContrastOpV2Base(OpKernelConstruction* context)
      : OpKernel(context) {}
  struct ComputeOptions {
    const Tensor* input = nullptr;
    const Tensor* factor = nullptr;
    Tensor* output = nullptr;
    int64_t batch = 0;
    int64_t height = 0;
    int64_t width = 0;
    int64_t channels = 0;
  };
  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& factor = context->input(1);
    OP_REQUIRES(context, input.dims() >= 3,
                errors::InvalidArgument("input must be at least 3-D, got shape",
                                        input.shape().DebugString()));
    const int64_t height = input.dim_size(input.dims() - 3);
    const int64_t width = input.dim_size(input.dims() - 2);
    const int64_t channels = input.dim_size(input.dims() - 1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(factor.shape()),
                errors::InvalidArgument("contrast_factor must be scalar: ",
                                        factor.shape().DebugString()));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    if (input.NumElements() > 0) {
      const int64_t batch = input.NumElements() / (height * width * channels);
      ComputeOptions options;
      options.input = &input;
      options.factor = &factor;
      options.output = output;
      options.batch = batch;
      options.height = height;
      options.width = width;
      options.channels = channels;
      DoCompute(context, options);
    }
  }
  virtual void DoCompute(OpKernelContext* context,
                         const ComputeOptions& options) = 0;
};
template <typename Device, typename T>
class AdjustContrastOpv2;
template <>
class AdjustContrastOpv2<CPUDevice, float> : public AdjustContrastOpV2Base {
 public:
  explicit AdjustContrastOpv2(OpKernelConstruction* context)
      : AdjustContrastOpV2Base(context) {}
  void DoCompute(OpKernelContext* context,
                 const ComputeOptions& options) override {
    const int64_t batch = options.batch;
    const int64_t height = options.height;
    const int64_t width = options.width;
    const int64_t channels = options.channels;
    const int64_t image_size = height * width;
    const Tensor* input = options.input;
    const Tensor* factor = options.factor;
    Tensor* output = options.output;
    Tensor mean_values;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<float>::value,
                                TensorShape({batch, channels}), &mean_values));
    auto input_data = input->shaped<float, 3>({batch, image_size, channels});
    auto mean_data = mean_values.tensor<float, 2>();
    auto output_data = output->shaped<float, 3>({batch, image_size, channels});
    ReduceMeanAcrossImage(input_data, mean_data, output_data);
    BroadcastAcrossImage(mean_data, output_data);
    IncrementWithScaling(input_data, factor->scalar<float>(), output_data);
  }
 private:
  void ReduceMeanAcrossImage(typename TTypes<float, 3>::ConstTensor input,
                             typename TTypes<float, 2>::Tensor mean,
                             typename TTypes<float, 3>::Tensor scratch) {
    const int64_t batch = input.dimension(0);
    const int64_t image_size = input.dimension(1);
    const int64_t channels = input.dimension(2);
    TTypes<float, 1>::ConstTensor input_flat(&input(0, 0, 0), input.size());
    TTypes<float, 1>::Tensor mean_flat(&mean(0, 0), mean.size());
    TTypes<float, 1>::Tensor summation_scratch(&scratch(0, 0, 0),
                                               scratch.size());
    using Eigen::DenseIndex;
    typedef Eigen::array<Eigen::DenseIndex, 1> Index;
    const int64_t plane_size = image_size * channels;
    for (int64_t i = 0; i < batch; i++) {
      auto input_plane = input_flat.slice(Index{DenseIndex(i * plane_size)},
                                          Index{DenseIndex(plane_size)});
      auto summation_plane = summation_scratch.slice(
          Index{DenseIndex(i * plane_size)}, Index{DenseIndex(plane_size)});
      int64_t remaining_size = image_size;
      int round = 0;
      do {
        int64_t right_size = remaining_size / 2;
        int64_t left_size = remaining_size - right_size;
        DCHECK(left_size == right_size || left_size == right_size + 1);
        if (round == 0) {
          summation_plane.slice(Index{0},
                                Index{DenseIndex(right_size * channels)}) =
              input_plane.slice(Index{DenseIndex(left_size * channels)},
                                Index{DenseIndex(right_size * channels)}) +
              input_plane.slice(Index{0},
                                Index{DenseIndex(right_size * channels)});
          if (left_size > right_size) {
            DCHECK_EQ(left_size - right_size, 1);
            summation_plane.slice(Index{DenseIndex(right_size * channels)},
                                  Index{DenseIndex(channels)}) =
                input_plane.slice(Index{DenseIndex(right_size * channels)},
                                  Index{DenseIndex(channels)});
          }
        } else {
          summation_plane.slice(Index{0},
                                Index{DenseIndex(right_size * channels)}) +=
              summation_plane.slice(Index{DenseIndex(left_size * channels)},
                                    Index{DenseIndex(right_size * channels)});
        }
        remaining_size = left_size;
        round++;
      } while (remaining_size > 1);
      const float mean_scaling = 1.0f / image_size;
      auto mean_plane = mean_flat.slice(Index{DenseIndex(i * channels)},
                                        Index{DenseIndex(channels)});
      mean_plane =
          summation_plane.slice(Index{0}, Index{DenseIndex(channels)}) *
          mean_scaling;
    }
  }
  void BroadcastAcrossImage(typename TTypes<float, 2>::Tensor inputs,
                            typename TTypes<float, 3>::Tensor outputs) {
    int64_t batch = outputs.dimension(0);
    int64_t image_size = outputs.dimension(1);
    int64_t channels = outputs.dimension(2);
    for (int64_t i = 0; i < batch; i++) {
      const float* mean_p = &inputs(i, 0);
      float* output_p = &outputs(i, 0, 0);
      memcpy(output_p, mean_p, sizeof(float) * channels);
      int64_t copied = 1;
      while (copied < image_size) {
        const int64_t kMaxToCopy = 1024;
        int64_t to_copy = std::min({copied, image_size - copied, kMaxToCopy});
        memcpy(output_p + channels * copied, output_p,
               to_copy * channels * sizeof(float));
        copied += to_copy;
      }
    }
  }
  void IncrementWithScaling(typename TTypes<float, 3>::ConstTensor input,
                            typename TTypes<float>::ConstScalar factor,
                            typename TTypes<float, 3>::Tensor output) {
    const float factor_value = factor();
    float* p = output.data();
    const float* q = input.data();
    for (int64_t n = 0; n < input.size(); ++n) {
      p[n] += factor_value * (q[n] - p[n]);
    }
  }
};
REGISTER_KERNEL_BUILDER(
    Name("AdjustContrastv2").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    AdjustContrastOpv2<CPUDevice, float>);
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
namespace functor {
#define DECLARE_GPU_SPEC(T)                                         \
  template <>                                                       \
  void AdjustContrastv2<GPUDevice, T>::operator()(                  \
      const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input, \
      typename TTypes<float>::ConstScalar contrast_factor,          \
      typename TTypes<T, 4>::Tensor output);                        \
  extern template struct AdjustContrastv2<GPUDevice, T>;
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(Eigen::half);
#undef DECLARE_GPU_SPEC
}  
template <typename T>
class AdjustContrastOpv2<GPUDevice, T> : public AdjustContrastOpV2Base {
 public:
  explicit AdjustContrastOpv2(OpKernelConstruction* context)
      : AdjustContrastOpV2Base(context) {}
  void DoCompute(OpKernelContext* context,
                 const ComputeOptions& options) override {
    const int64_t shape[4] = {options.batch, options.height, options.width,
                              options.channels};
    OP_REQUIRES(
        context, !OpDeterminismRequired(),
        errors::Unimplemented(
            "A deterministic GPU implementation of AdjustContrastv2 is not"
            " currently available."));
    functor::AdjustContrastv2<GPUDevice, T>()(
        context->eigen_device<GPUDevice>(), options.input->shaped<T, 4>(shape),
        options.factor->scalar<float>(), options.output->shaped<T, 4>(shape));
  }
};
#define REGISTER_GPU(T)                                                   \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("AdjustContrastv2").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      AdjustContrastOpv2<GPUDevice, T>);
REGISTER_GPU(float)
REGISTER_GPU(Eigen::half)
#undef REGISTER_GPU
#endif  
}  