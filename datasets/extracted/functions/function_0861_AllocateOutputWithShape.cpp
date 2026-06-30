#define EIGEN_USE_THREADS
#include <algorithm>
#include <cmath>
#include <memory>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/kernels/random_op_cpu.h"
#include "tensorflow/core/lib/hash/crc32c.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/core/util/work_sharder.h"
#if EIGEN_COMP_GNUC && __cplusplus > 199711L
#define DISABLE_FLOAT_EQUALITY_WARNING \
  _Pragma("GCC diagnostic push")       \
      _Pragma("GCC diagnostic ignored \"-Wfloat-equal\"")
#define ENABLE_FLOAT_EQUALITY_WARNING _Pragma("GCC diagnostic pop")
#else
#define DISABLE_FLOAT_EQUALITY_WARNING
#define ENABLE_FLOAT_EQUALITY_WARNING
#endif
namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
namespace {
static Status AllocateOutputWithShape(OpKernelContext* ctx, const Tensor& shape,
                                      int index, Tensor** output) {
  TensorShape tensor_shape;
  TF_RETURN_IF_ERROR(tensor::MakeShape(shape, &tensor_shape));
  return ctx->allocate_output(index, tensor_shape, output);
}
template <typename Device, class Distribution>
class PhiloxRandomOp : public OpKernel {
 public:
  typedef typename Distribution::ResultElementType T;
  explicit PhiloxRandomOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, generator_.Init(ctx));
  }
  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape = ctx->input(0);
    Tensor* output;
    OP_REQUIRES_OK(ctx, AllocateOutputWithShape(ctx, shape, 0, &output));
    auto output_flat = output->flat<T>();
    functor::FillPhiloxRandom<Device, Distribution>()(
        ctx, ctx->eigen_device<Device>(), nullptr, nullptr,
        generator_.ReserveRandomOutputs(output_flat.size(), 256),
        output_flat.data(), output_flat.size(), Distribution());
  }
 private:
  GuardedPhiloxRandom generator_;
};
template <typename Device, class IntType>
class RandomUniformIntOp : public OpKernel {
 public:
  explicit RandomUniformIntOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, generator_.Init(ctx));
  }
  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape = ctx->input(0);
    const Tensor& minval = ctx->input(1);
    const Tensor& maxval = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(minval.shape()),
                errors::InvalidArgument("minval must be 0-D, got shape ",
                                        minval.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(maxval.shape()),
                errors::InvalidArgument("maxval must be 0-D, got shape ",
                                        maxval.shape().DebugString()));
    Tensor* output;
    OP_REQUIRES_OK(ctx, AllocateOutputWithShape(ctx, shape, 0, &output));
    if (output->NumElements() == 0) return;
    IntType lo = minval.scalar<IntType>()();
    IntType hi = maxval.scalar<IntType>()();
    OP_REQUIRES(
        ctx, lo < hi,
        errors::InvalidArgument("Need minval < maxval, got ", lo, " >= ", hi));
    typedef random::UniformDistribution<random::PhiloxRandom, IntType>
        Distribution;
    Distribution dist(lo, hi);
    auto output_flat = output->flat<IntType>();
    functor::FillPhiloxRandom<Device, Distribution>()(
        ctx, ctx->eigen_device<Device>(), nullptr, nullptr,
        generator_.ReserveRandomOutputs(output_flat.size(), 256),
        output_flat.data(), output_flat.size(), dist);
  }
 private:
  GuardedPhiloxRandom generator_;
};
template <typename T>
class RandomGammaOp : public OpKernel {
 public:
  explicit RandomGammaOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, generator_.Init(context));
  }
  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape_t = ctx->input(0);
    const Tensor& alpha_t = ctx->input(1);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsVector(shape_t.shape()) &&
                    (shape_t.dtype() == DataType::DT_INT32 ||
                     shape_t.dtype() == DataType::DT_INT64),
                errors::InvalidArgument(
                    "shape must be a vector of {int32,int64}, got shape: ",
                    shape_t.DebugString()));
    TensorShape samples_shape;
    if (shape_t.dtype() == DataType::DT_INT32) {
      auto vec = shape_t.flat<int32>();
      OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(vec.data(), vec.size(),
                                                      &samples_shape));
    } else if (shape_t.dtype() == DataType::DT_INT64) {
      auto vec = shape_t.flat<int64_t>();
      OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(vec.data(), vec.size(),
                                                      &samples_shape));
    }
    const int64_t samples_per_alpha = samples_shape.num_elements();
    OP_REQUIRES_OK(ctx, samples_shape.AppendShapeWithStatus(alpha_t.shape()));
    Tensor* samples_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, samples_shape, &samples_t));
    if (samples_shape.num_elements() == 0) return;
    using random::PhiloxRandom;
    typedef random::NormalDistribution<PhiloxRandom, double> Normal;
    typedef random::UniformDistribution<PhiloxRandom, double> Uniform;
#define UNIFORM(X)                                    \
  if (uniform_remaining == 0) {                       \
    uniform_remaining = Uniform::kResultElementCount; \
    uniform_result = uniform(&gen);                   \
  }                                                   \
  uniform_remaining--;                                \
  double X = uniform_result[uniform_remaining]
    static constexpr int kReservedSamplesPerOutput = 256;
    const auto alpha_flat = alpha_t.flat<T>().data();
    const int64_t num_alphas = alpha_t.NumElements();
    OP_REQUIRES(ctx, num_alphas > 0,
                errors::InvalidArgument(
                    "Input alpha should have non-zero element count, got: ",
                    num_alphas));
    auto samples_flat = samples_t->flat<T>().data();
    PhiloxRandom rng = generator_.ReserveRandomOutputs(
        samples_per_alpha * num_alphas, kReservedSamplesPerOutput);
    auto DoWork = [samples_per_alpha, num_alphas, &rng, samples_flat,
                   alpha_flat](int64_t start_output, int64_t limit_output) {
      using Eigen::numext::exp;
      using Eigen::numext::log;
      using Eigen::numext::log1p;
      using Eigen::numext::pow;
      Normal normal;
      Uniform uniform;
      typename Normal::ResultType norm_result;
      typename Uniform::ResultType uniform_result;
      for (int64_t output_idx = start_output; output_idx < limit_output;
           ) {
        int64_t alpha_idx = output_idx / samples_per_alpha;
        T* const samples_alpha_offset = samples_flat + alpha_idx;
        const double alpha = static_cast<double>(alpha_flat[alpha_idx]);
        DISABLE_FLOAT_EQUALITY_WARNING
        if (alpha == static_cast<double>(1.0)) {
          ENABLE_FLOAT_EQUALITY_WARNING
          for (int64_t sample_idx = output_idx % samples_per_alpha;
               sample_idx < samples_per_alpha && output_idx < limit_output;
               sample_idx++, output_idx++) {
            PhiloxRandom gen = rng;
            gen.Skip(kReservedSamplesPerOutput * output_idx);
            int16_t uniform_remaining = 0;
            UNIFORM(u);
            const double res = -log1p(-u);
            samples_alpha_offset[sample_idx * num_alphas] = static_cast<T>(res);
          }       
        } else {  
          const bool alpha_less_than_one = alpha < 1;
          const double d = alpha + (alpha_less_than_one ? 2.0 / 3 : -1.0 / 3);
          const double c = 1.0 / 3 / sqrt(d);
          for (int64_t sample_idx = output_idx % samples_per_alpha;
               sample_idx < samples_per_alpha && output_idx < limit_output;
               sample_idx++, output_idx++) {
            PhiloxRandom gen = rng;
            gen.Skip(kReservedSamplesPerOutput * output_idx);
            int16_t norm_remaining = 0;
            int16_t uniform_remaining = 0;
            while (true) {
              if (norm_remaining == 0) {
                norm_remaining = Normal::kResultElementCount;
                norm_result = normal(&gen);
              }
              norm_remaining--;
              const double x = norm_result[norm_remaining];
              double v = 1 + c * x;
              if (v <= 0) {
                continue;
              }
              v = v * v * v;
              UNIFORM(u);
              if ((u < 1 - 0.0331 * (x * x) * (x * x)) ||
                  (log(u) < 0.5 * x * x + d * (1 - v + log(v)))) {
                double res = d * v;
                if (alpha_less_than_one) {
                  UNIFORM(b);
                  res *= pow(b, 1 / alpha);
                }
                samples_alpha_offset[sample_idx * num_alphas] =
                    static_cast<T>(res);
                break;
              }
            }  
          }    
        }      
      }        
    };         
#undef UNIFORM
    static const int kElementCost = 85 + 2 * Normal::kElementCost +
                                    Uniform::kElementCost +
                                    3 * PhiloxRandom::kElementCost;
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers,
          num_alphas * samples_per_alpha, kElementCost, DoWork);
  }
 private:
  GuardedPhiloxRandom generator_;
  RandomGammaOp(const RandomGammaOp&) = delete;
  void operator=(const RandomGammaOp&) = delete;
};
}  
#define REGISTER(TYPE)                                                         \
  template struct functor::FillPhiloxRandom<                                   \
      CPUDevice, random::UniformDistribution<random::PhiloxRandom, TYPE>>;     \
  template struct functor::FillPhiloxRandom<                                   \
      CPUDevice, random::NormalDistribution<random::PhiloxRandom, TYPE>>;      \
  template struct functor::FillPhiloxRandom<                                   \
      CPUDevice,                                                               \
      random::TruncatedNormalDistribution<                                     \
          random::SingleSampleAdapter<random::PhiloxRandom>, TYPE>>;           \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("RandomUniform")                                                    \
          .Device(DEVICE_CPU)                                                  \
          .HostMemory("shape")                                                 \
          .TypeConstraint<TYPE>("dtype"),                                      \
      PhiloxRandomOp<CPUDevice, random::UniformDistribution<                   \
                                    random::PhiloxRandom, TYPE>>);             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("RandomStandardNormal")                                             \
          .Device(DEVICE_CPU)                                                  \
          .HostMemory("shape")                                                 \
          .TypeConstraint<TYPE>("dtype"),                                      \
      PhiloxRandomOp<CPUDevice,                                                \
                     random::NormalDistribution<random::PhiloxRandom, TYPE>>); \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TruncatedNormal")                                                  \
          .Device(DEVICE_CPU)                                                  \
          .HostMemory("shape")                                                 \
          .TypeConstraint<TYPE>("dtype"),                                      \
      PhiloxRandomOp<                                                          \
          CPUDevice,                                                           \
          random::TruncatedNormalDistribution<                                 \
              random::SingleSampleAdapter<random::PhiloxRandom>, TYPE>>);      \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("RandomGamma").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"),        \
      RandomGammaOp<TYPE>)
#define REGISTER_FULL_INT(IntType)           \
  template struct functor::FillPhiloxRandom< \
      CPUDevice,                             \
      random::UniformFullIntDistribution<random::PhiloxRandom, IntType>>
#define REGISTER_INT(IntType)                                                 \
  REGISTER_FULL_INT(IntType);                                                 \
  template struct functor::FillPhiloxRandom<                                  \
      CPUDevice, random::UniformDistribution<random::PhiloxRandom, IntType>>; \
  REGISTER_KERNEL_BUILDER(Name("RandomUniformInt")                            \
                              .Device(DEVICE_CPU)                             \
                              .HostMemory("shape")                            \
                              .HostMemory("minval")                           \
                              .HostMemory("maxval")                           \
                              .TypeConstraint<IntType>("Tout"),               \
                          RandomUniformIntOp<CPUDevice, IntType>);
TF_CALL_half(REGISTER);
TF_CALL_bfloat16(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
TF_CALL_int32(REGISTER_INT);
TF_CALL_int64(REGISTER_INT);
TF_CALL_uint32(REGISTER_FULL_INT);
TF_CALL_uint64(REGISTER_FULL_INT);
#undef REGISTER
#undef REGISTER_INT
#undef REGISTER_FULL_INT
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER(TYPE)                                                         \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("RandomUniform")                                                    \
          .Device(DEVICE_GPU)                                                  \
          .HostMemory("shape")                                                 \
          .TypeConstraint<int32>("T")                                          \
          .TypeConstraint<TYPE>("dtype"),                                      \
      PhiloxRandomOp<GPUDevice, random::UniformDistribution<                   \
                                    random::PhiloxRandom, TYPE>>);             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("RandomStandardNormal")                                             \
          .Device(DEVICE_GPU)                                                  \
          .HostMemory("shape")                                                 \
          .TypeConstraint<int32>("T")                                          \
          .TypeConstraint<TYPE>("dtype"),                                      \
      PhiloxRandomOp<GPUDevice,                                                \
                     random::NormalDistribution<random::PhiloxRandom, TYPE>>); \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("TruncatedNormal")                                                  \
          .Device(DEVICE_GPU)                                                  \
          .HostMemory("shape")                                                 \
          .TypeConstraint<int32>("T")                                          \
          .TypeConstraint<TYPE>("dtype"),                                      \
      PhiloxRandomOp<                                                          \
          GPUDevice,                                                           \
          random::TruncatedNormalDistribution<                                 \
              random::SingleSampleAdapter<random::PhiloxRandom>, TYPE>>);
#define REGISTER_FULL_INT(IntType)           \
  template struct functor::FillPhiloxRandom< \
      GPUDevice,                             \
      random::UniformFullIntDistribution<random::PhiloxRandom, IntType>>
#define REGISTER_INT(IntType)                                                 \
  REGISTER_FULL_INT(IntType);                                                 \
  template struct functor::FillPhiloxRandom<                                  \
      GPUDevice, random::UniformDistribution<random::PhiloxRandom, IntType>>; \
  REGISTER_KERNEL_BUILDER(Name("RandomUniformInt")                            \
                              .Device(DEVICE_GPU)                             \
                              .HostMemory("shape")                            \
                              .HostMemory("minval")                           \
                              .HostMemory("maxval")                           \
                              .TypeConstraint<int32>("T")                     \
                              .TypeConstraint<IntType>("Tout"),               \
                          RandomUniformIntOp<GPUDevice, IntType>);
TF_CALL_half(REGISTER);
TF_CALL_bfloat16(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
TF_CALL_int32(REGISTER_INT);
TF_CALL_int64(REGISTER_INT);
TF_CALL_uint32(REGISTER_FULL_INT);
TF_CALL_uint64(REGISTER_FULL_INT);
#undef REGISTER
#undef REGISTER_INT
#undef REGISTER_FULL_INT
#endif  
}  