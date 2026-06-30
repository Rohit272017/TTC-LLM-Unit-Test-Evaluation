#define EIGEN_USE_THREADS
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/meta_support.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/errors.h"
namespace tensorflow {
template <typename T>
class QuantizedReluOp : public OpKernel {
 public:
  explicit QuantizedReluOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& min_input_tensor = context->input(1);
    const Tensor& max_input_tensor = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(min_input_tensor.shape()),
        errors::InvalidArgument("`min_input` must be rank 0 but is rank ",
                                min_input_tensor.dims()));
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_input_tensor.shape()),
        errors::InvalidArgument("`max_input` must be rank 0 but is rank ",
                                max_input_tensor.dims()));
    const float min_input = min_input_tensor.scalar<float>()();
    const float max_input = max_input_tensor.scalar<float>()();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    const T min_as_quantized = FloatToQuantized<T>(0.0f, min_input, max_input);
    if (meta::IsSupportedAndEnabled() && std::is_same<T, quint8>()) {
      auto input_ui8_array = input.flat<quint8>();
      meta::Clamp(context, input_ui8_array.data(), input_ui8_array.size(),
                  min_as_quantized, 255, output->flat<quint8>().data());
    } else {
      output->flat<T>().device(context->eigen_cpu_device()) =
          input.flat<T>().cwiseMax(min_as_quantized).template cast<T>();
    }
    Tensor* output_min = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {}, &output_min));
    output_min->flat<float>()(0) = min_input;
    Tensor* output_max = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, {}, &output_max));
    output_max->flat<float>()(0) = max_input;
  }
};
template <typename T>
class QuantizedRelu6Op : public OpKernel {
 public:
  explicit QuantizedRelu6Op(OpKernelConstruction* context)
      : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& min_input_tensor = context->input(1);
    const Tensor& max_input_tensor = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(min_input_tensor.shape()),
        errors::InvalidArgument("`min_input` must be rank 0 but is rank ",
                                min_input_tensor.dims()));
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_input_tensor.shape()),
        errors::InvalidArgument("`max_input` must be rank 0 but is rank ",
                                max_input_tensor.dims()));
    const float min_input = min_input_tensor.scalar<float>()();
    const float max_input = max_input_tensor.scalar<float>()();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    const T min_as_quantized = FloatToQuantized<T>(0.0f, min_input, max_input);
    const T max_as_quantized = FloatToQuantized<T>(6.0f, min_input, max_input);
    if (meta::IsSupportedAndEnabled() && std::is_same<T, quint8>()) {
      auto input_ui8_array = input.flat<quint8>();
      meta::Clamp(context, input_ui8_array.data(), input_ui8_array.size(),
                  min_as_quantized, max_as_quantized,
                  output->flat<quint8>().data());
    } else {
      output->flat<T>().device(context->eigen_cpu_device()) =
          input.flat<T>()
              .cwiseMax(min_as_quantized)
              .cwiseMin(max_as_quantized)
              .template cast<T>();
    }
    Tensor* output_min = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {}, &output_min));
    output_min->flat<float>()(0) = min_input;
    Tensor* output_max = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, {}, &output_max));
    output_max->flat<float>()(0) = max_input;
  }
};
REGISTER_KERNEL_BUILDER(Name("QuantizedRelu")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint32>("Tinput")
                            .TypeConstraint<qint32>("out_type"),
                        QuantizedReluOp<qint32>);
REGISTER_KERNEL_BUILDER(Name("QuantizedRelu")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<quint8>("out_type"),
                        QuantizedReluOp<quint8>);
REGISTER_KERNEL_BUILDER(Name("QuantizedRelu6")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint32>("Tinput")
                            .TypeConstraint<qint32>("out_type"),
                        QuantizedRelu6Op<qint32>);
REGISTER_KERNEL_BUILDER(Name("QuantizedRelu6")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<quint8>("out_type"),
                        QuantizedRelu6Op<quint8>);
}  