#ifdef INTEL_MKL
#include "tensorflow/core/kernels/mkl/mkl_eltwise_activation_base_op.h"
namespace tensorflow {
template <typename Device, typename T>
class MklSwishOp
    : public MklEltwiseFwdActivationOpBase<Device, T,
                                           dnnl::algorithm::eltwise_swish> {
 public:
  ~MklSwishOp() {}
  explicit MklSwishOp(OpKernelConstruction* context)
      : MklEltwiseFwdActivationOpBase<Device, T,
                                      dnnl::algorithm::eltwise_swish>(
            context, 1.0f, 0.0f) {}
  virtual void Compute_Scalar(OpKernelContext* context) {
    const Tensor& src_tensor = context->input(0);
    TensorShape src_shape = src_tensor.shape();
    Tensor* dst_tensor = nullptr;
    void* user_i =
        static_cast<void*>(const_cast<T*>(src_tensor.flat<T>().data()));
    TensorShape dst_shape = src_shape;
    OP_REQUIRES_OK(context, context->allocate_output(
                                GetTensorDataIndex(0, context->num_outputs()),
                                dst_shape, &dst_tensor));
    void* out_o = static_cast<void*>(dst_tensor->flat<T>().data());
    T feature = (static_cast<T*>(user_i))[0];
    T e1 = Eigen::numext::exp(-feature);
    (static_cast<T*>(out_o))[0] = feature / (static_cast<T>(1) + e1);
    return;
  }
};
#define REGISTER_SWISH_MKL_SUPPORTED_KERNELS_TYPES(type)              \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("_MklSwish").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      MklSwishOp<CPUDevice, type>);
TF_CALL_float(REGISTER_SWISH_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_SWISH_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_half(REGISTER_SWISH_MKL_SUPPORTED_KERNELS_TYPES);
}  
#endif  