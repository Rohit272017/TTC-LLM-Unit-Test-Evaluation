#include "tensorflow/lite/kernels/shim/test_op/tmpl_tf_op.h"
#include <cstdint>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/lite/kernels/shim/tf_op_shim.h"
namespace tflite {
namespace shim {
using TmplOpKernelInstance = TmplOpKernel<float, int32_t>;
REGISTER_TF_OP_SHIM(TmplOpKernelInstance);
REGISTER_KERNEL_BUILDER(Name(TmplOpKernelInstance::OpName())
                            .Device(::tensorflow::DEVICE_CPU)
                            .TypeConstraint<float>("AType")
                            .TypeConstraint<int32_t>("BType"),
                        TmplOpKernel<float, int32_t>);
REGISTER_KERNEL_BUILDER(Name(TmplOpKernelInstance::OpName())
                            .Device(::tensorflow::DEVICE_CPU)
                            .TypeConstraint<int32_t>("AType")
                            .TypeConstraint<int64_t>("BType"),
                        TmplOpKernel<int32_t, int64_t>);
}  
}  