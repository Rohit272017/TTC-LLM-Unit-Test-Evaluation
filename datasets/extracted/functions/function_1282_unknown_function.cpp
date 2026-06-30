#include "tensorflow/lite/kernels/shim/test_op/simple_tf_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/lite/kernels/shim/tf_op_shim.h"
namespace tflite {
namespace shim {
REGISTER_TF_OP_SHIM(SimpleOpKernel);
REGISTER_KERNEL_BUILDER(
    Name(SimpleOpKernel::OpName()).Device(::tensorflow::DEVICE_CPU),
    SimpleOpKernel);
}  
}  