#include "tensorflow/core/kernels/identity_n_op.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
namespace tensorflow {
REGISTER_KERNEL_BUILDER(Name("IdentityN").Device(DEVICE_DEFAULT), IdentityNOp);
REGISTER_KERNEL_BUILDER(Name("IdentityN").Device(DEVICE_TPU_SYSTEM),
                        IdentityNOp);
REGISTER_INPUT_COLOCATION_EXEMPTION("IdentityN");
}  