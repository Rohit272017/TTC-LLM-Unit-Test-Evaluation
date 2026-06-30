#include "tensorflow/lite/kernels/shim/test_op/simple_tflite_op.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/shim/test_op/simple_op.h"
#include "tensorflow/lite/kernels/shim/tflite_op_shim.h"
#include "tensorflow/lite/mutable_op_resolver.h"
namespace tflite {
namespace ops {
namespace custom {
using OpKernel = ::tflite::shim::TfLiteOpKernel<tflite::shim::SimpleOp>;
void AddSimpleOp(MutableOpResolver* resolver) { OpKernel::Add(resolver); }
TfLiteRegistration* Register_SIMPLE_OP() {
  return OpKernel::GetTfLiteRegistration();
}
const char* OpName_SIMPLE_OP() { return OpKernel::OpName(); }
}  
}  
}  