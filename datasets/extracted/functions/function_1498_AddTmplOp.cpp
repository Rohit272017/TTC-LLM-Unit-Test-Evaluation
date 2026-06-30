#include "tensorflow/lite/kernels/shim/test_op/tmpl_tflite_op.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/test_op/tmpl_op.h"
#include "tensorflow/lite/kernels/shim/tflite_op_shim.h"
#include "tensorflow/lite/kernels/shim/tflite_op_wrapper.h"
#include "tensorflow/lite/mutable_op_resolver.h"
namespace tflite {
namespace ops {
namespace custom {
namespace {
const char a_type[]("AType"), b_type[]("BType");
}  
using ::tflite::shim::op_wrapper::Attr;
using ::tflite::shim::op_wrapper::AttrName;
using ::tflite::shim::op_wrapper::OpWrapper;
template <shim::Runtime Rt>
using Op = OpWrapper<Rt, shim::TmplOp, Attr<AttrName<a_type>, int32_t, float>,
                     Attr<AttrName<b_type>, int32_t, int64_t, bool>>;
using OpKernel = ::tflite::shim::TfLiteOpKernel<Op>;
void AddTmplOp(MutableOpResolver* resolver) { OpKernel::Add(resolver); }
TfLiteRegistration* Register_TMPL_OP() {
  return OpKernel::GetTfLiteRegistration();
}
const char* OpName_TMPL_OP() { return OpKernel::OpName(); }
}  
}  
}  