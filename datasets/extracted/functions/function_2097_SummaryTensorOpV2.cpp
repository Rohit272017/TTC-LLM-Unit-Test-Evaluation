#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
namespace tensorflow {
template <typename T>
class SummaryTensorOpV2 : public OpKernel {
 public:
  explicit SummaryTensorOpV2(OpKernelConstruction* context)
      : OpKernel(context) {}
  void Compute(OpKernelContext* c) override {
    const Tensor& tag = c->input(0);
    OP_REQUIRES(c, TensorShapeUtils::IsScalar(tag.shape()),
                errors::InvalidArgument("tag must be scalar"));
    const Tensor& tensor = c->input(1);
    const Tensor& serialized_summary_metadata_tensor = c->input(2);
    OP_REQUIRES(
        c,
        TensorShapeUtils::IsScalar(serialized_summary_metadata_tensor.shape()),
        errors::InvalidArgument("serialized_summary_metadata must be scalar"));
    Summary s;
    Summary::Value* v = s.add_value();
    v->set_tag(string(tag.scalar<tstring>()()));  
    if (tensor.dtype() == DT_STRING) {
      tensor.AsProtoField(v->mutable_tensor());
    } else {
      tensor.AsProtoTensorContent(v->mutable_tensor());
    }
    ParseFromTString(serialized_summary_metadata_tensor.scalar<tstring>()(),
                     v->mutable_metadata());
    Tensor* summary_tensor = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({}), &summary_tensor));
    CHECK(SerializeToTString(s, &summary_tensor->scalar<tstring>()()));
  }
};
#define REGISTER(T)                                                      \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("TensorSummaryV2").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SummaryTensorOpV2<T>);
TF_CALL_ALL_TYPES(REGISTER)
#undef REGISTER
template <typename T>
class SummaryTensorOp : public OpKernel {
 public:
  explicit SummaryTensorOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* c) override {
    const Tensor& tensor = c->input(0);
    Summary s;
    Summary::Value* v = s.add_value();
    v->set_node_name(c->op_kernel().name());
    if (tensor.dtype() == DT_STRING) {
      tensor.AsProtoField(v->mutable_tensor());
    } else {
      tensor.AsProtoTensorContent(v->mutable_tensor());
    }
    Tensor* summary_tensor = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({}), &summary_tensor));
    CHECK(SerializeToTString(s, &summary_tensor->scalar<tstring>()()));
  }
};
#define REGISTER(T)                                                    \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("TensorSummary").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SummaryTensorOp<T>);
TF_CALL_ALL_TYPES(REGISTER)
#undef REGISTER
}  