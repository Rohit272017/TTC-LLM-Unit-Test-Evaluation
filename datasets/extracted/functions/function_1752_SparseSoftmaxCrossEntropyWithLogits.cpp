#include "tensorflow/c/experimental/ops/nn_ops.h"
#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_operation.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/tracing_utils.h"
#include "tensorflow/core/platform/status.h"
#include "tsl/platform/errors.h"
using tensorflow::tracing::MaybeSetOpName;
namespace tensorflow {
namespace ops {
Status SparseSoftmaxCrossEntropyWithLogits(AbstractContext* ctx,
                                           AbstractTensorHandle* const features,
                                           AbstractTensorHandle* const labels,
                                           AbstractTensorHandle** loss,
                                           AbstractTensorHandle** backprop,
                                           const char* name,
                                           const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(
      op_ptr->Reset("SparseSoftmaxCrossEntropyWithLogits", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(features));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(labels));
  int num_retvals = 2;
  AbstractTensorHandle* temp_outputs[2];
  Status status = op_ptr->Execute(temp_outputs, &num_retvals);
  *loss = temp_outputs[0];
  *backprop = temp_outputs[1];
  return status;
}
Status ReluGrad(AbstractContext* ctx, AbstractTensorHandle* const gradients,
                AbstractTensorHandle* const features,
                AbstractTensorHandle** backprops, const char* name,
                const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("ReluGrad", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(gradients));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(features));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(backprops, 1), &num_retvals);
}
Status Relu(AbstractContext* ctx, AbstractTensorHandle* const features,
            AbstractTensorHandle** activations, const char* name,
            const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Relu", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(features));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(activations, 1), &num_retvals);
}
Status BiasAdd(AbstractContext* ctx, AbstractTensorHandle* const value,
               AbstractTensorHandle* const bias, AbstractTensorHandle** output,
               const char* data_format, const char* name,
               const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("BiasAdd", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(value));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(bias));
  TF_RETURN_IF_ERROR(
      op_ptr->SetAttrString("data_format", data_format, strlen(data_format)));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(output, 1), &num_retvals);
}
Status BiasAddGrad(AbstractContext* ctx,
                   AbstractTensorHandle* const out_backprop,
                   AbstractTensorHandle** output, const char* data_format,
                   const char* name, const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("BiasAddGrad", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(out_backprop));
  TF_RETURN_IF_ERROR(
      op_ptr->SetAttrString("data_format", data_format, strlen(data_format)));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(output, 1), &num_retvals);
}
}  
}  