#include "tensorflow/c/experimental/ops/math_ops.h"
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
Status Mul(AbstractContext* ctx, AbstractTensorHandle* const x,
           AbstractTensorHandle* const y, AbstractTensorHandle** z,
           const char* name, const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Mul", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(y));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(z, 1), &num_retvals);
}
Status Conj(AbstractContext* ctx, AbstractTensorHandle* const input,
            AbstractTensorHandle** output, const char* name,
            const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Conj", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(input));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(output, 1), &num_retvals);
}
Status AddV2(AbstractContext* ctx, AbstractTensorHandle* const x,
             AbstractTensorHandle* const y, AbstractTensorHandle** z,
             const char* name, const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("AddV2", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(y));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(z, 1), &num_retvals);
}
Status MatMul(AbstractContext* ctx, AbstractTensorHandle* const a,
              AbstractTensorHandle* const b, AbstractTensorHandle** product,
              bool transpose_a, bool transpose_b, const char* name,
              const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("MatMul", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(a));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(b));
  TF_RETURN_IF_ERROR(op_ptr->SetAttrBool("transpose_a", transpose_a));
  TF_RETURN_IF_ERROR(op_ptr->SetAttrBool("transpose_b", transpose_b));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(product, 1), &num_retvals);
}
Status Neg(AbstractContext* ctx, AbstractTensorHandle* const x,
           AbstractTensorHandle** y, const char* name,
           const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Neg", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(y, 1), &num_retvals);
}
Status Sum(AbstractContext* ctx, AbstractTensorHandle* const input,
           AbstractTensorHandle* const reduction_indices,
           AbstractTensorHandle** output, bool keep_dims, const char* name,
           const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Sum", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(input));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(reduction_indices));
  TF_RETURN_IF_ERROR(op_ptr->SetAttrBool("keep_dims", keep_dims));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(output, 1), &num_retvals);
}
Status Sub(AbstractContext* ctx, AbstractTensorHandle* const x,
           AbstractTensorHandle* const y, AbstractTensorHandle** z,
           const char* name, const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Sub", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(y));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(z, 1), &num_retvals);
}
Status Div(AbstractContext* ctx, AbstractTensorHandle* const x,
           AbstractTensorHandle* const y, AbstractTensorHandle** z,
           const char* name, const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Div", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(y));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(z, 1), &num_retvals);
}
Status DivNoNan(AbstractContext* ctx, AbstractTensorHandle* const x,
                AbstractTensorHandle* const y, AbstractTensorHandle** z,
                const char* name, const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("DivNoNan", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(y));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(z, 1), &num_retvals);
}
Status Exp(AbstractContext* ctx, AbstractTensorHandle* const x,
           AbstractTensorHandle** y, const char* name,
           const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Exp", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(y, 1), &num_retvals);
}
Status Sqrt(AbstractContext* ctx, AbstractTensorHandle* const x,
            AbstractTensorHandle** y, const char* name,
            const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Sqrt", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(y, 1), &num_retvals);
}
Status SqrtGrad(AbstractContext* ctx, AbstractTensorHandle* const y,
                AbstractTensorHandle* const dy, AbstractTensorHandle** z,
                const char* name, const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("SqrtGrad", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(y));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(dy));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(z, 1), &num_retvals);
}
Status Log1p(AbstractContext* ctx, AbstractTensorHandle* const x,
             AbstractTensorHandle** y, const char* name,
             const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("Log1p", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(x));
  int num_retvals = 1;
  return op_ptr->Execute(absl::MakeSpan(y, 1), &num_retvals);
}
}  
}  