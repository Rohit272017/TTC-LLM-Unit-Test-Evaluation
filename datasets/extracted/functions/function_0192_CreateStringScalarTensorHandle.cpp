#include "tensorflow/c/experimental/saved_model/core/ops/restore_ops.h"
#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/tstring.h"
#include "tsl/platform/errors.h"
namespace tensorflow {
namespace internal {
namespace {
Status CreateStringScalarTensorHandle(ImmediateExecutionContext* ctx,
                                      const std::string& s,
                                      ImmediateTensorHandlePtr* out) {
  AbstractTensorPtr tensor(ctx->CreateStringScalar(s));
  if (tensor.get() == nullptr) {
    return errors::Internal(
        "Failed to create scalar string tensor for checkpoint restore");
  }
  out->reset(ctx->CreateLocalHandle(tensor.get()));
  return Status();
}
Status CreateStringVectorTensorHandle(ImmediateExecutionContext* ctx,
                                      const std::string& s,
                                      ImmediateTensorHandlePtr* out) {
  int64_t flat_shape[] = {1};
  AbstractTensorPtr tensor(ctx->CreateTensor(DT_STRING, flat_shape));
  if (tensor.get() == nullptr) {
    return errors::Internal(
        "Failed to create vector string tensor for checkpoint restore");
  }
  new (tensor->Data()) tstring(s);
  out->reset(ctx->CreateLocalHandle(tensor.get()));
  return Status();
}
}  
Status SingleRestore(ImmediateExecutionContext* ctx, const std::string& prefix,
                     const std::string& checkpoint_key, DataType dtype,
                     ImmediateTensorHandlePtr* out) {
  ImmediateOpPtr restore_op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(restore_op->Reset("RestoreV2", "/cpu:0"));
  TF_RETURN_IF_ERROR(restore_op->SetAttrTypeList("dtypes", &dtype, 1));
  ImmediateTensorHandlePtr prefix_handle;
  TF_RETURN_IF_ERROR(
      CreateStringScalarTensorHandle(ctx, prefix, &prefix_handle));
  ImmediateTensorHandlePtr names_handle;
  TF_RETURN_IF_ERROR(
      CreateStringVectorTensorHandle(ctx, checkpoint_key, &names_handle));
  ImmediateTensorHandlePtr shapes_and_slices_handle;
  TF_RETURN_IF_ERROR(
      CreateStringVectorTensorHandle(ctx, "", &shapes_and_slices_handle));
  TF_RETURN_IF_ERROR(restore_op->AddInput(prefix_handle.get()));
  TF_RETURN_IF_ERROR(restore_op->AddInput(names_handle.get()));
  TF_RETURN_IF_ERROR(restore_op->AddInput(shapes_and_slices_handle.get()));
  AbstractTensorHandle* restored_handle = nullptr;
  int num_retvals = 1;
  TF_RETURN_IF_ERROR(restore_op->Execute(
      absl::MakeSpan(&restored_handle, num_retvals), &num_retvals));
  AbstractTensorHandlePtr owned_restored_handle(restored_handle);
  if (!tensorflow::isa<ImmediateExecutionTensorHandle>(
          owned_restored_handle.get())) {
    return errors::Internal("Unexpected tensor handle kind.");
  }
  out->reset(reinterpret_cast<ImmediateExecutionTensorHandle*>(
      owned_restored_handle.release()));
  return Status();
}
}  
}  