#include "tensorflow/c/experimental/ops/io_ops.h"
#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_operation.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/tracing_utils.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tsl/platform/errors.h"
using tensorflow::tracing::MaybeSetOpName;
namespace tensorflow {
namespace ops {
Status RestoreV2(AbstractContext* ctx, AbstractTensorHandle* const prefix,
                 AbstractTensorHandle* const tensor_names,
                 AbstractTensorHandle* const shape_and_slices,
                 absl::Span<AbstractTensorHandle*> tensors,
                 absl::Span<DataType> dtypes, const char* name,
                 const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("RestoreV2", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(prefix));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(tensor_names));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(shape_and_slices));
  TF_RETURN_IF_ERROR(
      op_ptr->SetAttrTypeList("dtypes", dtypes.data(), dtypes.length()));
  int num_retvals = tensors.size();
  return op_ptr->Execute(tensors, &num_retvals);
}
Status SaveV2(AbstractContext* ctx, AbstractTensorHandle* const prefix,
              AbstractTensorHandle* const tensor_names,
              AbstractTensorHandle* const shape_and_slices,
              absl::Span<AbstractTensorHandle* const> tensors, const char* name,
              const char* raw_device_name) {
  AbstractOperationPtr op_ptr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op_ptr->Reset("SaveV2", raw_device_name));
  TF_RETURN_IF_ERROR(MaybeSetOpName(op_ptr.get(), name));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(prefix));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(tensor_names));
  TF_RETURN_IF_ERROR(op_ptr->AddInput(shape_and_slices));
  TF_RETURN_IF_ERROR(op_ptr->AddInputList(tensors));
  int num_retvals = 0;
  std::vector<AbstractTensorHandle*> dummy_outputs;
  return op_ptr->Execute(absl::MakeSpan(dummy_outputs), &num_retvals);
}
}  
}  