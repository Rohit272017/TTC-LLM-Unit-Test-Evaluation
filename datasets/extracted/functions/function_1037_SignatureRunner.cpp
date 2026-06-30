#include "tensorflow/lite/core/signature_runner.h"
#include <vector>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/internal/signature_def.h"
namespace tflite {
namespace impl {
SignatureRunner::SignatureRunner(const internal::SignatureDef* signature_def,
                                 Subgraph* subgraph)
    : signature_def_(signature_def), subgraph_(subgraph) {
  for (const auto& it : signature_def_->inputs) {
    input_names_.push_back(it.first.c_str());
  }
  for (const auto& it : signature_def_->outputs) {
    output_names_.push_back(it.first.c_str());
  }
}
TfLiteTensor* SignatureRunner::input_tensor(const char* input_name) {
  const auto& it = signature_def_->inputs.find(input_name);
  if (it == signature_def_->inputs.end()) {
    subgraph_->ReportError("Input name %s was not found", input_name);
    return nullptr;
  }
  return subgraph_->tensor(it->second);
}
const TfLiteTensor* SignatureRunner::output_tensor(
    const char* output_name) const {
  const auto& it = signature_def_->outputs.find(output_name);
  if (it == signature_def_->outputs.end()) {
    subgraph_->ReportError("Output name %s was not found", output_name);
    return nullptr;
  }
  return subgraph_->tensor(it->second);
}
TfLiteStatus SignatureRunner::SetInputBufferHandle(
    const char* input_name, TfLiteBufferHandle buffer_handle,
    TfLiteDelegate* delegate, bool release_existing_buffer_handle) {
  const auto& it = signature_def_->inputs.find(input_name);
  if (it == signature_def_->inputs.end()) {
    subgraph_->ReportError("Input name %s was not found", input_name);
    return kTfLiteError;
  }
  return subgraph_->SetBufferHandle(it->second, buffer_handle, delegate,
                                    release_existing_buffer_handle);
}
TfLiteStatus SignatureRunner::SetOutputBufferHandle(
    const char* output_name, TfLiteBufferHandle buffer_handle,
    TfLiteDelegate* delegate, bool release_existing_buffer_handle) {
  const auto& it = signature_def_->outputs.find(output_name);
  if (it == signature_def_->outputs.end()) {
    subgraph_->ReportError("Output name %s was not found", output_name);
    return kTfLiteError;
  }
  return subgraph_->SetBufferHandle(it->second, buffer_handle, delegate,
                                    release_existing_buffer_handle);
}
TfLiteStatus SignatureRunner::ResizeInputTensor(
    const char* input_name, const std::vector<int>& new_size) {
  const auto& it = signature_def_->inputs.find(input_name);
  if (it == signature_def_->inputs.end()) {
    subgraph_->ReportError("Input name %s was not found", input_name);
    return kTfLiteError;
  }
  return subgraph_->ResizeInputTensor(it->second, new_size);
}
TfLiteStatus SignatureRunner::ResizeInputTensorStrict(
    const char* input_name, const std::vector<int>& new_size) {
  const auto& it = signature_def_->inputs.find(input_name);
  if (it == signature_def_->inputs.end()) {
    subgraph_->ReportError("Input name %s was not found", input_name);
    return kTfLiteError;
  }
  return subgraph_->ResizeInputTensorStrict(it->second, new_size);
}
TfLiteStatus SignatureRunner::Invoke() {
  if (subgraph_->continue_invocation_)
    (void)subgraph_->continue_invocation_->test_and_set();
  TF_LITE_ENSURE_STATUS(subgraph_->Invoke());
  if (!allow_buffer_handle_output_) {
    for (int tensor_index : subgraph_->outputs()) {
      TF_LITE_ENSURE_STATUS(
          subgraph_->EnsureTensorDataIsReadable(tensor_index));
    }
  }
  return kTfLiteOk;
}
TfLiteStatus SignatureRunner::SetCustomAllocationForInputTensor(
    const char* input_name, const TfLiteCustomAllocation& allocation,
    int64_t flags) {
  const auto& it = signature_def_->inputs.find(input_name);
  if (it == signature_def_->inputs.end()) {
    subgraph_->ReportError("Input name %s was not found", input_name);
    return kTfLiteError;
  }
  return subgraph_->SetCustomAllocationForTensor(it->second, allocation, flags);
}
TfLiteStatus SignatureRunner::SetCustomAllocationForOutputTensor(
    const char* output_name, const TfLiteCustomAllocation& allocation,
    int64_t flags) {
  const auto& it = signature_def_->outputs.find(output_name);
  if (it == signature_def_->outputs.end()) {
    subgraph_->ReportError("Output name %s was not found", output_name);
    return kTfLiteError;
  }
  return subgraph_->SetCustomAllocationForTensor(it->second, allocation, flags);
}
}  
}  