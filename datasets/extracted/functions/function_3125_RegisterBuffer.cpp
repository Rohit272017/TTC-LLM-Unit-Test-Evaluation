#include "tensorflow/lite/async/backend_async_kernel_interface.h"
#include <vector>
#include "tensorflow/lite/async/c/async_kernel.h"
#include "tensorflow/lite/async/c/types.h"
namespace tflite {
namespace delegates {
namespace internal {
TfLiteStatus RegisterBuffer(TfLiteAsyncKernel* async_kernel,
                            TfLiteOpaqueContext* context, TfLiteIoType io_type,
                            const TfLiteBackendBuffer* buffer,
                            const TfLiteAttributeMap* attrs,
                            TfLiteBufferHandle handle) {
  return reinterpret_cast<BackendAsyncKernelInterface*>(
             TfLiteAsyncKernelGetKernelData(async_kernel))
      ->RegisterBuffer(context, io_type, buffer, attrs, handle);
}
TfLiteStatus RegisterBufferSlice(TfLiteAsyncKernel* async_kernel,
                                 TfLiteOpaqueContext* context,
                                 TfLiteBufferHandle buffer,
                                 const TfLiteAttributeMap* attrs,
                                 TfLiteBufferHandle handle) {
  return reinterpret_cast<BackendAsyncKernelInterface*>(
             TfLiteAsyncKernelGetKernelData(async_kernel))
      ->RegisterBufferSlice(context, buffer, attrs, handle);
}
TfLiteStatus UnregisterBuffer(TfLiteAsyncKernel* async_kernel,
                              TfLiteOpaqueContext* context,
                              const TfLiteBufferHandle handle) {
  return reinterpret_cast<BackendAsyncKernelInterface*>(
             TfLiteAsyncKernelGetKernelData(async_kernel))
      ->UnregisterBuffer(context, handle);
}
void SupportedBufferTypes(const TfLiteAsyncKernel* async_kernel,
                          TfLiteIoType io_type, const char* const** types,
                          size_t* n_types) {
  if (types == nullptr || n_types == nullptr) return;
  const auto& buf_types = reinterpret_cast<const BackendAsyncKernelInterface*>(
                              TfLiteAsyncKernelGetKernelData(async_kernel))
                              ->SupportedBufferTypes(io_type);
  *types = buf_types.data();
  *n_types = buf_types.size();
}
void SupportedSynchronizations(const TfLiteAsyncKernel* async_kernel,
                               TfLiteIoType io_type, const char* const** types,
                               size_t* n_types) {
  if (types == nullptr || n_types == nullptr) return;
  const auto& sync_types = reinterpret_cast<const BackendAsyncKernelInterface*>(
                               TfLiteAsyncKernelGetKernelData(async_kernel))
                               ->SupportedSynchronizations(io_type);
  *types = sync_types.data();
  *n_types = sync_types.size();
}
bool ReconcileRestrictions(const TfLiteAsyncKernel* async_kernel,
                           const TfLiteOpaqueContext* context,
                           const TfLiteOpaqueNode* node, int tensor_index,
                           const TfLiteAttributeMap* user_provided_attributes,
                           TfLiteAttributeMap* merged,
                           TfLiteAttributeMap* conflict) {
  return reinterpret_cast<const BackendAsyncKernelInterface*>(
             TfLiteAsyncKernelGetKernelData(async_kernel))
      ->ReconcileRestrictions(context, node, tensor_index,
                              user_provided_attributes, merged, conflict);
}
TfLiteStatus SetAttributes(TfLiteAsyncKernel* async_kernel,
                           TfLiteOpaqueContext* context, TfLiteOpaqueNode* node,
                           int tensor_index, const TfLiteAttributeMap* attrs) {
  return reinterpret_cast<BackendAsyncKernelInterface*>(
             TfLiteAsyncKernelGetKernelData(async_kernel))
      ->SetAttributes(context, node, tensor_index, attrs);
}
TfLiteStatus Prepare(TfLiteAsyncKernel* async_kernel,
                     TfLiteOpaqueContext* context, TfLiteOpaqueNode* node) {
  return reinterpret_cast<BackendAsyncKernelInterface*>(
             TfLiteAsyncKernelGetKernelData(async_kernel))
      ->Prepare(context, node);
}
TfLiteStatus Eval(TfLiteAsyncKernel* async_kernel, TfLiteOpaqueContext* context,
                  TfLiteOpaqueNode* node, TfLiteExecutionTask* task) {
  return reinterpret_cast<BackendAsyncKernelInterface*>(
             TfLiteAsyncKernelGetKernelData(async_kernel))
      ->Eval(context, node, task);
}
TfLiteStatus Wait(TfLiteAsyncKernel* async_kernel, TfLiteOpaqueContext* context,
                  TfLiteExecutionTask* task) {
  return reinterpret_cast<BackendAsyncKernelInterface*>(
             TfLiteAsyncKernelGetKernelData(async_kernel))
      ->Wait(context, task);
}
TfLiteStatus Finish(TfLiteAsyncKernel* async_kernel,
                    TfLiteOpaqueContext* context, TfLiteExecutionTask* task) {
  return reinterpret_cast<BackendAsyncKernelInterface*>(
             TfLiteAsyncKernelGetKernelData(async_kernel))
      ->Finish(context, task);
}
TfLiteStatus SetBufferAttributes(TfLiteAsyncKernel* async_kernel,
                                 const TfLiteBackendBuffer* buffer,
                                 const TfLiteAttributeMap* attrs) {
  return reinterpret_cast<BackendAsyncKernelInterface*>(
             TfLiteAsyncKernelGetKernelData(async_kernel))
      ->SetBufferAttributes(buffer, attrs);
}
TfLiteStatus GetBufferAttributes(TfLiteAsyncKernel* async_kernel,
                                 const TfLiteBackendBuffer* buffer,
                                 TfLiteAttributeMap* attrs) {
  return reinterpret_cast<BackendAsyncKernelInterface*>(
             TfLiteAsyncKernelGetKernelData(async_kernel))
      ->GetBufferAttributes(buffer, attrs);
}
}  
BackendAsyncKernelInterface::BackendAsyncKernelInterface() {
  kernel_ = TfLiteAsyncKernelCreate(this);
  TfLiteAsyncKernelSetRegisterBuffer(kernel_, internal::RegisterBuffer);
  TfLiteAsyncKernelSetRegisterBufferSlice(kernel_,
                                          internal::RegisterBufferSlice);
  TfLiteAsyncKernelSetUnregisterBuffer(kernel_, internal::UnregisterBuffer);
  TfLiteAsyncKernelSetSupportedBufferTypes(kernel_,
                                           internal::SupportedBufferTypes);
  TfLiteAsyncKernelSetSupportedSynchronizations(
      kernel_, internal::SupportedSynchronizations);
  TfLiteAsyncKernelSetReconcileRestrictions(kernel_,
                                            internal::ReconcileRestrictions);
  TfLiteAsyncKernelSetSetAttributes(kernel_, internal::SetAttributes);
  TfLiteAsyncKernelSetSetBufferAttributes(kernel_,
                                          internal::SetBufferAttributes);
  TfLiteAsyncKernelSetGetBufferAttributes(kernel_,
                                          internal::GetBufferAttributes);
  TfLiteAsyncKernelSetPrepare(kernel_, internal::Prepare);
  TfLiteAsyncKernelSetEval(kernel_, internal::Eval);
  TfLiteAsyncKernelSetWait(kernel_, internal::Wait);
  TfLiteAsyncKernelSetFinish(kernel_, internal::Finish);
}
}  
}  