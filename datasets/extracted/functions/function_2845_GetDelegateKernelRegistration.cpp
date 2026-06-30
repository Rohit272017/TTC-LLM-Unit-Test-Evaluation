#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include <stddef.h>
#include <stdint.h>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "tensorflow/lite/array.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/utils.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"
namespace tflite {
namespace {
TfLiteRegistration GetDelegateKernelRegistration(
    SimpleDelegateInterface* delegate) {
  TfLiteRegistration kernel_registration{};
  kernel_registration.profiling_string = nullptr;
  kernel_registration.builtin_code = kTfLiteBuiltinDelegate;
  kernel_registration.custom_name = delegate->Name();
  kernel_registration.version = 1;
  kernel_registration.free = [](TfLiteContext* context, void* buffer) -> void {
    delete reinterpret_cast<SimpleDelegateKernelInterface*>(buffer);
  };
  kernel_registration.init = [](TfLiteContext* context, const char* buffer,
                                size_t length) -> void* {
    const TfLiteDelegateParams* params =
        reinterpret_cast<const TfLiteDelegateParams*>(buffer);
    if (params == nullptr) {
      TF_LITE_KERNEL_LOG(context, "NULL TfLiteDelegateParams passed.");
      return nullptr;
    }
    auto* delegate =
        reinterpret_cast<SimpleDelegateInterface*>(params->delegate->data_);
    std::unique_ptr<SimpleDelegateKernelInterface> delegate_kernel(
        delegate->CreateDelegateKernelInterface());
    if (delegate_kernel->Init(context, params) != kTfLiteOk) {
      return nullptr;
    }
    return delegate_kernel.release();
  };
  kernel_registration.prepare = [](TfLiteContext* context,
                                   TfLiteNode* node) -> TfLiteStatus {
    if (node->user_data == nullptr) {
      TF_LITE_KERNEL_LOG(context, "Delegate kernel was not initialized");
      return kTfLiteError;
    }
    SimpleDelegateKernelInterface* delegate_kernel =
        reinterpret_cast<SimpleDelegateKernelInterface*>(node->user_data);
    return delegate_kernel->Prepare(context, node);
  };
  kernel_registration.invoke = [](TfLiteContext* context,
                                  TfLiteNode* node) -> TfLiteStatus {
    SimpleDelegateKernelInterface* delegate_kernel =
        reinterpret_cast<SimpleDelegateKernelInterface*>(node->user_data);
    TFLITE_DCHECK(delegate_kernel != nullptr);
    return delegate_kernel->Eval(context, node);
  };
  return kernel_registration;
}
TfLiteStatus DelegatePrepare(TfLiteContext* context,
                             TfLiteDelegate* base_delegate) {
  auto* delegate =
      reinterpret_cast<SimpleDelegateInterface*>(base_delegate->data_);
  auto delegate_options = delegate->DelegateOptions();
  if (delegate_options.max_delegated_partitions <= 0)
    delegate_options.max_delegated_partitions = std::numeric_limits<int>::max();
  TF_LITE_ENSURE_STATUS(delegate->Initialize(context));
  delegates::IsNodeSupportedFn node_supported_fn =
      [=](TfLiteContext* context, TfLiteNode* node,
          TfLiteRegistration* registration,
          std::string* unsupported_details) -> bool {
    return delegate->IsNodeSupportedByDelegate(registration, node, context);
  };
  delegates::GraphPartitionHelper helper(context, node_supported_fn);
  TF_LITE_ENSURE_STATUS(helper.Partition(nullptr));
  std::vector<int> supported_nodes = helper.GetNodesOfFirstNLargestPartitions(
      delegate_options.max_delegated_partitions,
      delegate_options.min_nodes_per_partition);
  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                       "%s delegate: %d nodes delegated out of %d nodes with "
                       "%d partitions.\n",
                       delegate->Name(), supported_nodes.size(),
                       helper.num_total_nodes(), helper.num_partitions());
  TfLiteRegistration delegate_kernel_registration =
      GetDelegateKernelRegistration(delegate);
  return context->ReplaceNodeSubsetsWithDelegateKernels(
      context, delegate_kernel_registration,
      BuildTfLiteArray(supported_nodes).get(), base_delegate);
}
}  
TfLiteDelegate* TfLiteDelegateFactory::CreateSimpleDelegate(
    std::unique_ptr<SimpleDelegateInterface> simple_delegate, int64_t flag) {
  if (simple_delegate == nullptr) {
    return nullptr;
  }
  auto delegate = new TfLiteDelegate{};
  delegate->Prepare = &DelegatePrepare;
  delegate->flags = flag;
  delegate->data_ = simple_delegate.release();
  delegate->CopyFromBufferHandle = [](TfLiteContext* context,
                                      TfLiteDelegate* delegate,
                                      TfLiteBufferHandle buffer_handle,
                                      TfLiteTensor* tensor) -> TfLiteStatus {
    auto* simple_delegate =
        reinterpret_cast<SimpleDelegateInterface*>(delegate->data_);
    return simple_delegate->CopyFromBufferHandle(context, buffer_handle,
                                                 tensor);
  };
  delegate->CopyToBufferHandle = [](TfLiteContext* context,
                                    TfLiteDelegate* delegate,
                                    TfLiteBufferHandle buffer_handle,
                                    TfLiteTensor* tensor) -> TfLiteStatus {
    auto* simple_delegate =
        reinterpret_cast<SimpleDelegateInterface*>(delegate->data_);
    return simple_delegate->CopyToBufferHandle(context, buffer_handle, tensor);
  };
  delegate->FreeBufferHandle = [](TfLiteContext* context,
                                  TfLiteDelegate* delegate,
                                  TfLiteBufferHandle* buffer_handle) {
    auto* simple_delegate =
        reinterpret_cast<SimpleDelegateInterface*>(delegate->data_);
    simple_delegate->FreeBufferHandle(context, buffer_handle);
  };
  return delegate;
}
void TfLiteDelegateFactory::DeleteSimpleDelegate(TfLiteDelegate* delegate) {
  if (!delegate) return;
  SimpleDelegateInterface* simple_delegate =
      reinterpret_cast<SimpleDelegateInterface*>(delegate->data_);
  delete simple_delegate;
  delete delegate;
}
}  