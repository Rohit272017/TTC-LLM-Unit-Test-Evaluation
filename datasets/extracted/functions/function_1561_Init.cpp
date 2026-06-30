#include <stddef.h>
#include <cstring>
#include <memory>
#include <vector>
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/experimental/resource/initialization_status.h"
#include "tensorflow/lite/kernels/kernel_util.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace call_once_kernel {
struct OpData {
  int init_subgraph_index;
};
void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData;
  const auto* params = reinterpret_cast<const TfLiteCallOnceParams*>(buffer);
  op_data->init_subgraph_index = params->init_subgraph_index;
  return op_data;
}
void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  resource::InitializationStatusMap* map =
      &this_subgraph->initialization_status_map();
  resource::InitializationStatus* status =
      resource::GetInitializationStatus(map, op_data->init_subgraph_index);
  if (status->IsInitialized()) return kTfLiteOk;
  auto* subgraphs = this_subgraph->GetSubgraphs();
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 0);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 0);
  TF_LITE_ENSURE(context, op_data->init_subgraph_index < subgraphs->size());
  Subgraph* init_subgraph = (*subgraphs)[op_data->init_subgraph_index].get();
  TF_LITE_ENSURE_EQ(context, init_subgraph->inputs().size(), 0);
  TF_LITE_ENSURE_EQ(context, init_subgraph->outputs().size(), 0);
  return kTfLiteOk;
}
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  resource::InitializationStatusMap* map =
      &this_subgraph->initialization_status_map();
  resource::InitializationStatus* status =
      resource::GetInitializationStatus(map, op_data->init_subgraph_index);
  if (status->IsInitialized()) return kTfLiteOk;
  auto* subgraphs = this_subgraph->GetSubgraphs();
  Subgraph& init_subgraph = *(*subgraphs)[op_data->init_subgraph_index];
  TF_LITE_ENSURE_OK(context, init_subgraph.AllocateTensors());
  TF_LITE_ENSURE_OK(context, init_subgraph.Invoke());
  TF_LITE_ENSURE_OK(context, init_subgraph.ReleaseNonPersistentMemory());
  status->MarkInitializationIsDone();
  return kTfLiteOk;
}
}  
TfLiteRegistration* Register_CALL_ONCE() {
  static TfLiteRegistration r = {call_once_kernel::Init, call_once_kernel::Free,
                                 call_once_kernel::Prepare,
                                 call_once_kernel::Eval};
  return &r;
}
}  
}  
}  