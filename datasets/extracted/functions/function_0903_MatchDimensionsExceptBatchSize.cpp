#include <stddef.h>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#include "flatbuffers/flexbuffers.h"  
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/call_register.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
namespace tflite {
namespace acceleration {
namespace ops {
namespace call_kernel {
namespace {
bool MatchDimensionsExceptBatchSize(TfLiteTensor* a, TfLiteTensor* b) {
  if (a->dims->size != b->dims->size) {
    return false;
  }
  for (int i = 1; i < a->dims->size; ++i) {
    if (a->dims->data[i] != b->dims->data[i]) {
      return false;
    }
  }
  return true;
}
TfLiteStatus ValidateAndResizeInputsIfNeeded(TfLiteContext* context,
                                             TfLiteNode* node,
                                             Subgraph* subgraph,
                                             int loop_count) {
  TF_LITE_ENSURE_EQ(context, subgraph->inputs().size(), node->inputs->size);
  for (int i = 0; i < node->inputs->size; ++i) {
    TfLiteTensor* node_input = context->tensors + node->inputs->data[i];
    TfLiteTensor* subgraph_input = subgraph->tensor(subgraph->inputs()[i]);
    TF_LITE_ENSURE_TYPES_EQ(context, node_input->type, subgraph_input->type);
    TF_LITE_ENSURE_MSG(
        context, node_input->dims->size > 0,
        "Dimensions of all of call node's inputs should be non-zero.");
    TF_LITE_ENSURE_EQ(context, node_input->dims->data[0], loop_count);
    if (!subgraph_input->dims->size) {
      std::vector<int> new_dims;
      new_dims.reserve(node_input->dims->size);
      new_dims.push_back(1);  
      new_dims.insert(new_dims.end(), node_input->dims->data + 1,
                      node_input->dims->data + node_input->dims->size);
      subgraph->ResizeInputTensor(subgraph->inputs()[i], new_dims);
    } else {
      if (!MatchDimensionsExceptBatchSize(node_input, subgraph_input)) {
        std::stringstream node_input_dims, subgraph_input_dims;
        for (int i = 0; i < node_input->dims->size; i++) {
          node_input_dims << node_input->dims->data[i] << " ";
          subgraph_input_dims << subgraph_input->dims->data[i] << " ";
        }
        TF_LITE_KERNEL_LOG(
            context,
            "%s:%d: All dimensions except the batch size should match for call "
            "node and the subgraph to invoke (input tensor %s[ %s], subgraph "
            "tensor %s[ %s])",
            __FILE__, __LINE__, node_input->name, node_input_dims.str().c_str(),
            subgraph_input->name, subgraph_input_dims.str().c_str());
        return kTfLiteError;
      }
      TF_LITE_ENSURE_EQ(context, subgraph_input->dims->data[0], 1);
    }
  }
  return kTfLiteOk;
}
TfLiteStatus ValidateAndResizeOutputs(TfLiteContext* context, TfLiteNode* node,
                                      Subgraph* subgraph, int loop_count) {
  TF_LITE_ENSURE_EQ(context, subgraph->outputs().size(), node->outputs->size);
  for (int i = 0; i < node->outputs->size; ++i) {
    const TfLiteTensor* subgraph_output =
        subgraph->tensor(subgraph->outputs()[i]);
    TfLiteTensor* node_output = context->tensors + node->outputs->data[i];
    TF_LITE_ASSERT(subgraph_output->dims->size > 0);
    TfLiteIntArray* new_dims_array = TfLiteIntArrayCopy(subgraph_output->dims);
    new_dims_array->data[0] = loop_count;
    TF_LITE_ENSURE_OK(
        context, context->ResizeTensor(context, node_output, new_dims_array));
    node_output->type = subgraph_output->type;
  }
  return kTfLiteOk;
}
TfLiteStatus CopyInputTensorsData(TfLiteContext* context, TfLiteNode* node,
                                  Subgraph* dst_subgraph, int loop_index,
                                  int loop_count) {
  const std::vector<int>& dst_tensor_indices = dst_subgraph->inputs();
  TF_LITE_ENSURE_EQ(context, node->inputs->size, dst_tensor_indices.size());
  for (int i = 0; i < dst_tensor_indices.size(); ++i) {
    TfLiteTensor* src_tensor = context->tensors + node->inputs->data[i];
    TfLiteTensor* dst_tensor = dst_subgraph->tensor(dst_tensor_indices[i]);
    size_t offset = src_tensor->bytes / loop_count * loop_index;
    TF_LITE_ENSURE_EQ(context, src_tensor->bytes / loop_count,
                      dst_tensor->bytes);
    memcpy(dst_tensor->data.raw, src_tensor->data.raw + offset,
           src_tensor->bytes / loop_count);
  }
  return kTfLiteOk;
}
TfLiteStatus CopyOutputTensorsData(TfLiteContext* context,
                                   Subgraph* src_subgraph, TfLiteNode* node,
                                   int loop_index, int loop_count) {
  const std::vector<int>& src_tensor_indices = src_subgraph->outputs();
  TF_LITE_ENSURE_EQ(context, src_tensor_indices.size(), node->outputs->size);
  for (int i = 0; i < src_tensor_indices.size(); ++i) {
    const TfLiteTensor* src_tensor =
        src_subgraph->tensor(src_tensor_indices[i]);
    TfLiteTensor* dst_tensor = context->tensors + node->outputs->data[i];
    size_t offset = dst_tensor->bytes / loop_count * loop_index;
    TF_LITE_ENSURE_EQ(context, src_tensor->bytes,
                      dst_tensor->bytes / loop_count);
    memcpy(dst_tensor->data.raw + offset, src_tensor->data.raw,
           src_tensor->bytes);
  }
  return kTfLiteOk;
}
}  
struct OpData {
  int subgraph_index;
  int loop_count;
};
void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  if (!buffer) {
    return nullptr;
  }
  auto* op_data = new OpData;
  const uint8_t* buffer_fixed_width = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& map =
      flexbuffers::GetRoot(buffer_fixed_width, length).AsMap();
  op_data->subgraph_index = map["subgraph_index"].AsInt32();
  op_data->loop_count = map["loop_count"].AsInt32();
  return op_data;
}
void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE(context, op_data);
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();
  TF_LITE_ENSURE_MSG(context,
                     (op_data->subgraph_index < subgraphs->size()) &&
                         (op_data->subgraph_index >= 0),
                     "Index of subgraph to be invoked is invalid.");
  Subgraph* subgraph = (*subgraphs)[op_data->subgraph_index].get();
  TF_LITE_ENSURE_MSG(
      context, subgraph != this_subgraph,
      "Subgraph to invoke must be different from the invoking graph.");
  int loop_count = op_data->loop_count;
  TF_LITE_ENSURE_MSG(context, loop_count >= 0, "Loop count must be positive. ");
  TF_LITE_ENSURE_OK(context, ValidateAndResizeInputsIfNeeded(
                                 context, node, subgraph, loop_count));
  TF_LITE_ENSURE_OK(context, subgraph->AllocateTensors());
  TF_LITE_ENSURE_OK(
      context, ValidateAndResizeOutputs(context, node, subgraph, loop_count));
  TF_LITE_ENSURE(context, !subgraph->HasDynamicTensors());
  return kTfLiteOk;
}
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();
  Subgraph* subgraph = (*subgraphs)[op_data->subgraph_index].get();
  for (int loop_index = 0; loop_index < op_data->loop_count; loop_index++) {
    TF_LITE_ENSURE_OK(context,
                      CopyInputTensorsData(context, node, subgraph, loop_index,
                                           op_data->loop_count));
    TF_LITE_ENSURE_OK(context, subgraph->Invoke());
    for (int tensor_index : subgraph->outputs()) {
      subgraph->EnsureTensorDataIsReadable(tensor_index);
    }
    TF_LITE_ENSURE_OK(context,
                      CopyOutputTensorsData(context, subgraph, node, loop_index,
                                            op_data->loop_count));
  }
  return kTfLiteOk;
}
}  
TfLiteRegistration* Register_CALL() {
  static TfLiteRegistration r = {call_kernel::Init, call_kernel::Free,
                                 call_kernel::Prepare, call_kernel::Eval};
  return &r;
}
}  
}  
}  