#include "tensorflow/lite/simple_planner.h"
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/graph_info.h"
namespace tflite {
namespace {
constexpr int32_t kNodeNotAssigned = std::numeric_limits<int32_t>::max();
}  
SimplePlanner::SimplePlanner(TfLiteContext* context,
                             std::unique_ptr<GraphInfo> graph_info)
    : context_(context), graph_info_(std::move(graph_info)) {}
SimplePlanner::~SimplePlanner() { FreeAllAllocations(); }
void SimplePlanner::FreeAllAllocations() {
  for (int i = 0; i < static_cast<int>(allocs_.size()); ++i) {
    allocs_[i].free();
  }
}
TfLiteStatus SimplePlanner::ResetAllocations() {
  FreeAllAllocations();
  allocs_.clear();
  allocs_.resize(graph_info_->num_tensors());
  return kTfLiteOk;
}
TfLiteStatus SimplePlanner::ResetAllocationsAfter(int node) {
  TfLiteTensor* tensors = graph_info_->tensors();
  for (int i = 0; i < static_cast<int>(allocs_.size()); ++i) {
    if (allocs_[i].node > node && allocs_[i].size > 0) {
      TfLiteTensor& tensor = tensors[i];
      if (tensor.allocation_type == kTfLiteArenaRw) {
        allocs_[i].free();
        tensor.data.raw = nullptr;
      }
    }
  }
  return kTfLiteOk;
}
TfLiteStatus SimplePlanner::PlanAllocations() {
  TF_LITE_ENSURE_STATUS(ResetAllocations());
  alloc_node_.assign(graph_info_->num_tensors(), kNodeNotAssigned);
  dealloc_node_.assign(graph_info_->num_tensors(), kNodeNotAssigned);
  std::vector<int> refcounts(graph_info_->num_tensors(), 0);
  auto allocate = [this](int node, int tensor) -> TfLiteStatus {
    if (alloc_node_[tensor] != kNodeNotAssigned) {
      return kTfLiteOk;
    }
    TF_LITE_ENSURE(context_, dealloc_node_[tensor] == kNodeNotAssigned);
    alloc_node_[tensor] = node;
    return kTfLiteOk;
  };
  auto deallocate = [this](int node, int tensor) -> TfLiteStatus {
    if (alloc_node_[tensor] == kNodeNotAssigned) {
      return kTfLiteOk;
    }
    TF_LITE_ENSURE(context_, dealloc_node_[tensor] == kNodeNotAssigned);
    dealloc_node_[tensor] = node;
    return kTfLiteOk;
  };
  for (int tensor_index : graph_info_->outputs()) {
    if (tensor_index != kTfLiteOptionalTensor) {
      refcounts[tensor_index]++;
    }
  }
  for (int tensor_index : graph_info_->variables()) {
    refcounts[tensor_index]++;
    TF_LITE_ENSURE(context_, tensor_index != kTfLiteOptionalTensor);
    TF_LITE_ENSURE_STATUS(allocate(0, tensor_index));
  }
  for (int tensor_index : graph_info_->inputs()) {
    if (tensor_index != kTfLiteOptionalTensor) {
      refcounts[tensor_index]++;
      TF_LITE_ENSURE_STATUS(allocate(0, tensor_index));
    }
  }
  const size_t num_execution_nodes = graph_info_->num_execution_nodes();
  for (size_t i = 0; i < num_execution_nodes; ++i) {
    const TfLiteNode& node = graph_info_->node(i);
    TfLiteIntArray* node_inputs = node.inputs;
    for (int j = 0; j < node_inputs->size; ++j) {
      int tensor_index = node_inputs->data[j];
      if (tensor_index != kTfLiteOptionalTensor) {
        refcounts[tensor_index]++;
      }
    }
  }
  for (size_t i = 0; i < num_execution_nodes; ++i) {
    const TfLiteNode& node = graph_info_->node(i);
    TfLiteIntArray* node_outputs = node.outputs;
    for (int j = 0; j < node_outputs->size; ++j) {
      int tensor_index = node_outputs->data[j];
      TF_LITE_ENSURE_STATUS(allocate(i, tensor_index));
    }
    TfLiteIntArray* node_inputs = node.inputs;
    for (int j = 0; j < node_inputs->size; ++j) {
      int tensor_index = node_inputs->data[j];
      if (tensor_index != kTfLiteOptionalTensor) {
        refcounts[tensor_index]--;
        if (refcounts[tensor_index] == 0) {
          TF_LITE_ENSURE_STATUS(deallocate(i, tensor_index));
        }
      }
    }
  }
  return kTfLiteOk;
}
TfLiteStatus SimplePlanner::ExecuteAllocations(int first_node, int last_node) {
  alloc_node_.resize(graph_info_->num_tensors(), kNodeNotAssigned);
  dealloc_node_.resize(graph_info_->num_tensors(), kNodeNotAssigned);
  allocs_.resize(graph_info_->num_tensors());
  const size_t num_execution_nodes = graph_info_->num_execution_nodes();
  for (size_t i = first_node;
       i <= static_cast<size_t>(last_node) && i < num_execution_nodes; ++i) {
    const TfLiteNode& node = graph_info_->node(i);
    TfLiteIntArray* node_temporaries = node.temporaries;
    for (int j = 0; j < node_temporaries->size; ++j) {
      int tensor_index = node_temporaries->data[j];
      alloc_node_[tensor_index] = i;
      dealloc_node_[tensor_index] = i;
    }
  }
  const int num_tensors = static_cast<int>(graph_info_->num_tensors());
  TfLiteTensor* tensors = graph_info_->tensors();
  for (int i = 0; i < num_tensors; ++i) {
    bool allocated = false;
    if (alloc_node_[i] >= first_node && alloc_node_[i] <= last_node) {
      TfLiteTensor& tensor = tensors[i];
      if (tensor.allocation_type == kTfLiteArenaRw) {
        if (allocs_[i].size != 0) {
          allocs_[i].free();
        }
        allocated = allocs_[i].alloc(tensor.bytes, alloc_node_[i]);
      } else if (tensor.allocation_type == kTfLiteArenaRwPersistent &&
                 allocs_[i].size == 0) {
        allocated = allocs_[i].alloc(tensor.bytes, alloc_node_[i]);
      }
    }
    if (allocated) {
      TF_LITE_ENSURE_STATUS(ResolveTensorAllocation(i));
    }
  }
  return kTfLiteOk;
}
TfLiteStatus SimplePlanner::ReleaseNonPersistentMemory() {
  const int num_tensors = static_cast<int>(graph_info_->num_tensors());
  TfLiteTensor* tensors = graph_info_->tensors();
  for (int i = 0; i < num_tensors; ++i) {
    TfLiteTensor& tensor = tensors[i];
    if (tensor.allocation_type == kTfLiteArenaRw) {
      allocs_[i].free();
      tensor.data.raw = nullptr;
    }
  }
  return kTfLiteOk;
}
TfLiteStatus SimplePlanner::AcquireNonPersistentMemory() {
  const int num_tensors = static_cast<int>(graph_info_->num_tensors());
  TfLiteTensor* tensors = graph_info_->tensors();
  for (int i = 0; i < num_tensors; ++i) {
    TfLiteTensor& tensor = tensors[i];
    if (tensor.allocation_type == kTfLiteArenaRw) {
      TF_LITE_ENSURE_STATUS(ResolveTensorAllocation(i));
    }
  }
  return kTfLiteOk;
}
TfLiteStatus SimplePlanner::ResolveTensorAllocation(int tensor_index) {
  TfLiteTensor& tensor = *graph_info_->tensor(tensor_index);
  if (tensor.allocation_type == kTfLiteArenaRw) {
    if (allocs_[tensor_index].size != 0) {
      tensor.data.raw = allocs_[tensor_index].ptr;
    }
  }
  if (tensor.allocation_type == kTfLiteArenaRwPersistent) {
    tensor.data.raw = allocs_[tensor_index].ptr;
  }
  return kTfLiteOk;
}
}  