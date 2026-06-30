#include "tensorflow/lite/arena_planner.h"
#include <stddef.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/graph_info.h"
#include "tensorflow/lite/simple_memory_arena.h"
namespace tflite {
constexpr int32_t kLastActiveNodeUndefined =
    std::numeric_limits<int32_t>::max();
constexpr int32_t kNodeNotAssigned = std::numeric_limits<int32_t>::max();
constexpr int32_t kScalarTensorBytes = 4;
ArenaPlanner::ArenaPlanner(TfLiteContext* context,
                           std::unique_ptr<GraphInfo> graph_info,
                           bool preserve_all_tensors, int tensor_alignment,
                           int subgraph_index)
    : context_(context),
      graph_info_(std::move(graph_info)),
      arena_(kDefaultArenaAlignment, subgraph_index),
      has_nonpersistent_memory_(false),
      persistent_arena_(kDefaultArenaAlignment, subgraph_index),
      preserve_all_tensors_(preserve_all_tensors),
      tensor_alignment_(tensor_alignment),
      last_active_node_(kLastActiveNodeUndefined) {}
ArenaPlanner::~ArenaPlanner() {
  arena_.ReleaseBuffer();
  persistent_arena_.ReleaseBuffer();
}
std::intptr_t ArenaPlanner::BasePointer(TfLiteAllocationType type) {
  if (type == kTfLiteArenaRwPersistent) {
    return persistent_arena_.BasePointer();
  }
  if (type == kTfLiteArenaRw) {
    return arena_.BasePointer();
  }
  return 0;
}
TfLiteStatus ArenaPlanner::ResetAllocations() {
  TF_LITE_ENSURE_STATUS(arena_.ClearPlan());
  TF_LITE_ENSURE_STATUS(persistent_arena_.ClearPlan());
  allocs_.clear();
  allocs_.resize(graph_info_->num_tensors());
  last_active_node_ = kLastActiveNodeUndefined;
  return kTfLiteOk;
}
TfLiteStatus ArenaPlanner::ResetAllocationsAfter(int node) {
  TfLiteTensor* tensors = graph_info_->tensors();
  for (int i = 0; i < static_cast<int>(allocs_.size()); ++i) {
    if (allocs_[i].first_node > node && allocs_[i].size > 0) {
      TfLiteTensor& tensor = tensors[i];
      if (tensor.allocation_type == kTfLiteArenaRw) {
        allocs_[i].reset();
        tensor.data.raw = nullptr;
      }
    }
  }
  if (last_active_node_ > node) {
    arena_.CalculateActiveAllocs(allocs_, node);
  } else {
    arena_.PurgeAfter(node);
  }
  last_active_node_ = node;
  return kTfLiteOk;
}
int ArenaPlanner::FindSharedTensor(int tensor_index) {
  auto actual_tensor_it = actual_tensor_id_.find(tensor_index);
  if (actual_tensor_it != actual_tensor_id_.end()) {
    tensor_index = actual_tensor_it->second;
  }
  return tensor_index;
}
bool ArenaPlanner::InputTensorCanBeShared(const TfLiteTensor& input_tensor,
                                          const TfLiteTensor& output_tensor,
                                          int input_id, int output_id,
                                          bool tensor_changed) {
  if (tensor_changed) {
    if (input_tensor.bytes != output_tensor.bytes ||
        input_tensor.bytes <= kScalarTensorBytes) {
      return false;
    }
    if (refcounts_[input_id] > 1) {
      return false;
    }
  }
  for (int input : graph_info_->inputs()) {
    if (input == input_id) {
      return false;
    }
  }
  for (int output : graph_info_->outputs()) {
    if (output == output_id) {
      return false;
    }
  }
  TfLiteAllocationType input_allocation_type = input_tensor.allocation_type;
  TfLiteAllocationType output_allocation_type = output_tensor.allocation_type;
  if (input_allocation_type != output_allocation_type &&
      input_allocation_type != kTfLiteArenaRw) {
    return false;
  }
  if (preserve_all_tensors_) {
    return false;
  }
  return true;
}
void ArenaPlanner::IdentifyInPlaceTensors() {
  actual_tensor_id_.clear();
  const int num_execution_nodes = graph_info_->num_execution_nodes();
  TfLiteTensor* tensors = graph_info_->tensors();
  for (int i = 0; i < num_execution_nodes; ++i) {
    const TfLiteRegistration& registration = graph_info_->registration(i);
    const TfLiteNode& node = graph_info_->node(i);
    if (node.outputs->size < 1) continue;
    bool tensor_changed =
        !(registration.inplace_operator & kTfLiteInplaceOpDataUnmodified);
    if (registration.inplace_operator == kTfLiteInplaceOpNone) {
      continue;
    }
    int32_t input_id = -1;
    int32_t output_id = node.outputs->data[0];
    const TfLiteTensor& output_tensor = tensors[output_id];
    const int loop_end =
        std::min(kTfLiteMaxSharableOpInputs, node.inputs->size);
    for (int i = 0; i < loop_end; ++i) {
      if (node.inputs->data[i] == kTfLiteOptionalTensor) {
        continue;
      }
      const bool input_shareable =
          registration.inplace_operator & (kTfLiteInplaceOpInput0Shared << i);
      if (input_shareable) {
        const TfLiteTensor& input_tensor = tensors[node.inputs->data[i]];
        if (InputTensorCanBeShared(input_tensor, output_tensor,
                                   node.inputs->data[i], output_id,
                                   tensor_changed)) {
          input_id = node.inputs->data[i];
          break;
        }
      }
    }
    if (input_id == -1) {
      continue;
    }
    int32_t actual_output_tensor_id = FindSharedTensor(input_id);
    if (tensor_changed) {
      if (refcounts_[actual_output_tensor_id] > 1) {
        continue;
      }
    }
    actual_tensor_id_[output_id] = actual_output_tensor_id;
  }
}
TfLiteStatus ArenaPlanner::PlanAllocations() {
  const size_t num_tensors = graph_info_->num_tensors();
  TF_LITE_ENSURE_STATUS(ResetAllocations());
  alloc_node_.assign(num_tensors, kNodeNotAssigned);
  dealloc_node_.assign(num_tensors, kNodeNotAssigned);
  nodes_to_tensors_.clear();
  nodes_to_tensors_.resize(
      std::max(graph_info_->num_execution_nodes(), (size_t)1), {});
  refcounts_.assign(num_tensors, 0);
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
      ++refcounts_[tensor_index];
    }
  }
  for (int tensor_index : graph_info_->variables()) {
    ++refcounts_[tensor_index];
    TF_LITE_ENSURE(context_, tensor_index != kTfLiteOptionalTensor);
    TF_LITE_ENSURE_STATUS(allocate(0, tensor_index));
    nodes_to_tensors_[0].insert(tensor_index);
  }
  for (int tensor_index : graph_info_->inputs()) {
    if (tensor_index != kTfLiteOptionalTensor) {
      ++refcounts_[tensor_index];
      TF_LITE_ENSURE_STATUS(allocate(0, tensor_index));
      nodes_to_tensors_[0].insert(tensor_index);
    }
  }
  std::vector<int> refcounts = refcounts_;
  const int num_execution_nodes = graph_info_->num_execution_nodes();
  for (size_t i = 0; i < num_execution_nodes; ++i) {
    const TfLiteNode& node = graph_info_->node(i);
    TfLiteIntArray* node_inputs = node.inputs;
    for (int j = 0; j < node_inputs->size; ++j) {
      int tensor_index = node_inputs->data[j];
      if (tensor_index != kTfLiteOptionalTensor) {
        ++refcounts_[tensor_index];
      }
    }
  }
  IdentifyInPlaceTensors();
  for (size_t i = 0; i < num_execution_nodes; ++i) {
    const TfLiteNode& node = graph_info_->node(i);
    TfLiteIntArray* node_inputs = node.inputs;
    for (int j = 0; j < node_inputs->size; ++j) {
      int tensor_index = node_inputs->data[j];
      if (tensor_index != kTfLiteOptionalTensor) {
        tensor_index = FindSharedTensor(tensor_index);
        ++refcounts[tensor_index];
      }
    }
  }
  for (size_t i = 0; i < num_execution_nodes; ++i) {
    const TfLiteNode& node = graph_info_->node(i);
    TfLiteIntArray* node_outputs = node.outputs;
    for (int j = 0; j < node_outputs->size; ++j) {
      int tensor_index = node_outputs->data[j];
      if (tensor_index == kTfLiteOptionalTensor) continue;
      nodes_to_tensors_[i].insert(tensor_index);
      TF_LITE_ENSURE_STATUS(allocate(i, tensor_index));
    }
    if (!preserve_all_tensors_) {
      TfLiteIntArray* node_inputs = node.inputs;
      for (int j = 0; j < node_inputs->size; ++j) {
        int tensor_index = node_inputs->data[j];
        if (tensor_index != kTfLiteOptionalTensor) {
          tensor_index = FindSharedTensor(tensor_index);
          --refcounts[tensor_index];
          if (refcounts[tensor_index] == 0) {
            TF_LITE_ENSURE_STATUS(deallocate(i, tensor_index));
          }
        }
      }
    }
  }
  return kTfLiteOk;
}
TfLiteStatus ArenaPlanner::ExecuteAllocations(int first_node, int last_node) {
  const size_t num_tensors = graph_info_->num_tensors();
  TF_LITE_ENSURE(context_, num_tensors >= allocs_.size());
  alloc_node_.resize(num_tensors, kNodeNotAssigned);
  dealloc_node_.resize(num_tensors, kNodeNotAssigned);
  allocs_.resize(num_tensors);
  const int num_execution_nodes = graph_info_->num_execution_nodes();
  for (size_t i = first_node;
       i <= static_cast<size_t>(last_node) && i < num_execution_nodes; ++i) {
    const TfLiteNode& node = graph_info_->node(i);
    TfLiteIntArray* node_temporaries = node.temporaries;
    for (int j = 0; j < node_temporaries->size; ++j) {
      int tensor_index = node_temporaries->data[j];
      alloc_node_[tensor_index] = i;
      nodes_to_tensors_[i].insert(tensor_index);
      if (!preserve_all_tensors_) {
        dealloc_node_[tensor_index] = i;
      }
    }
  }
  std::vector<int32_t> tensors_allocated;
  TF_LITE_ENSURE_STATUS(
      CalculateAllocations(first_node, last_node, &tensors_allocated));
  bool arena_reallocated = false;
  TF_LITE_ENSURE_STATUS(Commit(&arena_reallocated));
  TfLiteTensor* tensors = graph_info_->tensors();
  if (arena_reallocated) {
    for (int i = 0; i < static_cast<int>(num_tensors); ++i) {
      TF_LITE_ENSURE_STATUS(ResolveTensorAllocation(i, tensors));
    }
  } else {
    for (int i = 0; i < static_cast<int>(tensors_allocated.size()); ++i) {
      TF_LITE_ENSURE_STATUS(
          ResolveTensorAllocation(tensors_allocated[i], tensors));
    }
  }
  return kTfLiteOk;
}
TfLiteStatus ArenaPlanner::ReleaseNonPersistentMemory() {
  TF_LITE_ENSURE_STATUS(arena_.ReleaseBuffer());
  has_nonpersistent_memory_ = false;
  TfLiteTensor* tensors = graph_info_->tensors();
  for (int i = 0; i < static_cast<int>(graph_info_->num_tensors()); ++i) {
    TfLiteTensor& tensor = tensors[i];
    if (tensor.allocation_type == kTfLiteArenaRw) {
      tensor.data.raw = nullptr;
    }
  }
  return kTfLiteOk;
}
TfLiteStatus ArenaPlanner::AcquireNonPersistentMemory() {
  bool reallocated;
  TF_LITE_ENSURE_STATUS(arena_.Commit(&reallocated));
  has_nonpersistent_memory_ = true;
  TfLiteTensor* tensors = graph_info_->tensors();
  for (int i = 0; i < static_cast<int>(graph_info_->num_tensors()); ++i) {
    TfLiteTensor& tensor = tensors[i];
    if (tensor.allocation_type == kTfLiteArenaRw) {
      TF_LITE_ENSURE_STATUS(ResolveTensorAllocation(i, tensors));
    }
  }
  return kTfLiteOk;
}
bool ArenaPlanner::HasNonPersistentMemory() {
  return has_nonpersistent_memory_;
}
void ArenaPlanner::DumpDebugInfo(const std::vector<int>& execution_plan) const {
  arena_.DumpDebugInfo("kTfLiteArenaRw Dump:", execution_plan);
  persistent_arena_.DumpDebugInfo("kTfLiteArenaRwPersistent Dump:",
                                  execution_plan);
}
void ArenaPlanner::GetAllocInfo(size_t* arena_size,
                                size_t* arena_persist_size) const {
  *arena_size = arena_.GetBufferSize();
  *arena_persist_size = persistent_arena_.GetBufferSize();
}
TfLiteStatus ArenaPlanner::Commit(bool* reallocated) {
  bool arena_reallocated, persistent_arena_reallocated;
  TF_LITE_ENSURE_STATUS(arena_.Commit(&arena_reallocated));
  has_nonpersistent_memory_ = true;
  TF_LITE_ENSURE_STATUS(
      persistent_arena_.Commit(&persistent_arena_reallocated));
  *reallocated = arena_reallocated;
  *reallocated |= persistent_arena_reallocated;
  return kTfLiteOk;
}
void ArenaPlanner::CreateTensorAllocationVector(
    std::vector<int32_t>* tensors_to_allocate) {
  const TfLiteTensor* tensors = this->graph_info_->tensors();
  auto tensor_compare = [&](int idx1, int idx2) {
    if (alloc_node_[idx1] == 0 && dealloc_node_[idx1] == kNodeNotAssigned) {
      if (alloc_node_[idx2] == 0 && dealloc_node_[idx2] == kNodeNotAssigned) {
        return idx1 < idx2;
      }
      return true;
    }
    if (alloc_node_[idx2] == 0 && dealloc_node_[idx2] == kNodeNotAssigned) {
      return false;
    }
    auto size1 = tensors[idx1].bytes;
    auto size2 = tensors[idx2].bytes;
    if (size1 != size2) {
      return size1 > size2;
    }
    return alloc_node_[idx1] < alloc_node_[idx2];
  };
  std::sort(tensors_to_allocate->begin(), tensors_to_allocate->end(),
            tensor_compare);
}
std::vector<int32_t> ArenaPlanner::GetTensorsToAllocate(int first_node,
                                                        int last_node) {
  int num_tensors = static_cast<int>(graph_info_->num_tensors());
  std::vector<int32_t> tensors_to_allocate;
  tensors_to_allocate.reserve(num_tensors);
  for (int i = first_node; i <= last_node; ++i) {
    tensors_to_allocate.insert(tensors_to_allocate.end(),
                               nodes_to_tensors_[i].begin(),
                               nodes_to_tensors_[i].end());
  }
  return tensors_to_allocate;
}
TfLiteStatus ArenaPlanner::CalculateAllocations(
    int first_node, int last_node, std::vector<int32_t>* tensors_allocated) {
  const std::vector<int32_t> tensors_to_allocate =
      GetTensorsToAllocate(first_node, last_node);
  tensors_allocated->reserve(tensors_to_allocate.size());
  TfLiteTensor* tensors = graph_info_->tensors();
  for (const auto& tensor_index : tensors_to_allocate) {
    TfLiteTensor& tensor = tensors[tensor_index];
    if (tensor.allocation_type == kTfLiteArenaRw) {
      if (allocs_[tensor_index].size < tensor.bytes) {
        tensors_allocated->push_back(tensor_index);
      }
    } else if (tensor.allocation_type == kTfLiteArenaRwPersistent) {
      tensors_allocated->push_back(tensor_index);
    }
  }
  if (tensors_allocated->empty()) {
    last_active_node_ = last_node;
    return kTfLiteOk;
  }
  if (first_node < last_active_node_) {
    arena_.ResetAllocs();
    last_active_node_ = first_node;
  } else {
    arena_.PurgeActiveAllocs(first_node);
  }
  CreateTensorAllocationVector(tensors_allocated);
  for (const auto& tensor_index : *tensors_allocated) {
    TfLiteTensor& tensor = tensors[tensor_index];
    auto it = actual_tensor_id_.find(tensor_index);
    if (it != actual_tensor_id_.end()) {
      TfLiteAllocationType allocation_type =
          tensors[it->second].allocation_type;
      if (allocation_type != kTfLiteArenaRw ||
          tensors[it->second].bytes != tensors[it->first].bytes) {
        actual_tensor_id_.erase(it);
      } else {
        continue;
      }
    }
    if (tensor.allocation_type == kTfLiteArenaRw) {
      TF_LITE_ENSURE_STATUS(
          arena_.Allocate(context_, tensor_alignment_, tensor.bytes,
                          tensor_index, alloc_node_[tensor_index],
                          dealloc_node_[tensor_index], &allocs_[tensor_index]));
    }
    if (tensor.allocation_type == kTfLiteArenaRwPersistent &&
        allocs_[tensor_index].size == 0) {
      if (allocs_[tensor_index].size < tensor.bytes) {
        TF_LITE_ENSURE_STATUS(persistent_arena_.Allocate(
            context_, tensor_alignment_, tensor.bytes, tensor_index,
            alloc_node_[tensor_index],
            std::numeric_limits<int32_t>::max(),
            &allocs_[tensor_index]));
      }
    }
  }
  last_active_node_ = last_node;
  return kTfLiteOk;
}
bool AreTensorsAllocatedInSameArena(int32_t root_tensor_index,
                                    int32_t tensor_index,
                                    const TfLiteTensor* tensors) {
  if (tensors[root_tensor_index].allocation_type == kTfLiteArenaRw &&
      tensors[tensor_index].allocation_type == kTfLiteArenaRw) {
    return true;
  }
  if (tensors[root_tensor_index].allocation_type == kTfLiteArenaRwPersistent &&
      tensors[tensor_index].allocation_type == kTfLiteArenaRwPersistent) {
    return true;
  }
  return false;
}
TfLiteStatus ArenaPlanner::ResolveTensorAllocation(int32_t tensor_index,
                                                   TfLiteTensor* tensors) {
  auto actual_tensor_it = actual_tensor_id_.find(tensor_index);
  TfLiteTensor& tensor = tensors[tensor_index];
  int32_t root_tensor_index = actual_tensor_it == actual_tensor_id_.end()
                                  ? tensor_index
                                  : actual_tensor_it->second;
  const TfLiteTensor& root_tensor = tensors[root_tensor_index];
  if (root_tensor_index != tensor_index) {
    if (AreTensorsAllocatedInSameArena(root_tensor_index, tensor_index,
                                       tensors)) {
      ResolveTensorAllocation(root_tensor_index, tensors);
      tensor.data.data = root_tensor.data.data;
      return kTfLiteOk;
    }
  }
  if (tensor.allocation_type == kTfLiteArenaRw) {
    if (allocs_[tensor_index].size != 0) {
      return arena_.ResolveAlloc(context_, allocs_[tensor_index],
                                 &tensor.data.raw);
    }
  }
  if (tensor.allocation_type == kTfLiteArenaRwPersistent) {
    return persistent_arena_.ResolveAlloc(context_, allocs_[tensor_index],
                                          &tensor.data.raw);
  }
  return kTfLiteOk;
}
}  