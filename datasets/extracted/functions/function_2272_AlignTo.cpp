#include "tensorflow/lite/simple_memory_arena.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/macros.h"
#ifdef TF_LITE_TENSORFLOW_PROFILER
#include "tensorflow/lite/tensorflow_profiler_logger.h"
#endif  
#if defined(__ANDROID__)
#define TF_LITE_HAS_ALIGNED_ALLOC (__ANDROID_API__ >= 28)
#elif defined(__APPLE__)
#define TF_LITE_HAS_ALIGNED_ALLOC 0
#elif defined(_WIN32)
#define TF_LITE_HAS_ALIGNED_ALLOC 0
#elif __cplusplus >= 201703L || __STDC_VERSION__ >= 201112L
#define TF_LITE_HAS_ALIGNED_ALLOC 1
#endif
namespace {
template <typename T>
T AlignTo(size_t alignment, T offset) {
  return offset % alignment == 0 ? offset
                                 : offset + (alignment - offset % alignment);
}
tflite::PointerAlignedPointerPair AlignedAlloc(size_t size, size_t alignment);
void AlignedFree(const tflite::PointerAlignedPointerPair& buffer);
tflite::PointerAlignedPointerPair AlignedRealloc(
    const tflite::PointerAlignedPointerPair& old_buffer, size_t old_size,
    size_t new_size, size_t alignment);
#if defined(_WIN32)
tflite::PointerAlignedPointerPair AlignedAlloc(size_t size, size_t alignment) {
  char* pointer = reinterpret_cast<char*>(_aligned_malloc(size, alignment));
  char* aligned_ptr = pointer;
  return {pointer, aligned_ptr};
}
void AlignedFree(const tflite::PointerAlignedPointerPair& buffer) {
  _aligned_free(buffer.pointer);
}
tflite::PointerAlignedPointerPair AlignedRealloc(
    const tflite::PointerAlignedPointerPair& old_buffer, size_t old_size,
    size_t new_size, size_t alignment) {
  char* pointer = reinterpret_cast<char*>(
      _aligned_realloc(old_buffer.pointer, new_size, alignment));
  char* aligned_ptr = pointer;
  return {pointer, aligned_ptr};
}
#else
tflite::PointerAlignedPointerPair AlignedAlloc(size_t size, size_t alignment) {
#if TF_LITE_HAS_ALIGNED_ALLOC
  const size_t allocation_size = AlignTo(alignment, size + alignment - 1);
  char* pointer =
      reinterpret_cast<char*>(::aligned_alloc(alignment, allocation_size));
  char* aligned_ptr = pointer;
#else
  const size_t allocation_size = size + alignment - 1;
  char* pointer = reinterpret_cast<char*>(std::malloc(allocation_size));
  char* aligned_ptr = reinterpret_cast<char*>(
      AlignTo(alignment, reinterpret_cast<std::uintptr_t>(pointer)));
#endif
#if defined(__clang__)
#if __has_feature(memory_sanitizer)
  std::memset(pointer, 0, allocation_size);
#endif
#endif
  return {pointer, aligned_ptr};
}
void AlignedFree(const tflite::PointerAlignedPointerPair& buffer) {
  std::free(buffer.pointer);
}
tflite::PointerAlignedPointerPair AlignedRealloc(
    const tflite::PointerAlignedPointerPair& old_buffer, size_t old_size,
    size_t new_size, size_t alignment) {
  tflite::PointerAlignedPointerPair new_buffer =
      AlignedAlloc(new_size, alignment);
  if (new_size > 0 && old_size > 0) {
    const size_t copy_amount = std::min(new_size, old_size);
    std::memcpy(new_buffer.aligned_pointer, old_buffer.aligned_pointer,
                copy_amount);
  }
  AlignedFree(old_buffer);
  return new_buffer;
}
#endif
}  
namespace tflite {
bool ResizableAlignedBuffer::Resize(size_t new_size) {
  if (new_size <= data_size_) {
    return false;
  }
#ifdef TF_LITE_TENSORFLOW_PROFILER
  PauseHeapMonitoring(true);
  OnTfLiteArenaAlloc(subgraph_index_, reinterpret_cast<std::uintptr_t>(this),
                     new_size);
  if (data_size_ > 0) {
    OnTfLiteArenaDealloc(subgraph_index_,
                         reinterpret_cast<std::uintptr_t>(this), data_size_);
  }
#endif
  auto new_buffer = AlignedRealloc(buffer_, data_size_, new_size, alignment_);
  bool reallocated = (new_buffer.aligned_pointer != buffer_.aligned_pointer);
  buffer_ = new_buffer;
  data_size_ = new_size;
#ifdef TF_LITE_TENSORFLOW_PROFILER
  PauseHeapMonitoring(false);
#endif
  return reallocated;
}
void ResizableAlignedBuffer::Release() {
  if (buffer_.pointer == nullptr) {
    return;
  }
#ifdef TF_LITE_TENSORFLOW_PROFILER
  OnTfLiteArenaDealloc(subgraph_index_, reinterpret_cast<std::uintptr_t>(this),
                       data_size_);
#endif
  AlignedFree(buffer_);
  buffer_.pointer = nullptr;
  buffer_.aligned_pointer = nullptr;
  data_size_ = 0;
}
void SimpleMemoryArena::PurgeAfter(int32_t node) {
  for (int i = 0; i < active_allocs_.size(); ++i) {
    if (active_allocs_[i].first_node > node) {
      active_allocs_[i].tensor = -1;
    }
  }
  active_allocs_.erase(
      std::remove_if(active_allocs_.begin(), active_allocs_.end(),
                     [](ArenaAllocWithUsageInterval& alloc) {
                       return alloc.tensor == -1;
                     }),
      active_allocs_.end());
}
void SimpleMemoryArena::PurgeActiveAllocs(int32_t node) {
  for (int i = 0; i < active_allocs_.size(); ++i) {
    if (active_allocs_[i].last_node < node) {
      active_allocs_[i].tensor = -1;
    }
  }
  active_allocs_.erase(
      std::remove_if(active_allocs_.begin(), active_allocs_.end(),
                     [](ArenaAllocWithUsageInterval& alloc) {
                       return alloc.tensor == -1;
                     }),
      active_allocs_.end());
}
void SimpleMemoryArena::CalculateActiveAllocs(
    const std::vector<ArenaAllocWithUsageInterval>& allocs, int32_t node) {
  active_allocs_.clear();
  for (int i = 0; i < allocs.size(); ++i) {
    if (allocs[i].first_node <= node && allocs[i].last_node >= node) {
      active_allocs_.push_back(allocs[i]);
    }
  }
  std::sort(active_allocs_.begin(), active_allocs_.end());
}
void SimpleMemoryArena::ResetAllocs() { active_allocs_.clear(); }
TfLiteStatus SimpleMemoryArena::Allocate(
    TfLiteContext* context, size_t alignment, size_t size, int32_t tensor,
    int32_t first_node, int32_t last_node,
    ArenaAllocWithUsageInterval* new_alloc) {
  TF_LITE_ENSURE(context, alignment <= underlying_buffer_.GetAlignment());
  new_alloc->tensor = tensor;
  new_alloc->first_node = first_node;
  new_alloc->last_node = last_node;
  new_alloc->size = size;
  if (size == 0) {
    new_alloc->offset = 0;
    return kTfLiteOk;
  }
  const size_t kOffsetNotAssigned = std::numeric_limits<size_t>::max();
  size_t best_offset = kOffsetNotAssigned;
  size_t best_offset_fit = kOffsetNotAssigned;
  size_t current_offset = 0;
  for (const auto& alloc : active_allocs_) {
    if (alloc.last_node < first_node || alloc.first_node > last_node) {
      continue;
    }
    size_t aligned_current_offset = AlignTo(alignment, current_offset);
    if (aligned_current_offset + size <= alloc.offset &&
        alloc.offset - aligned_current_offset < best_offset_fit) {
      best_offset = aligned_current_offset;
      best_offset_fit = alloc.offset - current_offset;
    }
    current_offset = std::max(current_offset, alloc.offset + alloc.size);
    if (best_offset_fit == 0) {
      break;
    }
  }
  if (best_offset == kOffsetNotAssigned) {
    best_offset = AlignTo(alignment, current_offset);
  }
  high_water_mark_ = std::max(high_water_mark_, best_offset + size);
  new_alloc->offset = best_offset;
  auto insertion_it = std::upper_bound(active_allocs_.begin(),
                                       active_allocs_.end(), *new_alloc);
  active_allocs_.insert(insertion_it, *new_alloc);
  return kTfLiteOk;
}
TfLiteStatus SimpleMemoryArena::Commit(bool* arena_reallocated) {
  *arena_reallocated = underlying_buffer_.Resize(high_water_mark_);
  committed_ = true;
  return kTfLiteOk;
}
TfLiteStatus SimpleMemoryArena::ResolveAlloc(
    TfLiteContext* context, const ArenaAllocWithUsageInterval& alloc,
    char** output_ptr) {
  TF_LITE_ENSURE(context, committed_);
  TF_LITE_ENSURE(context, output_ptr != nullptr);
  TF_LITE_ENSURE(context,
                 underlying_buffer_.GetSize() >= (alloc.offset + alloc.size));
  if (alloc.size == 0) {
    *output_ptr = nullptr;
  } else {
    *output_ptr = underlying_buffer_.GetPtr() + alloc.offset;
  }
  return kTfLiteOk;
}
TfLiteStatus SimpleMemoryArena::ClearPlan() {
  committed_ = false;
  high_water_mark_ = 0;
  active_allocs_.clear();
  return kTfLiteOk;
}
TfLiteStatus SimpleMemoryArena::ReleaseBuffer() {
  committed_ = false;
  underlying_buffer_.Release();
  return kTfLiteOk;
}
TFLITE_ATTRIBUTE_WEAK void DumpArenaInfo(
    const std::string& name, const std::vector<int>& execution_plan,
    size_t arena_size, const std::vector<ArenaAllocWithUsageInterval>& allocs) {
}
void SimpleMemoryArena::DumpDebugInfo(
    const std::string& name, const std::vector<int>& execution_plan) const {
  tflite::DumpArenaInfo(name, execution_plan, underlying_buffer_.GetSize(),
                        active_allocs_);
}
}  