#include "tensorflow/lite/profiling/profile_buffer.h"
#include <utility>
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/profiling/memory_info.h"
#include "tensorflow/lite/profiling/time.h"
namespace tflite {
namespace profiling {
uint32_t ProfileBuffer::BeginEvent(const char* tag,
                                   ProfileEvent::EventType event_type,
                                   int64_t event_metadata1,
                                   int64_t event_metadata2) {
  if (!enabled_) {
    return kInvalidEventHandle;
  }
  uint64_t timestamp = time::NowMicros();
  const auto next_index = GetNextEntryIndex();
  if (next_index.second) {
    return next_index.first;
  }
  const int index = next_index.first;
  event_buffer_[index].tag = tag;
  event_buffer_[index].event_type = event_type;
  event_buffer_[index].event_metadata = event_metadata1;
  event_buffer_[index].extra_event_metadata = event_metadata2;
  event_buffer_[index].begin_timestamp_us = timestamp;
  event_buffer_[index].elapsed_time = 0;
  if (event_type != Profiler::EventType::OPERATOR_INVOKE_EVENT) {
    event_buffer_[index].begin_mem_usage = memory::GetMemoryUsage();
  }
  current_index_++;
  return index;
}
void ProfileBuffer::EndEvent(uint32_t event_handle,
                             const int64_t* event_metadata1,
                             const int64_t* event_metadata2) {
  if (!enabled_ || event_handle == kInvalidEventHandle ||
      event_handle > current_index_) {
    return;
  }
  const uint32_t max_size = event_buffer_.size();
  if (current_index_ > (max_size + event_handle)) {
    return;
  }
  int event_index = event_handle % max_size;
  event_buffer_[event_index].elapsed_time =
      time::NowMicros() - event_buffer_[event_index].begin_timestamp_us;
  if (event_buffer_[event_index].event_type !=
      Profiler::EventType::OPERATOR_INVOKE_EVENT) {
    event_buffer_[event_index].end_mem_usage = memory::GetMemoryUsage();
  }
  if (event_metadata1) {
    event_buffer_[event_index].event_metadata = *event_metadata1;
  }
  if (event_metadata2) {
    event_buffer_[event_index].extra_event_metadata = *event_metadata2;
  }
}
const struct ProfileEvent* ProfileBuffer::At(size_t index) const {
  size_t size = Size();
  if (index >= size) {
    return nullptr;
  }
  const uint32_t max_size = event_buffer_.size();
  uint32_t start =
      (current_index_ > max_size) ? current_index_ % max_size : max_size;
  index = (index + start) % max_size;
  return &event_buffer_[index];
}
void ProfileBuffer::AddEvent(const char* tag,
                             ProfileEvent::EventType event_type,
                             uint64_t elapsed_time, int64_t event_metadata1,
                             int64_t event_metadata2) {
  if (!enabled_) {
    return;
  }
  const auto next_index = GetNextEntryIndex();
  if (next_index.second) {
    return;
  }
  const int index = next_index.first;
  event_buffer_[index].tag = tag;
  event_buffer_[index].event_type = event_type;
  event_buffer_[index].event_metadata = event_metadata1;
  event_buffer_[index].extra_event_metadata = event_metadata2;
  event_buffer_[index].begin_timestamp_us = 0;
  event_buffer_[index].elapsed_time = elapsed_time;
  current_index_++;
}
std::pair<int, bool> ProfileBuffer::GetNextEntryIndex() {
  int index = current_index_ % event_buffer_.size();
  if (current_index_ == 0 || index != 0) {
    return std::make_pair(index, false);
  }
  if (!allow_dynamic_expansion_) {
    TFLITE_LOG_PROD_ONCE(TFLITE_LOG_INFO,
                         "Warning: Dropping ProfileBuffer event.");
    return std::make_pair(current_index_, true);
  } else {
    TFLITE_LOG_PROD_ONCE(TFLITE_LOG_INFO,
                         "Warning: Doubling internal profiling buffer.");
    event_buffer_.resize(current_index_ * 2);
    return std::make_pair(current_index_, false);
  }
}
}  
}  