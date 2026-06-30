#include "tensorflow/lite/delegates/gpu/common/memory_management/internal.h"
#include <algorithm>
#include <cstddef>
#include <vector>
#include "tensorflow/lite/delegates/gpu/common/memory_management/types.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
namespace tflite {
namespace gpu {
bool CompareBySize(const TensorUsageWithIndex<size_t>& first,
                   const TensorUsageWithIndex<size_t>& second) {
  return first.usage_record->tensor_size > second.usage_record->tensor_size;
}
bool IsCoveringObject(const uint2& first_object, const uint2& second_object) {
  return first_object.x >= second_object.x && first_object.y >= second_object.y;
}
bool IsCoveringObject(const uint3& first_object, const uint3& second_object) {
  return first_object.x >= second_object.x &&
         first_object.y >= second_object.y && first_object.z >= second_object.z;
}
size_t AbsDiffInElements(const size_t first_size, const size_t second_size) {
  return first_size >= second_size ? first_size - second_size
                                   : second_size - first_size;
}
size_t AbsDiffInElements(const uint2& first_size, const uint2& second_size) {
  const size_t first_elements_cnt = first_size.y * first_size.x;
  const size_t second_elements_cnt = second_size.y * second_size.x;
  return first_elements_cnt >= second_elements_cnt
             ? first_elements_cnt - second_elements_cnt
             : second_elements_cnt - first_elements_cnt;
}
size_t AbsDiffInElements(const uint3& first_size, const uint3& second_size) {
  const size_t first_elements_cnt = first_size.z * first_size.y * first_size.x;
  const size_t second_elements_cnt =
      second_size.z * second_size.y * second_size.x;
  return first_elements_cnt >= second_elements_cnt
             ? first_elements_cnt - second_elements_cnt
             : second_elements_cnt - first_elements_cnt;
}
std::vector<TaskProfile> CalculateTaskProfiles(
    const std::vector<TensorUsageRecord<size_t>>& usage_records) {
  TaskId num_tasks = 0;
  for (size_t i = 0; i < usage_records.size(); ++i) {
    num_tasks = std::max(num_tasks, usage_records[i].last_task + 1);
  }
  std::vector<TaskProfile> task_profiles(num_tasks);
  for (size_t rec_id = 0; rec_id < usage_records.size(); ++rec_id) {
    for (TaskId task_id = usage_records[rec_id].first_task;
         task_id <= usage_records[rec_id].last_task; ++task_id) {
      task_profiles[task_id].emplace_back(&usage_records[rec_id], rec_id);
    }
  }
  for (auto& task_profile : task_profiles) {
    std::stable_sort(task_profile.begin(), task_profile.end(), CompareBySize);
  }
  return task_profiles;
}
std::vector<size_t> CalculatePositionalMaximums(
    const std::vector<TensorUsageRecord<size_t>>& usage_records) {
  std::vector<TaskProfile> task_profiles = CalculateTaskProfiles(usage_records);
  std::vector<size_t> positional_max;
  for (const auto& task_profile : task_profiles) {
    size_t i = 0;
    for (; i < task_profile.size() && i < positional_max.size(); ++i) {
      positional_max[i] = std::max(positional_max[i],
                                   task_profile[i].usage_record->tensor_size);
    }
    for (; i < task_profile.size(); ++i) {
      positional_max.push_back(task_profile[i].usage_record->tensor_size);
    }
  }
  return positional_max;
}
}  
}  