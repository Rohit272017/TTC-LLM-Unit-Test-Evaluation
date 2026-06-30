#include "tensorflow/lite/profiling/subgraph_tensor_profiler.h"
#include <cstring>
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
namespace tflite::profiling {
SubgraphTensorProfiler::SubgraphTensorProfiler(const Interpreter& interpreter,
                                               CallbackT callback)
    : interpreter_(interpreter), callback_(callback) {
  events_.reserve(interpreter.subgraphs_size());
}
uint32_t SubgraphTensorProfiler::BeginEvent(const char* tag,
                                            EventType event_type,
                                            int64_t event_metadata1,
                                            int64_t event_metadata2) {
  if (strcmp(tag, "Invoke")) {
    return 0;
  }
  events_.push_back(event_metadata2);
  return events_.size();
}
void SubgraphTensorProfiler::EndEvent(uint32_t event_handle) {
  if (!event_handle || events_.size() < event_handle) {
    return;
  }
  const Subgraph* subgraph = interpreter_.subgraph(events_[event_handle - 1]);
  for (int i = 0; i < subgraph->tensors_size(); ++i) {
    callback_(subgraph->tensor(i));
  }
}
}  