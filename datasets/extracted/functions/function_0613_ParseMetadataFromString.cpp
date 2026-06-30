#include "xla/tsl/profiler/convert/trace_container.h"
#include <algorithm>
#include <string>
#include <string_view>
#include <vector>
#include "tsl/platform/protobuf.h"
namespace tsl {
namespace profiler {
bool TraceContainer::ParseMetadataFromString(const std::string& description) {
  return protobuf::TextFormat::ParseFromString(description, &metadata_);
}
void TraceContainer::CapEvents(const uint32_t max_count) {
  const size_t total_count = events_.size();
  if (total_count <= max_count) {
    return;
  }
  const std::vector<TraceEvent*>::iterator end = events_.begin() + max_count;
  std::partial_sort(
      events_.begin(), end, events_.end(),
      [](const TraceEvent* const lhs, const TraceEvent* const rhs) -> bool {
        return lhs->timestamp_ps() < rhs->timestamp_ps();
      });
  for (std::vector<TraceEvent*>::iterator i = end; i != events_.end(); ++i) {
    delete *i;
  }
  events_.erase(end, events_.end());
}
void TraceContainer::FlushAndSerializeEvents(std::string* const output) {
  Trace trace = metadata_;
  for (TraceEvent* const event : events_) {
    trace.mutable_trace_events()->AddAllocated(event);
  }
  events_.clear();
  trace.SerializeToString(output);
}
}  
}  