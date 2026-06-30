#include "tensorflow/core/profiler/convert/trace_viewer/trace_viewer_visibility.h"
#include <cstdint>
#include "absl/log/check.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "tensorflow/core/profiler/protobuf/trace_events.pb.h"
namespace tensorflow {
namespace profiler {
TraceViewerVisibility::TraceViewerVisibility(
    tsl::profiler::Timespan visible_span, uint64_t resolution_ps)
    : visible_span_(visible_span), resolution_ps_(resolution_ps) {}
bool TraceViewerVisibility::Visible(const TraceEvent& event) {
  if (visible_span_.Instant()) return true;
  tsl::profiler::Timespan span(event.timestamp_ps(), event.duration_ps());
  if (!visible_span_.Overlaps(span)) return false;
  if (resolution_ps_ == 0) return true;
  return VisibleAtResolution(event);
}
bool TraceViewerVisibility::VisibleAtResolution(const TraceEvent& event) {
  DCHECK_NE(resolution_ps_, 0);
  if (!event.has_resource_id()) {
#if 1
    return true;
#else
    CounterRowId counter_row_id(event.device_id(), event.name());
    auto iter = last_counter_timestamp_ps_.find(counter_row_id);
    bool found = (iter != last_counter_timestamp_ps_.end());
    bool visible =
        !found || ((event.timestamp_ps() - iter->second) >= resolution_ps_);
    if (visible) {
      if (found) {
        iter->second = event.timestamp_ps();
      } else {
        last_counter_timestamp_ps_.emplace(counter_row_id,
                                           event.timestamp_ps());
      }
    }
    return visible;
#endif
  }
  tsl::profiler::Timespan span(event.timestamp_ps(), event.duration_ps());
  bool visible = (span.duration_ps() >= resolution_ps_);
  auto& row = rows_[RowId(event.device_id(), event.resource_id())];
  size_t depth = row.Depth(span.begin_ps());
  if (!visible) {
    auto last_end_timestamp_ps = row.LastEndTimestampPs(depth);
    visible = !last_end_timestamp_ps ||
              (span.begin_ps() - *last_end_timestamp_ps >= resolution_ps_);
  }
  if (event.has_flow_id()) {
    auto result = flows_.try_emplace(event.flow_id(), visible);
    if (!visible) {
      if (result.second) {
        auto last_flow_timestamp_ps = row.LastFlowTimestampPs();
        result.first->second =
            !last_flow_timestamp_ps ||
            (span.end_ps() - *last_flow_timestamp_ps >= resolution_ps_);
      }
      visible = result.first->second;
    }
    if (event.flow_entry_type() == TraceEvent::FLOW_END) {
      flows_.erase(result.first);
    }
    if (visible) {
      row.SetLastFlowTimestampPs(span.end_ps());
    }
  }
  if (visible) {
    row.SetLastEndTimestampPs(depth, span.end_ps());
  }
  return visible;
}
void TraceViewerVisibility::SetVisibleAtResolution(const TraceEvent& event) {
  DCHECK_NE(resolution_ps_, 0);
  if (!event.has_resource_id()) {
    CounterRowId counter_row_id(event.device_id(), event.name());
    last_counter_timestamp_ps_.insert_or_assign(counter_row_id,
                                                event.timestamp_ps());
  } else {
    tsl::profiler::Timespan span(event.timestamp_ps(), event.duration_ps());
    auto& row = rows_[RowId(event.device_id(), event.resource_id())];
    if (event.has_flow_id()) {
      if (event.flow_entry_type() == TraceEvent::FLOW_END) {
        flows_.erase(event.flow_id());
      } else {
        flows_.try_emplace(event.flow_id(), true);
      }
      row.SetLastFlowTimestampPs(span.end_ps());
    }
    size_t depth = row.Depth(span.begin_ps());
    row.SetLastEndTimestampPs(depth, span.end_ps());
  }
}
size_t TraceViewerVisibility::RowVisibility::Depth(
    uint64_t begin_timestamp_ps) const {
  size_t depth = 0;
  for (; depth < last_end_timestamp_ps_.size(); ++depth) {
    if (last_end_timestamp_ps_[depth] <= begin_timestamp_ps) break;
  }
  return depth;
}
}  
}  