#include "tensorflow/core/profiler/utils/host_offload_utils.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "tensorflow/core/profiler/utils/trace_utils.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"
namespace tensorflow {
namespace profiler {
bool HostOffloadEventProcessor::IsHostOffloadOpName(
    const XEventVisitor& event) const {
  static constexpr absl::string_view keywords[] = {"copy-start",
                                                   "copy-done",
                                                   "dynamic-slice-start",
                                                   "dynamic-slice-done",
                                                   "dynamic-update-slice-start",
                                                   "dynamic-update-slice-done"};
  for (const auto& keyword : keywords) {
    if (absl::StrContains(event.DisplayName(), keyword) &&
        absl::StrContains(event.Name(), host_memory_label_)) {
      return true;
    }
  }
  return false;
}
std::string HostOffloadEventProcessor::GetOffloadInstructionID(
    absl::string_view op_name) const {
  std::vector<std::string> op_name_vec = absl::StrSplit(op_name, '.');
  if (op_name_vec.size() < 2) {
    return "0";
  }
  return op_name_vec.back();
}
std::string HostOffloadEventProcessor::GetOffloadInstructionName(
    absl::string_view op_name) const {
  std::string display_id = GetOffloadInstructionID(op_name);
  size_t startPos = op_name.find("-start");
  size_t donePos = op_name.find("-done");
  absl::string_view display_opname;
  if (startPos != absl::string_view::npos) {
    display_opname = op_name.substr(0, startPos);
  } else if (donePos != absl::string_view::npos) {
    display_opname = op_name.substr(0, donePos);
  } else {
    LOG(WARNING) << "Invalid op name: " << op_name;
    display_opname = op_name;
  }
  return absl::StrCat("offload-", display_opname, ".", display_id);
}
void HostOffloadEventProcessor::ProcessHostOffloadOpEvent(
    const XEventVisitor& event, std::optional<int64_t> group_id) {
  std::string display_opname = GetOffloadInstructionName(event.DisplayName());
  auto [iter, inserted] = seen_events_.try_emplace(display_opname);
  std::queue<const XEventVisitor*>& events = iter->second;
  if (absl::StrContains(event.DisplayName(), "-start")) {
    events.push(&event);
    return;
  } else if (absl::StrContains(event.DisplayName(), "-done")) {
    if (events.empty()) {
      LOG(INFO) << "No corresponding start event found for "
                << event.DisplayName();
      return;
    }
    const XEventVisitor* start_event = events.front();
    events.pop();
    tsl::profiler::Timespan event_span = tsl::profiler::Timespan::FromEndPoints(
        start_event->GetTimespan().begin_ps(), event.GetTimespan().end_ps());
    int line_builder_index = -1;
    uint64_t minimum_end_time_frontier = event_span.begin_ps();
    for (int i = 0; i < host_offload_op_line_builders_.size(); ++i) {
      if (host_offload_op_line_builders_[i].event_end_time_frontier_ns <=
          minimum_end_time_frontier) {
        line_builder_index = i;
        minimum_end_time_frontier =
            host_offload_op_line_builders_[i].event_end_time_frontier_ns;
      }
    }
    constexpr int kMaxHostOffloadOpLinesSize =
        kThreadIdHostOffloadOpEnd - kThreadIdHostOffloadOpStart + 1;
    if (line_builder_index == -1) {
      if (host_offload_op_line_builders_.size() < kMaxHostOffloadOpLinesSize) {
        XLineBuilder lb = plane_builder_->GetOrCreateLine(
            kThreadIdHostOffloadOpStart +
            host_offload_op_line_builders_.size());
        lb.SetName(absl::StrFormat("%s row %d", kHostOffloadOpLineName,
                                   host_offload_op_line_builders_.size()));
        lb.SetTimestampNs(start_timestamp_ns_);
        host_offload_op_line_builders_.push_back(
            {std::move(lb), event_span.end_ps()});
      }
      line_builder_index = host_offload_op_line_builders_.size() - 1;
    }
    host_offload_op_line_builders_[line_builder_index]
        .event_end_time_frontier_ns =
        std::max(host_offload_op_line_builders_[line_builder_index]
                     .event_end_time_frontier_ns,
                 event_span.end_ps());
    XEventMetadata* host_offload_copy_metadata =
        plane_builder_->CreateEventMetadata();
    host_offload_copy_metadata->set_display_name(display_opname);
    XEventBuilder event_builder =
        host_offload_op_line_builders_[line_builder_index]
            .line_builder.AddEvent(*host_offload_copy_metadata);
    event_builder.SetTimespan(event_span);
    const XStatMetadata& async_stat = *plane_builder_->GetOrCreateStatMetadata(
        GetStatTypeStr(StatType::kIsAsync));
    event_builder.AddStatValue(async_stat, 1);
  }
}
}  
}  