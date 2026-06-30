#include "tensorflow/core/profiler/convert/xplane_to_op_metrics_db.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "xla/tsl/profiler/utils/tf_op_utils.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/op_metrics_db_combiner.h"
#include "tensorflow/core/profiler/convert/op_stack.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/cost_utils.h"
#include "tensorflow/core/profiler/utils/op_metrics_db_utils.h"
#include "tensorflow/core/profiler/utils/op_utils.h"
#include "tensorflow/core/profiler/utils/trace_utils.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"
namespace tensorflow {
namespace profiler {
namespace {
constexpr uint64_t kRootSymbolId = 0;
enum TfActivityType { kTfOpBegin, kTfOpEnd };
struct TfActivity {
  uint64 timestamp_ps;
  uint32 tf_op_id;
  TfActivityType activity_type;
  tsl::profiler::TfOp tf_op;
  bool is_eager;
};
struct TfOpInfo {
  explicit TfOpInfo(uint64 ts) : start_timestamp_ps(ts) {}
  uint64 start_timestamp_ps;
  uint64 children_duration_ps = 0;
};
void ProcessOneTfActivity(const TfActivity& activity,
                          OpStack<TfOpInfo>* tf_op_stack,
                          TfMetricsDbData* tf_metrics_data) {
  uint32 tf_op_id = activity.tf_op_id;
  switch (activity.activity_type) {
    case kTfOpBegin: {
      tf_op_stack->Push(tf_op_id,
                        std::make_unique<TfOpInfo>(activity.timestamp_ps));
      break;
    }
    case kTfOpEnd: {
      std::unique_ptr<TfOpInfo> info = tf_op_stack->Pop(tf_op_id);
      if (info == nullptr) {
        VLOG(1) << "No begin event found for TF activity id=" << tf_op_id
                << " name=" << activity.tf_op.name
                << " type=" << activity.tf_op.type;
        break;
      }
      tsl::profiler::Timespan tf_op_span = tsl::profiler::PicoSpan(
          info->start_timestamp_ps, activity.timestamp_ps);
      tf_metrics_data->tf_metrics_db_builder.EnterOp(
          activity.tf_op.name, activity.tf_op.type, activity.is_eager,
          tf_op_span.duration_ps(), info->children_duration_ps);
      TfOpInfo* parent_info = tf_op_stack->Top();
      if (parent_info != nullptr) {
        parent_info->children_duration_ps += tf_op_span.duration_ps();
      }
      if (tsl::profiler::IsInfeedEnqueueOp(activity.tf_op.type)) {
        tf_metrics_data->tf_metrics_db_builder.EnterHostInfeedEnqueue(
            tf_op_span);
      }
      break;
    }
  }
}
void ProcessTfActivities(std::vector<TfActivity>* tf_activities,
                         TfMetricsDbData* tf_metrics_db_data) {
  if (tf_activities->empty()) return;
  absl::c_stable_sort(*tf_activities,
                      [](const TfActivity& a, const TfActivity& b) {
                        return a.timestamp_ps < b.timestamp_ps;
                      });
  OpStack<TfOpInfo> tf_op_stack;
  for (const auto& tf_activity : *tf_activities) {
    ProcessOneTfActivity(tf_activity, &tf_op_stack, tf_metrics_db_data);
  }
  SetTotalTimePs(
      tf_metrics_db_data->tf_metrics_db,
      tf_activities->back().timestamp_ps - tf_activities->front().timestamp_ps);
}
void CollectTfActivities(
    const XLineVisitor& line,
    const absl::flat_hash_map<int64_t, tsl::profiler::TfOp>& tf_ops,
    std::vector<TfActivity>* tf_activities) {
  uint32 tf_op_id = 0;
  if (IsDerivedThreadId(line.Id())) return;
  tf_activities->reserve(line.NumEvents() * 2);
  line.ForEachEvent(
      [&tf_ops, &tf_op_id, &tf_activities](const XEventVisitor& event) {
        const tsl::profiler::TfOp* tf_op = gtl::FindOrNull(tf_ops, event.Id());
        if (tf_op != nullptr) {
          ++tf_op_id;
          bool is_eager = false;
          if (std::optional<XStatVisitor> stat =
                  event.GetStat(StatType::kIsEager)) {
            is_eager = stat->IntValue();
          }
          tsl::profiler::Timespan span = event.GetTimespan();
          tf_activities->push_back(
              {span.begin_ps(), tf_op_id, kTfOpBegin, *tf_op, is_eager});
          tf_activities->push_back(
              {span.end_ps(), tf_op_id, kTfOpEnd, *tf_op, is_eager});
        }
        if (auto tf_op_stat = event.GetStat(StatType::kTfOp);
            tf_op_stat.has_value()) {
          ++tf_op_id;
          tsl::profiler::TfOp tf_op =
              tsl::profiler::ParseTfOpFullname(tf_op_stat->StrOrRefValue());
          tsl::profiler::Timespan span = event.GetTimespan();
          tf_activities->push_back(
              {span.begin_ps(), tf_op_id, kTfOpBegin, tf_op, false});
          tf_activities->push_back(
              {span.end_ps(), tf_op_id, kTfOpEnd, tf_op, false});
        }
      });
}
}  
absl::flat_hash_map<int64_t, tsl::profiler::TfOp>
CollectTfOpsFromHostThreadsXPlane(const XPlane& host_trace) {
  absl::flat_hash_map<int64_t, tsl::profiler::TfOp> tf_ops;
  for (const auto& id_metadata : host_trace.event_metadata()) {
    const XEventMetadata& metadata = id_metadata.second;
    tsl::profiler::TfOp tf_op =
        tsl::profiler::ParseTfOpFullname(metadata.name());
    if (tf_op.category != tsl::profiler::Category::kUnknown) {
      tf_ops.try_emplace(metadata.id(), tf_op);
    }
  }
  return tf_ops;
}
TfMetricsDbData ConvertHostThreadsXLineToTfMetricsDbData(
    const XLineVisitor& line,
    const absl::flat_hash_map<int64_t, tsl::profiler::TfOp>& tf_ops) {
  TfMetricsDbData tf_metrics_db_data;
  std::vector<TfActivity> tf_activities;
  CollectTfActivities(line, tf_ops, &tf_activities);
  ProcessTfActivities(&tf_activities, &tf_metrics_db_data);
  return tf_metrics_db_data;
}
void ConsumeTfMetricsDbData(TfMetricsDbData src, OpMetricsDbCombiner* dst) {
  AddIdleOp(src.tf_metrics_db);
  dst->Combine(src.tf_metrics_db, false);
  src.tf_metrics_db.Clear();
}
OpMetricsDb ConvertHostThreadsXPlaneToOpMetricsDb(const XPlane& host_trace) {
  absl::flat_hash_map<int64_t, tsl::profiler::TfOp> tf_ops =
      CollectTfOpsFromHostThreadsXPlane(host_trace);
  OpMetricsDb result;
  OpMetricsDbCombiner combiner(&result);
  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(&host_trace);
  plane.ForEachLine([&tf_ops, &combiner](const XLineVisitor& line) {
    ConsumeTfMetricsDbData(
        ConvertHostThreadsXLineToTfMetricsDbData(line, tf_ops), &combiner);
  });
  return result;
}
OpMetricsDb ConvertTpuDeviceTraceXPlaneToOpMetricsDb(
    const XPlane& device_trace) {
  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(&device_trace);
  using OpMetricBySymbol =
      absl::flat_hash_map<uint64_t, OpMetrics>;
  XEventsOpMetricsDbBuilder builder;
  plane.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent(
        [&](const XEventVisitor& event) { builder.AddOpMetric(event); });
  });
  return builder.Finalize(
      plane.GetStat(StatType::kTotalProfileDurationPs)->IntOrUintValue());
}
OpMetricsDb ConvertDeviceTraceXPlaneToOpMetricsDb(const XPlane& device_trace) {
  OpMetricsDb result;
  DeviceOpMetricsDbBuilder device_op_metrics_db_builder(&result);
  int64_t first_op_offset_ps = kint64max;
  int64_t last_op_offset_ps = 0;
  TfOpRoofLineCostEstimator op_level_cost_estimator;
  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(&device_trace);
  plane.ForEachLine([&](const XLineVisitor& line) {
    if (IsDerivedThreadId(line.Id())) return;
    line.ForEachEvent([&](const XEventVisitor& event) {
      first_op_offset_ps = std::min(first_op_offset_ps, event.OffsetPs());
      last_op_offset_ps = std::max(last_op_offset_ps, event.EndOffsetPs());
      absl::string_view tf_op_full_name;
      bool is_eager = false;
      int64_t program_id = 0;
      absl::string_view deduplicated_name = "";
      event.ForEachStat([&](const XStatVisitor& stat) {
        if (stat.Type() == StatType::kTfOp) {
          tf_op_full_name = stat.StrOrRefValue();
        } else if (stat.Type() == StatType::kIsEager) {
          is_eager = stat.IntValue();
        } else if (stat.Type() == StatType::kProgramId) {
          program_id = stat.IntOrUintValue();
        } else if (stat.Type() == StatType::kDeduplicatedName) {
          deduplicated_name = stat.StrOrRefValue();
        }
      });
      if (tf_op_full_name.empty()) return;
      tsl::profiler::TfOp tf_op =
          tsl::profiler::ParseTfOpFullname(tf_op_full_name);
      TfOpRoofLineCostEstimator::OpRoofLineStats costs;
      if (tf_op.category != tsl::profiler::Category::kUnknown) {
        costs = op_level_cost_estimator.Predict(event);
      }
      device_op_metrics_db_builder.EnterOp(
          program_id,
          absl::StrCat(tf_op.name, "/", event.Name()),
          tf_op.type,
          tf_op_full_name, deduplicated_name, is_eager,
          1, event.DurationPs(),
          0, costs.flops, costs.bytes_accessed);
    });
  });
  SetTotalTimePs(
      result, last_op_offset_ps ? last_op_offset_ps - first_op_offset_ps : 0);
  AddIdleOp(result);
  return result;
}
}  
}  