#include "tensorflow/core/profiler/convert/op_stats_to_pod_viewer.h"
#include <utility>
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/profiler/convert/op_stats_to_pod_stats.h"
#include "tensorflow/core/profiler/protobuf/pod_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tensorflow/core/profiler/utils/diagnostics.h"
namespace tensorflow {
namespace profiler {
namespace {
PodStatsSequence ConvertOpStatsToPodStatsSequence(const OpStats& op_stats,
                                                  PodStatsDatabase pod_stats) {
  PodStatsSequence result_db;
  int i = 0;
  for (const auto& step_sequence : op_stats.step_db().step_sequence()) {
    PodStatsMap* pod_stats_map = result_db.add_pod_stats_map();
    pod_stats_map->set_step_num(step_sequence.step_num());
    for (const auto& entry : step_sequence.step_info_per_core()) {
      PodStatsRecord& record =
          (*pod_stats_map->mutable_pod_stats_per_core())[entry.first];
      DCHECK_LE(i, pod_stats.pod_stats_record_size());
      record = std::move(*pod_stats.mutable_pod_stats_record(i++));
    }
  }
  return result_db;
}
}  
PodViewerDatabase ConvertOpStatsToPodViewer(const OpStats& op_stats) {
  PodViewerDatabase database;
  database.set_device_type(op_stats.run_environment().device_type());
  PodStatsDatabase pod_stats = ConvertOpStatsToPodStats(op_stats);
  database.mutable_step_breakdown_events()->Swap(
      pod_stats.mutable_step_breakdown_events());
  *database.mutable_pod_stats_sequence() =
      ConvertOpStatsToPodStatsSequence(op_stats, std::move(pod_stats));
  PopulateStepDiagnostics(op_stats, database.mutable_diagnostics());
  return database;
}
}  
}  