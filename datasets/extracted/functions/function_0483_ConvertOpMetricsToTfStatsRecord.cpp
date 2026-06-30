#include "tensorflow/core/profiler/convert/op_stats_to_tf_stats.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/op_metrics_to_record.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/tf_stats.pb.h"
#include "tensorflow/core/profiler/utils/kernel_stats_utils.h"
#include "tensorflow/core/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/utils/op_metrics_db_utils.h"
namespace tensorflow {
namespace profiler {
namespace {
const int kMaxNumOfOps = 500;
TfStatsRecord ConvertOpMetricsToTfStatsRecord(
    bool on_device, const OpMetrics& metrics,
    double ridge_point_operational_intensity) {
  TfStatsRecord record;
  record.set_host_or_device(on_device ? "Device" : "Host");
  record.set_is_eager(metrics.is_eager());
  record.set_op_type(metrics.category());
  record.set_op_name(metrics.name());
  SetExecutionTimes(metrics, &record);
  SetRooflineMetrics(metrics, ridge_point_operational_intensity, &record);
  return record;
}
TfStatsTable GenerateTfStatsTable(
    const OpMetricsDb& host_tf_metrics_db,
    const OpMetricsDb& device_tf_metrics_db,
    const KernelStatsByOpName& kernel_stats_by_op_name, double ridge_point,
    bool exclude_idle) {
  TfStatsTable tf_stats_table;
  TfStatsRecord sentinel;
  sentinel.set_rank(0);
  sentinel.set_device_cumulative_total_self_time_as_fraction(0.0);
  sentinel.set_host_cumulative_total_self_time_as_fraction(0.0);
  const TfStatsRecord* prev_record = &sentinel;
  uint64 total_device_time_ps = TotalTimePs(device_tf_metrics_db, exclude_idle);
  double total_device_time_us =
      tsl::profiler::PicoToMicro(total_device_time_ps);
  for (const OpMetrics* metrics :
       SortedOpMetricsDb(device_tf_metrics_db, kMaxNumOfOps)) {
    if (exclude_idle && IsIdleOp(*metrics)) continue;
    TfStatsRecord* record = tf_stats_table.add_tf_stats_record();
    *record = ConvertOpMetricsToTfStatsRecord(
        true, *metrics, ridge_point);
    auto iter = kernel_stats_by_op_name.find(record->op_name());
    if (iter != kernel_stats_by_op_name.end()) {
      record->set_gpu_tensorcore_utilization(
          tsl::profiler::SafeDivide(iter->second.tensor_core_duration_ns,
                                    iter->second.total_duration_ns));
    } else {
      record->set_gpu_tensorcore_utilization(0.0);
    }
    SetRankAndDeviceTimeFractions(total_device_time_us, *prev_record, record);
    prev_record = record;
  }
  uint64 total_host_time_ps = TotalTimePs(host_tf_metrics_db, exclude_idle);
  double total_host_time_us = tsl::profiler::PicoToMicro(total_host_time_ps);
  for (const OpMetrics* metrics : tensorflow::profiler::SortedOpMetricsDb(
           host_tf_metrics_db, kMaxNumOfOps)) {
    if (exclude_idle && IsIdleOp(*metrics)) continue;
    TfStatsRecord* record = tf_stats_table.add_tf_stats_record();
    *record = ConvertOpMetricsToTfStatsRecord(
        false, *metrics, ridge_point);
    record->set_gpu_tensorcore_utilization(0.0);
    SetRankAndHostTimeFractions(total_host_time_us, *prev_record, record);
    prev_record = record;
  }
  return tf_stats_table;
}
}  
TfStatsDatabase ConvertOpStatsToTfStats(const OpStats& op_stats) {
  const OpMetricsDb& host_tf_metrics_db = op_stats.host_op_metrics_db();
  OpMetricsDb device_tf_metrics_db =
      CreateTfMetricsDbFromDeviceOpMetricsDb(op_stats.device_op_metrics_db());
  double ridge_point = op_stats.perf_env().ridge_point();
  KernelStatsByOpName kernel_stats_by_op_name =
      GroupKernelReportsByOpName(op_stats.kernel_stats_db());
  TfStatsDatabase tf_stats_db;
  *tf_stats_db.mutable_with_idle() = GenerateTfStatsTable(
      host_tf_metrics_db, device_tf_metrics_db, kernel_stats_by_op_name,
      ridge_point, false);
  *tf_stats_db.mutable_without_idle() = GenerateTfStatsTable(
      host_tf_metrics_db, device_tf_metrics_db, kernel_stats_by_op_name,
      ridge_point, true);
  tf_stats_db.set_device_type(op_stats.run_environment().device_type());
  return tf_stats_db;
}
}  
}  