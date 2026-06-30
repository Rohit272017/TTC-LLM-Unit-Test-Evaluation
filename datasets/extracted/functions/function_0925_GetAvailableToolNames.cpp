#include "tensorflow/core/profiler/convert/xplane_to_tool_names.h"
#include <memory>
#include <string>
#include <vector>
#include "absl/strings/str_join.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/profiler/convert/repository.h"
#include "tensorflow/core/profiler/convert/xplane_to_dcn_collective_stats.h"
#include "tensorflow/core/profiler/convert/xplane_to_hlo.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
namespace tensorflow {
namespace profiler {
absl::StatusOr<std::string> GetAvailableToolNames(
    const SessionSnapshot& session_snapshot) {
  std::vector<std::string> tools;
  bool is_cloud_vertex_ai = !session_snapshot.HasAccessibleRunDir();
  if (session_snapshot.XSpaceSize() != 0) {
    tools.reserve(11);
    tools.push_back(is_cloud_vertex_ai ? "trace_viewer" : "trace_viewer@");
    tools.push_back("overview_page");
    tools.push_back("input_pipeline_analyzer");
    tools.push_back("framework_op_stats");
    tools.push_back("memory_profile");
    tools.push_back("pod_viewer");
    tools.push_back("tf_data_bottleneck_analysis");
    tools.push_back("op_profile");
    TF_ASSIGN_OR_RETURN(std::unique_ptr<XSpace> xspace,
                        session_snapshot.GetXSpace(0));
    if (!FindPlanesWithPrefix(*xspace, kGpuPlanePrefix).empty()) {
      tools.push_back("kernel_stats");
    }
    TF_ASSIGN_OR_RETURN(bool has_hlo,
                        ConvertMultiXSpaceToHloProto(session_snapshot));
    if (has_hlo) {
      tools.push_back("memory_viewer");
      tools.push_back("graph_viewer");
    }
    TF_ASSIGN_OR_RETURN(bool has_dcn_collective_stats,
                        HasDcnCollectiveStatsInMultiXSpace(session_snapshot));
    if (has_dcn_collective_stats) {
      tools.push_back("dcn_collective_stats");
    }
  }
  return absl::StrJoin(tools, ",");
}
}  
}  