#include "xla/tsl/profiler/utils/timestamp_utils.h"
#include <cstdint>
#include "absl/log/log.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
namespace tsl {
namespace profiler {
void SetSessionTimestamps(uint64_t start_walltime_ns, uint64_t stop_walltime_ns,
                          tensorflow::profiler::XSpace& space) {
  if (start_walltime_ns != 0 && stop_walltime_ns != 0) {
    tsl::profiler::XPlaneBuilder plane(
        tsl::profiler::FindOrAddMutablePlaneWithName(
            &space, tsl::profiler::kTaskEnvPlaneName));
    plane.AddStatValue(*plane.GetOrCreateStatMetadata(
                           GetTaskEnvStatTypeStr(kEnvProfileStartTime)),
                       start_walltime_ns);
    plane.AddStatValue(*plane.GetOrCreateStatMetadata(
                           GetTaskEnvStatTypeStr(kEnvProfileStopTime)),
                       stop_walltime_ns);
  } else {
    LOG(WARNING) << "Not Setting Session Timestamps, (start_walltime_ns, "
                    "stop_walltime_ns) : "
                 << start_walltime_ns << ", " << stop_walltime_ns;
  }
}
}  
}  