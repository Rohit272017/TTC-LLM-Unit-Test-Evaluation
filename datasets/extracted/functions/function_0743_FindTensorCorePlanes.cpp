#include "xla/tsl/profiler/utils/tpu_xplane_utils.h"
#include <optional>
#include <vector>
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/platform/regexp.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
namespace tsl {
namespace profiler {
std::vector<const XPlane*> FindTensorCorePlanes(const XSpace& xspace) {
  return FindPlanes(xspace, [](const XPlane& xplane) {
    static const LazyRE2 re = {kTpuPlaneRegex};
    return RE2::FullMatch(xplane.name(), *re);
  });
}
std::vector<XPlane*> FindMutableTensorCorePlanes(XSpace* xspace) {
  return FindMutablePlanes(xspace, [](const XPlane& xplane) {
    static const LazyRE2 re = {kTpuPlaneRegex};
    return RE2::FullMatch(xplane.name(), *re);
  });
}
std::optional<int> GetTensorCoreId(absl::string_view plane_name) {
  int core_id = -1;
  if (RE2::FullMatch(plane_name, {kTpuPlaneRegex}, &core_id)) {
    return core_id;
  }
  return std::nullopt;
}
std::optional<int> GetSparseCoreId(absl::string_view plane_name) {
  std::optional<int> core_id;
  RE2::FullMatch(plane_name, {kSparseCorePlaneRegex}, &core_id);
  return core_id;
}
}  
}  