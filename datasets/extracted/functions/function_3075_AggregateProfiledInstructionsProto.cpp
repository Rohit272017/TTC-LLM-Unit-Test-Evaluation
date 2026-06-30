#include "xla/python/aggregate_profile.h"
#include <algorithm>
#include <string>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "xla/python/xplane_to_profile_instructions.h"
namespace xla {
void AggregateProfiledInstructionsProto(
    absl::Span<const tensorflow::profiler::ProfiledInstructionsProto> profiles,
    int percentile,
    tensorflow::profiler::ProfiledInstructionsProto *result_profile) {
  if (percentile < 0 || percentile > 100) return;
  absl::flat_hash_map<std::string, HloLatencyInfo> hlo_latency_info;
  for (const auto &profile : profiles) {
    for (const auto &cost : profile.costs()) {
      hlo_latency_info[cost.name()].durations.emplace_back(cost.cost_us());
    }
  }
  for (const auto &iter : hlo_latency_info) {
    auto *cost = result_profile->add_costs();
    std::vector<double> durations = iter.second.durations;
    int index = 0;
    if (durations.size() > 1) {
      std::sort(durations.begin(), durations.end());
      index = percentile / 100.0 * (durations.size() - 1);
    }
    cost->set_cost_us(durations[index]);
    cost->set_name(iter.first);
  }
}
}  