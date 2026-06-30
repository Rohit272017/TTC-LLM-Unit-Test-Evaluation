#include "tensorflow/core/common_runtime/cost_measurement_registry.h"
#include <string>
#include <utility>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/cost_measurement.h"
#include "tensorflow/core/platform/logging.h"
namespace tensorflow {
namespace {
using RegistrationMap =
    absl::flat_hash_map<std::string, CostMeasurementRegistry::Creator>;
RegistrationMap* GetRegistrationMap() {
  static RegistrationMap* registered_cost_measurements = new RegistrationMap;
  return registered_cost_measurements;
}
}  
std::unique_ptr<CostMeasurement> CostMeasurementRegistry::CreateByNameOrNull(
    const std::string& name, const CostMeasurement::Context& context) {
  const auto it = GetRegistrationMap()->find(name);
  if (it == GetRegistrationMap()->end()) {
    LOG_FIRST_N(ERROR, 1) << "Cost type " << name << " is unregistered.";
    return nullptr;
  }
  return it->second(context);
}
void CostMeasurementRegistry::RegisterCostMeasurement(absl::string_view name,
                                                      Creator creator) {
  const auto it = GetRegistrationMap()->find(name);
  CHECK(it == GetRegistrationMap()->end())  
      << "CostMeasurement " << name << " is registered twice.";
  GetRegistrationMap()->emplace(name, std::move(creator));
}
}  