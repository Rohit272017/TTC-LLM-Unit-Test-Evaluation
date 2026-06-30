#include "tensorflow/core/common_runtime/cost_util.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "tensorflow/core/common_runtime/cost_measurement.h"
#include "tensorflow/core/common_runtime/cost_measurement_registry.h"
#include "tensorflow/core/common_runtime/request_cost_accessor_registry.h"
#include "tensorflow/core/platform/str_util.h"
namespace tensorflow {
namespace {
std::vector<std::string> GetCostMeasurementTypes() {
  const char* types = std::getenv("TF_COST_MEASUREMENT_TYPE");
  if (types == nullptr) return {};
  return str_util::Split(types, " ,");
}
const char* GetRequestCostAccessorType() {
  static const char* accessor = std::getenv("TF_REQUEST_COST_ACCESSOR_TYPE");
  return accessor;
}
}  
std::vector<std::unique_ptr<CostMeasurement>> CreateCostMeasurements(
    const CostMeasurement::Context& context) {
  static const std::vector<std::string>& types =
      *new std::vector<std::string>(GetCostMeasurementTypes());
  std::vector<std::unique_ptr<CostMeasurement>> measurements;
  for (const auto& type : types) {
    std::unique_ptr<CostMeasurement> measurement =
        CostMeasurementRegistry::CreateByNameOrNull(type, context);
    if (measurement != nullptr) {
      measurements.push_back(std::move(measurement));
    }
  }
  return measurements;
}
std::unique_ptr<RequestCostAccessor> CreateRequestCostAccessor() {
  const char* request_cost_accessor_type = GetRequestCostAccessorType();
  return request_cost_accessor_type
             ? RequestCostAccessorRegistry::CreateByNameOrNull(
                   request_cost_accessor_type)
             : nullptr;
}
}  