#include "tensorflow/core/common_runtime/no_op_cost_measurement.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/cost_constants.h"
namespace tensorflow {
absl::Duration NoOpCostMeasurement::GetTotalCost() { return absl::Duration(); }
absl::string_view NoOpCostMeasurement::GetCostType() const {
  return kNoOpCostName;
}
REGISTER_COST_MEASUREMENT(kNoOpCostName, NoOpCostMeasurement);
}  