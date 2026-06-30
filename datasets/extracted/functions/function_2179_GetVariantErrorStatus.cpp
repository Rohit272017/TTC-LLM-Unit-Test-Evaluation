#include <stddef.h>
#include <string>
#include "absl/status/status.h"
#include "tensorstore/util/span.h"
namespace tensorstore {
namespace internal_json_binding {
absl::Status GetVariantErrorStatus(span<const absl::Status> status_values) {
  std::string error = "No matching value binder: ";
  for (size_t i = 0; i < status_values.size(); ++i) {
    if (i != 0) error += "; ";
    error += status_values[i].message();
  }
  return absl::InvalidArgumentError(error);
}
}  
}  