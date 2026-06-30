#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
namespace tensorflow {
namespace serving {
absl::StatusOr<MixedPriorityBatchingPolicy> GetMixedPriorityBatchingPolicy(
    absl::string_view attr_value) {
  if (attr_value == kLowPriorityPaddingWithMaxBatchSizeAttrValue) {
    return MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize;
  } else if (attr_value ==
             kLowPriorityPaddingWithNextAllowedBatchSizeAttrValue) {
    return MixedPriorityBatchingPolicy::
        kLowPriorityPaddingWithNextAllowedBatchSize;
  } else if (attr_value == kPriorityIsolationAttrValue) {
    return MixedPriorityBatchingPolicy::kPriorityIsolation;
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Unknown mixed priority batching policy: %s", attr_value));
}
}  
}  