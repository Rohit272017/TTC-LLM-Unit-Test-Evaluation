#include "tensorstore/internal/dimension_labels.h"
#include <stddef.h>
#include <algorithm>
#include <string>
#include <string_view>
#include "absl/container/fixed_array.h"
#include "absl/status/status.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/str_cat.h"
namespace tensorstore {
namespace internal {
namespace {
absl::Status ValidateDimensionLabelsAreUniqueImpl(
    tensorstore::span<std::string_view> sorted_labels) {
  std::sort(sorted_labels.begin(), sorted_labels.end());
  size_t i;
  for (i = 1; i < sorted_labels.size() && sorted_labels[i].empty(); ++i)
    continue;
  std::string error;
  for (; i < sorted_labels.size(); ++i) {
    std::string_view label = sorted_labels[i];
    if (label == sorted_labels[i - 1]) {
      tensorstore::StrAppend(&error, error.empty() ? "" : ", ",
                             QuoteString(label));
    }
  }
  if (!error.empty()) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Dimension label(s) ", error, " not unique"));
  }
  return absl::OkStatus();
}
}  
absl::Status ValidateDimensionLabelsAreUnique(
    tensorstore::span<const std::string> labels) {
  absl::FixedArray<std::string_view, kMaxRank> sorted_labels(labels.begin(),
                                                             labels.end());
  return ValidateDimensionLabelsAreUniqueImpl(sorted_labels);
}
absl::Status ValidateDimensionLabelsAreUnique(
    tensorstore::span<const std::string_view> labels) {
  absl::FixedArray<std::string_view, kMaxRank> sorted_labels(labels.begin(),
                                                             labels.end());
  return ValidateDimensionLabelsAreUniqueImpl(sorted_labels);
}
}  
}  