#include "tensorflow/core/util/bad_indices_policy.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
namespace tensorflow {
constexpr char kDefault[] = "DEFAULT";
constexpr char kErrorStr[] = "ERROR";
constexpr char kIgnoreStr[] = "IGNORE";
absl::StatusOr<BadIndicesPolicy> BadIndicesPolicyFromString(
    absl::string_view str) {
  if (str.empty()) return BadIndicesPolicy::kDefault;
  if (str == kDefault) return BadIndicesPolicy::kDefault;
  if (str == kErrorStr) return BadIndicesPolicy::kError;
  if (str == kIgnoreStr) return BadIndicesPolicy::kIgnore;
  return absl::InvalidArgumentError(
      absl::StrCat("Unknown bad indices handling attribute: ", str));
}
}  