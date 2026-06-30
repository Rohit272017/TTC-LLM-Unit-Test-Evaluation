#include "xla/stream_executor/semantic_version.h"
#include <string>
#include <vector>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/statusor.h"
namespace stream_executor {
std::string SemanticVersion::ToString() const {
  return absl::StrFormat("%d.%d.%d", major_, minor_, patch_);
}
static absl::StatusOr<unsigned> ParseUnsignedNumber(
    absl::string_view component) {
  unsigned number;
  if (!absl::SimpleAtoi(component, &number)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("'%s' is not an unsigned number.", component));
  }
  return number;
}
absl::StatusOr<SemanticVersion> SemanticVersion::ParseFromString(
    absl::string_view str) {
  std::vector<absl::string_view> components = absl::StrSplit(str, '.');
  if (components.size() != 3) {
    return absl::InvalidArgumentError(
        "Version does not match the format X.Y.Z");
  }
  SemanticVersion result{0, 0, 0};
  TF_ASSIGN_OR_RETURN(result.major(), ParseUnsignedNumber(components[0]));
  TF_ASSIGN_OR_RETURN(result.minor(), ParseUnsignedNumber(components[1]));
  TF_ASSIGN_OR_RETURN(result.patch(), ParseUnsignedNumber(components[2]));
  return result;
}
}  