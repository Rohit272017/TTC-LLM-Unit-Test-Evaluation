#include "xla/stream_executor/rocm/rocm_version_parser.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/semantic_version.h"
namespace stream_executor {
absl::StatusOr<SemanticVersion> ParseRocmVersion(int rocm_version) {
  if (rocm_version < 0) {
    return absl::InvalidArgumentError("Version numbers cannot be negative.");
  }
  int major = rocm_version / 10'000'000;
  int minor = (rocm_version % 10'000'000) / 100'000;
  int patch = rocm_version % 100'000;
  return SemanticVersion(major, minor, patch);
}
}  