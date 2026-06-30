#include "xla/stream_executor/cuda/cuda_version_parser.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/semantic_version.h"
namespace stream_executor {
absl::StatusOr<SemanticVersion> ParseCudaVersion(int cuda_version) {
  if (cuda_version < 0) {
    return absl::InvalidArgumentError("Version numbers cannot be negative!");
  }
  int major = cuda_version / 1000;
  int minor = (cuda_version % 1000) / 10;
  return SemanticVersion(major, minor, 0);
}
}  