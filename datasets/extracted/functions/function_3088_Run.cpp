#include "xla/service/gpu/transforms/pgle_accuracy_checker.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "tsl/platform/errors.h"
namespace xla::gpu {
absl::StatusOr<bool> PGLEAccuracyChecker::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_RETURN_IF_ERROR(pgle_estimator_.CheckAccuracy(*module));
  return false;
}
}  