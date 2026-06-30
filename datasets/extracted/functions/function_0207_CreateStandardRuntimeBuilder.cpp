#include "runtime/standard_runtime_builder_factory.h"
#include "absl/status/statusor.h"
#include "internal/status_macros.h"
#include "runtime/runtime_builder.h"
#include "runtime/runtime_builder_factory.h"
#include "runtime/runtime_options.h"
#include "runtime/standard_functions.h"
namespace cel {
absl::StatusOr<RuntimeBuilder> CreateStandardRuntimeBuilder(
    const RuntimeOptions& options) {
  RuntimeBuilder result = CreateRuntimeBuilder(options);
  CEL_RETURN_IF_ERROR(
      RegisterStandardFunctions(result.function_registry(), options));
  return result;
}
}  