#include "eval/public/comparison_functions.h"
#include "absl/status/status.h"
#include "eval/public/cel_function_registry.h"
#include "eval/public/cel_options.h"
#include "runtime/function_registry.h"
#include "runtime/runtime_options.h"
#include "runtime/standard/comparison_functions.h"
namespace google::api::expr::runtime {
absl::Status RegisterComparisonFunctions(CelFunctionRegistry* registry,
                                         const InterpreterOptions& options) {
  cel::RuntimeOptions modern_options = ConvertToRuntimeOptions(options);
  cel::FunctionRegistry& modern_registry = registry->InternalGetRegistry();
  return cel::RegisterComparisonFunctions(modern_registry, modern_options);
}
}  