#include "eval/public/equality_function_registrar.h"
#include "absl/status/status.h"
#include "eval/public/cel_function_registry.h"
#include "eval/public/cel_options.h"
#include "runtime/runtime_options.h"
#include "runtime/standard/equality_functions.h"
namespace google::api::expr::runtime {
absl::Status RegisterEqualityFunctions(CelFunctionRegistry* registry,
                                       const InterpreterOptions& options) {
  cel::RuntimeOptions runtime_options = ConvertToRuntimeOptions(options);
  return cel::RegisterEqualityFunctions(registry->InternalGetRegistry(),
                                        runtime_options);
}
}  