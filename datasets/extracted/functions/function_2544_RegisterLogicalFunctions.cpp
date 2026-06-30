#include "eval/public/logical_function_registrar.h"
#include "absl/status/status.h"
#include "eval/public/cel_function_registry.h"
#include "eval/public/cel_options.h"
#include "runtime/standard/logical_functions.h"
namespace google::api::expr::runtime {
absl::Status RegisterLogicalFunctions(CelFunctionRegistry* registry,
                                      const InterpreterOptions& options) {
  return cel::RegisterLogicalFunctions(registry->InternalGetRegistry(),
                                       ConvertToRuntimeOptions(options));
}
}  