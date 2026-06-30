#include "eval/public/container_function_registrar.h"
#include "eval/public/cel_options.h"
#include "runtime/runtime_options.h"
#include "runtime/standard/container_functions.h"
namespace google::api::expr::runtime {
absl::Status RegisterContainerFunctions(CelFunctionRegistry* registry,
                                        const InterpreterOptions& options) {
  cel::RuntimeOptions runtime_options = ConvertToRuntimeOptions(options);
  return cel::RegisterContainerFunctions(registry->InternalGetRegistry(),
                                         runtime_options);
}
}  