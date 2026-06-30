#include "eval/public/string_extension_func_registrar.h"
#include "absl/status/status.h"
#include "eval/public/cel_function_registry.h"
#include "eval/public/cel_options.h"
#include "extensions/strings.h"
namespace google::api::expr::runtime {
absl::Status RegisterStringExtensionFunctions(
    CelFunctionRegistry* registry, const InterpreterOptions& options) {
  return cel::extensions::RegisterStringsFunctions(registry, options);
}
}  