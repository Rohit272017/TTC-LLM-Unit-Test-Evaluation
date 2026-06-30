#include "runtime/standard/logical_functions.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "base/builtins.h"
#include "base/function_adapter.h"
#include "common/casting.h"
#include "common/value.h"
#include "common/value_manager.h"
#include "internal/status_macros.h"
#include "runtime/function_registry.h"
#include "runtime/internal/errors.h"
#include "runtime/register_function_helper.h"
namespace cel {
namespace {
using ::cel::runtime_internal::CreateNoMatchingOverloadError;
Value NotStrictlyFalseImpl(ValueManager& value_factory, const Value& value) {
  if (InstanceOf<BoolValue>(value)) {
    return value;
  }
  if (InstanceOf<ErrorValue>(value) || InstanceOf<UnknownValue>(value)) {
    return value_factory.CreateBoolValue(true);
  }
  return value_factory.CreateErrorValue(
      CreateNoMatchingOverloadError(builtin::kNotStrictlyFalse));
}
}  
absl::Status RegisterLogicalFunctions(FunctionRegistry& registry,
                                      const RuntimeOptions& options) {
  CEL_RETURN_IF_ERROR(
      (RegisterHelper<UnaryFunctionAdapter<bool, bool>>::RegisterGlobalOverload(
          builtin::kNot,
          [](ValueManager&, bool value) -> bool { return !value; }, registry)));
  using StrictnessHelper = RegisterHelper<UnaryFunctionAdapter<Value, Value>>;
  CEL_RETURN_IF_ERROR(StrictnessHelper::RegisterNonStrictOverload(
      builtin::kNotStrictlyFalse, &NotStrictlyFalseImpl, registry));
  CEL_RETURN_IF_ERROR(StrictnessHelper::RegisterNonStrictOverload(
      builtin::kNotStrictlyFalseDeprecated, &NotStrictlyFalseImpl, registry));
  return absl::OkStatus();
}
}  