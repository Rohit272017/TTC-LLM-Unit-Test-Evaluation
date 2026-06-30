#include "extensions/sets_functions.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "base/function_adapter.h"
#include "common/value.h"
#include "common/value_manager.h"
#include "internal/status_macros.h"
#include "runtime/function_registry.h"
#include "runtime/runtime_options.h"
namespace cel::extensions {
namespace {
absl::StatusOr<Value> SetsContains(ValueManager& value_factory,
                                   const ListValue& list,
                                   const ListValue& sublist) {
  bool any_missing = false;
  CEL_RETURN_IF_ERROR(sublist.ForEach(
      value_factory,
      [&list, &value_factory,
       &any_missing](const Value& sublist_element) -> absl::StatusOr<bool> {
        CEL_ASSIGN_OR_RETURN(auto contains,
                             list.Contains(value_factory, sublist_element));
        any_missing =
            !contains->Is<BoolValue>() || !contains.GetBool().NativeValue();
        return !any_missing;
      }));
  return value_factory.CreateBoolValue(!any_missing);
}
absl::StatusOr<Value> SetsIntersects(ValueManager& value_factory,
                                     const ListValue& list,
                                     const ListValue& sublist) {
  bool exists = false;
  CEL_RETURN_IF_ERROR(list.ForEach(
      value_factory,
      [&value_factory, &sublist,
       &exists](const Value& list_element) -> absl::StatusOr<bool> {
        CEL_ASSIGN_OR_RETURN(auto contains,
                             sublist.Contains(value_factory, list_element));
        exists = contains->Is<BoolValue>() && contains.GetBool().NativeValue();
        return !exists;
      }));
  return value_factory.CreateBoolValue(exists);
}
absl::StatusOr<Value> SetsEquivalent(ValueManager& value_factory,
                                     const ListValue& list,
                                     const ListValue& sublist) {
  CEL_ASSIGN_OR_RETURN(auto contains_sublist,
                       SetsContains(value_factory, list, sublist));
  if (contains_sublist.Is<BoolValue>() &&
      !contains_sublist.GetBool().NativeValue()) {
    return contains_sublist;
  }
  return SetsContains(value_factory, sublist, list);
}
absl::Status RegisterSetsContainsFunction(FunctionRegistry& registry) {
  return registry.Register(
      BinaryFunctionAdapter<
          absl::StatusOr<Value>, const ListValue&,
          const ListValue&>::CreateDescriptor("sets.contains",
                                              false),
      BinaryFunctionAdapter<absl::StatusOr<Value>, const ListValue&,
                            const ListValue&>::WrapFunction(SetsContains));
}
absl::Status RegisterSetsIntersectsFunction(FunctionRegistry& registry) {
  return registry.Register(
      BinaryFunctionAdapter<
          absl::StatusOr<Value>, const ListValue&,
          const ListValue&>::CreateDescriptor("sets.intersects",
                                              false),
      BinaryFunctionAdapter<absl::StatusOr<Value>, const ListValue&,
                            const ListValue&>::WrapFunction(SetsIntersects));
}
absl::Status RegisterSetsEquivalentFunction(FunctionRegistry& registry) {
  return registry.Register(
      BinaryFunctionAdapter<
          absl::StatusOr<Value>, const ListValue&,
          const ListValue&>::CreateDescriptor("sets.equivalent",
                                              false),
      BinaryFunctionAdapter<absl::StatusOr<Value>, const ListValue&,
                            const ListValue&>::WrapFunction(SetsEquivalent));
}
}  
absl::Status RegisterSetsFunctions(FunctionRegistry& registry,
                                   const RuntimeOptions& options) {
  CEL_RETURN_IF_ERROR(RegisterSetsContainsFunction(registry));
  CEL_RETURN_IF_ERROR(RegisterSetsIntersectsFunction(registry));
  CEL_RETURN_IF_ERROR(RegisterSetsEquivalentFunction(registry));
  return absl::OkStatus();
}
}  