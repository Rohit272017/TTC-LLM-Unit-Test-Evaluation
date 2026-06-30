#include "runtime/standard/container_functions.h"
#include <utility>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "base/builtins.h"
#include "base/function_adapter.h"
#include "common/value.h"
#include "common/value_manager.h"
#include "internal/status_macros.h"
#include "runtime/function_registry.h"
#include "runtime/internal/mutable_list_impl.h"
#include "runtime/runtime_options.h"
namespace cel {
namespace {
using cel::runtime_internal::MutableListValue;
absl::StatusOr<int64_t> MapSizeImpl(ValueManager&, const MapValue& value) {
  return value.Size();
}
absl::StatusOr<int64_t> ListSizeImpl(ValueManager&, const ListValue& value) {
  return value.Size();
}
absl::StatusOr<ListValue> ConcatList(ValueManager& factory,
                                     const ListValue& value1,
                                     const ListValue& value2) {
  CEL_ASSIGN_OR_RETURN(auto size1, value1.Size());
  if (size1 == 0) {
    return value2;
  }
  CEL_ASSIGN_OR_RETURN(auto size2, value2.Size());
  if (size2 == 0) {
    return value1;
  }
  CEL_ASSIGN_OR_RETURN(auto list_builder,
                       factory.NewListValueBuilder(factory.GetDynListType()));
  list_builder->Reserve(size1 + size2);
  for (int i = 0; i < size1; i++) {
    CEL_ASSIGN_OR_RETURN(Value elem, value1.Get(factory, i));
    CEL_RETURN_IF_ERROR(list_builder->Add(std::move(elem)));
  }
  for (int i = 0; i < size2; i++) {
    CEL_ASSIGN_OR_RETURN(Value elem, value2.Get(factory, i));
    CEL_RETURN_IF_ERROR(list_builder->Add(std::move(elem)));
  }
  return std::move(*list_builder).Build();
}
absl::StatusOr<OpaqueValue> AppendList(ValueManager& factory,
                                       OpaqueValue value1,
                                       const ListValue& value2) {
  if (!MutableListValue::Is(value1)) {
    return absl::InvalidArgumentError(
        "Unexpected call to runtime list append.");
  }
  MutableListValue& mutable_list = MutableListValue::Cast(value1);
  CEL_ASSIGN_OR_RETURN(auto size2, value2.Size());
  for (int i = 0; i < size2; i++) {
    CEL_ASSIGN_OR_RETURN(Value elem, value2.Get(factory, i));
    CEL_RETURN_IF_ERROR(mutable_list.Append(std::move(elem)));
  }
  return value1;
}
}  
absl::Status RegisterContainerFunctions(FunctionRegistry& registry,
                                        const RuntimeOptions& options) {
  for (bool receiver_style : {true, false}) {
    CEL_RETURN_IF_ERROR(registry.Register(
        cel::UnaryFunctionAdapter<absl::StatusOr<int64_t>, const ListValue&>::
            CreateDescriptor(cel::builtin::kSize, receiver_style),
        UnaryFunctionAdapter<absl::StatusOr<int64_t>,
                             const ListValue&>::WrapFunction(ListSizeImpl)));
    CEL_RETURN_IF_ERROR(registry.Register(
        UnaryFunctionAdapter<absl::StatusOr<int64_t>, const MapValue&>::
            CreateDescriptor(cel::builtin::kSize, receiver_style),
        UnaryFunctionAdapter<absl::StatusOr<int64_t>,
                             const MapValue&>::WrapFunction(MapSizeImpl)));
  }
  if (options.enable_list_concat) {
    CEL_RETURN_IF_ERROR(registry.Register(
        BinaryFunctionAdapter<
            absl::StatusOr<Value>, const ListValue&,
            const ListValue&>::CreateDescriptor(cel::builtin::kAdd, false),
        BinaryFunctionAdapter<absl::StatusOr<Value>, const ListValue&,
                              const ListValue&>::WrapFunction(ConcatList)));
  }
  return registry.Register(
      BinaryFunctionAdapter<
          absl::StatusOr<OpaqueValue>, OpaqueValue,
          const ListValue&>::CreateDescriptor(cel::builtin::kRuntimeListAppend,
                                              false),
      BinaryFunctionAdapter<absl::StatusOr<OpaqueValue>, OpaqueValue,
                            const ListValue&>::WrapFunction(AppendList));
}
}  