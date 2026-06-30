#ifndef THIRD_PARTY_CEL_CPP_EVAL_PUBLIC_CEL_FUNCTION_ADAPTER_H_
#define THIRD_PARTY_CEL_CPP_EVAL_PUBLIC_CEL_FUNCTION_ADAPTER_H_
#include <cstdint>
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>
#include "google/protobuf/message.h"
#include "absl/status/status.h"
#include "eval/public/cel_function_adapter_impl.h"
#include "eval/public/cel_value.h"
#include "eval/public/structs/cel_proto_wrapper.h"
namespace google::api::expr::runtime {
namespace internal {
struct ProtoAdapterTypeCodeMatcher {
  template <typename T>
  constexpr static std::optional<CelValue::Type> type_code() {
    if constexpr (std::is_same_v<T, const google::protobuf::Message*>) {
      return CelValue::Type::kMessage;
    } else {
      return internal::TypeCodeMatcher().type_code<T>();
    }
  }
};
struct ProtoAdapterValueConverter
    : public internal::ValueConverterBase<ProtoAdapterValueConverter> {
  using BaseType = internal::ValueConverterBase<ProtoAdapterValueConverter>;
  using BaseType::NativeToValue;
  using BaseType::ValueToNative;
  absl::Status NativeToValue(const ::google::protobuf::Message* value,
                             ::google::protobuf::Arena* arena, CelValue* result) {
    if (value == nullptr) {
      return absl::Status(absl::StatusCode::kInvalidArgument,
                          "Null Message pointer returned");
    }
    *result = CelProtoWrapper::CreateMessage(value, arena);
    return absl::OkStatus();
  }
};
}  
template <typename ReturnType, typename... Arguments>
using FunctionAdapter =
    internal::FunctionAdapterImpl<internal::ProtoAdapterTypeCodeMatcher,
                                  internal::ProtoAdapterValueConverter>::
        FunctionAdapter<ReturnType, Arguments...>;
template <typename ReturnType, typename T>
using UnaryFunctionAdapter = internal::FunctionAdapterImpl<
    internal::ProtoAdapterTypeCodeMatcher,
    internal::ProtoAdapterValueConverter>::UnaryFunction<ReturnType, T>;
template <typename ReturnType, typename T, typename U>
using BinaryFunctionAdapter = internal::FunctionAdapterImpl<
    internal::ProtoAdapterTypeCodeMatcher,
    internal::ProtoAdapterValueConverter>::BinaryFunction<ReturnType, T, U>;
}  
#endif  