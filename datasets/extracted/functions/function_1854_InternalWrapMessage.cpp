#include "eval/public/structs/cel_proto_wrapper.h"
#include "absl/types/optional.h"
#include "eval/public/cel_value.h"
#include "eval/public/message_wrapper.h"
#include "eval/public/structs/cel_proto_wrap_util.h"
#include "eval/public/structs/proto_message_type_adapter.h"
#include "google/protobuf/arena.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
namespace google::api::expr::runtime {
namespace {
using ::google::protobuf::Arena;
using ::google::protobuf::Descriptor;
using ::google::protobuf::Message;
}  
CelValue CelProtoWrapper::InternalWrapMessage(const Message* message) {
  return CelValue::CreateMessageWrapper(
      MessageWrapper(message, &GetGenericProtoTypeInfoInstance()));
}
CelValue CelProtoWrapper::CreateMessage(const Message* value, Arena* arena) {
  return internal::UnwrapMessageToValue(value, &InternalWrapMessage, arena);
}
absl::optional<CelValue> CelProtoWrapper::MaybeWrapValue(
    const Descriptor* descriptor, google::protobuf::MessageFactory* factory,
    const CelValue& value, Arena* arena) {
  const Message* msg =
      internal::MaybeWrapValueToMessage(descriptor, factory, value, arena);
  if (msg != nullptr) {
    return InternalWrapMessage(msg);
  } else {
    return absl::nullopt;
  }
}
}  