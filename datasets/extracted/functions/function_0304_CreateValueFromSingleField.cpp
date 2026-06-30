#include "eval/public/containers/field_access.h"
#include "google/protobuf/arena.h"
#include "google/protobuf/map_field.h"
#include "absl/status/status.h"
#include "eval/public/structs/cel_proto_wrapper.h"
#include "eval/public/structs/field_access_impl.h"
#include "internal/status_macros.h"
namespace google::api::expr::runtime {
using ::google::protobuf::Arena;
using ::google::protobuf::FieldDescriptor;
using ::google::protobuf::MapValueConstRef;
using ::google::protobuf::Message;
absl::Status CreateValueFromSingleField(const google::protobuf::Message* msg,
                                        const FieldDescriptor* desc,
                                        google::protobuf::Arena* arena,
                                        CelValue* result) {
  return CreateValueFromSingleField(
      msg, desc, ProtoWrapperTypeOptions::kUnsetProtoDefault, arena, result);
}
absl::Status CreateValueFromSingleField(const google::protobuf::Message* msg,
                                        const FieldDescriptor* desc,
                                        ProtoWrapperTypeOptions options,
                                        google::protobuf::Arena* arena,
                                        CelValue* result) {
  CEL_ASSIGN_OR_RETURN(
      *result,
      internal::CreateValueFromSingleField(
          msg, desc, options, &CelProtoWrapper::InternalWrapMessage, arena));
  return absl::OkStatus();
}
absl::Status CreateValueFromRepeatedField(const google::protobuf::Message* msg,
                                          const FieldDescriptor* desc,
                                          google::protobuf::Arena* arena, int index,
                                          CelValue* result) {
  CEL_ASSIGN_OR_RETURN(
      *result,
      internal::CreateValueFromRepeatedField(
          msg, desc, index, &CelProtoWrapper::InternalWrapMessage, arena));
  return absl::OkStatus();
}
absl::Status CreateValueFromMapValue(const google::protobuf::Message* msg,
                                     const FieldDescriptor* desc,
                                     const MapValueConstRef* value_ref,
                                     google::protobuf::Arena* arena, CelValue* result) {
  CEL_ASSIGN_OR_RETURN(
      *result,
      internal::CreateValueFromMapValue(
          msg, desc, value_ref, &CelProtoWrapper::InternalWrapMessage, arena));
  return absl::OkStatus();
}
absl::Status SetValueToSingleField(const CelValue& value,
                                   const FieldDescriptor* desc, Message* msg,
                                   Arena* arena) {
  return internal::SetValueToSingleField(value, desc, msg, arena);
}
absl::Status AddValueToRepeatedField(const CelValue& value,
                                     const FieldDescriptor* desc, Message* msg,
                                     Arena* arena) {
  return internal::AddValueToRepeatedField(value, desc, msg, arena);
}
}  