#ifndef THIRD_PARTY_CEL_CPP_EVAL_PUBLIC_CONTAINERS_FIELD_BACKED_MAP_IMPL_H_
#define THIRD_PARTY_CEL_CPP_EVAL_PUBLIC_CONTAINERS_FIELD_BACKED_MAP_IMPL_H_
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "absl/status/statusor.h"
#include "eval/public/cel_value.h"
#include "eval/public/containers/internal_field_backed_map_impl.h"
#include "eval/public/structs/cel_proto_wrapper.h"
namespace google::api::expr::runtime {
class FieldBackedMapImpl : public internal::FieldBackedMapImpl {
 public:
  FieldBackedMapImpl(const google::protobuf::Message* message,
                     const google::protobuf::FieldDescriptor* descriptor,
                     google::protobuf::Arena* arena)
      : internal::FieldBackedMapImpl(
            message, descriptor, &CelProtoWrapper::InternalWrapMessage, arena) {
  }
};
}  
#endif  