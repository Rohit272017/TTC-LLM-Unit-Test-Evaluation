#ifndef THIRD_PARTY_CEL_CPP_INTERNAL_MESSAGE_TYPE_NAME_H_
#define THIRD_PARTY_CEL_CPP_INTERNAL_MESSAGE_TYPE_NAME_H_
#include <string>
#include <type_traits>
#include "absl/base/no_destructor.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/message.h"
#include "google/protobuf/message_lite.h"
namespace cel::internal {
template <typename T>
std::enable_if_t<
    std::conjunction_v<std::is_base_of<google::protobuf::MessageLite, T>,
                       std::negation<std::is_base_of<google::protobuf::Message, T>>>,
    absl::string_view>
MessageTypeNameFor() {
  static_assert(!std::is_const_v<T>, "T must not be const qualified");
  static_assert(!std::is_volatile_v<T>, "T must not be volatile qualified");
  static_assert(!std::is_reference_v<T>, "T must not be a reference");
  static const absl::NoDestructor<std::string> kTypeName(T().GetTypeName());
  return *kTypeName;
}
template <typename T>
std::enable_if_t<std::is_base_of_v<google::protobuf::Message, T>, absl::string_view>
MessageTypeNameFor() {
  static_assert(!std::is_const_v<T>, "T must not be const qualified");
  static_assert(!std::is_volatile_v<T>, "T must not be volatile qualified");
  static_assert(!std::is_reference_v<T>, "T must not be a reference");
  return T::descriptor()->full_name();
}
}  
#endif  