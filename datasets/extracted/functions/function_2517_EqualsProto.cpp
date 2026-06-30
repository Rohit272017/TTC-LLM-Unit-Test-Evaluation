#ifndef AROLLA_UTIL_TESTING_EQUALS_PROTO_H_
#define AROLLA_UTIL_TESTING_EQUALS_PROTO_H_
#include <string>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
namespace arolla::testing {
template <typename TypeProto>
::testing::AssertionResult EqualsProto(const TypeProto& actual_proto,
                                       absl::string_view expected_proto_text) {
  using ::google::protobuf::TextFormat;
  using ::google::protobuf::util::MessageDifferencer;
  TypeProto expected_proto;
  if (!TextFormat::ParseFromString(std::string(expected_proto_text),  
                                   &expected_proto)) {
    return ::testing::AssertionFailure()
           << "could not parse proto: " << expected_proto_text;
  }
  MessageDifferencer differencer;
  std::string differences;
  differencer.ReportDifferencesToString(&differences);
  if (!differencer.Compare(expected_proto, actual_proto)) {
    return ::testing::AssertionFailure() << "the protos are different:\n"
                                         << differences;
  }
  return ::testing::AssertionSuccess();
}
inline auto EqualsProto(absl::string_view expected_proto_text) {
  return ::testing::Truly([expected_proto_text = std::string(
                               expected_proto_text)](const auto& actual_proto) {
    return EqualsProto(actual_proto, expected_proto_text);
  });
}
}  
#endif  