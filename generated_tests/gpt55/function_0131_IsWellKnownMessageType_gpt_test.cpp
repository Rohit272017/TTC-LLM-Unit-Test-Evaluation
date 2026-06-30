#include "common/type.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <regex>
#include <string>

#include "google/protobuf/any.pb.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/duration.pb.h"
#include "google/protobuf/empty.pb.h"
#include "google/protobuf/field_mask.pb.h"
#include "google/protobuf/struct.pb.h"
#include "google/protobuf/timestamp.pb.h"
#include "google/protobuf/wrappers.pb.h"

namespace cel {
namespace {

using google::protobuf::Descriptor;
using google::protobuf::DescriptorPool;
using google::protobuf::FieldDescriptor;
using google::protobuf::FileDescriptor;
using google::protobuf::FileDescriptorProto;

TEST(IsWellKnownMessageTypeTest, WrapperTypesReturnTrue) {
  EXPECT_TRUE(IsWellKnownMessageType(google::protobuf::BoolValue::descriptor()));
  EXPECT_TRUE(IsWellKnownMessageType(google::protobuf::Int32Value::descriptor()));
  EXPECT_TRUE(IsWellKnownMessageType(google::protobuf::Int64Value::descriptor()));
  EXPECT_TRUE(IsWellKnownMessageType(google::protobuf::UInt32Value::descriptor()));
  EXPECT_TRUE(IsWellKnownMessageType(google::protobuf::UInt64Value::descriptor()));
  EXPECT_TRUE(IsWellKnownMessageType(google::protobuf::FloatValue::descriptor()));
  EXPECT_TRUE(IsWellKnownMessageType(google::protobuf::DoubleValue::descriptor()));
  EXPECT_TRUE(IsWellKnownMessageType(google::protobuf::BytesValue::descriptor()));
  EXPECT_TRUE(IsWellKnownMessageType(google::protobuf::StringValue::descriptor()));
}

TEST(IsWellKnownMessageTypeTest, AnyDurationAndTimestampReturnTrue) {
  EXPECT_TRUE(IsWellKnownMessageType(google::protobuf::Any::descriptor()));
  EXPECT_TRUE(IsWellKnownMessageType(google::protobuf::Duration::descriptor()));
  EXPECT_TRUE(IsWellKnownMessageType(google::protobuf::Timestamp::descriptor()));
}

TEST(IsWellKnownMessageTypeTest, StructValueAndListValueReturnTrue) {
  EXPECT_TRUE(IsWellKnownMessageType(google::protobuf::Value::descriptor()));
  EXPECT_TRUE(IsWellKnownMessageType(google::protobuf::ListValue::descriptor()));
  EXPECT_TRUE(IsWellKnownMessageType(google::protobuf::Struct::descriptor()));
}

TEST(IsWellKnownMessageTypeTest, EmptyReturnsFalse) {
  EXPECT_FALSE(IsWellKnownMessageType(google::protobuf::Empty::descriptor()));
}

TEST(IsWellKnownMessageTypeTest, FieldMaskReturnsFalse) {
  EXPECT_FALSE(IsWellKnownMessageType(google::protobuf::FieldMask::descriptor()));
}

TEST(IsWellKnownMessageTypeTest, CustomMessageReturnsFalse) {
  FileDescriptorProto file_proto;
  file_proto.set_name("custom_message.proto");
  file_proto.set_package("test");

  auto* message = file_proto.add_message_type();
  message->set_name("CustomMessage");

  DescriptorPool pool;
  const FileDescriptor* file = pool.BuildFile(file_proto);
  ASSERT_NE(file, nullptr);

  const Descriptor* descriptor = file->FindMessageTypeByName("CustomMessage");
  ASSERT_NE(descriptor, nullptr);

  EXPECT_FALSE(IsWellKnownMessageType(descriptor));
}

TEST(IsWellKnownMessageTypeTest, CustomMessageNamedLikeWellKnownButWrongPackageReturnsFalse) {
  FileDescriptorProto file_proto;
  file_proto.set_name("fake_wrappers.proto");
  file_proto.set_package("fake.google.protobuf");

  auto* message = file_proto.add_message_type();
  message->set_name("BoolValue");

  DescriptorPool pool;
  const FileDescriptor* file = pool.BuildFile(file_proto);
  ASSERT_NE(file, nullptr);

  const Descriptor* descriptor = file->FindMessageTypeByName("BoolValue");
  ASSERT_NE(descriptor, nullptr);

  EXPECT_FALSE(IsWellKnownMessageType(descriptor));
}

TEST(MessageTypeDebugStringTest, DefaultConstructedMessageTypeReturnsEmptyString) {
  MessageType message_type;

  EXPECT_EQ(message_type.DebugString(), "");
}

TEST(MessageTypeDebugStringTest, ValidMessageTypeContainsNameAndPointer) {
  MessageType message_type(google::protobuf::Any::descriptor());

  const std::string debug_string = message_type.DebugString();

  EXPECT_NE(debug_string.find("google.protobuf.Any@0x"), std::string::npos);
#if INTPTR_MAX == INT64_MAX
  EXPECT_TRUE(std::regex_match(
      debug_string,
      std::regex("google\\.protobuf\\.Any@0x[0-9a-fA-F]{16}")));
#else
  EXPECT_TRUE(std::regex_match(
      debug_string,
      std::regex("google\\.protobuf\\.Any@0x[0-9a-fA-F]{8}")));
#endif
}

TEST(MessageTypeDebugStringTest, DifferentMessageTypesProduceDifferentDebugStrings) {
  MessageType any_type(google::protobuf::Any::descriptor());
  MessageType duration_type(google::protobuf::Duration::descriptor());

  EXPECT_NE(any_type.DebugString(), duration_type.DebugString());
  EXPECT_NE(any_type.DebugString().find("google.protobuf.Any@0x"),
            std::string::npos);
  EXPECT_NE(duration_type.DebugString().find("google.protobuf.Duration@0x"),
            std::string::npos);
}

TEST(MessageTypeFieldDebugStringTest, DefaultConstructedFieldReturnsEmptyString) {
  MessageTypeField field;

  EXPECT_EQ(field.DebugString(), "");
}

TEST(MessageTypeFieldDebugStringTest, ValidFieldDebugStringContainsNumberNameAndPointer) {
  const FieldDescriptor* field_descriptor =
      google::protobuf::Any::descriptor()->FindFieldByName("type_url");
  ASSERT_NE(field_descriptor, nullptr);

  MessageTypeField field(field_descriptor);

  const std::string debug_string = field.DebugString();

  EXPECT_NE(debug_string.find("[1]type_url@0x"), std::string::npos);
#if INTPTR_MAX == INT64_MAX
  EXPECT_TRUE(std::regex_match(
      debug_string,
      std::regex("\\[1\\]type_url@0x[0-9a-fA-F]{16}")));
#else
  EXPECT_TRUE(std::regex_match(
      debug_string,
      std::regex("\\[1\\]type_url@0x[0-9a-fA-F]{8}")));
#endif
}

TEST(MessageTypeFieldDebugStringTest, DifferentFieldsProduceDifferentDebugStrings) {
  const FieldDescriptor* type_url_field =
      google::protobuf::Any::descriptor()->FindFieldByName("type_url");
  const FieldDescriptor* value_field =
      google::protobuf::Any::descriptor()->FindFieldByName("value");

  ASSERT_NE(type_url_field, nullptr);
  ASSERT_NE(value_field, nullptr);

  MessageTypeField type_url(type_url_field);
  MessageTypeField value(value_field);

  EXPECT_NE(type_url.DebugString(), value.DebugString());
  EXPECT_NE(type_url.DebugString().find("[1]type_url@0x"), std::string::npos);
  EXPECT_NE(value.DebugString().find("[2]value@0x"), std::string::npos);
}

TEST(MessageTypeFieldGetTypeTest, ValidFieldReturnsFieldType) {
  const FieldDescriptor* field_descriptor =
      google::protobuf::Any::descriptor()->FindFieldByName("type_url");
  ASSERT_NE(field_descriptor, nullptr);

  MessageTypeField field(field_descriptor);

  Type type = field.GetType();

  EXPECT_EQ(type, Type::Field(field_descriptor));
}

TEST(MessageTypeFieldGetTypeTest, DifferentFieldsReturnDifferentTypes) {
  const FieldDescriptor* type_url_field =
      google::protobuf::Any::descriptor()->FindFieldByName("type_url");
  const FieldDescriptor* value_field =
      google::protobuf::Any::descriptor()->FindFieldByName("value");

  ASSERT_NE(type_url_field, nullptr);
  ASSERT_NE(value_field, nullptr);

  MessageTypeField type_url(type_url_field);
  MessageTypeField value(value_field);

  EXPECT_EQ(type_url.GetType(), Type::Field(type_url_field));
  EXPECT_EQ(value.GetType(), Type::Field(value_field));
  EXPECT_NE(type_url.GetType(), value.GetType());
}

}  // namespace
}  // namespace cel