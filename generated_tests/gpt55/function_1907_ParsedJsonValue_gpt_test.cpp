#include "common/values/parsed_json_value.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>

#include "absl/status/status.h"
#include "common/allocator.h"
#include "common/value.h"
#include "google/protobuf/struct.pb.h"

namespace cel::common_internal {
namespace {

using ::google::protobuf::ListValue;
using ::google::protobuf::Struct;
using ::google::protobuf::Value;

class ParsedJsonValueTest : public ::testing::Test {
 protected:
  Allocator<> allocator_ = NewDeleteAllocator();
};

TEST_F(ParsedJsonValueTest, ECP_KindNotSetReturnsNullValue) {
  Value proto_value;

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  EXPECT_TRUE(result.IsNull());
}

TEST_F(ParsedJsonValueTest, ECP_ExplicitNullReturnsNullValue) {
  Value proto_value;
  proto_value.set_null_value(google::protobuf::NULL_VALUE);

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  EXPECT_TRUE(result.IsNull());
}

TEST_F(ParsedJsonValueTest, ECP_BoolTrueReturnsBoolValueTrue) {
  Value proto_value;
  proto_value.set_bool_value(true);

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsBool());
  EXPECT_TRUE(result.GetBool().NativeValue());
}

TEST_F(ParsedJsonValueTest, ECP_BoolFalseReturnsBoolValueFalse) {
  Value proto_value;
  proto_value.set_bool_value(false);

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsBool());
  EXPECT_FALSE(result.GetBool().NativeValue());
}

TEST_F(ParsedJsonValueTest, ECP_PositiveNumberReturnsDoubleValue) {
  Value proto_value;
  proto_value.set_number_value(123.75);

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsDouble());
  EXPECT_DOUBLE_EQ(result.GetDouble().NativeValue(), 123.75);
}

TEST_F(ParsedJsonValueTest, ECP_NegativeNumberReturnsDoubleValue) {
  Value proto_value;
  proto_value.set_number_value(-456.25);

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsDouble());
  EXPECT_DOUBLE_EQ(result.GetDouble().NativeValue(), -456.25);
}

TEST_F(ParsedJsonValueTest, BVA_NumberZeroReturnsDoubleValueZero) {
  Value proto_value;
  proto_value.set_number_value(0.0);

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsDouble());
  EXPECT_DOUBLE_EQ(result.GetDouble().NativeValue(), 0.0);
}

TEST_F(ParsedJsonValueTest, BVA_NumberNegativeZeroReturnsDoubleValueNegativeZero) {
  Value proto_value;
  proto_value.set_number_value(-0.0);

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsDouble());
  EXPECT_DOUBLE_EQ(result.GetDouble().NativeValue(), -0.0);
  EXPECT_TRUE(std::signbit(result.GetDouble().NativeValue()));
}

TEST_F(ParsedJsonValueTest, BVA_NumberMaxDoubleReturnsDoubleValue) {
  Value proto_value;
  proto_value.set_number_value(std::numeric_limits<double>::max());

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsDouble());
  EXPECT_DOUBLE_EQ(result.GetDouble().NativeValue(),
                   std::numeric_limits<double>::max());
}

TEST_F(ParsedJsonValueTest, BVA_NumberLowestDoubleReturnsDoubleValue) {
  Value proto_value;
  proto_value.set_number_value(std::numeric_limits<double>::lowest());

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsDouble());
  EXPECT_DOUBLE_EQ(result.GetDouble().NativeValue(),
                   std::numeric_limits<double>::lowest());
}

TEST_F(ParsedJsonValueTest, Edge_NumberPositiveInfinityReturnsDoubleValue) {
  Value proto_value;
  proto_value.set_number_value(std::numeric_limits<double>::infinity());

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsDouble());
  EXPECT_TRUE(std::isinf(result.GetDouble().NativeValue()));
  EXPECT_GT(result.GetDouble().NativeValue(), 0.0);
}

TEST_F(ParsedJsonValueTest, Edge_NumberNegativeInfinityReturnsDoubleValue) {
  Value proto_value;
  proto_value.set_number_value(-std::numeric_limits<double>::infinity());

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsDouble());
  EXPECT_TRUE(std::isinf(result.GetDouble().NativeValue()));
  EXPECT_LT(result.GetDouble().NativeValue(), 0.0);
}

TEST_F(ParsedJsonValueTest, Edge_NumberNaNReturnsDoubleValueNaN) {
  Value proto_value;
  proto_value.set_number_value(std::numeric_limits<double>::quiet_NaN());

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsDouble());
  EXPECT_TRUE(std::isnan(result.GetDouble().NativeValue()));
}

TEST_F(ParsedJsonValueTest, ECP_StringReturnsStringValue) {
  Value proto_value;
  proto_value.set_string_value("hello");

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsString());
  EXPECT_EQ(result.GetString().ToString(), "hello");
}

TEST_F(ParsedJsonValueTest, BVA_EmptyStringReturnsEmptyStringValue) {
  Value proto_value;
  proto_value.set_string_value("");

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsString());
  EXPECT_TRUE(result.GetString().IsEmpty());
  EXPECT_EQ(result.GetString().ToString(), "");
}

TEST_F(ParsedJsonValueTest, BVA_SingleCharacterStringReturnsStringValue) {
  Value proto_value;
  proto_value.set_string_value("a");

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsString());
  EXPECT_EQ(result.GetString().ToString(), "a");
}

TEST_F(ParsedJsonValueTest, Edge_StringWithWhitespaceIsPreserved) {
  Value proto_value;
  proto_value.set_string_value("  hello world  ");

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsString());
  EXPECT_EQ(result.GetString().ToString(), "  hello world  ");
}

TEST_F(ParsedJsonValueTest, Edge_StringWithEscapedCharactersIsPreserved) {
  Value proto_value;
  proto_value.set_string_value("line1\nline2\t\"quoted\"");

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsString());
  EXPECT_EQ(result.GetString().ToString(), "line1\nline2\t\"quoted\"");
}

TEST_F(ParsedJsonValueTest, Edge_StringWithEmbeddedNullIsPreserved) {
  std::string input("abc\0def", 7);
  Value proto_value;
  proto_value.set_string_value(input);

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsString());
  EXPECT_EQ(result.GetString().ToString(), input);
}

TEST_F(ParsedJsonValueTest, Edge_LongStringReturnsStringValue) {
  std::string input(4096, 'x');
  Value proto_value;
  proto_value.set_string_value(input);

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsString());
  EXPECT_EQ(result.GetString().ToString(), input);
}

TEST_F(ParsedJsonValueTest, ECP_EmptyListReturnsListValue) {
  Value proto_value;
  proto_value.mutable_list_value();

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsList());
  EXPECT_EQ(result.GetList().Size(), 0);
}

TEST_F(ParsedJsonValueTest, ECP_ListWithPrimitiveValuesReturnsListValue) {
  Value proto_value;
  ListValue* list = proto_value.mutable_list_value();

  list->add_values()->set_null_value(google::protobuf::NULL_VALUE);
  list->add_values()->set_bool_value(true);
  list->add_values()->set_number_value(12.5);
  list->add_values()->set_string_value("abc");

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsList());
  EXPECT_EQ(result.GetList().Size(), 4);
}

TEST_F(ParsedJsonValueTest, BVA_ListWithSingleElementReturnsListValue) {
  Value proto_value;
  proto_value.mutable_list_value()->add_values()->set_string_value("only");

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsList());
  EXPECT_EQ(result.GetList().Size(), 1);
}

TEST_F(ParsedJsonValueTest, Edge_NestedListReturnsListValue) {
  Value proto_value;
  ListValue* outer_list = proto_value.mutable_list_value();
  ListValue* inner_list = outer_list->add_values()->mutable_list_value();
  inner_list->add_values()->set_number_value(1.0);
  inner_list->add_values()->set_number_value(2.0);

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsList());
  EXPECT_EQ(result.GetList().Size(), 1);
}

TEST_F(ParsedJsonValueTest, ECP_EmptyStructReturnsMapValue) {
  Value proto_value;
  proto_value.mutable_struct_value();

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsMap());
  EXPECT_EQ(result.GetMap().Size(), 0);
}

TEST_F(ParsedJsonValueTest, ECP_StructWithPrimitiveFieldsReturnsMapValue) {
  Value proto_value;
  Struct* object = proto_value.mutable_struct_value();

  (*object->mutable_fields())["null"].set_null_value(
      google::protobuf::NULL_VALUE);
  (*object->mutable_fields())["bool"].set_bool_value(true);
  (*object->mutable_fields())["number"].set_number_value(1.5);
  (*object->mutable_fields())["string"].set_string_value("value");

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsMap());
  EXPECT_EQ(result.GetMap().Size(), 4);
}

TEST_F(ParsedJsonValueTest, BVA_StructWithSingleFieldReturnsMapValue) {
  Value proto_value;
  (*proto_value.mutable_struct_value()->mutable_fields())["key"]
      .set_string_value("value");

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsMap());
  EXPECT_EQ(result.GetMap().Size(), 1);
}

TEST_F(ParsedJsonValueTest, Edge_NestedStructReturnsMapValue) {
  Value proto_value;
  Struct* object = proto_value.mutable_struct_value();

  Struct* nested = (*object->mutable_fields())["nested"].mutable_struct_value();
  (*nested->mutable_fields())["inner"].set_bool_value(false);

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsMap());
  EXPECT_EQ(result.GetMap().Size(), 1);
}

TEST_F(ParsedJsonValueTest, Edge_StructWithEmptyStringKeyReturnsMapValue) {
  Value proto_value;
  (*proto_value.mutable_struct_value()->mutable_fields())[""]
      .set_number_value(10.0);

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsMap());
  EXPECT_EQ(result.GetMap().Size(), 1);
}

TEST_F(ParsedJsonValueTest, Edge_StructWithListAndStructValuesReturnsMapValue) {
  Value proto_value;
  Struct* object = proto_value.mutable_struct_value();

  ListValue* list = (*object->mutable_fields())["list"].mutable_list_value();
  list->add_values()->set_string_value("a");
  list->add_values()->set_string_value("b");

  Struct* nested = (*object->mutable_fields())["object"].mutable_struct_value();
  (*nested->mutable_fields())["flag"].set_bool_value(true);

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsMap());
  EXPECT_EQ(result.GetMap().Size(), 2);
}

TEST_F(ParsedJsonValueTest, ECP_LastSetKindDeterminesReturnedValue) {
  Value proto_value;
  proto_value.set_bool_value(true);
  proto_value.set_number_value(99.0);
  proto_value.set_string_value("final");

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  ASSERT_TRUE(result.IsString());
  EXPECT_EQ(result.GetString().ToString(), "final");
}

TEST_F(ParsedJsonValueTest, Edge_ReusedProtoWithClearedKindReturnsNullValue) {
  Value proto_value;
  proto_value.set_string_value("temporary");
  proto_value.clear_kind();

  common::Value result = ParsedJsonValue(allocator_, Borrowed(proto_value));

  EXPECT_TRUE(result.IsNull());
}

}  // namespace
}  // namespace cel::common_internal