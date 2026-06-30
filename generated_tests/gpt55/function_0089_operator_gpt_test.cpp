#include "eval/public/cel_attribute.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/variant.h"
#include "eval/public/cel_value.h"

namespace google::api::expr::runtime {
namespace {

TEST(CreateCelAttributeQualifierPatternTest, CreatesInt64Pattern) {
  CelValue value = CelValue::CreateInt64(123);

  CelAttributeQualifierPattern result =
      CreateCelAttributeQualifierPattern(value);

  EXPECT_EQ(result, CelAttributeQualifierPattern::OfInt(123));
}

TEST(CreateCelAttributeQualifierPatternTest, CreatesInt64MinPattern) {
  CelValue value = CelValue::CreateInt64(std::numeric_limits<int64_t>::min());

  CelAttributeQualifierPattern result =
      CreateCelAttributeQualifierPattern(value);

  EXPECT_EQ(result,
            CelAttributeQualifierPattern::OfInt(
                std::numeric_limits<int64_t>::min()));
}

TEST(CreateCelAttributeQualifierPatternTest, CreatesInt64MaxPattern) {
  CelValue value = CelValue::CreateInt64(std::numeric_limits<int64_t>::max());

  CelAttributeQualifierPattern result =
      CreateCelAttributeQualifierPattern(value);

  EXPECT_EQ(result,
            CelAttributeQualifierPattern::OfInt(
                std::numeric_limits<int64_t>::max()));
}

TEST(CreateCelAttributeQualifierPatternTest, CreatesUint64Pattern) {
  CelValue value = CelValue::CreateUint64(123u);

  CelAttributeQualifierPattern result =
      CreateCelAttributeQualifierPattern(value);

  EXPECT_EQ(result, CelAttributeQualifierPattern::OfUint(123u));
}

TEST(CreateCelAttributeQualifierPatternTest, CreatesUint64ZeroPattern) {
  CelValue value = CelValue::CreateUint64(0u);

  CelAttributeQualifierPattern result =
      CreateCelAttributeQualifierPattern(value);

  EXPECT_EQ(result, CelAttributeQualifierPattern::OfUint(0u));
}

TEST(CreateCelAttributeQualifierPatternTest, CreatesUint64MaxPattern) {
  CelValue value = CelValue::CreateUint64(std::numeric_limits<uint64_t>::max());

  CelAttributeQualifierPattern result =
      CreateCelAttributeQualifierPattern(value);

  EXPECT_EQ(result,
            CelAttributeQualifierPattern::OfUint(
                std::numeric_limits<uint64_t>::max()));
}

TEST(CreateCelAttributeQualifierPatternTest, CreatesStringPattern) {
  CelValue value = CelValue::CreateStringView("field");

  CelAttributeQualifierPattern result =
      CreateCelAttributeQualifierPattern(value);

  EXPECT_EQ(result, CelAttributeQualifierPattern::OfString("field"));
}

TEST(CreateCelAttributeQualifierPatternTest, CreatesEmptyStringPattern) {
  CelValue value = CelValue::CreateStringView("");

  CelAttributeQualifierPattern result =
      CreateCelAttributeQualifierPattern(value);

  EXPECT_EQ(result, CelAttributeQualifierPattern::OfString(""));
}

TEST(CreateCelAttributeQualifierPatternTest, StarStringFromCelValueIsLiteralString) {
  CelValue value = CelValue::CreateStringView("*");

  CelAttributeQualifierPattern result =
      CreateCelAttributeQualifierPattern(value);

  EXPECT_EQ(result, CelAttributeQualifierPattern::OfString("*"));
}

TEST(CreateCelAttributeQualifierPatternTest, CreatesBoolTruePattern) {
  CelValue value = CelValue::CreateBool(true);

  CelAttributeQualifierPattern result =
      CreateCelAttributeQualifierPattern(value);

  EXPECT_EQ(result, CelAttributeQualifierPattern::OfBool(true));
}

TEST(CreateCelAttributeQualifierPatternTest, CreatesBoolFalsePattern) {
  CelValue value = CelValue::CreateBool(false);

  CelAttributeQualifierPattern result =
      CreateCelAttributeQualifierPattern(value);

  EXPECT_EQ(result, CelAttributeQualifierPattern::OfBool(false));
}

TEST(CreateCelAttributeQualifierPatternTest, UnsupportedNullValueCreatesDefaultPattern) {
  CelValue value = CelValue::CreateNull();

  CelAttributeQualifierPattern result =
      CreateCelAttributeQualifierPattern(value);

  EXPECT_EQ(result,
            CelAttributeQualifierPattern(CelAttributeQualifier()));
}

TEST(CreateCelAttributeQualifierTest, CreatesInt64Qualifier) {
  CelValue value = CelValue::CreateInt64(-10);

  CelAttributeQualifier result = CreateCelAttributeQualifier(value);

  EXPECT_EQ(result, CelAttributeQualifier::OfInt(-10));
}

TEST(CreateCelAttributeQualifierTest, CreatesInt64ZeroQualifier) {
  CelValue value = CelValue::CreateInt64(0);

  CelAttributeQualifier result = CreateCelAttributeQualifier(value);

  EXPECT_EQ(result, CelAttributeQualifier::OfInt(0));
}

TEST(CreateCelAttributeQualifierTest, CreatesInt64MinQualifier) {
  CelValue value = CelValue::CreateInt64(std::numeric_limits<int64_t>::min());

  CelAttributeQualifier result = CreateCelAttributeQualifier(value);

  EXPECT_EQ(result,
            CelAttributeQualifier::OfInt(std::numeric_limits<int64_t>::min()));
}

TEST(CreateCelAttributeQualifierTest, CreatesInt64MaxQualifier) {
  CelValue value = CelValue::CreateInt64(std::numeric_limits<int64_t>::max());

  CelAttributeQualifier result = CreateCelAttributeQualifier(value);

  EXPECT_EQ(result,
            CelAttributeQualifier::OfInt(std::numeric_limits<int64_t>::max()));
}

TEST(CreateCelAttributeQualifierTest, CreatesUint64Qualifier) {
  CelValue value = CelValue::CreateUint64(10u);

  CelAttributeQualifier result = CreateCelAttributeQualifier(value);

  EXPECT_EQ(result, CelAttributeQualifier::OfUint(10u));
}

TEST(CreateCelAttributeQualifierTest, CreatesUint64ZeroQualifier) {
  CelValue value = CelValue::CreateUint64(0u);

  CelAttributeQualifier result = CreateCelAttributeQualifier(value);

  EXPECT_EQ(result, CelAttributeQualifier::OfUint(0u));
}

TEST(CreateCelAttributeQualifierTest, CreatesUint64MaxQualifier) {
  CelValue value = CelValue::CreateUint64(std::numeric_limits<uint64_t>::max());

  CelAttributeQualifier result = CreateCelAttributeQualifier(value);

  EXPECT_EQ(result,
            CelAttributeQualifier::OfUint(
                std::numeric_limits<uint64_t>::max()));
}

TEST(CreateCelAttributeQualifierTest, CreatesStringQualifier) {
  CelValue value = CelValue::CreateStringView("name");

  CelAttributeQualifier result = CreateCelAttributeQualifier(value);

  EXPECT_EQ(result, CelAttributeQualifier::OfString("name"));
}

TEST(CreateCelAttributeQualifierTest, CreatesEmptyStringQualifier) {
  CelValue value = CelValue::CreateStringView("");

  CelAttributeQualifier result = CreateCelAttributeQualifier(value);

  EXPECT_EQ(result, CelAttributeQualifier::OfString(""));
}

TEST(CreateCelAttributeQualifierTest, CreatesStarStringAsLiteralQualifier) {
  CelValue value = CelValue::CreateStringView("*");

  CelAttributeQualifier result = CreateCelAttributeQualifier(value);

  EXPECT_EQ(result, CelAttributeQualifier::OfString("*"));
}

TEST(CreateCelAttributeQualifierTest, CreatesBoolTrueQualifier) {
  CelValue value = CelValue::CreateBool(true);

  CelAttributeQualifier result = CreateCelAttributeQualifier(value);

  EXPECT_EQ(result, CelAttributeQualifier::OfBool(true));
}

TEST(CreateCelAttributeQualifierTest, CreatesBoolFalseQualifier) {
  CelValue value = CelValue::CreateBool(false);

  CelAttributeQualifier result = CreateCelAttributeQualifier(value);

  EXPECT_EQ(result, CelAttributeQualifier::OfBool(false));
}

TEST(CreateCelAttributeQualifierTest, UnsupportedNullValueCreatesDefaultQualifier) {
  CelValue value = CelValue::CreateNull();

  CelAttributeQualifier result = CreateCelAttributeQualifier(value);

  EXPECT_EQ(result, CelAttributeQualifier());
}

TEST(CreateCelAttributePatternTest, EmptyPathCreatesVariableOnlyPattern) {
  CelAttributePattern result = CreateCelAttributePattern("request", {});

  CelAttributePattern expected("request", {});

  EXPECT_EQ(result, expected);
}

TEST(CreateCelAttributePatternTest, CreatesStringPathPattern) {
  CelAttributePattern result =
      CreateCelAttributePattern("request", {"auth", "claims"});

  CelAttributePattern expected(
      "request",
      {CelAttributeQualifierPattern::OfString("auth"),
       CelAttributeQualifierPattern::OfString("claims")});

  EXPECT_EQ(result, expected);
}

TEST(CreateCelAttributePatternTest, StarStringInPathCreatesWildcardPattern) {
  CelAttributePattern result =
      CreateCelAttributePattern("request", {"auth", "*"});

  CelAttributePattern expected(
      "request",
      {CelAttributeQualifierPattern::OfString("auth"),
       CelAttributeQualifierPattern::CreateWildcard()});

  EXPECT_EQ(result, expected);
}

TEST(CreateCelAttributePatternTest, EmptyStringPathElementIsLiteralString) {
  CelAttributePattern result =
      CreateCelAttributePattern("request", {""});

  CelAttributePattern expected(
      "request", {CelAttributeQualifierPattern::OfString("")});

  EXPECT_EQ(result, expected);
}

TEST(CreateCelAttributePatternTest, CreatesInt64PathPattern) {
  CelAttributePattern result =
      CreateCelAttributePattern("items", {int64_t{0}, int64_t{-1}});

  CelAttributePattern expected(
      "items",
      {CelAttributeQualifierPattern::OfInt(0),
       CelAttributeQualifierPattern::OfInt(-1)});

  EXPECT_EQ(result, expected);
}

TEST(CreateCelAttributePatternTest, CreatesInt64BoundaryPathPattern) {
  CelAttributePattern result = CreateCelAttributePattern(
      "items", {std::numeric_limits<int64_t>::min(),
                std::numeric_limits<int64_t>::max()});

  CelAttributePattern expected(
      "items",
      {CelAttributeQualifierPattern::OfInt(std::numeric_limits<int64_t>::min()),
       CelAttributeQualifierPattern::OfInt(std::numeric_limits<int64_t>::max())});

  EXPECT_EQ(result, expected);
}

TEST(CreateCelAttributePatternTest, CreatesUint64PathPattern) {
  CelAttributePattern result =
      CreateCelAttributePattern("items", {uint64_t{0}, uint64_t{10}});

  CelAttributePattern expected(
      "items",
      {CelAttributeQualifierPattern::OfUint(0),
       CelAttributeQualifierPattern::OfUint(10)});

  EXPECT_EQ(result, expected);
}

TEST(CreateCelAttributePatternTest, CreatesUint64MaxPathPattern) {
  CelAttributePattern result =
      CreateCelAttributePattern("items",
                                {std::numeric_limits<uint64_t>::max()});

  CelAttributePattern expected(
      "items",
      {CelAttributeQualifierPattern::OfUint(
          std::numeric_limits<uint64_t>::max())});

  EXPECT_EQ(result, expected);
}

TEST(CreateCelAttributePatternTest, CreatesBoolPathPattern) {
  CelAttributePattern result =
      CreateCelAttributePattern("flags", {true, false});

  CelAttributePattern expected(
      "flags",
      {CelAttributeQualifierPattern::OfBool(true),
       CelAttributeQualifierPattern::OfBool(false)});

  EXPECT_EQ(result, expected);
}

TEST(CreateCelAttributePatternTest, PreservesProvidedQualifierPattern) {
  CelAttributeQualifierPattern wildcard =
      CelAttributeQualifierPattern::CreateWildcard();

  CelAttributePattern result =
      CreateCelAttributePattern("request", {wildcard});

  CelAttributePattern expected("request", {wildcard});

  EXPECT_EQ(result, expected);
}

TEST(CreateCelAttributePatternTest, MixedPathElementsCreateExpectedPattern) {
  CelAttributePattern result =
      CreateCelAttributePattern("request",
                                {"auth", "*", int64_t{-1}, uint64_t{2}, true});

  CelAttributePattern expected(
      "request",
      {CelAttributeQualifierPattern::OfString("auth"),
       CelAttributeQualifierPattern::CreateWildcard(),
       CelAttributeQualifierPattern::OfInt(-1),
       CelAttributeQualifierPattern::OfUint(2),
       CelAttributeQualifierPattern::OfBool(true)});

  EXPECT_EQ(result, expected);
}

TEST(CreateCelAttributePatternTest, EmptyVariableNameIsAllowed) {
  CelAttributePattern result =
      CreateCelAttributePattern("", {"field"});

  CelAttributePattern expected(
      "", {CelAttributeQualifierPattern::OfString("field")});

  EXPECT_EQ(result, expected);
}

TEST(CreateCelAttributePatternTest, LongPathCreatesExpectedPattern) {
  CelAttributePattern result = CreateCelAttributePattern(
      "root", {"a", "b", "c", "d", "e", "f", "g", "h"});

  CelAttributePattern expected(
      "root",
      {CelAttributeQualifierPattern::OfString("a"),
       CelAttributeQualifierPattern::OfString("b"),
       CelAttributeQualifierPattern::OfString("c"),
       CelAttributeQualifierPattern::OfString("d"),
       CelAttributeQualifierPattern::OfString("e"),
       CelAttributeQualifierPattern::OfString("f"),
       CelAttributeQualifierPattern::OfString("g"),
       CelAttributeQualifierPattern::OfString("h")});

  EXPECT_EQ(result, expected);
}

}  // namespace
}  // namespace google::api::expr::runtime