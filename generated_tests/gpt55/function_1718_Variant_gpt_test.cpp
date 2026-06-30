#include "tensorflow/lite/core/async/interop/variant.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>

namespace tflite {
namespace interop {
namespace {

Variant MakeInvalid() {
  Variant v;
  v.type = kInvalid;
  v.val.i = 0;
  return v;
}

Variant MakeInt(int value) {
  Variant v;
  v.type = kInt;
  v.val.i = value;
  return v;
}

Variant MakeSizeT(size_t value) {
  Variant v;
  v.type = kSizeT;
  v.val.s = value;
  return v;
}

Variant MakeString(const char* value) {
  Variant v;
  v.type = kString;
  v.val.c = value;
  return v;
}

Variant MakeBool(bool value) {
  Variant v;
  v.type = kBool;
  v.val.b = value;
  return v;
}

TEST(VariantTest, DefaultConstructorCreatesInvalidVariantWithZeroValue) {
  Variant v;

  EXPECT_EQ(v.type, kInvalid);
  EXPECT_EQ(v.val.i, 0);
}

TEST(VariantTest, ECP_InvalidVariantsAreAlwaysEqual) {
  Variant lhs = MakeInvalid();
  Variant rhs = MakeInvalid();

  lhs.val.i = 10;
  rhs.val.i = -20;

  EXPECT_TRUE(lhs == rhs);
  EXPECT_FALSE(lhs != rhs);
}

TEST(VariantTest, ECP_DifferentTypesAreNotEqual) {
  EXPECT_FALSE(MakeInt(0) == MakeSizeT(0));
  EXPECT_FALSE(MakeInt(1) == MakeBool(true));
  EXPECT_FALSE(MakeString("1") == MakeInt(1));
  EXPECT_TRUE(MakeString("1") != MakeInt(1));
}

TEST(VariantTest, ECP_IntVariantsWithSameValueAreEqual) {
  Variant lhs = MakeInt(123);
  Variant rhs = MakeInt(123);

  EXPECT_TRUE(lhs == rhs);
  EXPECT_FALSE(lhs != rhs);
}

TEST(VariantTest, ECP_IntVariantsWithDifferentValuesAreNotEqual) {
  Variant lhs = MakeInt(123);
  Variant rhs = MakeInt(456);

  EXPECT_FALSE(lhs == rhs);
  EXPECT_TRUE(lhs != rhs);
}

TEST(VariantTest, BVA_IntMinimumValueComparesEqual) {
  Variant lhs = MakeInt(std::numeric_limits<int>::min());
  Variant rhs = MakeInt(std::numeric_limits<int>::min());

  EXPECT_TRUE(lhs == rhs);
}

TEST(VariantTest, BVA_IntMaximumValueComparesEqual) {
  Variant lhs = MakeInt(std::numeric_limits<int>::max());
  Variant rhs = MakeInt(std::numeric_limits<int>::max());

  EXPECT_TRUE(lhs == rhs);
}

TEST(VariantTest, BVA_IntMinAndMaxAreNotEqual) {
  Variant lhs = MakeInt(std::numeric_limits<int>::min());
  Variant rhs = MakeInt(std::numeric_limits<int>::max());

  EXPECT_FALSE(lhs == rhs);
  EXPECT_TRUE(lhs != rhs);
}

TEST(VariantTest, ECP_SizeTVariantsWithSameValueAreEqual) {
  Variant lhs = MakeSizeT(42u);
  Variant rhs = MakeSizeT(42u);

  EXPECT_TRUE(lhs == rhs);
  EXPECT_FALSE(lhs != rhs);
}

TEST(VariantTest, ECP_SizeTVariantsWithDifferentValuesAreNotEqual) {
  Variant lhs = MakeSizeT(42u);
  Variant rhs = MakeSizeT(43u);

  EXPECT_FALSE(lhs == rhs);
  EXPECT_TRUE(lhs != rhs);
}

TEST(VariantTest, BVA_SizeTZeroComparesEqual) {
  Variant lhs = MakeSizeT(0u);
  Variant rhs = MakeSizeT(0u);

  EXPECT_TRUE(lhs == rhs);
}

TEST(VariantTest, BVA_SizeTMaximumComparesEqual) {
  Variant lhs = MakeSizeT(std::numeric_limits<size_t>::max());
  Variant rhs = MakeSizeT(std::numeric_limits<size_t>::max());

  EXPECT_TRUE(lhs == rhs);
}

TEST(VariantTest, BVA_SizeTZeroAndMaximumAreNotEqual) {
  Variant lhs = MakeSizeT(0u);
  Variant rhs = MakeSizeT(std::numeric_limits<size_t>::max());

  EXPECT_FALSE(lhs == rhs);
  EXPECT_TRUE(lhs != rhs);
}

TEST(VariantTest, ECP_BoolTrueVariantsAreEqual) {
  Variant lhs = MakeBool(true);
  Variant rhs = MakeBool(true);

  EXPECT_TRUE(lhs == rhs);
  EXPECT_FALSE(lhs != rhs);
}

TEST(VariantTest, ECP_BoolFalseVariantsAreEqual) {
  Variant lhs = MakeBool(false);
  Variant rhs = MakeBool(false);

  EXPECT_TRUE(lhs == rhs);
  EXPECT_FALSE(lhs != rhs);
}

TEST(VariantTest, ECP_BoolTrueAndFalseAreNotEqual) {
  Variant lhs = MakeBool(true);
  Variant rhs = MakeBool(false);

  EXPECT_FALSE(lhs == rhs);
  EXPECT_TRUE(lhs != rhs);
}

TEST(VariantTest, ECP_StringSamePointerIsEqual) {
  const char* text = "same";

  Variant lhs = MakeString(text);
  Variant rhs = MakeString(text);

  EXPECT_TRUE(lhs == rhs);
  EXPECT_FALSE(lhs != rhs);
}

TEST(VariantTest, ECP_StringDifferentPointersSameContentAreEqual) {
  const char lhs_text[] = "hello";
  const char rhs_text[] = "hello";

  Variant lhs = MakeString(lhs_text);
  Variant rhs = MakeString(rhs_text);

  ASSERT_NE(lhs.val.c, rhs.val.c);
  EXPECT_TRUE(lhs == rhs);
  EXPECT_FALSE(lhs != rhs);
}

TEST(VariantTest, ECP_StringDifferentContentIsNotEqual) {
  const char lhs_text[] = "hello";
  const char rhs_text[] = "world";

  Variant lhs = MakeString(lhs_text);
  Variant rhs = MakeString(rhs_text);

  EXPECT_FALSE(lhs == rhs);
  EXPECT_TRUE(lhs != rhs);
}

TEST(VariantTest, BVA_EmptyStringsDifferentPointersAreEqual) {
  const char lhs_text[] = "";
  const char rhs_text[] = "";

  Variant lhs = MakeString(lhs_text);
  Variant rhs = MakeString(rhs_text);

  ASSERT_NE(lhs.val.c, rhs.val.c);
  EXPECT_TRUE(lhs == rhs);
}

TEST(VariantTest, BVA_EmptyStringAndNonEmptyStringAreNotEqual) {
  Variant lhs = MakeString("");
  Variant rhs = MakeString("a");

  EXPECT_FALSE(lhs == rhs);
  EXPECT_TRUE(lhs != rhs);
}

TEST(VariantTest, Edge_StringCaseSensitiveComparison) {
  Variant lhs = MakeString("Test");
  Variant rhs = MakeString("test");

  EXPECT_FALSE(lhs == rhs);
  EXPECT_TRUE(lhs != rhs);
}

TEST(VariantTest, Edge_StringWithWhitespaceComparedExactly) {
  Variant lhs = MakeString("abc");
  Variant rhs = MakeString(" abc ");

  EXPECT_FALSE(lhs == rhs);
  EXPECT_TRUE(lhs != rhs);
}

TEST(VariantTest, Edge_StringStopsAtNullTerminator) {
  const char lhs_text[] = {'a', 'b', 'c', '\0', 'x', '\0'};
  const char rhs_text[] = {'a', 'b', 'c', '\0', 'y', '\0'};

  Variant lhs = MakeString(lhs_text);
  Variant rhs = MakeString(rhs_text);

  EXPECT_TRUE(lhs == rhs);
  EXPECT_FALSE(lhs != rhs);
}

TEST(VariantTest, Edge_SelfComparisonIsEqualForEveryValidType) {
  Variant invalid = MakeInvalid();
  Variant int_value = MakeInt(-1);
  Variant size_value = MakeSizeT(1u);
  Variant string_value = MakeString("self");
  Variant bool_value = MakeBool(true);

  EXPECT_TRUE(invalid == invalid);
  EXPECT_TRUE(int_value == int_value);
  EXPECT_TRUE(size_value == size_value);
  EXPECT_TRUE(string_value == string_value);
  EXPECT_TRUE(bool_value == bool_value);
}

}  // namespace
}  // namespace interop
}  // namespace tflite