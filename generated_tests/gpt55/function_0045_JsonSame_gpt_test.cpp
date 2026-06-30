#include "tensorstore/internal/json/same.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace tensorstore {
namespace internal_json {
namespace {

TEST(JsonSameTest, NullValuesAreSame) {
  EXPECT_TRUE(JsonSame(nullptr, nullptr));
}

TEST(JsonSameTest, NullAndNonNullAreDifferent) {
  EXPECT_FALSE(JsonSame(nullptr, 0));
}

TEST(JsonSameTest, BooleanValuesSameAndDifferent) {
  EXPECT_TRUE(JsonSame(true, true));
  EXPECT_FALSE(JsonSame(true, false));
}

TEST(JsonSameTest, IntegerValuesSameAndDifferent) {
  EXPECT_TRUE(JsonSame(42, 42));
  EXPECT_FALSE(JsonSame(42, 43));
}

TEST(JsonSameTest, FloatingPointValuesSameAndDifferent) {
  EXPECT_TRUE(JsonSame(3.14, 3.14));
  EXPECT_FALSE(JsonSame(3.14, 2.71));
}

TEST(JsonSameTest, StringValuesSameAndDifferent) {
  EXPECT_TRUE(JsonSame("abc", "abc"));
  EXPECT_FALSE(JsonSame("abc", "abcd"));
}

TEST(JsonSameTest, DifferentPrimitiveTypesAreDifferent) {
  EXPECT_FALSE(JsonSame(1, "1"));
  EXPECT_FALSE(JsonSame(false, 0));
  EXPECT_FALSE(JsonSame("true", true));
}

TEST(JsonSameTest, EmptyArraysAreSame) {
  nlohmann::json a = nlohmann::json::array();
  nlohmann::json b = nlohmann::json::array();

  EXPECT_TRUE(JsonSame(a, b));
}

TEST(JsonSameTest, EmptyArrayAndEmptyObjectAreDifferent) {
  EXPECT_FALSE(JsonSame(nlohmann::json::array(), nlohmann::json::object()));
}

TEST(JsonSameTest, ArraysWithSameValuesAreSame) {
  nlohmann::json a = nlohmann::json::array({1, true, "x", nullptr});
  nlohmann::json b = nlohmann::json::array({1, true, "x", nullptr});

  EXPECT_TRUE(JsonSame(a, b));
}

TEST(JsonSameTest, ArraysWithDifferentSizesAreDifferent) {
  nlohmann::json a = nlohmann::json::array({1, 2});
  nlohmann::json b = nlohmann::json::array({1, 2, 3});

  EXPECT_FALSE(JsonSame(a, b));
}

TEST(JsonSameTest, ArraysWithDifferentElementValuesAreDifferent) {
  nlohmann::json a = nlohmann::json::array({1, 2, 3});
  nlohmann::json b = nlohmann::json::array({1, 9, 3});

  EXPECT_FALSE(JsonSame(a, b));
}

TEST(JsonSameTest, ArraysWithDifferentElementOrderAreDifferent) {
  nlohmann::json a = nlohmann::json::array({1, 2, 3});
  nlohmann::json b = nlohmann::json::array({3, 2, 1});

  EXPECT_FALSE(JsonSame(a, b));
}

TEST(JsonSameTest, EmptyObjectsAreSame) {
  nlohmann::json a = nlohmann::json::object();
  nlohmann::json b = nlohmann::json::object();

  EXPECT_TRUE(JsonSame(a, b));
}

TEST(JsonSameTest, ObjectsWithSameKeyValuesAreSame) {
  nlohmann::json a = {{"a", 1}, {"b", true}, {"c", "text"}};
  nlohmann::json b = {{"a", 1}, {"b", true}, {"c", "text"}};

  EXPECT_TRUE(JsonSame(a, b));
}

TEST(JsonSameTest, ObjectsWithDifferentSizesAreDifferent) {
  nlohmann::json a = {{"a", 1}};
  nlohmann::json b = {{"a", 1}, {"b", 2}};

  EXPECT_FALSE(JsonSame(a, b));
}

TEST(JsonSameTest, ObjectsWithDifferentKeysAreDifferent) {
  nlohmann::json a = {{"a", 1}};
  nlohmann::json b = {{"b", 1}};

  EXPECT_FALSE(JsonSame(a, b));
}

TEST(JsonSameTest, ObjectsWithDifferentValuesAreDifferent) {
  nlohmann::json a = {{"a", 1}, {"b", 2}};
  nlohmann::json b = {{"a", 1}, {"b", 3}};

  EXPECT_FALSE(JsonSame(a, b));
}

TEST(JsonSameTest, NestedArraysAreSame) {
  nlohmann::json a = nlohmann::json::array(
      {1, nlohmann::json::array({2, 3}), nlohmann::json::array({4})});
  nlohmann::json b = nlohmann::json::array(
      {1, nlohmann::json::array({2, 3}), nlohmann::json::array({4})});

  EXPECT_TRUE(JsonSame(a, b));
}

TEST(JsonSameTest, NestedArraysWithDifferentInnerValueAreDifferent) {
  nlohmann::json a = nlohmann::json::array(
      {1, nlohmann::json::array({2, 3})});
  nlohmann::json b = nlohmann::json::array(
      {1, nlohmann::json::array({2, 4})});

  EXPECT_FALSE(JsonSame(a, b));
}

TEST(JsonSameTest, NestedObjectsAreSame) {
  nlohmann::json a = {
      {"outer", {{"inner", 10}, {"flag", true}}},
      {"list", nlohmann::json::array({1, 2, 3})}};
  nlohmann::json b = {
      {"outer", {{"inner", 10}, {"flag", true}}},
      {"list", nlohmann::json::array({1, 2, 3})}};

  EXPECT_TRUE(JsonSame(a, b));
}

TEST(JsonSameTest, NestedObjectsWithDifferentInnerValueAreDifferent) {
  nlohmann::json a = {{"outer", {{"inner", 10}}}};
  nlohmann::json b = {{"outer", {{"inner", 11}}}};

  EXPECT_FALSE(JsonSame(a, b));
}

TEST(JsonSameTest, ArrayContainingObjectsAreSame) {
  nlohmann::json a = nlohmann::json::array(
      {{{"id", 1}, {"name", "one"}}, {{"id", 2}, {"name", "two"}}});
  nlohmann::json b = nlohmann::json::array(
      {{{"id", 1}, {"name", "one"}}, {{"id", 2}, {"name", "two"}}});

  EXPECT_TRUE(JsonSame(a, b));
}

TEST(JsonSameTest, ArrayContainingObjectsWithDifferentValueAreDifferent) {
  nlohmann::json a = nlohmann::json::array({{{"id", 1}}, {{"id", 2}}});
  nlohmann::json b = nlohmann::json::array({{{"id", 1}}, {{"id", 3}}});

  EXPECT_FALSE(JsonSame(a, b));
}

TEST(JsonSameTest, DeeplyNestedArrayBeyondInlineStackCapacityIsSame) {
  nlohmann::json a = 0;
  nlohmann::json b = 0;

  for (int i = 0; i < 100; ++i) {
    a = nlohmann::json::array({a});
    b = nlohmann::json::array({b});
  }

  EXPECT_TRUE(JsonSame(a, b));
}

TEST(JsonSameTest, DeeplyNestedArrayBeyondInlineStackCapacityDifferentLeaf) {
  nlohmann::json a = 0;
  nlohmann::json b = 1;

  for (int i = 0; i < 100; ++i) {
    a = nlohmann::json::array({a});
    b = nlohmann::json::array({b});
  }

  EXPECT_FALSE(JsonSame(a, b));
}

TEST(JsonSameTest, DeeplyNestedObjectBeyondInlineStackCapacityIsSame) {
  nlohmann::json a = 0;
  nlohmann::json b = 0;

  for (int i = 0; i < 100; ++i) {
    a = nlohmann::json{{"k", a}};
    b = nlohmann::json{{"k", b}};
  }

  EXPECT_TRUE(JsonSame(a, b));
}

TEST(JsonSameTest, DeeplyNestedObjectBeyondInlineStackCapacityDifferentLeaf) {
  nlohmann::json a = 0;
  nlohmann::json b = 1;

  for (int i = 0; i < 100; ++i) {
    a = nlohmann::json{{"k", a}};
    b = nlohmann::json{{"k", b}};
  }

  EXPECT_FALSE(JsonSame(a, b));
}

TEST(JsonSameTest, ComplexMixedJsonSame) {
  nlohmann::json a = {
      {"name", "tensorstore"},
      {"version", 1},
      {"features",
       nlohmann::json::array(
           {nlohmann::json{{"name", "json"}, {"enabled", true}},
            nlohmann::json{{"name", "kvstore"}, {"enabled", false}}})},
      {"metadata", {{"nullable", nullptr}, {"count", 2}}}};

  nlohmann::json b = {
      {"name", "tensorstore"},
      {"version", 1},
      {"features",
       nlohmann::json::array(
           {nlohmann::json{{"name", "json"}, {"enabled", true}},
            nlohmann::json{{"name", "kvstore"}, {"enabled", false}}})},
      {"metadata", {{"nullable", nullptr}, {"count", 2}}}};

  EXPECT_TRUE(JsonSame(a, b));
}

TEST(JsonSameTest, ComplexMixedJsonDifferent) {
  nlohmann::json a = {
      {"name", "tensorstore"},
      {"features",
       nlohmann::json::array(
           {nlohmann::json{{"name", "json"}, {"enabled", true}}})}};

  nlohmann::json b = {
      {"name", "tensorstore"},
      {"features",
       nlohmann::json::array(
           {nlohmann::json{{"name", "json"}, {"enabled", false}}})}};

  EXPECT_FALSE(JsonSame(a, b));
}

}  // namespace
}  // namespace internal_json
}  // namespace tensorstore