#include "tensorstore/internal/string_like.h"

#include <gtest/gtest.h>

#include <array>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {
namespace {

TEST(IsStringLikeTest, RecognizesSupportedStringLikeTypes) {
  EXPECT_TRUE(IsStringLike<std::string>);
  EXPECT_TRUE(IsStringLike<std::string_view>);
  EXPECT_TRUE(IsStringLike<const char*>);
}

TEST(IsStringLikeTest, RejectsUnsupportedTypes) {
  EXPECT_FALSE(IsStringLike<char*>);
  EXPECT_FALSE(IsStringLike<char>);
  EXPECT_FALSE(IsStringLike<int>);
  EXPECT_FALSE(IsStringLike<const std::string>);
  EXPECT_FALSE(IsStringLike<const std::string_view>);
}

TEST(StringLikeSpanTest, DefaultConstructedSpanHasZeroSize) {
  StringLikeSpan span;

  EXPECT_EQ(span.size(), 0);
}

TEST(StringLikeSpanTest, EmptyCStringSpanHasZeroSize) {
  std::array<const char*, 0> values{};

  StringLikeSpan span(tensorstore::span<const char* const>(values));

  EXPECT_EQ(span.size(), 0);
}

TEST(StringLikeSpanTest, EmptyStringSpanHasZeroSize) {
  std::array<std::string, 0> values{};

  StringLikeSpan span(tensorstore::span<const std::string>(values));

  EXPECT_EQ(span.size(), 0);
}

TEST(StringLikeSpanTest, EmptyStringViewSpanHasZeroSize) {
  std::array<std::string_view, 0> values{};

  StringLikeSpan span(tensorstore::span<const std::string_view>(values));

  EXPECT_EQ(span.size(), 0);
}

TEST(StringLikeSpanTest, SingleCStringElementIsAccessible) {
  std::array<const char*, 1> values = {"alpha"};

  StringLikeSpan span(tensorstore::span<const char* const>(values));

  ASSERT_EQ(span.size(), 1);
  EXPECT_EQ(span[0], "alpha");
}

TEST(StringLikeSpanTest, SingleStringElementIsAccessible) {
  std::array<std::string, 1> values = {std::string("alpha")};

  StringLikeSpan span(tensorstore::span<const std::string>(values));

  ASSERT_EQ(span.size(), 1);
  EXPECT_EQ(span[0], "alpha");
}

TEST(StringLikeSpanTest, SingleStringViewElementIsAccessible) {
  std::array<std::string_view, 1> values = {std::string_view("alpha")};

  StringLikeSpan span(tensorstore::span<const std::string_view>(values));

  ASSERT_EQ(span.size(), 1);
  EXPECT_EQ(span[0], "alpha");
}

TEST(StringLikeSpanTest, MultipleCStringElementsPreserveOrder) {
  std::array<const char*, 3> values = {"alpha", "beta", "gamma"};

  StringLikeSpan span(tensorstore::span<const char* const>(values));

  ASSERT_EQ(span.size(), 3);
  EXPECT_EQ(span[0], "alpha");
  EXPECT_EQ(span[1], "beta");
  EXPECT_EQ(span[2], "gamma");
}

TEST(StringLikeSpanTest, MultipleStringElementsPreserveOrder) {
  std::array<std::string, 3> values = {
      std::string("alpha"),
      std::string("beta"),
      std::string("gamma"),
  };

  StringLikeSpan span(tensorstore::span<const std::string>(values));

  ASSERT_EQ(span.size(), 3);
  EXPECT_EQ(span[0], "alpha");
  EXPECT_EQ(span[1], "beta");
  EXPECT_EQ(span[2], "gamma");
}

TEST(StringLikeSpanTest, MultipleStringViewElementsPreserveOrder) {
  std::array<std::string_view, 3> values = {
      std::string_view("alpha"),
      std::string_view("beta"),
      std::string_view("gamma"),
  };

  StringLikeSpan span(tensorstore::span<const std::string_view>(values));

  ASSERT_EQ(span.size(), 3);
  EXPECT_EQ(span[0], "alpha");
  EXPECT_EQ(span[1], "beta");
  EXPECT_EQ(span[2], "gamma");
}

TEST(StringLikeSpanTest, CStringSpanSupportsEmptyStringElement) {
  std::array<const char*, 3> values = {"", "non_empty", ""};

  StringLikeSpan span(tensorstore::span<const char* const>(values));

  ASSERT_EQ(span.size(), 3);
  EXPECT_EQ(span[0], "");
  EXPECT_EQ(span[1], "non_empty");
  EXPECT_EQ(span[2], "");
}

TEST(StringLikeSpanTest, StringSpanSupportsEmptyStringElement) {
  std::array<std::string, 3> values = {
      std::string(""),
      std::string("non_empty"),
      std::string(""),
  };

  StringLikeSpan span(tensorstore::span<const std::string>(values));

  ASSERT_EQ(span.size(), 3);
  EXPECT_EQ(span[0], "");
  EXPECT_EQ(span[1], "non_empty");
  EXPECT_EQ(span[2], "");
}

TEST(StringLikeSpanTest, StringViewSpanSupportsEmptyStringElement) {
  std::array<std::string_view, 3> values = {
      std::string_view(""),
      std::string_view("non_empty"),
      std::string_view(""),
  };

  StringLikeSpan span(tensorstore::span<const std::string_view>(values));

  ASSERT_EQ(span.size(), 3);
  EXPECT_EQ(span[0], "");
  EXPECT_EQ(span[1], "non_empty");
  EXPECT_EQ(span[2], "");
}

TEST(StringLikeSpanTest, CStringSpanSupportsSpecialCharacters) {
  std::array<const char*, 3> values = {
      "a b c",
      "path/to/file.txt",
      "symbols_!@#$%^&*()",
  };

  StringLikeSpan span(tensorstore::span<const char* const>(values));

  ASSERT_EQ(span.size(), 3);
  EXPECT_EQ(span[0], "a b c");
  EXPECT_EQ(span[1], "path/to/file.txt");
  EXPECT_EQ(span[2], "symbols_!@#$%^&*()");
}

TEST(StringLikeSpanTest, StringSpanSupportsSpecialCharacters) {
  std::array<std::string, 3> values = {
      std::string("a b c"),
      std::string("path/to/file.txt"),
      std::string("symbols_!@#$%^&*()"),
  };

  StringLikeSpan span(tensorstore::span<const std::string>(values));

  ASSERT_EQ(span.size(), 3);
  EXPECT_EQ(span[0], "a b c");
  EXPECT_EQ(span[1], "path/to/file.txt");
  EXPECT_EQ(span[2], "symbols_!@#$%^&*()");
}

TEST(StringLikeSpanTest, StringViewSpanSupportsSpecialCharacters) {
  std::array<std::string_view, 3> values = {
      std::string_view("a b c"),
      std::string_view("path/to/file.txt"),
      std::string_view("symbols_!@#$%^&*()"),
  };

  StringLikeSpan span(tensorstore::span<const std::string_view>(values));

  ASSERT_EQ(span.size(), 3);
  EXPECT_EQ(span[0], "a b c");
  EXPECT_EQ(span[1], "path/to/file.txt");
  EXPECT_EQ(span[2], "symbols_!@#$%^&*()");
}

TEST(StringLikeSpanTest, StringViewSpanPreservesEmbeddedNullCharacters) {
  const char value[] = {'a', '\0', 'b'};

  std::array<std::string_view, 1> values = {
      std::string_view(value, sizeof(value)),
  };

  StringLikeSpan span(tensorstore::span<const std::string_view>(values));

  ASSERT_EQ(span.size(), 1);
  EXPECT_EQ(span[0].size(), 3);
  EXPECT_EQ(span[0], std::string_view(value, sizeof(value)));
}

TEST(StringLikeSpanTest, StringSpanPreservesEmbeddedNullCharacters) {
  const std::string value("a\0b", 3);
  std::array<std::string, 1> values = {value};

  StringLikeSpan span(tensorstore::span<const std::string>(values));

  ASSERT_EQ(span.size(), 1);
  EXPECT_EQ(span[0].size(), 3);
  EXPECT_EQ(span[0], std::string_view(value));
}

TEST(StringLikeSpanTest, CStringSpanStopsAtFirstNullCharacter) {
  const char value[] = {'a', '\0', 'b', '\0'};
  std::array<const char*, 1> values = {value};

  StringLikeSpan span(tensorstore::span<const char* const>(values));

  ASSERT_EQ(span.size(), 1);
  EXPECT_EQ(span[0], "a");
  EXPECT_EQ(span[0].size(), 1);
}

TEST(StringLikeSpanTest, LargeCStringSpanReportsCorrectSizeAndBoundaryElements) {
  std::vector<const char*> values;
  values.reserve(1024);
  for (int i = 0; i < 1024; ++i) {
    values.push_back(i == 0 ? "first" : (i == 1023 ? "last" : "middle"));
  }

  StringLikeSpan span(tensorstore::span<const char* const>(values.data(),
                                                           values.size()));

  ASSERT_EQ(span.size(), 1024);
  EXPECT_EQ(span[0], "first");
  EXPECT_EQ(span[1], "middle");
  EXPECT_EQ(span[1022], "middle");
  EXPECT_EQ(span[1023], "last");
}

TEST(StringLikeSpanTest, LargeStringSpanReportsCorrectSizeAndBoundaryElements) {
  std::vector<std::string> values(1024, "middle");
  values.front() = "first";
  values.back() = "last";

  StringLikeSpan span(tensorstore::span<const std::string>(values.data(),
                                                           values.size()));

  ASSERT_EQ(span.size(), 1024);
  EXPECT_EQ(span[0], "first");
  EXPECT_EQ(span[1], "middle");
  EXPECT_EQ(span[1022], "middle");
  EXPECT_EQ(span[1023], "last");
}

TEST(StringLikeSpanTest, LargeStringViewSpanReportsCorrectSizeAndBoundaryElements) {
  std::vector<std::string_view> values(1024, "middle");
  values.front() = "first";
  values.back() = "last";

  StringLikeSpan span(tensorstore::span<const std::string_view>(values.data(),
                                                                values.size()));

  ASSERT_EQ(span.size(), 1024);
  EXPECT_EQ(span[0], "first");
  EXPECT_EQ(span[1], "middle");
  EXPECT_EQ(span[1022], "middle");
  EXPECT_EQ(span[1023], "last");
}

#ifndef NDEBUG

TEST(StringLikeSpanDeathTest, NegativeIndexTriggersAssert) {
  std::array<std::string_view, 1> values = {std::string_view("alpha")};
  StringLikeSpan span(tensorstore::span<const std::string_view>(values));

  EXPECT_DEATH(static_cast<void>(span[-1]), "");
}

TEST(StringLikeSpanDeathTest, IndexEqualToSizeTriggersAssert) {
  std::array<std::string_view, 1> values = {std::string_view("alpha")};
  StringLikeSpan span(tensorstore::span<const std::string_view>(values));

  EXPECT_DEATH(static_cast<void>(span[1]), "");
}

TEST(StringLikeSpanDeathTest, IndexGreaterThanSizeTriggersAssert) {
  std::array<std::string_view, 1> values = {std::string_view("alpha")};
  StringLikeSpan span(tensorstore::span<const std::string_view>(values));

  EXPECT_DEATH(static_cast<void>(span[2]), "");
}

TEST(StringLikeSpanDeathTest, AccessingDefaultConstructedSpanTriggersAssert) {
  StringLikeSpan span;

  EXPECT_DEATH(static_cast<void>(span[0]), "");
}

#endif  // NDEBUG

}  // namespace
}  // namespace internal
}  // namespace tensorstore