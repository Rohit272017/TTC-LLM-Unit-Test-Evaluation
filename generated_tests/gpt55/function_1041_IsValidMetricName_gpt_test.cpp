#include "tensorstore/internal/metrics/metadata.h"

#include <gtest/gtest.h>

#include <string>

namespace tensorstore {
namespace internal_metrics {
namespace {

TEST(IsValidMetricNameTest, RejectsEmptyAndSingleCharacterNames) {
  EXPECT_FALSE(IsValidMetricName(""));
  EXPECT_FALSE(IsValidMetricName("/"));
  EXPECT_FALSE(IsValidMetricName("a"));
}

TEST(IsValidMetricNameTest, RejectsNameWithoutLeadingSlash) {
  EXPECT_FALSE(IsValidMetricName("metric"));
  EXPECT_FALSE(IsValidMetricName("a/b"));
}

TEST(IsValidMetricNameTest, RejectsNameEndingWithSlash) {
  EXPECT_FALSE(IsValidMetricName("/metric/"));
  EXPECT_FALSE(IsValidMetricName("/a/b/"));
}

TEST(IsValidMetricNameTest, RejectsWhenFirstCharacterAfterSlashIsNotAlpha) {
  EXPECT_FALSE(IsValidMetricName("/1metric"));
  EXPECT_FALSE(IsValidMetricName("/_metric"));
}

TEST(IsValidMetricNameTest, AcceptsMinimumValidMetricName) {
  EXPECT_TRUE(IsValidMetricName("/a"));
  EXPECT_TRUE(IsValidMetricName("/Z"));
}

TEST(IsValidMetricNameTest, AcceptsAlnumAndUnderscoreCharacters) {
  EXPECT_TRUE(IsValidMetricName("/metric_123"));
  EXPECT_TRUE(IsValidMetricName("/Metric_ABC_123"));
}

TEST(IsValidMetricNameTest, AcceptsMultiplePathComponents) {
  EXPECT_TRUE(IsValidMetricName("/foo/bar"));
  EXPECT_TRUE(IsValidMetricName("/foo_1/bar_2/baz3"));
}

TEST(IsValidMetricNameTest, RejectsEmptyPathComponent) {
  EXPECT_FALSE(IsValidMetricName("/foo//bar"));
  EXPECT_FALSE(IsValidMetricName("//foo"));
}

TEST(IsValidMetricNameTest, RejectsInvalidCharacters) {
  EXPECT_FALSE(IsValidMetricName("/foo-bar"));
  EXPECT_FALSE(IsValidMetricName("/foo.bar"));
  EXPECT_FALSE(IsValidMetricName("/foo bar"));
  EXPECT_FALSE(IsValidMetricName("/foo@bar"));
}

TEST(IsValidMetricNameTest, ComponentLengthExactlySixtyThreeIsValid) {
  std::string component(63, 'a');

  EXPECT_TRUE(IsValidMetricName("/" + component));
  EXPECT_TRUE(IsValidMetricName("/" + component + "/b"));
}

TEST(IsValidMetricNameTest, ComponentLengthSixtyFourIsInvalidBeforeSlash) {
  std::string component(64, 'a');

  EXPECT_FALSE(IsValidMetricName("/" + component + "/b"));
}

TEST(IsValidMetricNameTest, FinalComponentLengthGreaterThanSixtyThreeIsAllowed) {
  std::string component(64, 'a');

  EXPECT_TRUE(IsValidMetricName("/" + component));
}

TEST(IsValidMetricNameTest, RejectsNonAsciiAlphabeticStart) {
  EXPECT_FALSE(IsValidMetricName("/émetric"));
}

TEST(IsValidMetricLabelTest, RejectsEmptyLabel) {
  EXPECT_FALSE(IsValidMetricLabel(""));
}

TEST(IsValidMetricLabelTest, RejectsLabelStartingWithNonAlpha) {
  EXPECT_FALSE(IsValidMetricLabel("1label"));
  EXPECT_FALSE(IsValidMetricLabel("_label"));
}

TEST(IsValidMetricLabelTest, AcceptsSingleAlphabeticCharacter) {
  EXPECT_TRUE(IsValidMetricLabel("a"));
  EXPECT_TRUE(IsValidMetricLabel("Z"));
}

TEST(IsValidMetricLabelTest, AcceptsAlnumAndUnderscoreCharacters) {
  EXPECT_TRUE(IsValidMetricLabel("label_123"));
  EXPECT_TRUE(IsValidMetricLabel("LabelABC_123"));
}

TEST(IsValidMetricLabelTest, RejectsInvalidCharacters) {
  EXPECT_FALSE(IsValidMetricLabel("label-name"));
  EXPECT_FALSE(IsValidMetricLabel("label.name"));
  EXPECT_FALSE(IsValidMetricLabel("label name"));
  EXPECT_FALSE(IsValidMetricLabel("label/name"));
  EXPECT_FALSE(IsValidMetricLabel("label@name"));
}

TEST(IsValidMetricLabelTest, RejectsNonAsciiAlphabeticStart) {
  EXPECT_FALSE(IsValidMetricLabel("élabel"));
}

TEST(IsValidMetricLabelTest, LongValidLabelIsAccepted) {
  std::string label = "a" + std::string(1000, '_');

  EXPECT_TRUE(IsValidMetricLabel(label));
}

TEST(UnitsToStringTest, UnknownUnitReturnsEmptyStringView) {
  EXPECT_EQ(UnitsToString(Units::kUnknown), "");
}

TEST(UnitsToStringTest, TimeUnitsReturnExpectedStrings) {
  EXPECT_EQ(UnitsToString(Units::kSeconds), "seconds");
  EXPECT_EQ(UnitsToString(Units::kMilliseconds), "milliseconds");
  EXPECT_EQ(UnitsToString(Units::kMicroseconds), "microseconds");
  EXPECT_EQ(UnitsToString(Units::kNanoseconds), "nanoseconds");
}

TEST(UnitsToStringTest, DataUnitsReturnExpectedStrings) {
  EXPECT_EQ(UnitsToString(Units::kBits), "bits");
  EXPECT_EQ(UnitsToString(Units::kBytes), "bytes");
  EXPECT_EQ(UnitsToString(Units::kKilobytes), "kilobytes");
  EXPECT_EQ(UnitsToString(Units::kMegabytes), "megabytes");
}

}  // namespace
}  // namespace internal_metrics
}  // namespace tensorstore