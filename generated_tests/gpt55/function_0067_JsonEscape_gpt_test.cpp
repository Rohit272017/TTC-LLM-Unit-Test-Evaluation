#include "tensorflow/core/profiler/convert/trace_viewer/trace_events_to_json.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <string>

#include "tensorflow/core/profiler/protobuf/trace_events.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

TEST(JsonEscapeTest, EmptyStringReturnsQuotedEmptyString) {
  EXPECT_EQ(JsonEscape(""), "\"\"");
}

TEST(JsonEscapeTest, PlainAsciiStringIsOnlyQuoted) {
  EXPECT_EQ(JsonEscape("abcXYZ123_-"), "\"abcXYZ123_-\"");
}

TEST(JsonEscapeTest, EscapesDoubleQuote) {
  EXPECT_EQ(JsonEscape("a\"b"), "\"a\\\"b\"");
}

TEST(JsonEscapeTest, EscapesBackslash) {
  EXPECT_EQ(JsonEscape("a\\b"), "\"a\\\\b\"");
}

TEST(JsonEscapeTest, EscapesBackspace) {
  EXPECT_EQ(JsonEscape(std::string("a\b", 2)), "\"a\\b\"");
}

TEST(JsonEscapeTest, EscapesFormFeed) {
  EXPECT_EQ(JsonEscape(std::string("a\f", 2)), "\"a\\f\"");
}

TEST(JsonEscapeTest, EscapesNewline) {
  EXPECT_EQ(JsonEscape("a\nb"), "\"a\\nb\"");
}

TEST(JsonEscapeTest, EscapesCarriageReturn) {
  EXPECT_EQ(JsonEscape("a\rb"), "\"a\\rb\"");
}

TEST(JsonEscapeTest, EscapesTab) {
  EXPECT_EQ(JsonEscape("a\tb"), "\"a\\tb\"");
}

TEST(JsonEscapeTest, EscapesOtherControlCharactersAsUnicode) {
  std::string input;
  input.push_back(static_cast<char>(0x00));
  input.push_back(static_cast<char>(0x01));
  input.push_back(static_cast<char>(0x1F));

  EXPECT_EQ(JsonEscape(input), "\"\\u0000\\u0001\\u001f\"");
}

TEST(JsonEscapeTest, BoundaryControlCharacters) {
  std::string input;
  input.push_back(static_cast<char>(0x1F));
  input.push_back(static_cast<char>(0x20));

  EXPECT_EQ(JsonEscape(input), "\"\\u001f \"");
}

TEST(JsonEscapeTest, EscapesLessThanGreaterThanAndAmpersand) {
  EXPECT_EQ(JsonEscape("<>&"), "\"\\u003c\\u003e\\u0026\"");
}

TEST(JsonEscapeTest, EscapesUnicodeLineSeparator) {
  const std::string input = std::string("a") + "\xE2\x80\xA8" + "b";

  EXPECT_EQ(JsonEscape(input), "\"a\\u2028b\"");
}

TEST(JsonEscapeTest, EscapesUnicodeParagraphSeparator) {
  const std::string input = std::string("a") + "\xE2\x80\xA9" + "b";

  EXPECT_EQ(JsonEscape(input), "\"a\\u2029b\"");
}

TEST(JsonEscapeTest, LeavesOtherUtf8SequencesUnchanged) {
  const std::string input = std::string("a") + "\xE2\x82\xAC" + "b";

  EXPECT_EQ(JsonEscape(input), "\"" + input + "\"");
}

TEST(JsonEscapeTest, IncompleteE2SequenceAtEndIsPreserved) {
  const std::string input = std::string("a") + "\xE2";

  EXPECT_EQ(JsonEscape(input), "\"" + input + "\"");
}

TEST(JsonEscapeTest, E2SequenceWithout80SecondByteIsPreserved) {
  const std::string input = std::string("a") + "\xE2\x81\xA8" + "b";

  EXPECT_EQ(JsonEscape(input), "\"" + input + "\"");
}

TEST(JsonEscapeTest, E280SequenceWithoutA8OrA9IsPreserved) {
  const std::string input = std::string("a") + "\xE2\x80\xAA" + "b";

  EXPECT_EQ(JsonEscape(input), "\"" + input + "\"");
}

TEST(JsonEscapeTest, MixedEscapingCases) {
  const std::string input =
      std::string("a\"b\\c\nd\t") + "<>&" + "\xE2\x80\xA8";

  EXPECT_EQ(JsonEscape(input),
            "\"a\\\"b\\\\c\\nd\\t\\u003c\\u003e\\u0026\\u2028\"");
}

TEST(JsonEscapeTest, LongStringIsEscapedCharacterByCharacter) {
  std::string input;
  std::string expected = "\"";

  for (int i = 0; i < 1000; ++i) {
    input += "a\n";
    expected += "a\\n";
  }

  expected += "\"";

  EXPECT_EQ(JsonEscape(input), expected);
}

TEST(BuildStackFrameReferencesTest, EmptyTraceReturnsEmptyMap) {
  Trace trace;

  const std::map<uint64_t, uint64_t> result =
      BuildStackFrameReferences(trace);

  EXPECT_TRUE(result.empty());
}

TEST(BuildStackFrameReferencesTest, IgnoresNamesWithoutAtAtPrefix) {
  Trace trace;
  (*trace.mutable_name_table())[1] = "main";
  (*trace.mutable_name_table())[2] = "@single_at";
  (*trace.mutable_name_table())[3] = "prefix@@name";

  const std::map<uint64_t, uint64_t> result =
      BuildStackFrameReferences(trace);

  EXPECT_TRUE(result.empty());
}

TEST(BuildStackFrameReferencesTest, IncludesNamesWithAtAtPrefix) {
  Trace trace;
  (*trace.mutable_name_table())[10] = "@@main";

  const std::map<uint64_t, uint64_t> result =
      BuildStackFrameReferences(trace);

  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result.at(10), 1u);
}

TEST(BuildStackFrameReferencesTest, AssignsSequentialReferencesInFramePointerOrder) {
  Trace trace;
  (*trace.mutable_name_table())[30] = "@@third";
  (*trace.mutable_name_table())[10] = "@@first";
  (*trace.mutable_name_table())[20] = "@@second";

  const std::map<uint64_t, uint64_t> result =
      BuildStackFrameReferences(trace);

  ASSERT_EQ(result.size(), 3u);
  EXPECT_EQ(result.at(10), 1u);
  EXPECT_EQ(result.at(20), 2u);
  EXPECT_EQ(result.at(30), 3u);
}

TEST(BuildStackFrameReferencesTest, FiltersMixedNamesAndKeepsOnlyStackFrames) {
  Trace trace;
  (*trace.mutable_name_table())[1] = "main";
  (*trace.mutable_name_table())[2] = "@@frame_a";
  (*trace.mutable_name_table())[3] = "@frame_b";
  (*trace.mutable_name_table())[4] = "@@frame_c";
  (*trace.mutable_name_table())[5] = "frame@@d";

  const std::map<uint64_t, uint64_t> result =
      BuildStackFrameReferences(trace);

  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(result.at(2), 1u);
  EXPECT_EQ(result.at(4), 2u);
  EXPECT_EQ(result.count(1), 0u);
  EXPECT_EQ(result.count(3), 0u);
  EXPECT_EQ(result.count(5), 0u);
}

TEST(BuildStackFrameReferencesTest, HandlesMinimumAndMaximumFramePointerValues) {
  Trace trace;
  (*trace.mutable_name_table())[0] = "@@zero";
  (*trace.mutable_name_table())[UINT64_MAX] = "@@max";

  const std::map<uint64_t, uint64_t> result =
      BuildStackFrameReferences(trace);

  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(result.at(0), 1u);
  EXPECT_EQ(result.at(UINT64_MAX), 2u);
}

TEST(BuildStackFrameReferencesTest, AtAtOnlyNameIsValidStackFrameName) {
  Trace trace;
  (*trace.mutable_name_table())[42] = "@@";

  const std::map<uint64_t, uint64_t> result =
      BuildStackFrameReferences(trace);

  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result.at(42), 1u);
}

TEST(BuildStackFrameReferencesTest, LargeNumberOfStackFramesGetsSequentialIds) {
  Trace trace;

  for (uint64_t i = 1; i <= 100; ++i) {
    (*trace.mutable_name_table())[i * 10] = "@@frame";
  }

  const std::map<uint64_t, uint64_t> result =
      BuildStackFrameReferences(trace);

  ASSERT_EQ(result.size(), 100u);

  uint64_t expected_reference = 1;
  for (uint64_t i = 1; i <= 100; ++i) {
    EXPECT_EQ(result.at(i * 10), expected_reference++);
  }
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow