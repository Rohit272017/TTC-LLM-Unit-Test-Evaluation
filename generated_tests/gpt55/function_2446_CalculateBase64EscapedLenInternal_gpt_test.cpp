#include "absl/strings/internal/escaping.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace absl {
ABSL_NAMESPACE_BEGIN
namespace strings_internal {
namespace {

std::string EncodeBase64(const std::string& input,
                         const char* alphabet = kBase64Chars,
                         bool do_padding = true,
                         size_t extra_capacity = 0) {
  const size_t encoded_len =
      CalculateBase64EscapedLenInternal(input.size(), do_padding);
  std::string output(encoded_len + extra_capacity, '\0');

  size_t written = Base64EscapeInternal(
      reinterpret_cast<const unsigned char*>(input.data()), input.size(),
      output.data(), output.size(), alphabet, do_padding);

  output.resize(written);
  return output;
}

TEST(Base64EscapingInternalTest, BVA_CalculateLenEmptyInput) {
  EXPECT_EQ(CalculateBase64EscapedLenInternal(0, true), 0u);
  EXPECT_EQ(CalculateBase64EscapedLenInternal(0, false), 0u);
}

TEST(Base64EscapingInternalTest, BVA_CalculateLenOneByteWithAndWithoutPadding) {
  EXPECT_EQ(CalculateBase64EscapedLenInternal(1, true), 4u);
  EXPECT_EQ(CalculateBase64EscapedLenInternal(1, false), 2u);
}

TEST(Base64EscapingInternalTest, BVA_CalculateLenTwoBytesWithAndWithoutPadding) {
  EXPECT_EQ(CalculateBase64EscapedLenInternal(2, true), 4u);
  EXPECT_EQ(CalculateBase64EscapedLenInternal(2, false), 3u);
}

TEST(Base64EscapingInternalTest, BVA_CalculateLenThreeBytesWithAndWithoutPadding) {
  EXPECT_EQ(CalculateBase64EscapedLenInternal(3, true), 4u);
  EXPECT_EQ(CalculateBase64EscapedLenInternal(3, false), 4u);
}

TEST(Base64EscapingInternalTest, ECP_CalculateLenMultipleCompleteGroups) {
  EXPECT_EQ(CalculateBase64EscapedLenInternal(6, true), 8u);
  EXPECT_EQ(CalculateBase64EscapedLenInternal(9, false), 12u);
  EXPECT_EQ(CalculateBase64EscapedLenInternal(12, true), 16u);
}

TEST(Base64EscapingInternalTest, ECP_CalculateLenRemainderOne) {
  EXPECT_EQ(CalculateBase64EscapedLenInternal(4, true), 8u);
  EXPECT_EQ(CalculateBase64EscapedLenInternal(4, false), 6u);
  EXPECT_EQ(CalculateBase64EscapedLenInternal(7, true), 12u);
  EXPECT_EQ(CalculateBase64EscapedLenInternal(7, false), 10u);
}

TEST(Base64EscapingInternalTest, ECP_CalculateLenRemainderTwo) {
  EXPECT_EQ(CalculateBase64EscapedLenInternal(5, true), 8u);
  EXPECT_EQ(CalculateBase64EscapedLenInternal(5, false), 7u);
  EXPECT_EQ(CalculateBase64EscapedLenInternal(8, true), 12u);
  EXPECT_EQ(CalculateBase64EscapedLenInternal(8, false), 11u);
}

TEST(Base64EscapingInternalTest, BVA_CalculateLenAtMaximumAllowedInput) {
  constexpr size_t kMaxSize =
      (std::numeric_limits<size_t>::max() - 1) / 4 * 3;

  EXPECT_NO_FATAL_FAILURE(
      static_cast<void>(CalculateBase64EscapedLenInternal(kMaxSize, true)));
}

TEST(Base64EscapingInternalDeathTest, Invalid_CalculateLenOverflowDies) {
  constexpr size_t kMaxSize =
      (std::numeric_limits<size_t>::max() - 1) / 4 * 3;

  EXPECT_DEATH(
      static_cast<void>(CalculateBase64EscapedLenInternal(kMaxSize + 1, true)),
      "overflow");
}

TEST(Base64EscapingInternalTest, BVA_EncodeEmptyInputReturnsZeroBytes) {
  std::string output(1, 'x');

  size_t written = Base64EscapeInternal(
      reinterpret_cast<const unsigned char*>(""), 0, output.data(),
      output.size(), kBase64Chars, true);

  EXPECT_EQ(written, 0u);
  EXPECT_EQ(output[0], 'x');
}

TEST(Base64EscapingInternalTest, BVA_EncodeOneByteWithPadding) {
  EXPECT_EQ(EncodeBase64("f", kBase64Chars, true), "Zg==");
}

TEST(Base64EscapingInternalTest, BVA_EncodeOneByteWithoutPadding) {
  EXPECT_EQ(EncodeBase64("f", kBase64Chars, false), "Zg");
}

TEST(Base64EscapingInternalTest, BVA_EncodeTwoBytesWithPadding) {
  EXPECT_EQ(EncodeBase64("fo", kBase64Chars, true), "Zm8=");
}

TEST(Base64EscapingInternalTest, BVA_EncodeTwoBytesWithoutPadding) {
  EXPECT_EQ(EncodeBase64("fo", kBase64Chars, false), "Zm8");
}

TEST(Base64EscapingInternalTest, BVA_EncodeThreeBytesWithPadding) {
  EXPECT_EQ(EncodeBase64("foo", kBase64Chars, true), "Zm9v");
}

TEST(Base64EscapingInternalTest, BVA_EncodeThreeBytesWithoutPadding) {
  EXPECT_EQ(EncodeBase64("foo", kBase64Chars, false), "Zm9v");
}

TEST(Base64EscapingInternalTest, ECP_EncodeSeveralCompleteAndPartialGroups) {
  EXPECT_EQ(EncodeBase64("foob", kBase64Chars, true), "Zm9vYg==");
  EXPECT_EQ(EncodeBase64("fooba", kBase64Chars, true), "Zm9vYmE=");
  EXPECT_EQ(EncodeBase64("foobar", kBase64Chars, true), "Zm9vYmFy");

  EXPECT_EQ(EncodeBase64("foob", kBase64Chars, false), "Zm9vYg");
  EXPECT_EQ(EncodeBase64("fooba", kBase64Chars, false), "Zm9vYmE");
  EXPECT_EQ(EncodeBase64("foobar", kBase64Chars, false), "Zm9vYmFy");
}

TEST(Base64EscapingInternalTest, ECP_EncodeWebSafeAlphabetUsesDashAndUnderscore) {
  const std::string input("\xFB\xFF\xFF", 3);

  EXPECT_EQ(EncodeBase64(input, kBase64Chars, true), "+///");
  EXPECT_EQ(EncodeBase64(input, kWebSafeBase64Chars, true), "-___");
}

TEST(Base64EscapingInternalTest, ECP_EncodeBinaryDataContainingNullBytes) {
  const std::string input("\0\0\0\0", 4);

  EXPECT_EQ(EncodeBase64(input, kBase64Chars, true), "AAAAAA==");
  EXPECT_EQ(EncodeBase64(input, kBase64Chars, false), "AAAAAA");
}

TEST(Base64EscapingInternalTest, Edge_EncodeAllByteValuesProducesNonEmptyOutput) {
  std::string input;
  for (int i = 0; i <= 255; ++i) {
    input.push_back(static_cast<char>(i));
  }

  std::string output = EncodeBase64(input, kBase64Chars, true);

  EXPECT_EQ(output.size(),
            CalculateBase64EscapedLenInternal(input.size(), true));
  EXPECT_FALSE(output.empty());
}

TEST(Base64EscapingInternalTest, Invalid_InsufficientDestinationForOneByteWithPaddingFails) {
  std::string input = "f";
  std::array<char, 3> output = {'x', 'x', 'x'};

  size_t written = Base64EscapeInternal(
      reinterpret_cast<const unsigned char*>(input.data()), input.size(),
      output.data(), output.size(), kBase64Chars, true);

  EXPECT_EQ(written, 0u);
}

TEST(Base64EscapingInternalTest, Invalid_InsufficientDestinationForTwoBytesWithPaddingFails) {
  std::string input = "fo";
  std::array<char, 3> output = {'x', 'x', 'x'};

  size_t written = Base64EscapeInternal(
      reinterpret_cast<const unsigned char*>(input.data()), input.size(),
      output.data(), output.size(), kBase64Chars, true);

  EXPECT_EQ(written, 0u);
}

TEST(Base64EscapingInternalTest, Invalid_InsufficientDestinationForThreeBytesFails) {
  std::string input = "foo";
  std::array<char, 3> output = {'x', 'x', 'x'};

  size_t written = Base64EscapeInternal(
      reinterpret_cast<const unsigned char*>(input.data()), input.size(),
      output.data(), output.size(), kBase64Chars, true);

  EXPECT_EQ(written, 0u);
}

TEST(Base64EscapingInternalTest, Invalid_EarlyCapacityCheckFailsWhenDestinationTooSmall) {
  std::string input = "foobar";
  std::array<char, 7> output = {};

  size_t written = Base64EscapeInternal(
      reinterpret_cast<const unsigned char*>(input.data()), input.size(),
      output.data(), output.size(), kBase64Chars, true);

  EXPECT_EQ(written, 0u);
}

TEST(Base64EscapingInternalTest, BVA_ExactDestinationSizeSucceeds) {
  std::string input = "foobar";
  std::string output(CalculateBase64EscapedLenInternal(input.size(), true),
                     '\0');

  size_t written = Base64EscapeInternal(
      reinterpret_cast<const unsigned char*>(input.data()), input.size(),
      output.data(), output.size(), kBase64Chars, true);

  EXPECT_EQ(written, output.size());
  EXPECT_EQ(output, "Zm9vYmFy");
}

TEST(Base64EscapingInternalTest, Edge_LargerDestinationSizeDoesNotAffectWrittenLength) {
  std::string input = "foobar";
  std::string output(32, '\0');

  size_t written = Base64EscapeInternal(
      reinterpret_cast<const unsigned char*>(input.data()), input.size(),
      output.data(), output.size(), kBase64Chars, true);

  EXPECT_EQ(written, 8u);
  output.resize(written);
  EXPECT_EQ(output, "Zm9vYmFy");
}

TEST(Base64EscapingInternalTest, Edge_DestinationBytesAfterWrittenLengthRemainUnchanged) {
  std::string input = "f";
  std::string output(8, 'x');

  size_t written = Base64EscapeInternal(
      reinterpret_cast<const unsigned char*>(input.data()), input.size(),
      output.data(), output.size(), kBase64Chars, true);

  ASSERT_EQ(written, 4u);
  EXPECT_EQ(output.substr(0, written), "Zg==");
  EXPECT_EQ(output.substr(written), "xxxx");
}

TEST(Base64EscapingInternalTest, Edge_OutputMayUseCustomAlphabet) {
  constexpr char kCustomAlphabet[] =
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/";

  EXPECT_EQ(EncodeBase64("foo", kCustomAlphabet, true), "6m9V");
}

TEST(Base64EscapingInternalTest, ECP_InputLengthFourExercisesLoopAndRemainderOne) {
  EXPECT_EQ(EncodeBase64("abcd", kBase64Chars, true), "YWJjZA==");
}

TEST(Base64EscapingInternalTest, ECP_InputLengthFiveExercisesLoopAndRemainderTwo) {
  EXPECT_EQ(EncodeBase64("abcde", kBase64Chars, true), "YWJjZGU=");
}

TEST(Base64EscapingInternalTest, ECP_InputLengthSixExercisesTwoFullGroups) {
  EXPECT_EQ(EncodeBase64("abcdef", kBase64Chars, true), "YWJjZGVm");
}

}  // namespace
}  // namespace strings_internal
ABSL_NAMESPACE_END
}  // namespace absl