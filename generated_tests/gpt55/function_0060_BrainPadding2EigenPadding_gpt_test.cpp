#include "tensorflow/core/util/padding.h"

#include <gtest/gtest.h>

#include <string>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace {

TEST(BrainPadding2EigenPaddingTest, ValidPaddingReturnsEigenValid) {
  EXPECT_EQ(BrainPadding2EigenPadding(Padding::VALID), Eigen::PADDING_VALID);
}

TEST(BrainPadding2EigenPaddingTest, SamePaddingReturnsEigenSame) {
  EXPECT_EQ(BrainPadding2EigenPadding(Padding::SAME), Eigen::PADDING_SAME);
}

TEST(BrainPadding2EigenPaddingTest, ExplicitPaddingCausesFatalFailure) {
  EXPECT_DEATH(
      { BrainPadding2EigenPadding(Padding::EXPLICIT); },
      "Eigen does not have explicit padding enum value");
}

TEST(GetBroadcastSizeTest, NoPaddingFullKernelAtStart) {
  int bindex = -1;
  int bsize = -1;

  TF_EXPECT_OK(GetBroadcastSize(
      /*index=*/0,
      /*in_size=*/10,
      /*ksize=*/3,
      /*stride=*/1,
      /*pad_size=*/0,
      &bindex,
      &bsize));

  EXPECT_EQ(bindex, 0);
  EXPECT_EQ(bsize, 3);
}

TEST(GetBroadcastSizeTest, NoPaddingMiddleWindow) {
  int bindex = -1;
  int bsize = -1;

  TF_EXPECT_OK(GetBroadcastSize(2, 10, 3, 2, 0, &bindex, &bsize));

  EXPECT_EQ(bindex, 4);
  EXPECT_EQ(bsize, 3);
}

TEST(GetBroadcastSizeTest, LeftPaddingReducesBroadcastSize) {
  int bindex = -1;
  int bsize = -1;

  TF_EXPECT_OK(GetBroadcastSize(
      /*index=*/0,
      /*in_size=*/10,
      /*ksize=*/5,
      /*stride=*/1,
      /*pad_size=*/2,
      &bindex,
      &bsize));

  EXPECT_EQ(bindex, 0);
  EXPECT_EQ(bsize, 3);
}

TEST(GetBroadcastSizeTest, LeftPaddingBoundaryExactlyAtPadSize) {
  int bindex = -1;
  int bsize = -1;

  TF_EXPECT_OK(GetBroadcastSize(
      /*index=*/2,
      /*in_size=*/10,
      /*ksize=*/5,
      /*stride=*/1,
      /*pad_size=*/2,
      &bindex,
      &bsize));

  EXPECT_EQ(bindex, 0);
  EXPECT_EQ(bsize, 5);
}

TEST(GetBroadcastSizeTest, RightBoundaryReducesBroadcastSize) {
  int bindex = -1;
  int bsize = -1;

  TF_EXPECT_OK(GetBroadcastSize(
      /*index=*/4,
      /*in_size=*/10,
      /*ksize=*/5,
      /*stride=*/3,
      /*pad_size=*/2,
      &bindex,
      &bsize));

  EXPECT_EQ(bindex, 10);
  EXPECT_EQ(bsize, 0);
}

TEST(GetBroadcastSizeTest, WindowPartiallyExceedsRightBoundary) {
  int bindex = -1;
  int bsize = -1;

  TF_EXPECT_OK(GetBroadcastSize(
      /*index=*/3,
      /*in_size=*/10,
      /*ksize=*/5,
      /*stride=*/3,
      /*pad_size=*/2,
      &bindex,
      &bsize));

  EXPECT_EQ(bindex, 7);
  EXPECT_EQ(bsize, 3);
}

TEST(GetBroadcastSizeTest, IndexTimesStrideEqualToInputSizeIsValidBoundary) {
  int bindex = -1;
  int bsize = -1;

  TF_EXPECT_OK(GetBroadcastSize(
      /*index=*/5,
      /*in_size=*/10,
      /*ksize=*/3,
      /*stride=*/2,
      /*pad_size=*/0,
      &bindex,
      &bsize));

  EXPECT_EQ(bindex, 10);
  EXPECT_EQ(bsize, 0);
}

TEST(GetBroadcastSizeTest, IndexTimesStrideGreaterThanInputSizeReturnsError) {
  int bindex = -1;
  int bsize = -1;

  const Status status = GetBroadcastSize(
      /*index=*/6,
      /*in_size=*/10,
      /*ksize=*/3,
      /*stride=*/2,
      /*pad_size=*/0,
      &bindex,
      &bsize);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  EXPECT_NE(status.message().find(
                "index * stride must be less than or equal to input size"),
            std::string::npos);
  EXPECT_EQ(bindex, -1);
  EXPECT_EQ(bsize, -1);
}

TEST(GetBroadcastSizeTest, KernelSizeOneValidBoundary) {
  int bindex = -1;
  int bsize = -1;

  TF_EXPECT_OK(GetBroadcastSize(
      /*index=*/1,
      /*in_size=*/5,
      /*ksize=*/1,
      /*stride=*/1,
      /*pad_size=*/0,
      &bindex,
      &bsize));

  EXPECT_EQ(bindex, 1);
  EXPECT_EQ(bsize, 1);
}

TEST(GetBroadcastSizeTest, InputSizeOneWithPadding) {
  int bindex = -1;
  int bsize = -1;

  TF_EXPECT_OK(GetBroadcastSize(
      /*index=*/0,
      /*in_size=*/1,
      /*ksize=*/3,
      /*stride=*/1,
      /*pad_size=*/1,
      &bindex,
      &bsize));

  EXPECT_EQ(bindex, 0);
  EXPECT_EQ(bsize, 1);
}

TEST(GetBroadcastSizeTest, ZeroKernelSizeReturnsZeroBroadcastSize) {
  int bindex = -1;
  int bsize = -1;

  TF_EXPECT_OK(GetBroadcastSize(
      /*index=*/0,
      /*in_size=*/10,
      /*ksize=*/0,
      /*stride=*/1,
      /*pad_size=*/0,
      &bindex,
      &bsize));

  EXPECT_EQ(bindex, 0);
  EXPECT_EQ(bsize, 0);
}

TEST(GetBroadcastSizeTest, ZeroInputSizeBoundary) {
  int bindex = -1;
  int bsize = -1;

  TF_EXPECT_OK(GetBroadcastSize(
      /*index=*/0,
      /*in_size=*/0,
      /*ksize=*/3,
      /*stride=*/1,
      /*pad_size=*/0,
      &bindex,
      &bsize));

  EXPECT_EQ(bindex, 0);
  EXPECT_EQ(bsize, 0);
}

TEST(GetBroadcastSizeTest, LargeStrideValidCase) {
  int bindex = -1;
  int bsize = -1;

  TF_EXPECT_OK(GetBroadcastSize(
      /*index=*/2,
      /*in_size=*/100,
      /*ksize=*/7,
      /*stride=*/20,
      /*pad_size=*/5,
      &bindex,
      &bsize));

  EXPECT_EQ(bindex, 35);
  EXPECT_EQ(bsize, 7);
}

TEST(SanitizeThreadSuffixTest, EmptyStringReturnsEmptyString) {
  EXPECT_EQ(SanitizeThreadSuffix(""), "");
}

TEST(SanitizeThreadSuffixTest, KeepsLowercaseLetters) {
  EXPECT_EQ(SanitizeThreadSuffix("abcxyz"), "abcxyz");
}

TEST(SanitizeThreadSuffixTest, KeepsUppercaseLetters) {
  EXPECT_EQ(SanitizeThreadSuffix("ABCXYZ"), "ABCXYZ");
}

TEST(SanitizeThreadSuffixTest, KeepsDigits) {
  EXPECT_EQ(SanitizeThreadSuffix("0123456789"), "0123456789");
}

TEST(SanitizeThreadSuffixTest, KeepsUnderscoreAndHyphen) {
  EXPECT_EQ(SanitizeThreadSuffix("_-"), "_-");
}

TEST(SanitizeThreadSuffixTest, ReplacesSpacesWithUnderscores) {
  EXPECT_EQ(SanitizeThreadSuffix("thread one two"), "thread_one_two");
}

TEST(SanitizeThreadSuffixTest, ReplacesSpecialCharactersWithUnderscores) {
  EXPECT_EQ(SanitizeThreadSuffix("a.b/c\\d:e*f?g!"), "a_b_c_d_e_f_g_");
}

TEST(SanitizeThreadSuffixTest, MixedValidAndInvalidCharacters) {
  EXPECT_EQ(SanitizeThreadSuffix("Worker-01/Main.Thread"),
            "Worker-01_Main_Thread");
}

TEST(SanitizeThreadSuffixTest, BoundaryCharactersAroundAllowedRanges) {
  EXPECT_EQ(SanitizeThreadSuffix("@AZ[`az{09:/_-"), "_AZ__az_09___-");
}

TEST(SanitizeThreadSuffixTest, AllInvalidCharactersBecomeUnderscores) {
  EXPECT_EQ(SanitizeThreadSuffix(" !@#$%^&*()+={}[]|;:'\",.<>?/"),
            "______________________________");
}

TEST(SanitizeThreadSuffixTest, LongStringIsSanitizedCharacterByCharacter) {
  std::string input;
  std::string expected;

  for (int i = 0; i < 1000; ++i) {
    input += "A.";
    expected += "A_";
  }

  EXPECT_EQ(SanitizeThreadSuffix(input), expected);
}

}  // namespace
}  // namespace tensorflow