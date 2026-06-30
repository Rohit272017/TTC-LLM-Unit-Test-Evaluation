#include "absl/log/internal/fnmatch.h"

#include <gtest/gtest.h>

#include "absl/base/config.h"

namespace absl {
ABSL_NAMESPACE_BEGIN
namespace log_internal {
namespace {

TEST(FNMatchTest, EmptyPatternMatchesOnlyEmptyString) {
  EXPECT_TRUE(FNMatch("", ""));
  EXPECT_FALSE(FNMatch("", "a"));
}

TEST(FNMatchTest, ExactMatchReturnsTrue) {
  EXPECT_TRUE(FNMatch("abc", "abc"));
}

TEST(FNMatchTest, ExactMatchDifferentLengthReturnsFalse) {
  EXPECT_FALSE(FNMatch("abc", "ab"));
  EXPECT_FALSE(FNMatch("abc", "abcd"));
}

TEST(FNMatchTest, ExactMatchDifferentCharacterReturnsFalse) {
  EXPECT_FALSE(FNMatch("abc", "axc"));
}

TEST(FNMatchTest, SingleQuestionMarkMatchesOneCharacter) {
  EXPECT_TRUE(FNMatch("?", "a"));
  EXPECT_TRUE(FNMatch("a?c", "abc"));
}

TEST(FNMatchTest, QuestionMarkDoesNotMatchEmptyString) {
  EXPECT_FALSE(FNMatch("?", ""));
  EXPECT_FALSE(FNMatch("a?", "a"));
}

TEST(FNMatchTest, MultipleQuestionMarksMatchSameNumberOfCharacters) {
  EXPECT_TRUE(FNMatch("???", "abc"));
  EXPECT_FALSE(FNMatch("???", "ab"));
  EXPECT_FALSE(FNMatch("???", "abcd"));
}

TEST(FNMatchTest, SingleStarMatchesEmptyString) {
  EXPECT_TRUE(FNMatch("*", ""));
}

TEST(FNMatchTest, SingleStarMatchesAnyString) {
  EXPECT_TRUE(FNMatch("*", "a"));
  EXPECT_TRUE(FNMatch("*", "abcdef"));
}

TEST(FNMatchTest, MultipleStarsMatchEmptyString) {
  EXPECT_TRUE(FNMatch("**", ""));
  EXPECT_TRUE(FNMatch("***", ""));
}

TEST(FNMatchTest, TrailingStarsMatchRemainingString) {
  EXPECT_TRUE(FNMatch("abc*", "abc"));
  EXPECT_TRUE(FNMatch("abc*", "abcdef"));
}

TEST(FNMatchTest, LeadingStarMatchesPrefix) {
  EXPECT_TRUE(FNMatch("*abc", "abc"));
  EXPECT_TRUE(FNMatch("*abc", "xyzabc"));
}

TEST(FNMatchTest, LeadingStarRequiresFixedSuffix) {
  EXPECT_FALSE(FNMatch("*abc", "xyzab"));
}

TEST(FNMatchTest, MiddleStarMatchesZeroCharacters) {
  EXPECT_TRUE(FNMatch("ab*cd", "abcd"));
}

TEST(FNMatchTest, MiddleStarMatchesMultipleCharacters) {
  EXPECT_TRUE(FNMatch("ab*cd", "abXYZcd"));
}

TEST(FNMatchTest, MiddleStarRequiresFollowingFixedPortion) {
  EXPECT_FALSE(FNMatch("ab*cd", "abXYZce"));
}

TEST(FNMatchTest, StarThenQuestionMatchesAtLeastOneRemainingCharacter) {
  EXPECT_TRUE(FNMatch("*?", "a"));
  EXPECT_TRUE(FNMatch("*?", "abc"));
  EXPECT_FALSE(FNMatch("*?", ""));
}

TEST(FNMatchTest, QuestionThenStarMatchesAtLeastOneStartingCharacter) {
  EXPECT_TRUE(FNMatch("?*", "a"));
  EXPECT_TRUE(FNMatch("?*", "abc"));
  EXPECT_FALSE(FNMatch("?*", ""));
}

TEST(FNMatchTest, StarQuestionFixedSuffix) {
  EXPECT_TRUE(FNMatch("*?c", "abc"));
  EXPECT_TRUE(FNMatch("*?c", "zzabc"));
  EXPECT_FALSE(FNMatch("*?c", "c"));
}

TEST(FNMatchTest, StarFindsFirstFixedPortionThenContinues) {
  EXPECT_TRUE(FNMatch("*ab?d", "xxabcd"));
  EXPECT_FALSE(FNMatch("*ab?d", "xxab"));
}

TEST(FNMatchTest, ConsecutiveStarsBehaveLikeSingleStar) {
  EXPECT_TRUE(FNMatch("a**c", "ac"));
  EXPECT_TRUE(FNMatch("a**c", "abbbbbc"));
  EXPECT_FALSE(FNMatch("a**c", "abbbbbd"));
}

TEST(FNMatchTest, MultipleWildcardSectionsMatch) {
  EXPECT_TRUE(FNMatch("a*b?d*e", "axxxbcdyye"));
  EXPECT_TRUE(FNMatch("a*b?d*e", "abcde"));
}

TEST(FNMatchTest, MultipleWildcardSectionsDoNotMatchWhenFixedPartMissing) {
  EXPECT_FALSE(FNMatch("a*b?d*e", "axxxbcxyye"));
}

TEST(FNMatchTest, PatternLongerThanStringWithOnlyTrailingStarsCanMatch) {
  EXPECT_TRUE(FNMatch("abc***", "abc"));
}

TEST(FNMatchTest, PatternLongerThanStringWithNonStarRemainderDoesNotMatch) {
  EXPECT_FALSE(FNMatch("abc?*", "abc"));
  EXPECT_FALSE(FNMatch("abcd*", "abc"));
}

TEST(FNMatchTest, CaseSensitiveMatching) {
  EXPECT_TRUE(FNMatch("AbC", "AbC"));
  EXPECT_FALSE(FNMatch("abc", "AbC"));
}

TEST(FNMatchTest, SpecialCharactersAreMatchedLiterally) {
  EXPECT_TRUE(FNMatch("a.b/c", "a.b/c"));
  EXPECT_FALSE(FNMatch("a.b/c", "axb/c"));
}

TEST(FNMatchTest, StarCanMatchSpecialCharacters) {
  EXPECT_TRUE(FNMatch("a*c", "a./-_123c"));
}

TEST(FNMatchTest, QuestionCanMatchSpecialCharacter) {
  EXPECT_TRUE(FNMatch("a?c", "a.c"));
  EXPECT_TRUE(FNMatch("a?c", "a/c"));
}

TEST(FNMatchTest, EmbeddedNullCharactersAreHandled) {
  EXPECT_TRUE(FNMatch(absl::string_view("a?c", 3),
                      absl::string_view("a\0c", 3)));
  EXPECT_TRUE(FNMatch(absl::string_view("a*c", 3),
                      absl::string_view("a\0\0c", 4)));
}

TEST(FNMatchTest, LongStringExactAndWildcardMatch) {
  std::string str(1000, 'x');
  std::string pattern = "a*" + str + "*z";
  std::string input = "a" + str + "middle" + str + "z";

  EXPECT_TRUE(FNMatch(pattern, input));
}

TEST(FNMatchTest, WildcardDoesNotBacktrackToLaterFixedMatch) {
  EXPECT_FALSE(FNMatch("*ab?c", "ababc"));
}

}  // namespace
}  // namespace log_internal
ABSL_NAMESPACE_END
}  // namespace absl