#include "tensorflow/lite/testing/tokenize.h"

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

namespace tflite {
namespace testing {
namespace {

class CollectingTokenProcessor : public TokenProcessor {
 public:
  void ConsumeToken(std::string* token) override { tokens.push_back(*token); }

  std::vector<std::string> tokens;
};

std::vector<std::string> TokenizeString(const std::string& input) {
  std::istringstream stream(input);
  CollectingTokenProcessor processor;

  Tokenize(&stream, &processor);

  return processor.tokens;
}

TEST(TokenizeTest, ECP_EmptyInputProducesNoTokens) {
  EXPECT_TRUE(TokenizeString("").empty());
}

TEST(TokenizeTest, ECP_WhitespaceOnlyProducesNoTokens) {
  EXPECT_TRUE(TokenizeString(" \t\n\r").empty());
}

TEST(TokenizeTest, ECP_SingleTokenProducesOneToken) {
  EXPECT_EQ(TokenizeString("abc"), std::vector<std::string>({"abc"}));
}

TEST(TokenizeTest, BVA_SingleCharacterTokenProducesOneToken) {
  EXPECT_EQ(TokenizeString("a"), std::vector<std::string>({"a"}));
}

TEST(TokenizeTest, ECP_MultipleTokensSeparatedByWhitespace) {
  EXPECT_EQ(TokenizeString("abc def ghi"),
            std::vector<std::string>({"abc", "def", "ghi"}));
}

TEST(TokenizeTest, Edge_LeadingAndTrailingWhitespaceIgnored) {
  EXPECT_EQ(TokenizeString("  \t abc def \n "),
            std::vector<std::string>({"abc", "def"}));
}

TEST(TokenizeTest, ECP_DelimitersAreSeparateTokens) {
  EXPECT_EQ(TokenizeString("{abc:def}"),
            std::vector<std::string>({"{", "abc", ":", "def", "}"}));
}

TEST(TokenizeTest, BVA_SingleDelimiterLeftBrace) {
  EXPECT_EQ(TokenizeString("{"), std::vector<std::string>({"{"}));
}

TEST(TokenizeTest, BVA_SingleDelimiterRightBrace) {
  EXPECT_EQ(TokenizeString("}"), std::vector<std::string>({"}"}));
}

TEST(TokenizeTest, BVA_SingleDelimiterColon) {
  EXPECT_EQ(TokenizeString(":"), std::vector<std::string>({":"}));
}

TEST(TokenizeTest, ECP_ConsecutiveDelimitersProduceSeparateTokens) {
  EXPECT_EQ(TokenizeString("{}::"),
            std::vector<std::string>({"{", "}", ":", ":"}));
}

TEST(TokenizeTest, ECP_DelimitersWithWhitespaceProduceSameTokens) {
  EXPECT_EQ(TokenizeString("{ abc : def }"),
            std::vector<std::string>({"{", "abc", ":", "def", "}"}));
}

TEST(TokenizeTest, ECP_QuotedTokenProducesTokenWithoutQuotes) {
  EXPECT_EQ(TokenizeString("\"hello world\""),
            std::vector<std::string>({"hello world"}));
}

TEST(TokenizeTest, BVA_EmptyQuotedTokenProducesEmptyToken) {
  EXPECT_EQ(TokenizeString("\"\""), std::vector<std::string>({""}));
}

TEST(TokenizeTest, BVA_SingleCharacterQuotedToken) {
  EXPECT_EQ(TokenizeString("\"a\""), std::vector<std::string>({"a"}));
}

TEST(TokenizeTest, Edge_QuotedTokenCanContainDelimiters) {
  EXPECT_EQ(TokenizeString("\"{a:b}\""),
            std::vector<std::string>({"{a:b}"}));
}

TEST(TokenizeTest, Edge_QuotedTokenCanContainWhitespace) {
  EXPECT_EQ(TokenizeString("\" a b \t c \""),
            std::vector<std::string>({" a b \t c "}));
}

TEST(TokenizeTest, Edge_QuotedTokenFollowedByNormalToken) {
  EXPECT_EQ(TokenizeString("\"abc\"def"),
            std::vector<std::string>({"abc", "def"}));
}

TEST(TokenizeTest, Edge_NormalTokenFollowedByQuotedToken) {
  EXPECT_EQ(TokenizeString("abc\"def\""),
            std::vector<std::string>({"abc", "def"}));
}

TEST(TokenizeTest, Edge_NormalTokenDelimiterQuotedToken) {
  EXPECT_EQ(TokenizeString("key:\"value with spaces\""),
            std::vector<std::string>({"key", ":", "value with spaces"}));
}

TEST(TokenizeTest, Invalid_UnclosedQuotedTokenIssuedAtEnd) {
  EXPECT_EQ(TokenizeString("\"abc def"),
            std::vector<std::string>({"abc def"}));
}

TEST(TokenizeTest, Invalid_UnclosedNormalTokenIssuedAtEnd) {
  EXPECT_EQ(TokenizeString("abc"),
            std::vector<std::string>({"abc"}));
}

TEST(TokenizeTest, Edge_QuoteStartsNewTokenAfterWhitespace) {
  EXPECT_EQ(TokenizeString("abc \"def\" ghi"),
            std::vector<std::string>({"abc", "def", "ghi"}));
}

TEST(TokenizeTest, Edge_QuoteInsideTokenEndsCurrentAndStartsQuotedToken) {
  EXPECT_EQ(TokenizeString("abc\"def ghi\"jkl"),
            std::vector<std::string>({"abc", "def ghi", "jkl"}));
}

TEST(TokenizeTest, Edge_AdjacentQuotedTokensProduceSeparateTokens) {
  EXPECT_EQ(TokenizeString("\"abc\"\"def\""),
            std::vector<std::string>({"abc", "def"}));
}

TEST(TokenizeTest, Edge_EmptyQuotedTokenBetweenTokensIsPreserved) {
  EXPECT_EQ(TokenizeString("a\"\"b"),
            std::vector<std::string>({"a", "", "b"}));
}

TEST(TokenizeTest, Edge_NewlinesAndTabsSeparateTokens) {
  EXPECT_EQ(TokenizeString("a\nb\tc\rd"),
            std::vector<std::string>({"a", "b", "c", "d"}));
}

TEST(TokenizeTest, Edge_LongTokenProducedCorrectly) {
  std::string long_token(4096, 'x');

  EXPECT_EQ(TokenizeString(long_token),
            std::vector<std::string>({long_token}));
}

TEST(TokenizeTest, Edge_LongQuotedTokenProducedCorrectly) {
  std::string long_token(4096, 'q');

  EXPECT_EQ(TokenizeString("\"" + long_token + "\""),
            std::vector<std::string>({long_token}));
}

TEST(TokenizeTest, Edge_ComplexJsonLikeInputTokenizedCorrectly) {
  EXPECT_EQ(TokenizeString("{ name: \"model name\" version: 1 }"),
            std::vector<std::string>(
                {"{", "name", ":", "model name", "version", ":", "1", "}"}));
}

TEST(TokenizeTest, Edge_ProcessorReceivesStableTokenCopies) {
  std::istringstream stream("a b c");
  CollectingTokenProcessor processor;

  Tokenize(&stream, &processor);

  ASSERT_EQ(processor.tokens.size(), 3u);
  EXPECT_EQ(processor.tokens[0], "a");
  EXPECT_EQ(processor.tokens[1], "b");
  EXPECT_EQ(processor.tokens[2], "c");
}

}  // namespace
}  // namespace testing
}  // namespace tflite