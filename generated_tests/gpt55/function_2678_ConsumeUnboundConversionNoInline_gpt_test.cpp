#include "absl/strings/internal/str_format/parser.h"

#include <gtest/gtest.h>

#include <string>

namespace absl {
ABSL_NAMESPACE_BEGIN
namespace str_format_internal {
namespace {

TEST(StrFormatParserTest, ECP_LengthModToStringKnownValues) {
  EXPECT_EQ(LengthModToString(LengthMod::h), "h");
  EXPECT_EQ(LengthModToString(LengthMod::hh), "hh");
  EXPECT_EQ(LengthModToString(LengthMod::l), "l");
  EXPECT_EQ(LengthModToString(LengthMod::ll), "ll");
  EXPECT_EQ(LengthModToString(LengthMod::L), "L");
  EXPECT_EQ(LengthModToString(LengthMod::j), "j");
  EXPECT_EQ(LengthModToString(LengthMod::z), "z");
  EXPECT_EQ(LengthModToString(LengthMod::t), "t");
  EXPECT_EQ(LengthModToString(LengthMod::q), "q");
  EXPECT_EQ(LengthModToString(LengthMod::none), "");
}

TEST(StrFormatParserTest, BVA_LengthModToStringInvalidValueReturnsEmptyString) {
  EXPECT_EQ(LengthModToString(static_cast<LengthMod>(-1)), "");
  EXPECT_EQ(LengthModToString(static_cast<LengthMod>(999)), "");
}

TEST(StrFormatParserTest, ECP_ConsumeUnboundConversionParsesIntegerConversion) {
  const char* begin = "%d";
  const char* end = begin + 2;
  UnboundConversion conv;
  int next_arg = 0;

  const char* result =
      ConsumeUnboundConversionNoInline(begin + 1, end, &conv, &next_arg);

  ASSERT_EQ(result, end);
  EXPECT_EQ(conv.arg_position, 1);
  EXPECT_EQ(FormatConversionCharToChar(conv.conv), 'd');
}

TEST(StrFormatParserTest, ECP_ConsumeUnboundConversionParsesStringConversion) {
  const char* begin = "%s";
  const char* end = begin + 2;
  UnboundConversion conv;
  int next_arg = 0;

  const char* result =
      ConsumeUnboundConversionNoInline(begin + 1, end, &conv, &next_arg);

  ASSERT_EQ(result, end);
  EXPECT_EQ(conv.arg_position, 1);
  EXPECT_EQ(FormatConversionCharToChar(conv.conv), 's');
}

TEST(StrFormatParserTest, ECP_ConsumeUnboundConversionParsesFloatConversion) {
  const char* begin = "%f";
  const char* end = begin + 2;
  UnboundConversion conv;
  int next_arg = 0;

  const char* result =
      ConsumeUnboundConversionNoInline(begin + 1, end, &conv, &next_arg);

  ASSERT_EQ(result, end);
  EXPECT_EQ(conv.arg_position, 1);
  EXPECT_EQ(FormatConversionCharToChar(conv.conv), 'f');
}

TEST(StrFormatParserTest, ECP_ConsumeUnboundConversionParsesExplicitArgPosition) {
  const char* begin = "%2$d";
  const char* end = begin + 4;
  UnboundConversion conv;
  int next_arg = 0;

  const char* result =
      ConsumeUnboundConversionNoInline(begin + 1, end, &conv, &next_arg);

  ASSERT_EQ(result, end);
  EXPECT_EQ(conv.arg_position, 2);
  EXPECT_EQ(FormatConversionCharToChar(conv.conv), 'd');
}

TEST(StrFormatParserTest, BVA_ConsumeUnboundConversionParsesFirstArgPosition) {
  const char* begin = "%1$s";
  const char* end = begin + 4;
  UnboundConversion conv;
  int next_arg = 0;

  const char* result =
      ConsumeUnboundConversionNoInline(begin + 1, end, &conv, &next_arg);

  ASSERT_EQ(result, end);
  EXPECT_EQ(conv.arg_position, 1);
  EXPECT_EQ(FormatConversionCharToChar(conv.conv), 's');
}

TEST(StrFormatParserTest, ECP_ConsumeUnboundConversionParsesWidthFromArg) {
  const char* begin = "%*d";
  const char* end = begin + 3;
  UnboundConversion conv;
  int next_arg = 0;

  const char* result =
      ConsumeUnboundConversionNoInline(begin + 1, end, &conv, &next_arg);

  ASSERT_EQ(result, end);
  EXPECT_TRUE(conv.width.is_from_arg());
  EXPECT_EQ(conv.width.get_from_arg(), 1);
  EXPECT_EQ(conv.arg_position, 2);
  EXPECT_EQ(FormatConversionCharToChar(conv.conv), 'd');
}

TEST(StrFormatParserTest, ECP_ConsumeUnboundConversionParsesPrecisionFromArg) {
  const char* begin = "%.*f";
  const char* end = begin + 4;
  UnboundConversion conv;
  int next_arg = 0;

  const char* result =
      ConsumeUnboundConversionNoInline(begin + 1, end, &conv, &next_arg);

  ASSERT_EQ(result, end);
  EXPECT_TRUE(conv.precision.is_from_arg());
  EXPECT_EQ(conv.precision.get_from_arg(), 1);
  EXPECT_EQ(conv.arg_position, 2);
  EXPECT_EQ(FormatConversionCharToChar(conv.conv), 'f');
}

TEST(StrFormatParserTest, ECP_ConsumeUnboundConversionParsesWidthAndPrecisionFromArgs) {
  const char* begin = "%*.*f";
  const char* end = begin + 5;
  UnboundConversion conv;
  int next_arg = 0;

  const char* result =
      ConsumeUnboundConversionNoInline(begin + 1, end, &conv, &next_arg);

  ASSERT_EQ(result, end);
  EXPECT_TRUE(conv.width.is_from_arg());
  EXPECT_TRUE(conv.precision.is_from_arg());
  EXPECT_EQ(conv.width.get_from_arg(), 1);
  EXPECT_EQ(conv.precision.get_from_arg(), 2);
  EXPECT_EQ(conv.arg_position, 3);
  EXPECT_EQ(FormatConversionCharToChar(conv.conv), 'f');
}

TEST(StrFormatParserTest, ECP_ConsumeUnboundConversionParsesExplicitWidthAndPrecisionArgs) {
  const char* begin = "%3$*1$.*2$f";
  const char* end = begin + 11;
  UnboundConversion conv;
  int next_arg = 0;

  const char* result =
      ConsumeUnboundConversionNoInline(begin + 1, end, &conv, &next_arg);

  ASSERT_EQ(result, end);
  EXPECT_TRUE(conv.width.is_from_arg());
  EXPECT_TRUE(conv.precision.is_from_arg());
  EXPECT_EQ(conv.width.get_from_arg(), 1);
  EXPECT_EQ(conv.precision.get_from_arg(), 2);
  EXPECT_EQ(conv.arg_position, 3);
  EXPECT_EQ(FormatConversionCharToChar(conv.conv), 'f');
}

TEST(StrFormatParserTest, ECP_ConsumeUnboundConversionParsesLengthModifiers) {
  {
    const char* begin = "%hd";
    const char* end = begin + 3;
    UnboundConversion conv;
    int next_arg = 0;

    const char* result =
        ConsumeUnboundConversionNoInline(begin + 1, end, &conv, &next_arg);

    ASSERT_EQ(result, end);
    EXPECT_EQ(conv.length_mod, LengthMod::h);
    EXPECT_EQ(FormatConversionCharToChar(conv.conv), 'd');
  }

  {
    const char* begin = "%lld";
    const char* end = begin + 4;
    UnboundConversion conv;
    int next_arg = 0;

    const char* result =
        ConsumeUnboundConversionNoInline(begin + 1, end, &conv, &next_arg);

    ASSERT_EQ(result, end);
    EXPECT_EQ(conv.length_mod, LengthMod::ll);
    EXPECT_EQ(FormatConversionCharToChar(conv.conv), 'd');
  }

  {
    const char* begin = "%Lf";
    const char* end = begin + 3;
    UnboundConversion conv;
    int next_arg = 0;

    const char* result =
        ConsumeUnboundConversionNoInline(begin + 1, end, &conv, &next_arg);

    ASSERT_EQ(result, end);
    EXPECT_EQ(conv.length_mod, LengthMod::L);
    EXPECT_EQ(FormatConversionCharToChar(conv.conv), 'f');
  }
}

TEST(StrFormatParserTest, Invalid_ConsumeUnboundConversionWithUnknownConversionFails) {
  const char* begin = "%?";
  const char* end = begin + 2;
  UnboundConversion conv;
  int next_arg = 0;

  const char* result =
      ConsumeUnboundConversionNoInline(begin + 1, end, &conv, &next_arg);

  EXPECT_EQ(result, nullptr);
}

TEST(StrFormatParserTest, Invalid_ConsumeUnboundConversionWithIncompletePercentFails) {
  const char* begin = "%";
  const char* end = begin + 1;
  UnboundConversion conv;
  int next_arg = 0;

  const char* result =
      ConsumeUnboundConversionNoInline(begin + 1, end, &conv, &next_arg);

  EXPECT_EQ(result, nullptr);
}

TEST(StrFormatParserTest, Invalid_ConsumeUnboundConversionWithIncompleteArgPositionFails) {
  const char* begin = "%1$";
  const char* end = begin + 3;
  UnboundConversion conv;
  int next_arg = 0;

  const char* result =
      ConsumeUnboundConversionNoInline(begin + 1, end, &conv, &next_arg);

  EXPECT_EQ(result, nullptr);
}

TEST(StrFormatParserTest, Invalid_ConsumeUnboundConversionWithIncompletePrecisionFails) {
  const char* begin = "%.";
  const char* end = begin + 2;
  UnboundConversion conv;
  int next_arg = 0;

  const char* result =
      ConsumeUnboundConversionNoInline(begin + 1, end, &conv, &next_arg);

  EXPECT_EQ(result, nullptr);
}

TEST(StrFormatParserTest, Invalid_ConsumeUnboundConversionWithIncompleteWidthArgFails) {
  const char* begin = "%*";
  const char* end = begin + 2;
  UnboundConversion conv;
  int next_arg = 0;

  const char* result =
      ConsumeUnboundConversionNoInline(begin + 1, end, &conv, &next_arg);

  EXPECT_EQ(result, nullptr);
}

TEST(StrFormatParserTest, ECP_ParsedFormatEmptyStringHasNoError) {
  ParsedFormatBase parsed("", false, {});

  EXPECT_FALSE(parsed.has_error());
}

TEST(StrFormatParserTest, ECP_ParsedFormatPlainTextHasNoError) {
  ParsedFormatBase parsed("plain text only", false, {});

  EXPECT_FALSE(parsed.has_error());
}

TEST(StrFormatParserTest, ECP_ParsedFormatIntegerConversionMatches) {
  ParsedFormatBase parsed("%d", false, {FormatConversionCharSet::d});

  EXPECT_FALSE(parsed.has_error());
}

TEST(StrFormatParserTest, ECP_ParsedFormatStringConversionMatches) {
  ParsedFormatBase parsed("%s", false, {FormatConversionCharSet::s});

  EXPECT_FALSE(parsed.has_error());
}

TEST(StrFormatParserTest, ECP_ParsedFormatFloatConversionMatches) {
  ParsedFormatBase parsed("%f", false, {FormatConversionCharSet::f});

  EXPECT_FALSE(parsed.has_error());
}

TEST(StrFormatParserTest, ECP_ParsedFormatMultipleConversionsMatchInOrder) {
  ParsedFormatBase parsed("name=%s value=%d ratio=%f", false,
                          {FormatConversionCharSet::s,
                           FormatConversionCharSet::d,
                           FormatConversionCharSet::f});

  EXPECT_FALSE(parsed.has_error());
}

TEST(StrFormatParserTest, ECP_ParsedFormatEscapedPercentDoesNotRequireArgument) {
  ParsedFormatBase parsed("progress=100%%", false, {});

  EXPECT_FALSE(parsed.has_error());
}

TEST(StrFormatParserTest, Invalid_ParsedFormatUnknownConversionHasError) {
  ParsedFormatBase parsed("%?", false, {});

  EXPECT_TRUE(parsed.has_error());
}

TEST(StrFormatParserTest, Invalid_ParsedFormatIncompleteConversionHasError) {
  ParsedFormatBase parsed("%", false, {});

  EXPECT_TRUE(parsed.has_error());
}

TEST(StrFormatParserTest, Invalid_ParsedFormatWrongConversionTypeHasError) {
  ParsedFormatBase parsed("%d", false, {FormatConversionCharSet::s});

  EXPECT_TRUE(parsed.has_error());
}

TEST(StrFormatParserTest, Invalid_ParsedFormatMissingExpectedConversionHasErrorWhenNotIgnored) {
  ParsedFormatBase parsed("%d", false,
                          {FormatConversionCharSet::d,
                           FormatConversionCharSet::s});

  EXPECT_TRUE(parsed.has_error());
}

TEST(StrFormatParserTest, ECP_ParsedFormatAllowsIgnoredExpectedConversionsWhenEnabled) {
  ParsedFormatBase parsed("%d", true,
                          {FormatConversionCharSet::d,
                           FormatConversionCharSet::s});

  EXPECT_FALSE(parsed.has_error());
}

TEST(StrFormatParserTest, Invalid_ParsedFormatTooManyActualConversionsHasError) {
  ParsedFormatBase parsed("%d %s", false, {FormatConversionCharSet::d});

  EXPECT_TRUE(parsed.has_error());
}

TEST(StrFormatParserTest, ECP_ParsedFormatExplicitArgPositionMatchesExpectedType) {
  ParsedFormatBase parsed("%2$s %1$d", false,
                          {FormatConversionCharSet::d,
                           FormatConversionCharSet::s});

  EXPECT_FALSE(parsed.has_error());
}

TEST(StrFormatParserTest, Invalid_ParsedFormatExplicitArgPositionWrongTypeHasError) {
  ParsedFormatBase parsed("%2$d %1$s", false,
                          {FormatConversionCharSet::d,
                           FormatConversionCharSet::s});

  EXPECT_TRUE(parsed.has_error());
}

TEST(StrFormatParserTest, ECP_ParsedFormatWidthFromArgRequiresStarConversion) {
  ParsedFormatBase parsed("%*d", false,
                          {FormatConversionCharSet::Star,
                           FormatConversionCharSet::d});

  EXPECT_FALSE(parsed.has_error());
}

TEST(StrFormatParserTest, ECP_ParsedFormatPrecisionFromArgRequiresStarConversion) {
  ParsedFormatBase parsed("%.*f", false,
                          {FormatConversionCharSet::Star,
                           FormatConversionCharSet::f});

  EXPECT_FALSE(parsed.has_error());
}

TEST(StrFormatParserTest, ECP_ParsedFormatWidthAndPrecisionFromArgsRequireStarConversions) {
  ParsedFormatBase parsed("%*.*f", false,
                          {FormatConversionCharSet::Star,
                           FormatConversionCharSet::Star,
                           FormatConversionCharSet::f});

  EXPECT_FALSE(parsed.has_error());
}

TEST(StrFormatParserTest, Invalid_ParsedFormatWidthFromArgWrongTypeHasError) {
  ParsedFormatBase parsed("%*d", false,
                          {FormatConversionCharSet::d,
                           FormatConversionCharSet::d});

  EXPECT_TRUE(parsed.has_error());
}

TEST(StrFormatParserTest, Invalid_ParsedFormatPrecisionFromArgWrongTypeHasError) {
  ParsedFormatBase parsed("%.*f", false,
                          {FormatConversionCharSet::f,
                           FormatConversionCharSet::f});

  EXPECT_TRUE(parsed.has_error());
}

TEST(StrFormatParserTest, ECP_ParsedFormatExplicitWidthPrecisionAndValuePositionsMatch) {
  ParsedFormatBase parsed("%3$*1$.*2$f", false,
                          {FormatConversionCharSet::Star,
                           FormatConversionCharSet::Star,
                           FormatConversionCharSet::f});

  EXPECT_FALSE(parsed.has_error());
}

TEST(StrFormatParserTest, Invalid_ParsedFormatExplicitWidthPositionOutOfRangeHasError) {
  ParsedFormatBase parsed("%2$*3$d", false,
                          {FormatConversionCharSet::Star,
                           FormatConversionCharSet::d});

  EXPECT_TRUE(parsed.has_error());
}

TEST(StrFormatParserTest, BVA_ParsedFormatSingleCharacterPlainText) {
  ParsedFormatBase parsed("a", false, {});

  EXPECT_FALSE(parsed.has_error());
}

TEST(StrFormatParserTest, BVA_ParsedFormatSingleEscapedPercent) {
  ParsedFormatBase parsed("%%", false, {});

  EXPECT_FALSE(parsed.has_error());
}

TEST(StrFormatParserTest, Edge_ParsedFormatTextBeforeAndAfterConversion) {
  ParsedFormatBase parsed("prefix %d suffix", false,
                          {FormatConversionCharSet::d});

  EXPECT_FALSE(parsed.has_error());
}

TEST(StrFormatParserTest, Edge_ParsedFormatAdjacentConversions) {
  ParsedFormatBase parsed("%d%s%f", false,
                          {FormatConversionCharSet::d,
                           FormatConversionCharSet::s,
                           FormatConversionCharSet::f});

  EXPECT_FALSE(parsed.has_error());
}

TEST(StrFormatParserTest, Edge_ParsedFormatRepeatedSameArgumentPositionCountsOnce) {
  ParsedFormatBase parsed("%1$d %1$d", false, {FormatConversionCharSet::d});

  EXPECT_FALSE(parsed.has_error());
}

TEST(StrFormatParserTest, Edge_ParsedFormatRepeatedSameArgumentWrongTypeFails) {
  ParsedFormatBase parsed("%1$d %1$s", false, {FormatConversionCharSet::d});

  EXPECT_TRUE(parsed.has_error());
}

TEST(StrFormatParserTest, Edge_ParsedFormatLongPlainTextHasNoError) {
  std::string format(4096, 'x');

  ParsedFormatBase parsed(format, false, {});

  EXPECT_FALSE(parsed.has_error());
}

TEST(StrFormatParserTest, Edge_ParsedFormatLongTextWithConversionHasNoError) {
  std::string format(2048, 'x');
  format += "%d";
  format += std::string(2048, 'y');

  ParsedFormatBase parsed(format, false, {FormatConversionCharSet::d});

  EXPECT_FALSE(parsed.has_error());
}

}  // namespace
}  // namespace str_format_internal
ABSL_NAMESPACE_END
}  // namespace absl