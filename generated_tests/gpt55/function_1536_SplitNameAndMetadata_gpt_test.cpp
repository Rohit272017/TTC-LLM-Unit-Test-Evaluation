#include "xla/tsl/profiler/utils/parse_annotation.h"

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

namespace tsl {
namespace profiler {
namespace {

TEST(ParseAnnotationTest, ECP_EmptyAnnotationReturnsEmptyResult) {
  Annotation result = ParseAnnotation("");

  EXPECT_TRUE(result.name.empty());
  EXPECT_TRUE(result.metadata.empty());
}

TEST(ParseAnnotationTest, ECP_NameOnlyWithoutMetadata) {
  Annotation result = ParseAnnotation("MatMul");

  EXPECT_EQ(result.name, "MatMul");
  EXPECT_TRUE(result.metadata.empty());
}

TEST(ParseAnnotationTest, BVA_SingleCharacterName) {
  Annotation result = ParseAnnotation("A");

  EXPECT_EQ(result.name, "A");
  EXPECT_TRUE(result.metadata.empty());
}

TEST(ParseAnnotationTest, BVA_NameWithLeadingAndTrailingWhitespace) {
  Annotation result = ParseAnnotation("   Conv2D   ");

  EXPECT_EQ(result.name, "Conv2D");
  EXPECT_TRUE(result.metadata.empty());
}

TEST(ParseAnnotationTest, ECP_SingleMetadataPair) {
  Annotation result = ParseAnnotation("Op#key=value#");

  ASSERT_EQ(result.metadata.size(), 1u);
  EXPECT_EQ(result.name, "Op");
  EXPECT_EQ(result.metadata[0].first, "key");
  EXPECT_EQ(result.metadata[0].second, "value");
}

TEST(ParseAnnotationTest, ECP_MultipleMetadataPairs) {
  Annotation result =
      ParseAnnotation("Op#key1=value1,key2=value2,key3=value3#");

  ASSERT_EQ(result.metadata.size(), 3u);

  EXPECT_EQ(result.metadata[0].first, "key1");
  EXPECT_EQ(result.metadata[0].second, "value1");

  EXPECT_EQ(result.metadata[1].first, "key2");
  EXPECT_EQ(result.metadata[1].second, "value2");

  EXPECT_EQ(result.metadata[2].first, "key3");
  EXPECT_EQ(result.metadata[2].second, "value3");
}

TEST(ParseAnnotationTest, ECP_MetadataWhitespaceIsTrimmed) {
  Annotation result =
      ParseAnnotation("Op#  key1 = value1  , key2 = value2  #");

  ASSERT_EQ(result.metadata.size(), 2u);

  EXPECT_EQ(result.metadata[0].first, "key1");
  EXPECT_EQ(result.metadata[0].second, "value1");

  EXPECT_EQ(result.metadata[1].first, "key2");
  EXPECT_EQ(result.metadata[1].second, "value2");
}

TEST(ParseAnnotationTest, Invalid_MetadataWithoutEqualsIgnored) {
  Annotation result =
      ParseAnnotation("Op#key1=value1,invalidpair,key2=value2#");

  ASSERT_EQ(result.metadata.size(), 2u);

  EXPECT_EQ(result.metadata[0].first, "key1");
  EXPECT_EQ(result.metadata[0].second, "value1");

  EXPECT_EQ(result.metadata[1].first, "key2");
  EXPECT_EQ(result.metadata[1].second, "value2");
}

TEST(ParseAnnotationTest, Invalid_EmptyKeyIgnored) {
  Annotation result = ParseAnnotation("Op#=value,key2=value2#");

  ASSERT_EQ(result.metadata.size(), 1u);
  EXPECT_EQ(result.metadata[0].first, "key2");
  EXPECT_EQ(result.metadata[0].second, "value2");
}

TEST(ParseAnnotationTest, Invalid_EmptyValueIgnored) {
  Annotation result = ParseAnnotation("Op#key1=,key2=value2#");

  ASSERT_EQ(result.metadata.size(), 1u);
  EXPECT_EQ(result.metadata[0].first, "key2");
  EXPECT_EQ(result.metadata[0].second, "value2");
}

TEST(ParseAnnotationTest, Invalid_OnlyMetadataDelimiterProducesNameOnly) {
  Annotation result = ParseAnnotation("Op##");

  EXPECT_EQ(result.name, "Op");
  EXPECT_TRUE(result.metadata.empty());
}

TEST(ParseAnnotationTest, Edge_MetadataValueContainsCommaInsideQuotes) {
  Annotation result =
      ParseAnnotation("Op#message=\"a,b,c\",id=123#");

  ASSERT_EQ(result.metadata.size(), 2u);

  EXPECT_EQ(result.metadata[0].first, "message");
  EXPECT_EQ(result.metadata[0].second, "\"a,b,c\"");

  EXPECT_EQ(result.metadata[1].first, "id");
  EXPECT_EQ(result.metadata[1].second, "123");
}

TEST(ParseAnnotationTest, Edge_MetadataValueContainsSingleQuotedComma) {
  Annotation result =
      ParseAnnotation("Op#message='a,b,c',id=123#");

  ASSERT_EQ(result.metadata.size(), 2u);

  EXPECT_EQ(result.metadata[0].first, "message");
  EXPECT_EQ(result.metadata[0].second, "'a,b,c'");

  EXPECT_EQ(result.metadata[1].first, "id");
  EXPECT_EQ(result.metadata[1].second, "123");
}

TEST(ParseAnnotationTest, Edge_MetadataValueContainsBracesAndCommas) {
  Annotation result =
      ParseAnnotation("Op#config={a,b,c},id=1#");

  ASSERT_EQ(result.metadata.size(), 2u);

  EXPECT_EQ(result.metadata[0].first, "config");
  EXPECT_EQ(result.metadata[0].second, "{a,b,c}");

  EXPECT_EQ(result.metadata[1].first, "id");
  EXPECT_EQ(result.metadata[1].second, "1");
}

TEST(ParseAnnotationTest, Edge_MetadataValueContainsParenthesesAndCommas) {
  Annotation result =
      ParseAnnotation("Op#shape=(1,2,3),id=99#");

  ASSERT_EQ(result.metadata.size(), 2u);

  EXPECT_EQ(result.metadata[0].first, "shape");
  EXPECT_EQ(result.metadata[0].second, "(1,2,3)");

  EXPECT_EQ(result.metadata[1].first, "id");
  EXPECT_EQ(result.metadata[1].second, "99");
}

TEST(ParseAnnotationTest, Edge_MetadataValueContainsSquareBracketsAndCommas) {
  Annotation result =
      ParseAnnotation("Op#dims=[1,2,3],id=42#");

  ASSERT_EQ(result.metadata.size(), 2u);

  EXPECT_EQ(result.metadata[0].first, "dims");
  EXPECT_EQ(result.metadata[0].second, "[1,2,3]");

  EXPECT_EQ(result.metadata[1].first, "id");
  EXPECT_EQ(result.metadata[1].second, "42");
}

TEST(ParseAnnotationTest, Edge_ValueContainingEqualsCharacter) {
  Annotation result =
      ParseAnnotation("Op#expr=a=b=c,id=1#");

  ASSERT_EQ(result.metadata.size(), 2u);

  EXPECT_EQ(result.metadata[0].first, "expr");
  EXPECT_EQ(result.metadata[0].second, "a=b=c");

  EXPECT_EQ(result.metadata[1].first, "id");
  EXPECT_EQ(result.metadata[1].second, "1");
}

TEST(ParseAnnotationTest, Edge_AdditionalHashCharactersAfterMetadataIgnored) {
  Annotation result =
      ParseAnnotation("Op#key=value#extra#ignored#");

  EXPECT_EQ(result.name, "Op");

  ASSERT_EQ(result.metadata.size(), 1u);
  EXPECT_EQ(result.metadata[0].first, "key");
  EXPECT_EQ(result.metadata[0].second, "value");
}

TEST(ParseAnnotationTest, Edge_NameWithWhitespaceAndMetadata) {
  Annotation result =
      ParseAnnotation("   MyOperation   #key=value#");

  EXPECT_EQ(result.name, "MyOperation");

  ASSERT_EQ(result.metadata.size(), 1u);
  EXPECT_EQ(result.metadata[0].first, "key");
  EXPECT_EQ(result.metadata[0].second, "value");
}

TEST(ParseAnnotationStackTest, ECP_EmptyStackReturnsEmptyVector) {
  auto result = ParseAnnotationStack("");

  EXPECT_TRUE(result.empty());
}

TEST(ParseAnnotationStackTest, ECP_SingleAnnotationInStack) {
  auto result = ParseAnnotationStack("MatMul");

  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].name, "MatMul");
}

TEST(ParseAnnotationStackTest, ECP_MultipleAnnotationsSeparatedByDelimiter) {
  auto result =
      ParseAnnotationStack("Op1::Op2::Op3");

  ASSERT_EQ(result.size(), 3u);

  EXPECT_EQ(result[0].name, "Op1");
  EXPECT_EQ(result[1].name, "Op2");
  EXPECT_EQ(result[2].name, "Op3");
}

TEST(ParseAnnotationStackTest, ECP_MultipleAnnotationsWithMetadata) {
  auto result =
      ParseAnnotationStack(
          "Op1#k1=v1#::Op2#k2=v2#");

  ASSERT_EQ(result.size(), 2u);

  EXPECT_EQ(result[0].name, "Op1");
  ASSERT_EQ(result[0].metadata.size(), 1u);
  EXPECT_EQ(result[0].metadata[0].first, "k1");
  EXPECT_EQ(result[0].metadata[0].second, "v1");

  EXPECT_EQ(result[1].name, "Op2");
  ASSERT_EQ(result[1].metadata.size(), 1u);
  EXPECT_EQ(result[1].metadata[0].first, "k2");
  EXPECT_EQ(result[1].metadata[0].second, "v2");
}

TEST(ParseAnnotationStackTest, Edge_ConsecutiveDelimitersSkipped) {
  auto result =
      ParseAnnotationStack("Op1::::Op2");

  ASSERT_EQ(result.size(), 2u);

  EXPECT_EQ(result[0].name, "Op1");
  EXPECT_EQ(result[1].name, "Op2");
}

TEST(ParseAnnotationStackTest, Edge_LeadingAndTrailingDelimitersSkipped) {
  auto result =
      ParseAnnotationStack("::Op1::Op2::");

  ASSERT_EQ(result.size(), 2u);

  EXPECT_EQ(result[0].name, "Op1");
  EXPECT_EQ(result[1].name, "Op2");
}

TEST(ParseAnnotationStackTest, BVA_SingleCharacterAnnotations) {
  auto result =
      ParseAnnotationStack("A::B::C");

  ASSERT_EQ(result.size(), 3u);

  EXPECT_EQ(result[0].name, "A");
  EXPECT_EQ(result[1].name, "B");
  EXPECT_EQ(result[2].name, "C");
}

}  // namespace
}  // namespace profiler
}  // namespace tsl