#include "arolla/expr/expr_node.h"

#include <gtest/gtest.h>

#include <sstream>
#include <string>

namespace arolla::expr {
namespace {

TEST(ExprNodeTypeStreamTest, PrintsLiteralType) {
  std::ostringstream os;

  os << ExprNodeType::kLiteral;

  EXPECT_EQ(os.str(), "kLiteral");
}

TEST(ExprNodeTypeStreamTest, PrintsLeafType) {
  std::ostringstream os;

  os << ExprNodeType::kLeaf;

  EXPECT_EQ(os.str(), "kLeaf");
}

TEST(ExprNodeTypeStreamTest, PrintsOperatorType) {
  std::ostringstream os;

  os << ExprNodeType::kOperator;

  EXPECT_EQ(os.str(), "kOperator");
}

TEST(ExprNodeTypeStreamTest, PrintsPlaceholderType) {
  std::ostringstream os;

  os << ExprNodeType::kPlaceholder;

  EXPECT_EQ(os.str(), "kPlaceholder");
}

TEST(ExprNodeTypeStreamTest, PrintsInvalidNegativeEnumValue) {
  std::ostringstream os;

  os << static_cast<ExprNodeType>(-1);

  EXPECT_EQ(os.str(), "ExprNodeType(-1)");
}

TEST(ExprNodeTypeStreamTest, PrintsInvalidPositiveEnumValue) {
  std::ostringstream os;

  os << static_cast<ExprNodeType>(100);

  EXPECT_EQ(os.str(), "ExprNodeType(100)");
}

TEST(ExprNodeLeafTest, EmptyLeafKeyCreatesLeafNode) {
  ExprNodePtr node = ExprNode::MakeLeafNode("");

  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->type(), ExprNodeType::kLeaf);
  EXPECT_EQ(node->leaf_key(), "");
}

TEST(ExprNodeLeafTest, SingleCharacterLeafKeyCreatesLeafNode) {
  ExprNodePtr node = ExprNode::MakeLeafNode("x");

  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->type(), ExprNodeType::kLeaf);
  EXPECT_EQ(node->leaf_key(), "x");
}

TEST(ExprNodeLeafTest, NormalLeafKeyCreatesLeafNode) {
  ExprNodePtr node = ExprNode::MakeLeafNode("input_feature");

  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->type(), ExprNodeType::kLeaf);
  EXPECT_EQ(node->leaf_key(), "input_feature");
}

TEST(ExprNodeLeafTest, LongLeafKeyCreatesLeafNode) {
  const std::string long_key(4096, 'a');

  ExprNodePtr node = ExprNode::MakeLeafNode(long_key);

  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->type(), ExprNodeType::kLeaf);
  EXPECT_EQ(node->leaf_key(), long_key);
}

TEST(ExprNodeLeafTest, LeafKeyWithSpecialCharactersIsPreserved) {
  const std::string key = "feature.name_1:/with-symbols";

  ExprNodePtr node = ExprNode::MakeLeafNode(key);

  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->type(), ExprNodeType::kLeaf);
  EXPECT_EQ(node->leaf_key(), key);
}

TEST(ExprNodeLeafTest, SameLeafKeyProducesSameFingerprint) {
  ExprNodePtr node1 = ExprNode::MakeLeafNode("x");
  ExprNodePtr node2 = ExprNode::MakeLeafNode("x");

  ASSERT_NE(node1, nullptr);
  ASSERT_NE(node2, nullptr);
  EXPECT_EQ(node1->fingerprint(), node2->fingerprint());
}

TEST(ExprNodeLeafTest, DifferentLeafKeysProduceDifferentFingerprints) {
  ExprNodePtr node1 = ExprNode::MakeLeafNode("x");
  ExprNodePtr node2 = ExprNode::MakeLeafNode("y");

  ASSERT_NE(node1, nullptr);
  ASSERT_NE(node2, nullptr);
  EXPECT_NE(node1->fingerprint(), node2->fingerprint());
}

TEST(ExprNodePlaceholderTest, EmptyPlaceholderKeyCreatesPlaceholderNode) {
  ExprNodePtr node = ExprNode::MakePlaceholderNode("");

  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->type(), ExprNodeType::kPlaceholder);
  EXPECT_EQ(node->placeholder_key(), "");
}

TEST(ExprNodePlaceholderTest, SingleCharacterPlaceholderKeyCreatesPlaceholderNode) {
  ExprNodePtr node = ExprNode::MakePlaceholderNode("p");

  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->type(), ExprNodeType::kPlaceholder);
  EXPECT_EQ(node->placeholder_key(), "p");
}

TEST(ExprNodePlaceholderTest, NormalPlaceholderKeyCreatesPlaceholderNode) {
  ExprNodePtr node = ExprNode::MakePlaceholderNode("placeholder_input");

  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->type(), ExprNodeType::kPlaceholder);
  EXPECT_EQ(node->placeholder_key(), "placeholder_input");
}

TEST(ExprNodePlaceholderTest, LongPlaceholderKeyCreatesPlaceholderNode) {
  const std::string long_key(4096, 'p');

  ExprNodePtr node = ExprNode::MakePlaceholderNode(long_key);

  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->type(), ExprNodeType::kPlaceholder);
  EXPECT_EQ(node->placeholder_key(), long_key);
}

TEST(ExprNodePlaceholderTest, PlaceholderKeyWithSpecialCharactersIsPreserved) {
  const std::string key = "$placeholder.name_1:/with-symbols";

  ExprNodePtr node = ExprNode::MakePlaceholderNode(key);

  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->type(), ExprNodeType::kPlaceholder);
  EXPECT_EQ(node->placeholder_key(), key);
}

TEST(ExprNodePlaceholderTest, SamePlaceholderKeyProducesSameFingerprint) {
  ExprNodePtr node1 = ExprNode::MakePlaceholderNode("p");
  ExprNodePtr node2 = ExprNode::MakePlaceholderNode("p");

  ASSERT_NE(node1, nullptr);
  ASSERT_NE(node2, nullptr);
  EXPECT_EQ(node1->fingerprint(), node2->fingerprint());
}

TEST(ExprNodePlaceholderTest, DifferentPlaceholderKeysProduceDifferentFingerprints) {
  ExprNodePtr node1 = ExprNode::MakePlaceholderNode("p1");
  ExprNodePtr node2 = ExprNode::MakePlaceholderNode("p2");

  ASSERT_NE(node1, nullptr);
  ASSERT_NE(node2, nullptr);
  EXPECT_NE(node1->fingerprint(), node2->fingerprint());
}

TEST(ExprNodeFingerprintTest, LeafAndPlaceholderWithSameKeyHaveDifferentFingerprints) {
  ExprNodePtr leaf = ExprNode::MakeLeafNode("x");
  ExprNodePtr placeholder = ExprNode::MakePlaceholderNode("x");

  ASSERT_NE(leaf, nullptr);
  ASSERT_NE(placeholder, nullptr);
  EXPECT_NE(leaf->fingerprint(), placeholder->fingerprint());
}

TEST(ExprNodeDestructorTest, DestroyLeafNodeWithoutDependenciesDoesNotCrash) {
  ExprNodePtr node = ExprNode::MakeLeafNode("x");

  node = nullptr;

  SUCCEED();
}

TEST(ExprNodeDestructorTest, DestroyPlaceholderNodeWithoutDependenciesDoesNotCrash) {
  ExprNodePtr node = ExprNode::MakePlaceholderNode("p");

  node = nullptr;

  SUCCEED();
}

}  // namespace
}  // namespace arolla::expr