#include "tensorstore/internal/box_difference.h"

#include <gtest/gtest.h>

#include <vector>

#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"

namespace tensorstore {
namespace internal {
namespace {

using ::tensorstore::Box;
using ::tensorstore::Index;
using ::tensorstore::IndexInterval;

std::vector<Box<>> CollectSubBoxes(BoxView<> outer, BoxView<> inner) {
  BoxDifference diff(outer, inner);

  std::vector<Box<>> result;
  for (Index i = 0; i < diff.size(); ++i) {
    Box<> out(outer.rank());
    diff.GetSubBox(i, out);
    result.push_back(out);
  }
  return result;
}

bool ContainsBox(const std::vector<Box<>>& boxes, BoxView<> expected) {
  for (const auto& box : boxes) {
    if (box == expected) {
      return true;
    }
  }
  return false;
}

TEST(BoxDifferenceTest, ECP_OneDimInnerFullyInsideOuterProducesBeforeAndAfter) {
  Box<> outer({IndexInterval::UncheckedHalfOpen(0, 10)});
  Box<> inner({IndexInterval::UncheckedHalfOpen(3, 7)});

  auto boxes = CollectSubBoxes(outer, inner);

  ASSERT_EQ(boxes.size(), 2);
  EXPECT_TRUE(
      ContainsBox(boxes, Box<>({IndexInterval::UncheckedHalfOpen(0, 3)})));
  EXPECT_TRUE(
      ContainsBox(boxes, Box<>({IndexInterval::UncheckedHalfOpen(7, 10)})));
}

TEST(BoxDifferenceTest, BVA_InnerTouchesOuterMinProducesOnlyAfterBox) {
  Box<> outer({IndexInterval::UncheckedHalfOpen(0, 10)});
  Box<> inner({IndexInterval::UncheckedHalfOpen(0, 4)});

  auto boxes = CollectSubBoxes(outer, inner);

  ASSERT_EQ(boxes.size(), 1);
  EXPECT_EQ(boxes[0], Box<>({IndexInterval::UncheckedHalfOpen(4, 10)}));
}

TEST(BoxDifferenceTest, BVA_InnerTouchesOuterMaxProducesOnlyBeforeBox) {
  Box<> outer({IndexInterval::UncheckedHalfOpen(0, 10)});
  Box<> inner({IndexInterval::UncheckedHalfOpen(6, 10)});

  auto boxes = CollectSubBoxes(outer, inner);

  ASSERT_EQ(boxes.size(), 1);
  EXPECT_EQ(boxes[0], Box<>({IndexInterval::UncheckedHalfOpen(0, 6)}));
}

TEST(BoxDifferenceTest, Edge_InnerEqualOuterProducesNoSubBoxes) {
  Box<> outer({IndexInterval::UncheckedHalfOpen(0, 10)});
  Box<> inner({IndexInterval::UncheckedHalfOpen(0, 10)});

  BoxDifference diff(outer, inner);

  EXPECT_EQ(diff.size(), 0);
}

TEST(BoxDifferenceTest, InvalidEquivalent_InnerCompletelyBeforeOuterReturnsOuter) {
  Box<> outer({IndexInterval::UncheckedHalfOpen(10, 20)});
  Box<> inner({IndexInterval::UncheckedHalfOpen(0, 5)});

  auto boxes = CollectSubBoxes(outer, inner);

  ASSERT_EQ(boxes.size(), 1);
  EXPECT_EQ(boxes[0], outer);
}

TEST(BoxDifferenceTest, InvalidEquivalent_InnerCompletelyAfterOuterReturnsOuter) {
  Box<> outer({IndexInterval::UncheckedHalfOpen(10, 20)});
  Box<> inner({IndexInterval::UncheckedHalfOpen(25, 30)});

  auto boxes = CollectSubBoxes(outer, inner);

  ASSERT_EQ(boxes.size(), 1);
  EXPECT_EQ(boxes[0], outer);
}

TEST(BoxDifferenceTest, BVA_InnerOverlapsOnlyOuterMinBoundary) {
  Box<> outer({IndexInterval::UncheckedHalfOpen(10, 20)});
  Box<> inner({IndexInterval::UncheckedHalfOpen(5, 11)});

  auto boxes = CollectSubBoxes(outer, inner);

  ASSERT_EQ(boxes.size(), 1);
  EXPECT_EQ(boxes[0], Box<>({IndexInterval::UncheckedHalfOpen(11, 20)}));
}

TEST(BoxDifferenceTest, BVA_InnerOverlapsOnlyOuterMaxBoundary) {
  Box<> outer({IndexInterval::UncheckedHalfOpen(10, 20)});
  Box<> inner({IndexInterval::UncheckedHalfOpen(19, 30)});

  auto boxes = CollectSubBoxes(outer, inner);

  ASSERT_EQ(boxes.size(), 1);
  EXPECT_EQ(boxes[0], Box<>({IndexInterval::UncheckedHalfOpen(10, 19)}));
}

TEST(BoxDifferenceTest, ECP_InnerCoversOuterCompletelyProducesNoSubBoxes) {
  Box<> outer({IndexInterval::UncheckedHalfOpen(10, 20)});
  Box<> inner({IndexInterval::UncheckedHalfOpen(0, 30)});

  BoxDifference diff(outer, inner);

  EXPECT_EQ(diff.size(), 0);
}

TEST(BoxDifferenceTest, Edge_EmptyInnerDoesNotIntersectAndReturnsOuter) {
  Box<> outer({IndexInterval::UncheckedHalfOpen(0, 10)});
  Box<> inner({IndexInterval::UncheckedHalfOpen(5, 5)});

  auto boxes = CollectSubBoxes(outer, inner);

  ASSERT_EQ(boxes.size(), 1);
  EXPECT_EQ(boxes[0], outer);
}

TEST(BoxDifferenceTest, Edge_EmptyOuterProducesNoSubBoxes) {
  Box<> outer({IndexInterval::UncheckedHalfOpen(0, 0)});
  Box<> inner({IndexInterval::UncheckedHalfOpen(0, 10)});

  BoxDifference diff(outer, inner);

  EXPECT_EQ(diff.size(), 0);
}

TEST(BoxDifferenceTest, ECP_TwoDimInnerFullyInsideProducesEightSubBoxes) {
  Box<> outer({
      IndexInterval::UncheckedHalfOpen(0, 10),
      IndexInterval::UncheckedHalfOpen(0, 10),
  });
  Box<> inner({
      IndexInterval::UncheckedHalfOpen(3, 7),
      IndexInterval::UncheckedHalfOpen(2, 8),
  });

  auto boxes = CollectSubBoxes(outer, inner);

  ASSERT_EQ(boxes.size(), 8);

  EXPECT_TRUE(ContainsBox(
      boxes, Box<>({IndexInterval::UncheckedHalfOpen(0, 3),
                    IndexInterval::UncheckedHalfOpen(2, 8)})));

  EXPECT_TRUE(ContainsBox(
      boxes, Box<>({IndexInterval::UncheckedHalfOpen(7, 10),
                    IndexInterval::UncheckedHalfOpen(2, 8)})));

  EXPECT_TRUE(ContainsBox(
      boxes, Box<>({IndexInterval::UncheckedHalfOpen(3, 7),
                    IndexInterval::UncheckedHalfOpen(0, 2)})));

  EXPECT_TRUE(ContainsBox(
      boxes, Box<>({IndexInterval::UncheckedHalfOpen(3, 7),
                    IndexInterval::UncheckedHalfOpen(8, 10)})));
}

TEST(BoxDifferenceTest, ECP_TwoDimNoIntersectionInOneDimensionReturnsOuter) {
  Box<> outer({
      IndexInterval::UncheckedHalfOpen(0, 10),
      IndexInterval::UncheckedHalfOpen(0, 10),
  });
  Box<> inner({
      IndexInterval::UncheckedHalfOpen(3, 7),
      IndexInterval::UncheckedHalfOpen(20, 30),
  });

  auto boxes = CollectSubBoxes(outer, inner);

  ASSERT_EQ(boxes.size(), 1);
  EXPECT_EQ(boxes[0], outer);
}

TEST(BoxDifferenceTest, BVA_TwoDimInnerTouchesMinAndMaxProducesOnlyOneSubBox) {
  Box<> outer({
      IndexInterval::UncheckedHalfOpen(0, 10),
      IndexInterval::UncheckedHalfOpen(0, 10),
  });
  Box<> inner({
      IndexInterval::UncheckedHalfOpen(0, 5),
      IndexInterval::UncheckedHalfOpen(0, 10),
  });

  auto boxes = CollectSubBoxes(outer, inner);

  ASSERT_EQ(boxes.size(), 1);
  EXPECT_EQ(boxes[0],
            Box<>({IndexInterval::UncheckedHalfOpen(5, 10),
                   IndexInterval::UncheckedHalfOpen(0, 10)}));
}

TEST(BoxDifferenceTest, Edge_RankZeroOuterAndInnerProduceNoSubBoxes) {
  Box<> outer(0);
  Box<> inner(0);

  BoxDifference diff(outer, inner);

  EXPECT_EQ(diff.size(), 0);
}

TEST(BoxDifferenceDeathTest, InvalidSubBoxIndexNegativeTriggersAssert) {
  Box<> outer({IndexInterval::UncheckedHalfOpen(0, 10)});
  Box<> inner({IndexInterval::UncheckedHalfOpen(3, 7)});
  Box<> out(1);

  BoxDifference diff(outer, inner);

#ifndef NDEBUG
  EXPECT_DEATH(diff.GetSubBox(-1, out), ".*");
#endif
}

TEST(BoxDifferenceDeathTest, InvalidSubBoxIndexEqualToSizeTriggersAssert) {
  Box<> outer({IndexInterval::UncheckedHalfOpen(0, 10)});
  Box<> inner({IndexInterval::UncheckedHalfOpen(3, 7)});
  Box<> out(1);

  BoxDifference diff(outer, inner);

#ifndef NDEBUG
  EXPECT_DEATH(diff.GetSubBox(diff.size(), out), ".*");
#endif
}

TEST(BoxDifferenceDeathTest, InvalidOutputRankTriggersAssert) {
  Box<> outer({
      IndexInterval::UncheckedHalfOpen(0, 10),
      IndexInterval::UncheckedHalfOpen(0, 10),
  });
  Box<> inner({
      IndexInterval::UncheckedHalfOpen(3, 7),
      IndexInterval::UncheckedHalfOpen(3, 7),
  });
  Box<> out(1);

  BoxDifference diff(outer, inner);

#ifndef NDEBUG
  EXPECT_DEATH(diff.GetSubBox(0, out), ".*");
#endif
}

}  // namespace
}  // namespace internal
}  // namespace tensorstore