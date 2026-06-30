#include "arolla/util/switch_index.h"

#include <gtest/gtest.h>

#include <type_traits>
#include <vector>

namespace arolla {
namespace {

struct ReturnIntegralConstantValue {
  template <int I>
  int operator()(std::integral_constant<int, I>) const {
    return I;
  }
};

struct ReturnIntegralConstantTypeProperty {
  template <int I>
  bool operator()(std::integral_constant<int, I>) const {
    return std::is_same_v<decltype(std::integral_constant<int, I>{}),
                          std::integral_constant<int, I>>;
  }
};

TEST(SwitchIndex32Test, ReturnsZeroForLowerBoundary) {
  EXPECT_EQ(switch_index_32(0, ReturnIntegralConstantValue{}), 0);
}

TEST(SwitchIndex32Test, ReturnsThirtyOneForUpperBoundary) {
  EXPECT_EQ(switch_index_32(31, ReturnIntegralConstantValue{}), 31);
}

TEST(SwitchIndex32Test, ReturnsExpectedValuesForRepresentativeIndices) {
  EXPECT_EQ(switch_index_32(1, ReturnIntegralConstantValue{}), 1);
  EXPECT_EQ(switch_index_32(15, ReturnIntegralConstantValue{}), 15);
  EXPECT_EQ(switch_index_32(16, ReturnIntegralConstantValue{}), 16);
  EXPECT_EQ(switch_index_32(30, ReturnIntegralConstantValue{}), 30);
}

TEST(SwitchIndex32Test, CoversAllValidIndices) {
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(switch_index_32(i, ReturnIntegralConstantValue{}), i);
  }
}

TEST(SwitchIndex32Test, CallbackReceivesIntegralConstantType) {
  for (int i = 0; i < 32; ++i) {
    EXPECT_TRUE(switch_index_32(i, ReturnIntegralConstantTypeProperty{}));
  }
}

TEST(SwitchIndex64Test, ReturnsZeroForLowerBoundary) {
  EXPECT_EQ(switch_index_64(0, ReturnIntegralConstantValue{}), 0);
}

TEST(SwitchIndex64Test, ReturnsSixtyThreeForUpperBoundary) {
  EXPECT_EQ(switch_index_64(63, ReturnIntegralConstantValue{}), 63);
}

TEST(SwitchIndex64Test, ReturnsExpectedValuesForRepresentativeIndices) {
  EXPECT_EQ(switch_index_64(1, ReturnIntegralConstantValue{}), 1);
  EXPECT_EQ(switch_index_64(15, ReturnIntegralConstantValue{}), 15);
  EXPECT_EQ(switch_index_64(16, ReturnIntegralConstantValue{}), 16);
  EXPECT_EQ(switch_index_64(31, ReturnIntegralConstantValue{}), 31);
  EXPECT_EQ(switch_index_64(32, ReturnIntegralConstantValue{}), 32);
  EXPECT_EQ(switch_index_64(47, ReturnIntegralConstantValue{}), 47);
  EXPECT_EQ(switch_index_64(48, ReturnIntegralConstantValue{}), 48);
  EXPECT_EQ(switch_index_64(62, ReturnIntegralConstantValue{}), 62);
}

TEST(SwitchIndex64Test, CoversAllValidIndices) {
  for (int i = 0; i < 64; ++i) {
    EXPECT_EQ(switch_index_64(i, ReturnIntegralConstantValue{}), i);
  }
}

TEST(SwitchIndex64Test, CallbackReceivesIntegralConstantType) {
  for (int i = 0; i < 64; ++i) {
    EXPECT_TRUE(switch_index_64(i, ReturnIntegralConstantTypeProperty{}));
  }
}

TEST(SwitchIndexTemplateTest, DispatchesTo32Case) {
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ((switch_index<32>(i, ReturnIntegralConstantValue{})), i);
  }
}

TEST(SwitchIndexTemplateTest, DispatchesTo64Case) {
  for (int i = 0; i < 64; ++i) {
    EXPECT_EQ((switch_index<64>(i, ReturnIntegralConstantValue{})), i);
  }
}

TEST(SwitchIndexTest, MutableCallbackCanRecordVisitedIndex32) {
  std::vector<int> visited;

  int result = switch_index_32(7, [&visited](auto index) {
    visited.push_back(decltype(index)::value);
    return decltype(index)::value * 2;
  });

  ASSERT_EQ(visited.size(), 1u);
  EXPECT_EQ(visited[0], 7);
  EXPECT_EQ(result, 14);
}

TEST(SwitchIndexTest, MutableCallbackCanRecordVisitedIndex64) {
  std::vector<int> visited;

  int result = switch_index_64(42, [&visited](auto index) {
    visited.push_back(decltype(index)::value);
    return decltype(index)::value + 1;
  });

  ASSERT_EQ(visited.size(), 1u);
  EXPECT_EQ(visited[0], 42);
  EXPECT_EQ(result, 43);
}

TEST(SwitchIndexDeathTest, SwitchIndex32RejectsNegativeIndex) {
  EXPECT_DEATH(
      {
        volatile int result =
            switch_index_32(-1, ReturnIntegralConstantValue{});
        (void)result;
      },
      "0 <= n");
}

TEST(SwitchIndexDeathTest, SwitchIndex32RejectsIndexEqualTo32) {
  EXPECT_DEATH(
      {
        volatile int result =
            switch_index_32(32, ReturnIntegralConstantValue{});
        (void)result;
      },
      "n < 32");
}

TEST(SwitchIndexDeathTest, SwitchIndex64RejectsNegativeIndex) {
  EXPECT_DEATH(
      {
        volatile int result =
            switch_index_64(-1, ReturnIntegralConstantValue{});
        (void)result;
      },
      "0 <= n");
}

TEST(SwitchIndexDeathTest, SwitchIndex64RejectsIndexEqualTo64) {
  EXPECT_DEATH(
      {
        volatile int result =
            switch_index_64(64, ReturnIntegralConstantValue{});
        (void)result;
      },
      "n < 64");
}

}  // namespace
}  // namespace arolla