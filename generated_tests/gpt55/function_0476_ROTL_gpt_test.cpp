#include "tensorflow/core/kernels/random_index_shuffle.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <set>
#include <vector>

namespace tensorflow {
namespace random {
namespace {

TEST(IndexShuffleTest, ReturnsValueWithinRangeForSmallestValidRange) {
  const std::array<uint32_t, 3> key = {1, 2, 3};

  const uint64_t result = index_shuffle(
      /*index=*/0,
      key,
      /*max_index=*/1,
      /*rounds=*/4);

  EXPECT_LE(result, 1u);
}

TEST(IndexShuffleTest, HandlesIndexAtZeroBoundary) {
  const std::array<uint32_t, 3> key = {10, 20, 30};

  const uint64_t result = index_shuffle(
      /*index=*/0,
      key,
      /*max_index=*/100,
      /*rounds=*/4);

  EXPECT_LE(result, 100u);
}

TEST(IndexShuffleTest, HandlesIndexAtMaxBoundary) {
  const std::array<uint32_t, 3> key = {10, 20, 30};

  const uint64_t result = index_shuffle(
      /*index=*/100,
      key,
      /*max_index=*/100,
      /*rounds=*/4);

  EXPECT_LE(result, 100u);
}

TEST(IndexShuffleTest, DeterministicForSameInputKeyAndRounds) {
  const std::array<uint32_t, 3> key = {123, 456, 789};

  const uint64_t first = index_shuffle(42, key, 1000, 6);
  const uint64_t second = index_shuffle(42, key, 1000, 6);

  EXPECT_EQ(first, second);
}

TEST(IndexShuffleTest, DifferentKeysUsuallyProduceDifferentResults) {
  const std::array<uint32_t, 3> key1 = {1, 2, 3};
  const std::array<uint32_t, 3> key2 = {4, 5, 6};

  const uint64_t result1 = index_shuffle(42, key1, 1000, 6);
  const uint64_t result2 = index_shuffle(42, key2, 1000, 6);

  EXPECT_NE(result1, result2);
}

TEST(IndexShuffleTest, DifferentRoundsUsuallyProduceDifferentResults) {
  const std::array<uint32_t, 3> key = {1, 2, 3};

  const uint64_t result_round_4 = index_shuffle(42, key, 1000, 4);
  const uint64_t result_round_6 = index_shuffle(42, key, 1000, 6);

  EXPECT_NE(result_round_4, result_round_6);
}

TEST(IndexShuffleTest, MinimumEvenRoundsAccepted) {
  const std::array<uint32_t, 3> key = {7, 8, 9};

  const uint64_t result = index_shuffle(5, key, 50, 4);

  EXPECT_LE(result, 50u);
}

TEST(IndexShuffleTest, LargerEvenRoundsAccepted) {
  const std::array<uint32_t, 3> key = {7, 8, 9};

  const uint64_t result = index_shuffle(5, key, 50, 20);

  EXPECT_LE(result, 50u);
}

TEST(IndexShuffleTest, ZeroKeyIsAccepted) {
  const std::array<uint32_t, 3> key = {0, 0, 0};

  const uint64_t result = index_shuffle(10, key, 100, 4);

  EXPECT_LE(result, 100u);
}

TEST(IndexShuffleTest, MaxKeyValuesAreAccepted) {
  const std::array<uint32_t, 3> key = {UINT32_MAX, UINT32_MAX, UINT32_MAX};

  const uint64_t result = index_shuffle(10, key, 100, 4);

  EXPECT_LE(result, 100u);
}

TEST(IndexShuffleTest, PermutesAllValuesInSmallDomain) {
  const std::array<uint32_t, 3> key = {1, 2, 3};
  constexpr uint64_t kMaxIndex = 15;

  std::set<uint64_t> outputs;

  for (uint64_t index = 0; index <= kMaxIndex; ++index) {
    const uint64_t shuffled = index_shuffle(index, key, kMaxIndex, 4);
    EXPECT_LE(shuffled, kMaxIndex);
    outputs.insert(shuffled);
  }

  EXPECT_EQ(outputs.size(), kMaxIndex + 1);
}

TEST(IndexShuffleTest, PermutesAllValuesForNonPowerOfTwoDomain) {
  const std::array<uint32_t, 3> key = {11, 22, 33};
  constexpr uint64_t kMaxIndex = 20;

  std::set<uint64_t> outputs;

  for (uint64_t index = 0; index <= kMaxIndex; ++index) {
    const uint64_t shuffled = index_shuffle(index, key, kMaxIndex, 6);
    EXPECT_LE(shuffled, kMaxIndex);
    outputs.insert(shuffled);
  }

  EXPECT_EQ(outputs.size(), kMaxIndex + 1);
}

TEST(IndexShuffleTest, HandlesBoundaryAroundMinimumBlockSize) {
  const std::array<uint32_t, 3> key = {3, 5, 7};

  EXPECT_LE(index_shuffle(0, key, 255, 4), 255u);
  EXPECT_LE(index_shuffle(255, key, 255, 4), 255u);
  EXPECT_LE(index_shuffle(0, key, 256, 4), 256u);
  EXPECT_LE(index_shuffle(256, key, 256, 4), 256u);
}

TEST(IndexShuffleTest, HandlesBlockSize18Boundary) {
  const std::array<uint32_t, 3> key = {3, 5, 7};

  EXPECT_LE(index_shuffle(0, key, 65536, 4), 65536u);
  EXPECT_LE(index_shuffle(65536, key, 65536, 4), 65536u);
  EXPECT_LE(index_shuffle(100, key, 100000, 4), 100000u);
}

TEST(IndexShuffleTest, HandlesLargeMaxIndex) {
  const std::array<uint32_t, 3> key = {101, 202, 303};

  const uint64_t max_index = (uint64_t{1} << 40) + 123;
  const uint64_t index = (uint64_t{1} << 39);

  const uint64_t result = index_shuffle(index, key, max_index, 8);

  EXPECT_LE(result, max_index);
}

TEST(IndexShuffleTest, HandlesVeryLargeMaxIndexUsingDefault64BitBlock) {
  const std::array<uint32_t, 3> key = {101, 202, 303};

  const uint64_t max_index = (uint64_t{1} << 63);
  const uint64_t index = (uint64_t{1} << 62);

  const uint64_t result = index_shuffle(index, key, max_index, 8);

  EXPECT_LE(result, max_index);
}

TEST(IndexShuffleTest, DifferentInputsInDomainProduceUniqueOutputsForSample) {
  const std::array<uint32_t, 3> key = {9, 8, 7};
  constexpr uint64_t kMaxIndex = 1000;

  std::set<uint64_t> outputs;

  for (uint64_t index = 0; index < 100; ++index) {
    const uint64_t shuffled = index_shuffle(index, key, kMaxIndex, 6);
    EXPECT_LE(shuffled, kMaxIndex);
    outputs.insert(shuffled);
  }

  EXPECT_EQ(outputs.size(), 100u);
}

TEST(IndexShuffleTest, DeathWhenRoundsLessThanFour) {
  const std::array<uint32_t, 3> key = {1, 2, 3};

  EXPECT_DEATH(
      {
        volatile uint64_t result = index_shuffle(0, key, 10, 2);
        (void)result;
      },
      "rounds >= 4");
}

TEST(IndexShuffleTest, DeathWhenRoundsOdd) {
  const std::array<uint32_t, 3> key = {1, 2, 3};

  EXPECT_DEATH(
      {
        volatile uint64_t result = index_shuffle(0, key, 10, 5);
        (void)result;
      },
      "rounds >= 4");
}

}  // namespace
}  // namespace random
}  // namespace tensorflow