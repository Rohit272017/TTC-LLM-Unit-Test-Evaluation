#include "arolla/util/binary_search.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "absl/types/span.h"

namespace arolla::binary_search_details {
namespace {

TEST(BinarySearchDetailsTest, BVA_LowerBoundInt32SingleElement) {
  std::vector<int32_t> array = {10};

  EXPECT_EQ(LowerBoundImpl(5, absl::MakeConstSpan(array)), 0u);
  EXPECT_EQ(LowerBoundImpl(10, absl::MakeConstSpan(array)), 0u);
  EXPECT_EQ(LowerBoundImpl(15, absl::MakeConstSpan(array)), 1u);
}

TEST(BinarySearchDetailsTest, BVA_UpperBoundInt32SingleElement) {
  std::vector<int32_t> array = {10};

  EXPECT_EQ(UpperBoundImpl(5, absl::MakeConstSpan(array)), 0u);
  EXPECT_EQ(UpperBoundImpl(10, absl::MakeConstSpan(array)), 1u);
  EXPECT_EQ(UpperBoundImpl(15, absl::MakeConstSpan(array)), 1u);
}

TEST(BinarySearchDetailsTest, ECP_LowerBoundInt32SortedUniqueValues) {
  std::vector<int32_t> array = {1, 3, 5, 7, 9};

  EXPECT_EQ(LowerBoundImpl(0, absl::MakeConstSpan(array)), 0u);
  EXPECT_EQ(LowerBoundImpl(1, absl::MakeConstSpan(array)), 0u);
  EXPECT_EQ(LowerBoundImpl(4, absl::MakeConstSpan(array)), 2u);
  EXPECT_EQ(LowerBoundImpl(9, absl::MakeConstSpan(array)), 4u);
  EXPECT_EQ(LowerBoundImpl(10, absl::MakeConstSpan(array)), 5u);
}

TEST(BinarySearchDetailsTest, ECP_UpperBoundInt32SortedUniqueValues) {
  std::vector<int32_t> array = {1, 3, 5, 7, 9};

  EXPECT_EQ(UpperBoundImpl(0, absl::MakeConstSpan(array)), 0u);
  EXPECT_EQ(UpperBoundImpl(1, absl::MakeConstSpan(array)), 1u);
  EXPECT_EQ(UpperBoundImpl(4, absl::MakeConstSpan(array)), 2u);
  EXPECT_EQ(UpperBoundImpl(9, absl::MakeConstSpan(array)), 5u);
  EXPECT_EQ(UpperBoundImpl(10, absl::MakeConstSpan(array)), 5u);
}

TEST(BinarySearchDetailsTest, ECP_LowerBoundInt32WithDuplicates) {
  std::vector<int32_t> array = {1, 2, 2, 2, 3, 4};

  EXPECT_EQ(LowerBoundImpl(2, absl::MakeConstSpan(array)), 1u);
  EXPECT_EQ(LowerBoundImpl(3, absl::MakeConstSpan(array)), 4u);
}

TEST(BinarySearchDetailsTest, ECP_UpperBoundInt32WithDuplicates) {
  std::vector<int32_t> array = {1, 2, 2, 2, 3, 4};

  EXPECT_EQ(UpperBoundImpl(2, absl::MakeConstSpan(array)), 4u);
  EXPECT_EQ(UpperBoundImpl(3, absl::MakeConstSpan(array)), 5u);
}

TEST(BinarySearchDetailsTest, BVA_Int32MinAndMaxValues) {
  std::vector<int32_t> array = {
      std::numeric_limits<int32_t>::min(),
      -1,
      0,
      1,
      std::numeric_limits<int32_t>::max(),
  };

  EXPECT_EQ(LowerBoundImpl(std::numeric_limits<int32_t>::min(),
                           absl::MakeConstSpan(array)),
            0u);
  EXPECT_EQ(UpperBoundImpl(std::numeric_limits<int32_t>::min(),
                           absl::MakeConstSpan(array)),
            1u);
  EXPECT_EQ(LowerBoundImpl(std::numeric_limits<int32_t>::max(),
                           absl::MakeConstSpan(array)),
            4u);
  EXPECT_EQ(UpperBoundImpl(std::numeric_limits<int32_t>::max(),
                           absl::MakeConstSpan(array)),
            5u);
}

TEST(BinarySearchDetailsTest, BVA_LowerBoundInt64SingleElement) {
  std::vector<int64_t> array = {10};

  EXPECT_EQ(LowerBoundImpl(5LL, absl::MakeConstSpan(array)), 0u);
  EXPECT_EQ(LowerBoundImpl(10LL, absl::MakeConstSpan(array)), 0u);
  EXPECT_EQ(LowerBoundImpl(15LL, absl::MakeConstSpan(array)), 1u);
}

TEST(BinarySearchDetailsTest, BVA_UpperBoundInt64SingleElement) {
  std::vector<int64_t> array = {10};

  EXPECT_EQ(UpperBoundImpl(5LL, absl::MakeConstSpan(array)), 0u);
  EXPECT_EQ(UpperBoundImpl(10LL, absl::MakeConstSpan(array)), 1u);
  EXPECT_EQ(UpperBoundImpl(15LL, absl::MakeConstSpan(array)), 1u);
}

TEST(BinarySearchDetailsTest, ECP_LowerBoundInt64WithDuplicates) {
  std::vector<int64_t> array = {-10, -5, -5, 0, 100};

  EXPECT_EQ(LowerBoundImpl(-5LL, absl::MakeConstSpan(array)), 1u);
  EXPECT_EQ(LowerBoundImpl(50LL, absl::MakeConstSpan(array)), 4u);
  EXPECT_EQ(LowerBoundImpl(101LL, absl::MakeConstSpan(array)), 5u);
}

TEST(BinarySearchDetailsTest, ECP_UpperBoundInt64WithDuplicates) {
  std::vector<int64_t> array = {-10, -5, -5, 0, 100};

  EXPECT_EQ(UpperBoundImpl(-5LL, absl::MakeConstSpan(array)), 3u);
  EXPECT_EQ(UpperBoundImpl(50LL, absl::MakeConstSpan(array)), 4u);
  EXPECT_EQ(UpperBoundImpl(100LL, absl::MakeConstSpan(array)), 5u);
}

TEST(BinarySearchDetailsTest, BVA_Int64MinAndMaxValues) {
  std::vector<int64_t> array = {
      std::numeric_limits<int64_t>::min(),
      -1,
      0,
      1,
      std::numeric_limits<int64_t>::max(),
  };

  EXPECT_EQ(LowerBoundImpl(std::numeric_limits<int64_t>::min(),
                           absl::MakeConstSpan(array)),
            0u);
  EXPECT_EQ(UpperBoundImpl(std::numeric_limits<int64_t>::min(),
                           absl::MakeConstSpan(array)),
            1u);
  EXPECT_EQ(LowerBoundImpl(std::numeric_limits<int64_t>::max(),
                           absl::MakeConstSpan(array)),
            4u);
  EXPECT_EQ(UpperBoundImpl(std::numeric_limits<int64_t>::max(),
                           absl::MakeConstSpan(array)),
            5u);
}

TEST(BinarySearchDetailsTest, ECP_LowerBoundFloatSortedUniqueValues) {
  std::vector<float> array = {-2.0f, -1.0f, 0.0f, 1.5f, 3.0f};

  EXPECT_EQ(LowerBoundImpl(-3.0f, absl::MakeConstSpan(array)), 0u);
  EXPECT_EQ(LowerBoundImpl(-2.0f, absl::MakeConstSpan(array)), 0u);
  EXPECT_EQ(LowerBoundImpl(1.0f, absl::MakeConstSpan(array)), 3u);
  EXPECT_EQ(LowerBoundImpl(3.0f, absl::MakeConstSpan(array)), 4u);
  EXPECT_EQ(LowerBoundImpl(4.0f, absl::MakeConstSpan(array)), 5u);
}

TEST(BinarySearchDetailsTest, ECP_UpperBoundFloatSortedUniqueValues) {
  std::vector<float> array = {-2.0f, -1.0f, 0.0f, 1.5f, 3.0f};

  EXPECT_EQ(UpperBoundImpl(-3.0f, absl::MakeConstSpan(array)), 0u);
  EXPECT_EQ(UpperBoundImpl(-2.0f, absl::MakeConstSpan(array)), 1u);
  EXPECT_EQ(UpperBoundImpl(1.0f, absl::MakeConstSpan(array)), 3u);
  EXPECT_EQ(UpperBoundImpl(3.0f, absl::MakeConstSpan(array)), 5u);
  EXPECT_EQ(UpperBoundImpl(4.0f, absl::MakeConstSpan(array)), 5u);
}

TEST(BinarySearchDetailsTest, ECP_LowerBoundFloatWithDuplicates) {
  std::vector<float> array = {-1.0f, 0.0f, 0.0f, 0.0f, 2.0f};

  EXPECT_EQ(LowerBoundImpl(0.0f, absl::MakeConstSpan(array)), 1u);
  EXPECT_EQ(LowerBoundImpl(1.0f, absl::MakeConstSpan(array)), 4u);
}

TEST(BinarySearchDetailsTest, ECP_UpperBoundFloatWithDuplicates) {
  std::vector<float> array = {-1.0f, 0.0f, 0.0f, 0.0f, 2.0f};

  EXPECT_EQ(UpperBoundImpl(0.0f, absl::MakeConstSpan(array)), 4u);
  EXPECT_EQ(UpperBoundImpl(1.0f, absl::MakeConstSpan(array)), 4u);
}

TEST(BinarySearchDetailsTest, BVA_FloatNegativeZeroAndPositiveZero) {
  std::vector<float> array = {-0.0f, 0.0f, 1.0f};

  EXPECT_EQ(LowerBoundImpl(0.0f, absl::MakeConstSpan(array)), 0u);
  EXPECT_EQ(UpperBoundImpl(0.0f, absl::MakeConstSpan(array)), 2u);
}

TEST(BinarySearchDetailsTest, Edge_UpperBoundFloatNaNReturnsArraySize) {
  std::vector<float> array = {-1.0f, 0.0f, 1.0f};

  EXPECT_EQ(UpperBoundImpl(std::numeric_limits<float>::quiet_NaN(),
                           absl::MakeConstSpan(array)),
            array.size());
}

TEST(BinarySearchDetailsTest, Edge_LowerBoundFloatNaNReturnsFirstNotLessThanNaN) {
  std::vector<float> array = {-1.0f, 0.0f, 1.0f};

  EXPECT_EQ(LowerBoundImpl(std::numeric_limits<float>::quiet_NaN(),
                           absl::MakeConstSpan(array)),
            0u);
}

TEST(BinarySearchDetailsTest, BVA_FloatInfinityValues) {
  std::vector<float> array = {
      -std::numeric_limits<float>::infinity(),
      -1.0f,
      0.0f,
      1.0f,
      std::numeric_limits<float>::infinity(),
  };

  EXPECT_EQ(LowerBoundImpl(-std::numeric_limits<float>::infinity(),
                           absl::MakeConstSpan(array)),
            0u);
  EXPECT_EQ(UpperBoundImpl(-std::numeric_limits<float>::infinity(),
                           absl::MakeConstSpan(array)),
            1u);
  EXPECT_EQ(LowerBoundImpl(std::numeric_limits<float>::infinity(),
                           absl::MakeConstSpan(array)),
            4u);
  EXPECT_EQ(UpperBoundImpl(std::numeric_limits<float>::infinity(),
                           absl::MakeConstSpan(array)),
            5u);
}

TEST(BinarySearchDetailsTest, ECP_LowerBoundDoubleSortedUniqueValues) {
  std::vector<double> array = {-2.0, -1.0, 0.0, 1.5, 3.0};

  EXPECT_EQ(LowerBoundImpl(-3.0, absl::MakeConstSpan(array)), 0u);
  EXPECT_EQ(LowerBoundImpl(-2.0, absl::MakeConstSpan(array)), 0u);
  EXPECT_EQ(LowerBoundImpl(1.0, absl::MakeConstSpan(array)), 3u);
  EXPECT_EQ(LowerBoundImpl(3.0, absl::MakeConstSpan(array)), 4u);
  EXPECT_EQ(LowerBoundImpl(4.0, absl::MakeConstSpan(array)), 5u);
}

TEST(BinarySearchDetailsTest, ECP_UpperBoundDoubleSortedUniqueValues) {
  std::vector<double> array = {-2.0, -1.0, 0.0, 1.5, 3.0};

  EXPECT_EQ(UpperBoundImpl(-3.0, absl::MakeConstSpan(array)), 0u);
  EXPECT_EQ(UpperBoundImpl(-2.0, absl::MakeConstSpan(array)), 1u);
  EXPECT_EQ(UpperBoundImpl(1.0, absl::MakeConstSpan(array)), 3u);
  EXPECT_EQ(UpperBoundImpl(3.0, absl::MakeConstSpan(array)), 5u);
  EXPECT_EQ(UpperBoundImpl(4.0, absl::MakeConstSpan(array)), 5u);
}

TEST(BinarySearchDetailsTest, ECP_LowerBoundDoubleWithDuplicates) {
  std::vector<double> array = {-1.0, 0.0, 0.0, 0.0, 2.0};

  EXPECT_EQ(LowerBoundImpl(0.0, absl::MakeConstSpan(array)), 1u);
  EXPECT_EQ(LowerBoundImpl(1.0, absl::MakeConstSpan(array)), 4u);
}

TEST(BinarySearchDetailsTest, ECP_UpperBoundDoubleWithDuplicates) {
  std::vector<double> array = {-1.0, 0.0, 0.0, 0.0, 2.0};

  EXPECT_EQ(UpperBoundImpl(0.0, absl::MakeConstSpan(array)), 4u);
  EXPECT_EQ(UpperBoundImpl(1.0, absl::MakeConstSpan(array)), 4u);
}

TEST(BinarySearchDetailsTest, Edge_UpperBoundDoubleNaNReturnsArraySize) {
  std::vector<double> array = {-1.0, 0.0, 1.0};

  EXPECT_EQ(UpperBoundImpl(std::numeric_limits<double>::quiet_NaN(),
                           absl::MakeConstSpan(array)),
            array.size());
}

TEST(BinarySearchDetailsTest, Edge_LowerBoundDoubleNaNReturnsFirstNotLessThanNaN) {
  std::vector<double> array = {-1.0, 0.0, 1.0};

  EXPECT_EQ(LowerBoundImpl(std::numeric_limits<double>::quiet_NaN(),
                           absl::MakeConstSpan(array)),
            0u);
}

TEST(BinarySearchDetailsTest, BVA_DoubleInfinityValues) {
  std::vector<double> array = {
      -std::numeric_limits<double>::infinity(),
      -1.0,
      0.0,
      1.0,
      std::numeric_limits<double>::infinity(),
  };

  EXPECT_EQ(LowerBoundImpl(-std::numeric_limits<double>::infinity(),
                           absl::MakeConstSpan(array)),
            0u);
  EXPECT_EQ(UpperBoundImpl(-std::numeric_limits<double>::infinity(),
                           absl::MakeConstSpan(array)),
            1u);
  EXPECT_EQ(LowerBoundImpl(std::numeric_limits<double>::infinity(),
                           absl::MakeConstSpan(array)),
            4u);
  EXPECT_EQ(UpperBoundImpl(std::numeric_limits<double>::infinity(),
                           absl::MakeConstSpan(array)),
            5u);
}

TEST(BinarySearchDetailsTest, Edge_ArraySizePowerOfTwoMinusOne) {
  std::vector<int32_t> array = {1, 2, 3, 4, 5, 6, 7};

  EXPECT_EQ(LowerBoundImpl(4, absl::MakeConstSpan(array)), 3u);
  EXPECT_EQ(UpperBoundImpl(4, absl::MakeConstSpan(array)), 4u);
}

TEST(BinarySearchDetailsTest, Edge_ArraySizePowerOfTwo) {
  std::vector<int32_t> array = {1, 2, 3, 4, 5, 6, 7, 8};

  EXPECT_EQ(LowerBoundImpl(4, absl::MakeConstSpan(array)), 3u);
  EXPECT_EQ(UpperBoundImpl(4, absl::MakeConstSpan(array)), 4u);
}

TEST(BinarySearchDetailsTest, Edge_ArraySizePowerOfTwoPlusOne) {
  std::vector<int32_t> array = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  EXPECT_EQ(LowerBoundImpl(4, absl::MakeConstSpan(array)), 3u);
  EXPECT_EQ(UpperBoundImpl(4, absl::MakeConstSpan(array)), 4u);
}

#ifndef NDEBUG

TEST(BinarySearchDetailsDeathTest, Invalid_LowerBoundEmptyInt32ArrayDies) {
  std::vector<int32_t> array;

  EXPECT_DEATH(LowerBoundImpl(1, absl::MakeConstSpan(array)), ".*");
}

TEST(BinarySearchDetailsDeathTest, Invalid_UpperBoundEmptyInt32ArrayDies) {
  std::vector<int32_t> array;

  EXPECT_DEATH(UpperBoundImpl(1, absl::MakeConstSpan(array)), ".*");
}

#endif

}  // namespace
}  // namespace arolla::binary_search_details