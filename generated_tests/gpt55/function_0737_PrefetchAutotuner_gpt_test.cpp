#include "tensorflow/core/kernels/data/prefetch_autotuner.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

#include "tensorflow/core/framework/model.h"

namespace tensorflow {
namespace data {
namespace {

TEST(PrefetchAutotunerTest, FixedInitialBufferSizeDisablesAutotuning) {
  PrefetchAutotuner autotuner(
      /*initial_buffer_size=*/8,
      /*buffer_size_min=*/1,
      /*ram_budget_manager=*/nullptr);

  EXPECT_EQ(autotuner.buffer_limit(), 8);

  autotuner.SetElementSize(10);
  autotuner.RecordConsumption(8);
  autotuner.RecordConsumption(0);

  EXPECT_EQ(autotuner.buffer_limit(), 8);
}

TEST(PrefetchAutotunerTest, AutotuneInitializesToOneWhenMinimumIsZero) {
  PrefetchAutotuner autotuner(
      /*initial_buffer_size=*/model::kAutotune,
      /*buffer_size_min=*/0,
      /*ram_budget_manager=*/nullptr);

  EXPECT_EQ(autotuner.buffer_limit(), 1);
}

TEST(PrefetchAutotunerTest, AutotuneInitializesToMinimumWhenMinimumIsPositive) {
  PrefetchAutotuner autotuner(
      /*initial_buffer_size=*/model::kAutotune,
      /*buffer_size_min=*/4,
      /*ram_budget_manager=*/nullptr);

  EXPECT_EQ(autotuner.buffer_limit(), 4);
}

TEST(PrefetchAutotunerTest, AutotuneInitializesToOneWhenMinimumIsNegative) {
  PrefetchAutotuner autotuner(
      /*initial_buffer_size=*/model::kAutotune,
      /*buffer_size_min=*/-5,
      /*ram_budget_manager=*/nullptr);

  EXPECT_EQ(autotuner.buffer_limit(), 1);
}

TEST(PrefetchAutotunerTest, UpswingDoesNotIncreaseUntilBufferBecomesFullThenEmpty) {
  PrefetchAutotuner autotuner(
      /*initial_buffer_size=*/model::kAutotune,
      /*buffer_size_min=*/4,
      /*ram_budget_manager=*/nullptr);

  autotuner.SetElementSize(10);

  autotuner.RecordConsumption(3);
  autotuner.RecordConsumption(0);

  EXPECT_EQ(autotuner.buffer_limit(), 4);
}

TEST(PrefetchAutotunerTest, FullBufferThenEmptyBufferDoublesLimitBelowThreshold) {
  PrefetchAutotuner autotuner(
      /*initial_buffer_size=*/model::kAutotune,
      /*buffer_size_min=*/4,
      /*ram_budget_manager=*/nullptr);

  autotuner.SetElementSize(10);

  autotuner.RecordConsumption(4);
  autotuner.RecordConsumption(0);

  EXPECT_EQ(autotuner.buffer_limit(), 8);
}

TEST(PrefetchAutotunerTest, EmptyBufferWithoutElementSizeDoesNotIncreaseLimit) {
  PrefetchAutotuner autotuner(
      /*initial_buffer_size=*/model::kAutotune,
      /*buffer_size_min=*/4,
      /*ram_budget_manager=*/nullptr);

  autotuner.RecordConsumption(4);
  autotuner.RecordConsumption(0);

  EXPECT_EQ(autotuner.buffer_limit(), 4);
}

TEST(PrefetchAutotunerTest, DownSwingWithNonZeroCurrentBufferDoesNotIncreaseLimit) {
  PrefetchAutotuner autotuner(
      /*initial_buffer_size=*/model::kAutotune,
      /*buffer_size_min=*/4,
      /*ram_budget_manager=*/nullptr);

  autotuner.SetElementSize(10);

  autotuner.RecordConsumption(4);
  autotuner.RecordConsumption(1);

  EXPECT_EQ(autotuner.buffer_limit(), 4);
}

TEST(PrefetchAutotunerTest, MultipleGrowthCyclesDoubleBelowThreshold) {
  PrefetchAutotuner autotuner(
      /*initial_buffer_size=*/model::kAutotune,
      /*buffer_size_min=*/2,
      /*ram_budget_manager=*/nullptr);

  autotuner.SetElementSize(10);

  autotuner.RecordConsumption(2);
  autotuner.RecordConsumption(0);
  EXPECT_EQ(autotuner.buffer_limit(), 4);

  autotuner.RecordConsumption(4);
  autotuner.RecordConsumption(0);
  EXPECT_EQ(autotuner.buffer_limit(), 8);

  autotuner.RecordConsumption(8);
  autotuner.RecordConsumption(0);
  EXPECT_EQ(autotuner.buffer_limit(), 16);
}

TEST(PrefetchAutotunerTest, GrowthAtThresholdAddsThresholdInsteadOfDoubling) {
  PrefetchAutotuner autotuner(
      /*initial_buffer_size=*/model::kAutotune,
      /*buffer_size_min=*/2048,
      /*ram_budget_manager=*/nullptr);

  autotuner.SetElementSize(1);

  autotuner.RecordConsumption(2048);
  autotuner.RecordConsumption(0);

  EXPECT_EQ(autotuner.buffer_limit(), 4096);
}

TEST(PrefetchAutotunerTest, GrowthAboveThresholdAddsThreshold) {
  PrefetchAutotuner autotuner(
      /*initial_buffer_size=*/model::kAutotune,
      /*buffer_size_min=*/4096,
      /*ram_budget_manager=*/nullptr);

  autotuner.SetElementSize(1);

  autotuner.RecordConsumption(4096);
  autotuner.RecordConsumption(0);

  EXPECT_EQ(autotuner.buffer_limit(), 6144);
}

TEST(PrefetchAutotunerTest, BoundaryJustBelowThresholdDoubles) {
  PrefetchAutotuner autotuner(
      /*initial_buffer_size=*/model::kAutotune,
      /*buffer_size_min=*/2047,
      /*ram_budget_manager=*/nullptr);

  autotuner.SetElementSize(1);

  autotuner.RecordConsumption(2047);
  autotuner.RecordConsumption(0);

  EXPECT_EQ(autotuner.buffer_limit(), 4094);
}

TEST(PrefetchAutotunerTest, ZeroElementSizeStillAllowsGrowthWithoutBudgetManager) {
  PrefetchAutotuner autotuner(
      /*initial_buffer_size=*/model::kAutotune,
      /*buffer_size_min=*/4,
      /*ram_budget_manager=*/nullptr);

  autotuner.SetElementSize(0);

  autotuner.RecordConsumption(4);
  autotuner.RecordConsumption(0);

  EXPECT_EQ(autotuner.buffer_limit(), 8);
}

TEST(PrefetchAutotunerTest, NegativeElementSizeStillAllowsGrowthWithoutBudgetManager) {
  PrefetchAutotuner autotuner(
      /*initial_buffer_size=*/model::kAutotune,
      /*buffer_size_min=*/4,
      /*ram_budget_manager=*/nullptr);

  autotuner.SetElementSize(-10);

  autotuner.RecordConsumption(4);
  autotuner.RecordConsumption(0);

  EXPECT_EQ(autotuner.buffer_limit(), 8);
}

TEST(PrefetchAutotunerTest, CurrentBufferSizeGreaterThanLimitDoesNotEnterDownswing) {
  PrefetchAutotuner autotuner(
      /*initial_buffer_size=*/model::kAutotune,
      /*buffer_size_min=*/4,
      /*ram_budget_manager=*/nullptr);

  autotuner.SetElementSize(10);

  autotuner.RecordConsumption(5);
  autotuner.RecordConsumption(0);

  EXPECT_EQ(autotuner.buffer_limit(), 4);
}

TEST(PrefetchAutotunerTest, CurrentBufferSizeEqualToLimitIsRequiredForDownswing) {
  PrefetchAutotuner autotuner(
      /*initial_buffer_size=*/model::kAutotune,
      /*buffer_size_min=*/1,
      /*ram_budget_manager=*/nullptr);

  autotuner.SetElementSize(1);

  autotuner.RecordConsumption(0);
  EXPECT_EQ(autotuner.buffer_limit(), 1);

  autotuner.RecordConsumption(1);
  autotuner.RecordConsumption(0);
  EXPECT_EQ(autotuner.buffer_limit(), 2);
}

TEST(PrefetchAutotunerTest, GrowthCycleReturnsToUpswingAfterEmptyBuffer) {
  PrefetchAutotuner autotuner(
      /*initial_buffer_size=*/model::kAutotune,
      /*buffer_size_min=*/4,
      /*ram_budget_manager=*/nullptr);

  autotuner.SetElementSize(10);

  autotuner.RecordConsumption(4);
  autotuner.RecordConsumption(0);
  EXPECT_EQ(autotuner.buffer_limit(), 8);

  autotuner.RecordConsumption(0);
  EXPECT_EQ(autotuner.buffer_limit(), 8);

  autotuner.RecordConsumption(8);
  autotuner.RecordConsumption(0);
  EXPECT_EQ(autotuner.buffer_limit(), 16);
}

TEST(PrefetchAutotunerTest, SetElementSizeCanBeCalledAfterFullBufferBeforeEmpty) {
  PrefetchAutotuner autotuner(
      /*initial_buffer_size=*/model::kAutotune,
      /*buffer_size_min=*/4,
      /*ram_budget_manager=*/nullptr);

  autotuner.RecordConsumption(4);
  autotuner.SetElementSize(10);
  autotuner.RecordConsumption(0);

  EXPECT_EQ(autotuner.buffer_limit(), 8);
}

TEST(PrefetchAutotunerTest, UpdatingElementSizeKeepsAutotuningFunctional) {
  PrefetchAutotuner autotuner(
      /*initial_buffer_size=*/model::kAutotune,
      /*buffer_size_min=*/4,
      /*ram_budget_manager=*/nullptr);

  autotuner.SetElementSize(10);
  autotuner.SetElementSize(20);

  autotuner.RecordConsumption(4);
  autotuner.RecordConsumption(0);

  EXPECT_EQ(autotuner.buffer_limit(), 8);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow