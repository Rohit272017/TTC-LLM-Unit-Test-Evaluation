#include "xla/service/gpu/model/affine_map_evaluator.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

namespace xla {
namespace gpu {
namespace {

class AffineMapEvaluatorTest : public ::testing::Test {
 protected:
  AffineMapEvaluatorTest() : builder_(&context_) {}

  mlir::MLIRContext context_;
  mlir::Builder builder_;
};

TEST_F(AffineMapEvaluatorTest, ECP_EvaluateConstantExpr) {
  mlir::AffineExpr expr = builder_.getAffineConstantExpr(42);

  EXPECT_EQ(EvaluateAffineExpr(expr, {}, {}), 42);
}

TEST_F(AffineMapEvaluatorTest, BVA_EvaluateZeroConstantExpr) {
  mlir::AffineExpr expr = builder_.getAffineConstantExpr(0);

  EXPECT_EQ(EvaluateAffineExpr(expr, {}, {}), 0);
}

TEST_F(AffineMapEvaluatorTest, BVA_EvaluateNegativeConstantExpr) {
  mlir::AffineExpr expr = builder_.getAffineConstantExpr(-7);

  EXPECT_EQ(EvaluateAffineExpr(expr, {}, {}), -7);
}

TEST_F(AffineMapEvaluatorTest, ECP_EvaluateDimExpr) {
  mlir::AffineExpr expr = builder_.getAffineDimExpr(1);

  std::vector<int64_t> dims = {10, 20, 30};

  EXPECT_EQ(EvaluateAffineExpr(expr, dims, {}), 20);
}

TEST_F(AffineMapEvaluatorTest, BVA_EvaluateFirstDimExpr) {
  mlir::AffineExpr expr = builder_.getAffineDimExpr(0);

  std::vector<int64_t> dims = {-5};

  EXPECT_EQ(EvaluateAffineExpr(expr, dims, {}), -5);
}

TEST_F(AffineMapEvaluatorTest, ECP_EvaluateSymbolExpr) {
  mlir::AffineExpr expr = builder_.getAffineSymbolExpr(1);

  std::vector<int64_t> symbols = {100, 200, 300};

  EXPECT_EQ(EvaluateAffineExpr(expr, {}, symbols), 200);
}

TEST_F(AffineMapEvaluatorTest, BVA_EvaluateFirstSymbolExpr) {
  mlir::AffineExpr expr = builder_.getAffineSymbolExpr(0);

  std::vector<int64_t> symbols = {-9};

  EXPECT_EQ(EvaluateAffineExpr(expr, {}, symbols), -9);
}

TEST_F(AffineMapEvaluatorTest, ECP_EvaluateAddExpr) {
  mlir::AffineExpr d0 = builder_.getAffineDimExpr(0);
  mlir::AffineExpr s0 = builder_.getAffineSymbolExpr(0);
  mlir::AffineExpr expr = d0 + s0 + 5;

  std::vector<int64_t> dims = {7};
  std::vector<int64_t> symbols = {3};

  EXPECT_EQ(EvaluateAffineExpr(expr, dims, symbols), 15);
}

TEST_F(AffineMapEvaluatorTest, BVA_EvaluateAddWithNegativeValues) {
  mlir::AffineExpr d0 = builder_.getAffineDimExpr(0);
  mlir::AffineExpr s0 = builder_.getAffineSymbolExpr(0);
  mlir::AffineExpr expr = d0 + s0;

  std::vector<int64_t> dims = {-10};
  std::vector<int64_t> symbols = {4};

  EXPECT_EQ(EvaluateAffineExpr(expr, dims, symbols), -6);
}

TEST_F(AffineMapEvaluatorTest, ECP_EvaluateMulExpr) {
  mlir::AffineExpr d0 = builder_.getAffineDimExpr(0);
  mlir::AffineExpr expr = d0 * 6;

  std::vector<int64_t> dims = {7};

  EXPECT_EQ(EvaluateAffineExpr(expr, dims, {}), 42);
}

TEST_F(AffineMapEvaluatorTest, BVA_EvaluateMulByZero) {
  mlir::AffineExpr d0 = builder_.getAffineDimExpr(0);
  mlir::AffineExpr expr = d0 * 0;

  std::vector<int64_t> dims = {123};

  EXPECT_EQ(EvaluateAffineExpr(expr, dims, {}), 0);
}

TEST_F(AffineMapEvaluatorTest, BVA_EvaluateMulWithNegativeValue) {
  mlir::AffineExpr d0 = builder_.getAffineDimExpr(0);
  mlir::AffineExpr expr = d0 * -3;

  std::vector<int64_t> dims = {8};

  EXPECT_EQ(EvaluateAffineExpr(expr, dims, {}), -24);
}

TEST_F(AffineMapEvaluatorTest, ECP_EvaluateFloorDivPositiveValues) {
  mlir::AffineExpr d0 = builder_.getAffineDimExpr(0);
  mlir::AffineExpr expr = d0.floorDiv(3);

  std::vector<int64_t> dims = {10};

  EXPECT_EQ(EvaluateAffineExpr(expr, dims, {}), 3);
}

TEST_F(AffineMapEvaluatorTest, BVA_EvaluateFloorDivExactDivision) {
  mlir::AffineExpr d0 = builder_.getAffineDimExpr(0);
  mlir::AffineExpr expr = d0.floorDiv(5);

  std::vector<int64_t> dims = {10};

  EXPECT_EQ(EvaluateAffineExpr(expr, dims, {}), 2);
}

TEST_F(AffineMapEvaluatorTest, Edge_EvaluateFloorDivNegativeDividend) {
  mlir::AffineExpr d0 = builder_.getAffineDimExpr(0);
  mlir::AffineExpr expr = d0.floorDiv(3);

  std::vector<int64_t> dims = {-10};

  EXPECT_EQ(EvaluateAffineExpr(expr, dims, {}), -4);
}

TEST_F(AffineMapEvaluatorTest, Edge_EvaluateFloorDivNegativeDivisor) {
  mlir::AffineExpr d0 = builder_.getAffineDimExpr(0);
  mlir::AffineExpr expr = d0.floorDiv(-3);

  std::vector<int64_t> dims = {10};

  EXPECT_EQ(EvaluateAffineExpr(expr, dims, {}), -4);
}

TEST_F(AffineMapEvaluatorTest, ECP_EvaluateModPositiveValues) {
  mlir::AffineExpr d0 = builder_.getAffineDimExpr(0);
  mlir::AffineExpr expr = d0 % 3;

  std::vector<int64_t> dims = {10};

  EXPECT_EQ(EvaluateAffineExpr(expr, dims, {}), 1);
}

TEST_F(AffineMapEvaluatorTest, BVA_EvaluateModExactDivisionReturnsZero) {
  mlir::AffineExpr d0 = builder_.getAffineDimExpr(0);
  mlir::AffineExpr expr = d0 % 5;

  std::vector<int64_t> dims = {10};

  EXPECT_EQ(EvaluateAffineExpr(expr, dims, {}), 0);
}

TEST_F(AffineMapEvaluatorTest, Edge_EvaluateModNegativeDividendUsesCppRemainder) {
  mlir::AffineExpr d0 = builder_.getAffineDimExpr(0);
  mlir::AffineExpr expr = d0 % 3;

  std::vector<int64_t> dims = {-10};

  EXPECT_EQ(EvaluateAffineExpr(expr, dims, {}), -1);
}

TEST_F(AffineMapEvaluatorTest, Edge_EvaluateNestedExpression) {
  mlir::AffineExpr d0 = builder_.getAffineDimExpr(0);
  mlir::AffineExpr d1 = builder_.getAffineDimExpr(1);
  mlir::AffineExpr s0 = builder_.getAffineSymbolExpr(0);

  mlir::AffineExpr expr = ((d0 + 2) * (d1 - 3)).floorDiv(2) + (s0 % 5);

  std::vector<int64_t> dims = {4, 9};
  std::vector<int64_t> symbols = {12};

  EXPECT_EQ(EvaluateAffineExpr(expr, dims, symbols), 20);
}

TEST_F(AffineMapEvaluatorTest, ECP_EvaluateMapWithSingleResult) {
  mlir::AffineExpr d0 = builder_.getAffineDimExpr(0);
  mlir::AffineMap map = mlir::AffineMap::get(
      /*dimCount=*/1, /*symbolCount=*/0, {d0 + 1}, &context_);

  std::vector<int64_t> dims = {9};

  llvm::SmallVector<int64_t> result = EvaluateAffineMap(map, dims, {});

  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0], 10);
}

TEST_F(AffineMapEvaluatorTest, ECP_EvaluateMapWithMultipleResults) {
  mlir::AffineExpr d0 = builder_.getAffineDimExpr(0);
  mlir::AffineExpr d1 = builder_.getAffineDimExpr(1);
  mlir::AffineExpr s0 = builder_.getAffineSymbolExpr(0);

  mlir::AffineMap map = mlir::AffineMap::get(
      /*dimCount=*/2, /*symbolCount=*/1,
      {d0 + d1, d0 * 2, s0.floorDiv(4), s0 % 4}, &context_);

  std::vector<int64_t> dims = {3, 5};
  std::vector<int64_t> symbols = {10};

  llvm::SmallVector<int64_t> result = EvaluateAffineMap(map, dims, symbols);

  ASSERT_EQ(result.size(), 4u);
  EXPECT_EQ(result[0], 8);
  EXPECT_EQ(result[1], 6);
  EXPECT_EQ(result[2], 2);
  EXPECT_EQ(result[3], 2);
}

TEST_F(AffineMapEvaluatorTest, BVA_EvaluateMapWithZeroResults) {
  mlir::AffineMap map = mlir::AffineMap::get(
      /*dimCount=*/0, /*symbolCount=*/0,
      llvm::ArrayRef<mlir::AffineExpr>{}, &context_);

  llvm::SmallVector<int64_t> result = EvaluateAffineMap(map, {}, {});

  EXPECT_TRUE(result.empty());
}

TEST_F(AffineMapEvaluatorTest, BVA_EvaluateMapWithZeroDimsAndOneSymbol) {
  mlir::AffineExpr s0 = builder_.getAffineSymbolExpr(0);
  mlir::AffineMap map = mlir::AffineMap::get(
      /*dimCount=*/0, /*symbolCount=*/1, {s0 + 5}, &context_);

  std::vector<int64_t> symbols = {7};

  llvm::SmallVector<int64_t> result = EvaluateAffineMap(map, {}, symbols);

  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0], 12);
}

TEST_F(AffineMapEvaluatorTest, BVA_EvaluateMapWithOneDimAndZeroSymbols) {
  mlir::AffineExpr d0 = builder_.getAffineDimExpr(0);
  mlir::AffineMap map = mlir::AffineMap::get(
      /*dimCount=*/1, /*symbolCount=*/0, {d0 - 1}, &context_);

  std::vector<int64_t> dims = {1};

  llvm::SmallVector<int64_t> result = EvaluateAffineMap(map, dims, {});

  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0], 0);
}

TEST_F(AffineMapEvaluatorTest, Edge_EvaluateMapPreservesResultOrder) {
  mlir::AffineExpr d0 = builder_.getAffineDimExpr(0);
  mlir::AffineMap map = mlir::AffineMap::get(
      /*dimCount=*/1, /*symbolCount=*/0, {d0 + 3, d0 + 2, d0 + 1, d0}, &context_);

  std::vector<int64_t> dims = {10};

  llvm::SmallVector<int64_t> result = EvaluateAffineMap(map, dims, {});

  ASSERT_EQ(result.size(), 4u);
  EXPECT_EQ(result[0], 13);
  EXPECT_EQ(result[1], 12);
  EXPECT_EQ(result[2], 11);
  EXPECT_EQ(result[3], 10);
}

TEST_F(AffineMapEvaluatorTest, Edge_EvaluateLargeValuesWithoutOverflowExpectation) {
  mlir::AffineExpr d0 = builder_.getAffineDimExpr(0);
  mlir::AffineExpr s0 = builder_.getAffineSymbolExpr(0);
  mlir::AffineExpr expr = d0 + s0;

  std::vector<int64_t> dims = {1000000000LL};
  std::vector<int64_t> symbols = {2000000000LL};

  EXPECT_EQ(EvaluateAffineExpr(expr, dims, symbols), 3000000000LL);
}

#ifndef NDEBUG

TEST_F(AffineMapEvaluatorTest, Invalid_MapDimCountMismatchDies) {
  mlir::AffineExpr d0 = builder_.getAffineDimExpr(0);
  mlir::AffineMap map = mlir::AffineMap::get(
      /*dimCount=*/1, /*symbolCount=*/0, {d0}, &context_);

  std::vector<int64_t> wrong_dims = {};

  EXPECT_DEATH(EvaluateAffineMap(map, wrong_dims, {}), ".*");
}

TEST_F(AffineMapEvaluatorTest, Invalid_MapSymbolCountMismatchDies) {
  mlir::AffineExpr s0 = builder_.getAffineSymbolExpr(0);
  mlir::AffineMap map = mlir::AffineMap::get(
      /*dimCount=*/0, /*symbolCount=*/1, {s0}, &context_);

  std::vector<int64_t> wrong_symbols = {};

  EXPECT_DEATH(EvaluateAffineMap(map, {}, wrong_symbols), ".*");
}

#endif

}  // namespace
}  // namespace gpu
}  // namespace xla