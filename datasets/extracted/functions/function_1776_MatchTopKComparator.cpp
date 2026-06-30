#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/sort.h"
#include <cstdint>
#include "llvm/ADT/ilist.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  
#include "mlir/IR/Block.h"  
#include "mlir/IR/Builders.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/BuiltinTypeInterfaces.h"  
#include "mlir/IR/MLIRContext.h"  
#include "mlir/IR/PatternMatch.h"  
#include "mlir/IR/Region.h"  
#include "mlir/Support/LogicalResult.h"  
#include "mlir/Transforms/DialectConversion.h"  
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/hlo_matchers.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
namespace mlir::odml {
namespace {
using OpListType = llvm::iplist<Operation>;
template <typename ReturnOpType>
bool MatchTopKComparator(Region& comparator) {
  if (!comparator.hasOneBlock()) return false;
  Block& comparator_blk = comparator.front();
  OpListType& operations = comparator_blk.getOperations();
  if (operations.size() != 2) return false;
  auto compare_op =
      llvm::dyn_cast_or_null<mhlo::CompareOp>(&operations.front());
  auto return_op = llvm::dyn_cast_or_null<ReturnOpType>(&operations.back());
  if (!compare_op || !return_op) return false;
  if (compare_op.getComparisonDirection() != mhlo::ComparisonDirection::GT) {
    return false;
  }
  if (compare_op.getOperands()[0] != comparator_blk.getArgument(0) ||
      compare_op.getOperands()[1] != comparator_blk.getArgument(1)) {
    return false;
  }
  return return_op.getOperands().front() == compare_op.getResult();
}
bool IsSortOpNotTopK(mhlo::SortOp op) {
  if (op->getNumOperands() != 2) {
    return true;
  }
  auto keys_opr = op.getInputs().front();
  auto keys_type = llvm::cast<ShapedType>(keys_opr.getType());
  if (!keys_type.hasStaticShape() ||
      !keys_type.getElementType().isIntOrFloat()) {
    return true;
  }
  auto indices_opr = op.getInputs().back();
  auto indices_type = llvm::cast<ShapedType>(indices_opr.getType());
  if (!indices_type.hasStaticShape() ||
      !indices_type.getElementType().isInteger(32)) {
    return true;
  }
  const int64_t sort_dim = op.getDimension();
  const auto k = indices_type.getDimSize(sort_dim);
  const auto rank = keys_type.getRank();
  if (sort_dim != rank - 1 || k < 1) {
    return true;
  }
  OpBuilder b(op->getContext());
  if (!MatchIota(b.getI64TensorAttr({sort_dim}), indices_opr)) {
    return true;
  }
  if (!MatchTopKComparator<mhlo::ReturnOp>(op.getComparator())) {
    return true;
  }
  return false;
}
class LegalizeSortOp : public OpConversionPattern<mhlo::SortOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::SortOp sort_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
};
LogicalResult LegalizeSortOp::matchAndRewrite(
    mhlo::SortOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  if (IsSortOpNotTopK(op)) {
    return failure();
  }
  auto keys = op.getInputs().front();
  auto indices = op.getInputs().back();
  auto indices_type = llvm::cast<ShapedType>(indices.getType());
  const int32_t k = indices_type.getShape().back();
  auto k_cst_attr = DenseIntElementsAttr::get(
      RankedTensorType::get({}, rewriter.getI32Type()), k);
  auto k_cst = rewriter.create<arith::ConstantOp>(op->getLoc(), k_cst_attr);
  rewriter.replaceOpWithNewOp<TFL::TopKV2Op>(op, keys.getType(),
                                             indices.getType(), keys, k_cst);
  return success();
}
}  
void PopulateSortPatterns(MLIRContext* ctx, RewritePatternSet& patterns,
                          ConversionTarget& target) {
  patterns.add<LegalizeSortOp>(ctx);
  target.addDynamicallyLegalOp<mhlo::SortOp>(IsSortOpNotTopK);
}
}  