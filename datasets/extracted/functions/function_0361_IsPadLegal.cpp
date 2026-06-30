#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/pad.h"
#include <cstdint>
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  
#include "mlir/IR/Builders.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/Matchers.h"  
#include "mlir/IR/PatternMatch.h"  
#include "mlir/Support/LLVM.h"  
#include "mlir/Support/LogicalResult.h"  
#include "mlir/Transforms/DialectConversion.h"  
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/op_util_common.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/pad_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
namespace mlir::odml {
namespace {
bool IsPadLegal(mhlo::PadOp op) {
  return AnyNegativePads(op) || !TrivialInterior(op);
}
bool IsPadValCstZero(mhlo::PadOp op) {
  if (matchPattern(op.getPaddingValue(), m_AnyZeroFloat())) {
    return true;
  }
  if (matchPattern(op.getPaddingValue(), m_Zero())) {
    return true;
  }
  return false;
}
DenseIntElementsAttr BuildTFLPaddingAttr(OpBuilder& b, mhlo::PadOp op) {
  auto lows = UnrollI64Splat(op.getEdgePaddingLow());
  auto highs = UnrollI64Splat(op.getEdgePaddingHigh());
  llvm::SmallVector<int64_t> res;
  for (auto [l, h] : llvm::zip(lows, highs)) {
    res.push_back(l);
    res.push_back(h);
  }
  const int64_t n_dims = res.size();
  auto tfl_padding_type =
      RankedTensorType::get({n_dims / 2, 2}, b.getI64Type());
  return DenseIntElementsAttr::get(tfl_padding_type, res);
}
class LegalizePad : public OpConversionPattern<mhlo::PadOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::PadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
};
LogicalResult LegalizePad::matchAndRewrite(
    mhlo::PadOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  if (IsPadLegal(op)) {
    return rewriter.notifyMatchFailure(op, "Matching an already legal pad op.");
  }
  if (!IsPadValCstZero(op)) {
    return rewriter.notifyMatchFailure(
        op, "Legalizing to padv1 requires zero const padding values.");
  }
  auto tfl_paddings = BuildTFLPaddingAttr(rewriter, op);
  auto paddings_op =
      rewriter.create<arith::ConstantOp>(op->getLoc(), tfl_paddings);
  rewriter.replaceOpWithNewOp<TFL::PadOp>(op, op.getType(), op.getOperand(),
                                          paddings_op);
  return success();
}
class LegalizePadV2 : public OpConversionPattern<mhlo::PadOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::PadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
};
LogicalResult LegalizePadV2::matchAndRewrite(
    mhlo::PadOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  if (IsPadLegal(op)) {
    return rewriter.notifyMatchFailure(op, "Matching an already legal pad op.");
  }
  if (IsPadValCstZero(op)) {
    return rewriter.notifyMatchFailure(
        op, "Legalizing to padv2 requires non zero const padding values.");
  }
  auto tfl_paddings = BuildTFLPaddingAttr(rewriter, op);
  auto paddings_op =
      rewriter.create<arith::ConstantOp>(op->getLoc(), tfl_paddings);
  rewriter.replaceOpWithNewOp<TFL::PadV2Op>(op, op.getType(), op.getOperand(),
                                            paddings_op, op.getPaddingValue());
  return success();
}
}  
void PopulatePadPatterns(MLIRContext* ctx, RewritePatternSet& patterns,
                         ConversionTarget& target) {
  patterns.add<LegalizePad>(ctx);
  patterns.add<LegalizePadV2>(ctx);
  target.addDynamicallyLegalOp<mhlo::PadOp>(IsPadLegal);
}
}  