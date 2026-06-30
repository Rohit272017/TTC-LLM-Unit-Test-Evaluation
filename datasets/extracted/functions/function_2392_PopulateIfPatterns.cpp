#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/if.h"
#include "mlir/IR/PatternMatch.h"  
#include "mlir/IR/Region.h"  
#include "mlir/Support/LLVM.h"  
#include "mlir/Support/LogicalResult.h"  
#include "mlir/Transforms/DialectConversion.h"  
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
namespace mlir::odml {
namespace {
class LegalizeIfOp : public OpConversionPattern<mhlo::IfOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::IfOp if_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto new_op = rewriter.create<TFL::IfOp>(
        if_op.getLoc(), if_op.getResultTypes(), if_op.getPred());
    new_op.getThenRegion().takeBody(if_op.getTrueBranch());
    new_op.getElseRegion().takeBody(if_op.getFalseBranch());
    ReplaceTerminatorWithYield(new_op.getThenRegion(), rewriter);
    ReplaceTerminatorWithYield(new_op.getElseRegion(), rewriter);
    rewriter.replaceOp(if_op, new_op.getResults());
    return success();
  }
};
}  
void PopulateIfPatterns(MLIRContext* ctx, RewritePatternSet& patterns,
                        ConversionTarget& target) {
  patterns.add<LegalizeIfOp>(ctx);
  target.addIllegalOp<mhlo::IfOp>();
}
}  