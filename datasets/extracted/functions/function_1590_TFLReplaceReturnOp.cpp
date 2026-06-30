#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/while.h"
#include "mlir/IR/Builders.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/PatternMatch.h"  
#include "mlir/IR/Region.h"  
#include "mlir/Support/LLVM.h"  
#include "mlir/Support/LogicalResult.h"  
#include "mlir/Transforms/DialectConversion.h"  
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
namespace mlir::odml {
namespace {
void TFLReplaceReturnOp(Region& region, PatternRewriter& rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  for (auto& block : region.getBlocks()) {
    Operation* terminator = block.getTerminator();
    rewriter.setInsertionPoint(terminator);
    rewriter.replaceOpWithNewOp<TFL::YieldOp>(terminator,
                                              terminator->getOperands());
  }
}
class LeagalizeWhileOp : public OpConversionPattern<mhlo::WhileOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::WhileOp while_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto is_stateless = rewriter.getBoolAttr(false);
    auto new_while = rewriter.create<TFL::WhileOp>(
        while_op.getLoc(), while_op->getResultTypes(), while_op->getOperands(),
        is_stateless);
    new_while.getCond().takeBody(while_op.getCond());
    new_while.getBody().takeBody(while_op.getBody());
    TFLReplaceReturnOp(new_while.getCond(), rewriter);
    TFLReplaceReturnOp(new_while.getBody(), rewriter);
    rewriter.replaceOp(while_op, new_while.getResults());
    return success();
  }
};
bool IsWhileLegal(mhlo::WhileOp while_op) {
  for (auto type : while_op->getOperandTypes()) {
    if (mlir::isa<TupleType>(type)) return true;
  }
  return false;
}
}  
void PopulateWhilePatterns(MLIRContext* ctx, RewritePatternSet& patterns,
                           ConversionTarget& target) {
  target.addDynamicallyLegalOp<mhlo::WhileOp>(IsWhileLegal);
  patterns.add<LeagalizeWhileOp>(ctx);
}
}  