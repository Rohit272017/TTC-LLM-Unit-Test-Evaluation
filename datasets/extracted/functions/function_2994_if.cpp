#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/custom_call.h"
#include <optional>
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/PatternMatch.h"  
#include "mlir/Support/LLVM.h"  
#include "mlir/Support/LogicalResult.h"  
#include "mlir/Transforms/DialectConversion.h"  
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
namespace mlir {
namespace odml {
class ConvertCustomCallOp : public OpConversionPattern<mhlo::CustomCallOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::CustomCallOp mhlo_custom_call, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
};
LogicalResult ConvertCustomCallOp::matchAndRewrite(
    mhlo::CustomCallOp mhlo_custom_call, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto call_target_name = mhlo_custom_call.getCallTargetName();
  if (!call_target_name.starts_with("custom_call.")) {
    return failure();
  }
  auto tfl_custom = rewriter.create<TFL::CustomOp>(
      mhlo_custom_call.getLoc(), mhlo_custom_call.getResultTypes(),
      mhlo_custom_call.getInputs());
  tfl_custom.setCustomCodeAttr(rewriter.getStringAttr(call_target_name));
  if (auto bc = mhlo_custom_call.getBackendConfig()) {
    if (auto stringattr = mlir::dyn_cast_or_null<mlir::StringAttr>(*bc)) {
      tfl_custom.setCustomOptionAttr(
          TFL::ConstBytesAttr::get(rewriter.getContext(), stringattr));
    }
  } else {
    tfl_custom.setCustomOptionAttr(
        TFL::ConstBytesAttr::get(rewriter.getContext(), ""));
  }
  rewriter.replaceOp(mhlo_custom_call, tfl_custom);
  return success();
}
class RemoveCustomCallWithShapeAssertion
    : public OpRewritePattern<mhlo::CustomCallOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::CustomCallOp op,
                                PatternRewriter& rewriter) const final;
};
LogicalResult RemoveCustomCallWithShapeAssertion::matchAndRewrite(
    mhlo::CustomCallOp op, PatternRewriter& rewriter) const {
  if (op.getCallTargetName() != "shape_assertion") {
    return mlir::failure();
  }
  rewriter.eraseOp(op);
  return success();
}
std::optional<bool> IsCustomCallLegal(mhlo::CustomCallOp op) {
  auto call_target_name = op.getCallTargetName();
  if (call_target_name.starts_with("custom_call.")) {
    auto bc = op.getBackendConfig();
    if (!bc || mlir::isa<mlir::StringAttr>(*bc)) {
      return false;
    }
  }
  return true;
}
void PopulateCustomCallPatterns(MLIRContext* ctx, RewritePatternSet& patterns,
                                ConversionTarget& target) {
  patterns.add<ConvertCustomCallOp>(ctx);
  target.addDynamicallyLegalOp<mhlo::CustomCallOp>(IsCustomCallLegal);
}
void PopulateCustomCallPreparePatterns(MLIRContext* ctx,
                                       RewritePatternSet& patterns) {
  patterns.add<RemoveCustomCallWithShapeAssertion>(ctx);
}
}  
}  