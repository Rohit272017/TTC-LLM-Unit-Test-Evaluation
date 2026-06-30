#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/get_dimension_size.h"
#include <cstdint>
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  
#include "mlir/IR/Builders.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/BuiltinTypeInterfaces.h"  
#include "mlir/IR/ImplicitLocOpBuilder.h"  
#include "mlir/IR/PatternMatch.h"  
#include "mlir/Support/LLVM.h"  
#include "mlir/Support/LogicalResult.h"  
#include "mlir/Transforms/DialectConversion.h"  
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
namespace mlir::odml {
namespace {
class LeagalizeDimensionSizeOp
    : public OpConversionPattern<mhlo::GetDimensionSizeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::GetDimensionSizeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    auto operand_type = llvm::cast<ShapedType>(op.getOperand().getType());
    auto shaped_op_type =
        RankedTensorType::get({operand_type.getRank()}, rewriter.getI64Type());
    Value shape_op = rewriter.create<TFL::ShapeOp>(op.getLoc(), shaped_op_type,
                                                   op.getOperand());
    Value size = BuildIntArrayConstOp<arith::ConstantOp>(builder, rewriter, {1},
                                                         rewriter.getI64Type());
    auto begin = BuildIntArrayConstOp<arith::ConstantOp>(
        builder, rewriter,
        llvm::SmallVector<int64_t>({static_cast<int64_t>(op.getDimension())}),
        rewriter.getI64Type());
    auto slice_type = RankedTensorType::get({1}, rewriter.getI64Type());
    Value slice = rewriter.create<TFL::SliceOp>(op.getLoc(), slice_type,
                                                shape_op, begin, size);
    auto op_el_type = llvm::cast<ShapedType>(op.getType()).getElementType();
    if (op_el_type != slice_type.getElementType()) {
      slice = rewriter.create<TFL::CastOp>(op->getLoc(),
                                           slice_type.clone(op_el_type), slice);
    }
    rewriter.replaceOpWithNewOp<TFL::SqueezeOp>(op, op.getType(), slice,
                                                rewriter.getI64ArrayAttr({0}));
    return success();
  }
};
}  
void PopulateGetDimensionSizePatterns(MLIRContext* ctx,
                                      RewritePatternSet& patterns,
                                      ConversionTarget& target) {
  target.addIllegalOp<mhlo::GetDimensionSizeOp>();
  patterns.add<LeagalizeDimensionSizeOp>(ctx);
}
}  