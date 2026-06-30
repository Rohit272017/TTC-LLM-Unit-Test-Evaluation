#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/slice.h"
#include <cstdint>
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  
#include "mlir/IR/Builders.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/BuiltinTypeInterfaces.h"  
#include "mlir/IR/PatternMatch.h"  
#include "mlir/IR/Value.h"  
#include "mlir/IR/ValueRange.h"  
#include "mlir/Support/LLVM.h"  
#include "mlir/Support/LogicalResult.h"  
#include "mlir/Transforms/DialectConversion.h"  
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/op_util_common.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
namespace mlir::odml {
namespace {
Value PackScalarIndices(mlir::ValueRange indices, OpBuilder& b) {
  auto e_type =
      llvm::cast<ShapedType>(indices.front().getType()).getElementType();
  const int64_t num_indices = indices.size();
  auto packed_indices_type = RankedTensorType::get({num_indices}, e_type);
  auto values_count_attr = b.getI32IntegerAttr(num_indices);
  auto pack_axis_attr = b.getI32IntegerAttr(0);
  return b.create<TFL::PackOp>(indices.back().getLoc(), packed_indices_type,
                               indices, values_count_attr, pack_axis_attr);
}
Value BuildTFLCastOp(OpBuilder& b, Value value) {
  return b.create<TFL::CastOp>(
      value.getLoc(),
      RankedTensorType::get(llvm::cast<ShapedType>(value.getType()).getShape(),
                            b.getI32Type()),
      value);
}
class LegalizeSliceOp : public OpConversionPattern<mhlo::SliceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::SliceOp slice_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto begin = rewriter.create<arith::ConstantOp>(slice_op.getLoc(),
                                                    slice_op.getStartIndices());
    auto end = rewriter.create<arith::ConstantOp>(slice_op.getLoc(),
                                                  slice_op.getLimitIndices());
    auto strides = rewriter.create<arith::ConstantOp>(slice_op.getLoc(),
                                                      slice_op.getStrides());
    auto zero = rewriter.getIntegerAttr(rewriter.getI32Type(), 0);
    auto no_offset = rewriter.getBoolAttr(false);
    rewriter.replaceOpWithNewOp<TFL::StridedSliceOp>(
        slice_op, slice_op.getType(), slice_op.getOperand(),
        BuildTFLCastOp(rewriter, begin), BuildTFLCastOp(rewriter, end),
        BuildTFLCastOp(rewriter, strides), zero, zero, zero, zero, zero,
        no_offset);
    return success();
  }
};
class CastSliceIndicesToSignless
    : public OpRewritePattern<mhlo::DynamicSliceOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::DynamicSliceOp op,
                                PatternRewriter& rewriter) const final;
};
LogicalResult CastSliceIndicesToSignless::matchAndRewrite(
    mhlo::DynamicSliceOp op, PatternRewriter& rewriter) const {
  auto start_type =
      llvm::cast<ShapedType>(op.getStartIndices().front().getType());
  auto start_e_type = start_type.getElementType();
  if (start_e_type.isSignlessIntOrFloat()) {
    return rewriter.notifyMatchFailure(op, "Already signless.");
  }
  auto new_start_e_type =
      rewriter.getIntegerType(start_e_type.getIntOrFloatBitWidth());
  llvm::SmallVector<Value> casted_start_inds;
  for (auto start_ind_opr : op.getStartIndices()) {
    auto casted_start_ind_opr = rewriter.create<mhlo::ConvertOp>(
        start_ind_opr.getLoc(), start_ind_opr, new_start_e_type);
    casted_start_inds.push_back(casted_start_ind_opr.getResult());
  }
  rewriter.replaceOpWithNewOp<mhlo::DynamicSliceOp>(
      op, op.getOperand(), casted_start_inds, op.getSliceSizes());
  return success();
}
bool IsDynamicSliceLegal(mhlo::DynamicSliceOp op) {
  return !llvm::cast<ShapedType>(op.getOperand().getType()).hasStaticShape();
}
class LegalizeDynamicSliceOp
    : public OpConversionPattern<mhlo::DynamicSliceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::DynamicSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
};
LogicalResult LegalizeDynamicSliceOp::matchAndRewrite(
    mhlo::DynamicSliceOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto start_type =
      llvm::cast<ShapedType>(op.getStartIndices().front().getType());
  auto start_e_type = start_type.getElementType();
  if (!start_e_type.isSignlessIntOrFloat()) {
    return rewriter.notifyMatchFailure(
        op, "Must be signless integer for start indices.");
  }
  auto input_type = llvm::cast<ShapedType>(op.getOperand().getType());
  if (!input_type.hasStaticShape()) {
    return rewriter.notifyMatchFailure(op, "Input must be statically shaped.");
  }
  Value clamp_left_cst = rewriter.create<arith::ConstantOp>(
      op->getLoc(), rewriter.getZeroAttr(start_type));
  llvm::SmallVector<Value> new_start_indices;
  const auto stride_sizes = UnrollI64Splat(op.getSliceSizes());
  for (auto [dim_size, start_ind_opr, stride_size] :
       llvm::zip(input_type.getShape(), op.getStartIndices(), stride_sizes)) {
    const int64_t clamp_right_val = dim_size - stride_size;
    auto clamp_right_cst = rewriter.create<arith::ConstantOp>(
        op->getLoc(),
        DenseElementsAttr::get(start_type, rewriter.getIntegerAttr(
                                               start_e_type, clamp_right_val)));
    Value new_start_ind = rewriter.create<TFL::MaximumOp>(
        op->getLoc(), start_type, clamp_left_cst, start_ind_opr);
    new_start_ind = rewriter.create<TFL::MinimumOp>(
        op->getLoc(), start_type, clamp_right_cst, new_start_ind);
    new_start_indices.push_back(new_start_ind);
  }
  auto packed_indices = PackScalarIndices(new_start_indices, rewriter);
  auto slice_sizes_cst =
      rewriter.create<arith::ConstantOp>(op->getLoc(), op.getSliceSizes());
  rewriter.replaceOpWithNewOp<TFL::SliceOp>(op, op.getType(), op.getOperand(),
                                            packed_indices, slice_sizes_cst);
  return success();
}
class LegalizeRealDynamicSliceOp
    : public OpConversionPattern<mhlo::RealDynamicSliceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::RealDynamicSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
};
LogicalResult LegalizeRealDynamicSliceOp::matchAndRewrite(
    mhlo::RealDynamicSliceOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto start_indices_type =
      mlir::cast<RankedTensorType>(op.getStartIndices().getType());
  auto end_indices_type =
      mlir::cast<RankedTensorType>(op.getLimitIndices().getType());
  if (start_indices_type.getNumDynamicDims() != 0 ||
      end_indices_type.getNumDynamicDims() != 0) {
    return rewriter.notifyMatchFailure(
        op,
        "Start indices and limit indices must not have dynamic dimensions.");
  }
  auto zero = rewriter.getIntegerAttr(rewriter.getI32Type(), 0);
  auto no_offset = rewriter.getBoolAttr(false);
  rewriter.replaceOpWithNewOp<TFL::StridedSliceOp>(
      op, op.getType(), op.getOperand(),
      BuildTFLCastOp(rewriter, op.getStartIndices()),
      BuildTFLCastOp(rewriter, op.getLimitIndices()),
      BuildTFLCastOp(rewriter, op.getStrides()), zero, zero, zero, zero, zero,
      no_offset);
  return success();
};
class LegalizeDynamicUpdateSliceOp
    : public OpConversionPattern<mhlo::DynamicUpdateSliceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::DynamicUpdateSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
};
LogicalResult LegalizeDynamicUpdateSliceOp::matchAndRewrite(
    mhlo::DynamicUpdateSliceOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto packed_indices = PackScalarIndices(op.getStartIndices(), rewriter);
  rewriter.replaceOpWithNewOp<TFL::DynamicUpdateSliceOp>(
      op, op.getType(), op.getOperand(), op.getUpdate(), packed_indices);
  return success();
};
}  
void PopulateLegalizeSlicePatterns(MLIRContext* ctx,
                                   RewritePatternSet& patterns,
                                   ConversionTarget& target) {
  patterns.add<LegalizeSliceOp, LegalizeDynamicSliceOp,
               LegalizeDynamicUpdateSliceOp, LegalizeRealDynamicSliceOp>(ctx);
  target.addIllegalOp<mhlo::SliceOp, mhlo::DynamicUpdateSliceOp,
                      mhlo::RealDynamicSliceOp>();
  target.addDynamicallyLegalOp<mhlo::DynamicSliceOp>(IsDynamicSliceLegal);
}
void PopulatePrepareSlicePatterns(MLIRContext* ctx,
                                  RewritePatternSet& patterns) {
  patterns.add<CastSliceIndicesToSignless>(ctx);
}
}  