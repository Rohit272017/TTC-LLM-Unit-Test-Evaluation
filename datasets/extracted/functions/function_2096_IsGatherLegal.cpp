#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/gather.h"
#include <cstdint>
#include <optional>
#include <vector>
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  
#include "mlir/IR/BuiltinTypeInterfaces.h"  
#include "mlir/IR/BuiltinTypes.h"  
#include "mlir/IR/ImplicitLocOpBuilder.h"  
#include "mlir/IR/PatternMatch.h"  
#include "mlir/Support/LLVM.h"  
#include "mlir/Support/LogicalResult.h"  
#include "mlir/Transforms/DialectConversion.h"  
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
namespace mlir::odml {
std::optional<bool> IsGatherLegal(mhlo::GatherOp op) { return std::nullopt; }
class LegalizeGatherToSlice : public OpConversionPattern<mhlo::GatherOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::GatherOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
};
LogicalResult LegalizeGatherToSlice::matchAndRewrite(
    mhlo::GatherOp gather_op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Value operand = gather_op.getOperand();
  Value start_indices = gather_op.getStartIndices();
  static const int rank_two = 2;
  static const int max_batch_size = 50;
  ShapedType operand_type = mlir::cast<ShapedType>(operand.getType());
  ShapedType start_indices_type =
      mlir::cast<ShapedType>(start_indices.getType());
  ShapedType result_type =
      mlir::cast<ShapedType>(gather_op.getResult().getType());
  if (!operand_type.hasStaticShape() || !start_indices_type.hasStaticShape() ||
      !result_type.hasStaticShape()) {
    return rewriter.notifyMatchFailure(
        gather_op,
        "Dynamic shaped inputs are not supported when legalizing mhlo.gather "
        "op to tf.slice.");
  }
  auto start_index_map = gather_op.getDimensionNumbers().getStartIndexMap();
  auto collapsed_slice_dims =
      gather_op.getDimensionNumbers().getCollapsedSliceDims();
  auto offset_dims = gather_op.getDimensionNumbers().getOffsetDims();
  auto slice_sizes = gather_op.getSliceSizes();
  llvm::SmallVector<int64_t, 2> slice_sizes_vector;
  slice_sizes_vector.reserve(slice_sizes.size());
  for (int64_t s : slice_sizes.getValues<int64_t>()) {
    slice_sizes_vector.push_back(s);
  }
  llvm::SmallVector<int64_t, 1> batch_dims;
  int offset_index = 0;
  for (int64_t i = 0; i < result_type.getRank(); ++i) {
    if (offset_index >= offset_dims.size() || offset_dims[offset_index] != i) {
      batch_dims.push_back(i);
    } else {
      ++offset_index;
    }
  }
  if (batch_dims.size() != 1 || batch_dims[0] != 0) {
    return failure();
  }
  int64_t batch_dim = batch_dims[0];
  if (operand_type.getDimSize(batch_dim) > max_batch_size ||
      operand_type.getRank() != rank_two ||
      start_indices_type.getRank() != rank_two ||
      operand_type.getDimSize(batch_dim) !=
          start_indices_type.getDimSize(batch_dim) ||
      slice_sizes_vector[batch_dim] != 1) {
    return failure();
  }
  for (int64_t i = 0; i < start_index_map.size(); i++) {
    if (start_index_map[i] != i) {
      return failure();
    }
  }
  if (collapsed_slice_dims.size() != start_index_map.size() - 1 ||
      collapsed_slice_dims.size() != 1 || collapsed_slice_dims[0] != 0) {
    return failure();
  }
  int64_t index_vector_dim =
      gather_op.getDimensionNumbers().getIndexVectorDim();
  if (failed(NormalizeIndexVector(gather_op, start_indices, start_indices_type,
                                  index_vector_dim, rewriter))) {
    return failure();
  }
  ImplicitLocOpBuilder builder(gather_op.getLoc(), rewriter);
  auto max_start_indices = BuildIntArrayConstOp<arith::ConstantOp>(
      builder, rewriter,
      llvm::SmallVector<int64_t>(
          {operand_type.getDimSize(0) - slice_sizes_vector[0],
           operand_type.getDimSize(1) - slice_sizes_vector[1]}),
      start_indices_type.getElementType());
  auto min_start_indices = BuildIntArrayConstOp<arith::ConstantOp>(
      builder, rewriter, llvm::SmallVector<int64_t>({0, 0}),
      start_indices_type.getElementType());
  auto start_indices_max_op = rewriter.create<TFL::MaximumOp>(
      gather_op.getLoc(), start_indices, min_start_indices);
  auto clamped_start_indices_op = rewriter.create<TFL::MinimumOp>(
      gather_op.getLoc(), start_indices_max_op, max_start_indices);
  int64_t batch_size = start_indices_type.getDimSize(batch_dim);
  auto slice_size = BuildIntArrayConstOp<arith::ConstantOp>(
      builder, rewriter, slice_sizes_vector, rewriter.getI32Type());
  if (batch_size == 1) {
    auto squeeze_op = rewriter.create<TFL::SqueezeOp>(
        gather_op.getLoc(),
        RankedTensorType::get({rank_two}, start_indices_type.getElementType()),
        clamped_start_indices_op,
        rewriter.getI64ArrayAttr(llvm::ArrayRef<int64_t>({batch_dim})));
    auto slice_op =
        rewriter.create<TFL::SliceOp>(gather_op.getLoc(), gather_op.getType(),
                                      operand, squeeze_op, slice_size);
    rewriter.replaceOp(gather_op, slice_op);
    return mlir::success();
  }
  llvm::SmallVector<Value, 1> slices;
  slices.reserve(batch_size);
  for (int64_t i = 0; i < batch_size; ++i) {
    auto zero = BuildIntArrayConstOp<arith::ConstantOp>(
        builder, rewriter, llvm::SmallVector<int64_t>({i, 0}),
        rewriter.getI32Type());
    auto two = BuildIntArrayConstOp<arith::ConstantOp>(
        builder, rewriter, llvm::SmallVector<int64_t>({1, 2}),
        rewriter.getI32Type());
    auto begin = rewriter.create<TFL::SliceOp>(
        gather_op.getLoc(),
        RankedTensorType::get({1, 2}, start_indices_type.getElementType()),
        clamped_start_indices_op, zero, two);
    auto squeeze_op = rewriter.create<TFL::SqueezeOp>(
        gather_op.getLoc(),
        RankedTensorType::get({rank_two}, start_indices_type.getElementType()),
        begin, rewriter.getI64ArrayAttr(llvm::ArrayRef<int64_t>({batch_dim})));
    auto slice_op = rewriter.create<TFL::SliceOp>(
        gather_op.getLoc(),
        RankedTensorType::get({1, slice_sizes_vector[1]},
                              operand_type.getElementType()),
        operand, squeeze_op, slice_size);
    slices.push_back(slice_op);
  }
  auto concat_op = rewriter.create<TFL::ConcatenationOp>(
      gather_op.getLoc(), result_type, slices, 0,
      rewriter.getStringAttr("NONE"));
  rewriter.replaceOp(gather_op, concat_op);
  return mlir::success();
}
struct TransposeParams {
  std::vector<int64_t> permutation;
  std::vector<int64_t> canonicalized_output_shape;
  std::vector<int64_t> canonicalized_offset_dims;
};
TransposeParams CanonicalizeOffset(ShapedType result_type,
                                   ArrayRef<int64_t> original_offset_dims) {
  TransposeParams transpose_params;
  int output_rank = result_type.getRank();
  for (int start = output_rank - original_offset_dims.size();
       start < output_rank; ++start) {
    transpose_params.canonicalized_offset_dims.push_back(start);
  }
  std::vector<int64_t> batch_dims;
  int offset_index = 0;
  for (int64_t i = 0; i < output_rank; ++i) {
    if (offset_index >= original_offset_dims.size() ||
        original_offset_dims[offset_index] != i) {
      batch_dims.push_back(i);
    } else {
      ++offset_index;
    }
  }
  int batch_idx = 0;
  int offset_idx = 0;
  int batch_dim_size = batch_dims.size();
  for (int i = 0; i < output_rank; ++i) {
    if (batch_idx >= batch_dims.size()) {
      transpose_params.permutation.push_back(batch_dim_size + offset_idx);
      ++offset_idx;
    } else if (offset_idx < original_offset_dims.size() &&
               original_offset_dims[offset_idx] < batch_dims[batch_idx]) {
      transpose_params.permutation.push_back(batch_dim_size + offset_idx);
      ++offset_idx;
    } else {
      transpose_params.permutation.push_back(batch_idx++);
    }
  }
  for (auto dim : batch_dims) {
    transpose_params.canonicalized_output_shape.push_back(
        result_type.getDimSize(dim));
  }
  for (auto dim : original_offset_dims) {
    transpose_params.canonicalized_output_shape.push_back(
        result_type.getDimSize(dim));
  }
  return transpose_params;
}
class LegalizeGatherToGatherND : public OpConversionPattern<mhlo::GatherOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::GatherOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
};
LogicalResult LegalizeGatherToGatherND::matchAndRewrite(
    mhlo::GatherOp gather_op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Value operand = gather_op.getOperand();
  Value start_indices = gather_op.getStartIndices();
  ShapedType operand_type = mlir::cast<ShapedType>(operand.getType());
  ShapedType start_indices_type =
      mlir::cast<ShapedType>(start_indices.getType());
  ShapedType result_type =
      mlir::cast<ShapedType>(gather_op.getResult().getType());
  if (!operand_type.hasStaticShape()) {
    gather_op.emitOpError() << "Dynamic shaped operand is not supported.";
    return failure();
  }
  int64_t index_vector_dim =
      gather_op.getDimensionNumbers().getIndexVectorDim();
  if (failed(NormalizeIndexVector(gather_op, start_indices, start_indices_type,
                                  index_vector_dim, rewriter))) {
    return failure();
  }
  auto start_index_map = gather_op.getDimensionNumbers().getStartIndexMap();
  auto collapsed_slice_dims =
      gather_op.getDimensionNumbers().getCollapsedSliceDims();
  if (start_index_map.size() != collapsed_slice_dims.size()) {
    return rewriter.notifyMatchFailure(
        gather_op,
        "different size for start index map and collapsed slice dims");
  }
  for (auto c : collapsed_slice_dims) {
    if (llvm::count(start_index_map, c) == 0) {
      return rewriter.notifyMatchFailure(
          gather_op, "collapsed slice dim isn't present in start index map");
    }
  }
  auto slice_sizes = gather_op.getSliceSizes();
  int64_t index = 0;
  for (int64_t s : slice_sizes.getValues<int64_t>()) {
    if (llvm::count(start_index_map, index)) {
      if (s != 1) {
        return rewriter.notifyMatchFailure(gather_op,
                                           "unsupported slice sizes");
      }
    } else {
      if (s != operand_type.getShape()[index]) {
        return rewriter.notifyMatchFailure(gather_op,
                                           "unsupported slice sizes");
      }
    }
    ++index;
  }
  auto offset_dims = gather_op.getDimensionNumbers().getOffsetDims();
  SmallVector<int64_t, 4> offset_dims_vector(offset_dims.begin(),
                                             offset_dims.end());
  const TransposeParams& transpose_params =
      CanonicalizeOffset(result_type,
                         offset_dims_vector);
  int64_t offset = start_indices_type.getRank() - 1;
  for (int64_t o : transpose_params.canonicalized_offset_dims) {
    if (o != offset) {
      return rewriter.notifyMatchFailure(gather_op, "unsupported offset dims");
    }
    ++offset;
  }
  llvm::SmallVector<int64_t, 4> transpose_dimensions;
  llvm::SmallVector<int64_t, 4> transpose_shape;
  for (auto s : start_index_map) {
    transpose_dimensions.push_back(s);
    transpose_shape.push_back(operand_type.getShape()[s]);
  }
  for (int64_t i = 0, e = operand_type.getRank(); i < e; ++i) {
    if (llvm::count(start_index_map, i) == 0) {
      transpose_dimensions.push_back(i);
      transpose_shape.push_back(operand_type.getShape()[i]);
    }
  }
  operand_type =
      RankedTensorType::get(transpose_shape, operand_type.getElementType());
  operand = rewriter.create<mhlo::TransposeOp>(
      gather_op.getLoc(), operand_type, operand,
      rewriter.getI64TensorAttr(transpose_dimensions));
  bool need_transpose_after = false;
  for (int i = 0; i < transpose_params.permutation.size(); ++i) {
    if (i != transpose_params.permutation[i]) {
      need_transpose_after = true;
      break;
    }
  }
  auto tf_gather_nd_result_type =
      RankedTensorType::get(transpose_params.canonicalized_output_shape,
                            result_type.getElementType());
  if (start_indices_type.getElementType().isUnsignedInteger(32)) {
    start_indices = rewriter.create<TFL::CastOp>(
        gather_op->getLoc(),
        RankedTensorType::get(start_indices_type.getShape(),
                              rewriter.getI64Type()),
        start_indices);
  }
  auto tf_gather_nd_op = rewriter.create<TFL::GatherNdOp>(
      gather_op->getLoc(), tf_gather_nd_result_type, operand, start_indices);
  if (!need_transpose_after) {
    rewriter.replaceOp(gather_op, tf_gather_nd_op->getOpResults());
    return success();
  }
  rewriter.replaceOpWithNewOp<mhlo::TransposeOp>(
      gather_op, result_type, tf_gather_nd_op,
      rewriter.getI64TensorAttr(transpose_params.permutation));
  return success();
}
void PopulateGatherPatterns(MLIRContext* ctx, RewritePatternSet& patterns,
                            ConversionTarget& target) {
  patterns.add<LegalizeGatherToSlice, LegalizeGatherToGatherND>(ctx);
  target.addDynamicallyLegalOp<mhlo::GatherOp>(IsGatherLegal);
}
}  