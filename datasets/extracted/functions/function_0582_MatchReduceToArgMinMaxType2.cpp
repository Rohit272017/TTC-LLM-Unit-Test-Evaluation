#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/reduce.h"
#include <cstdint>
#include <optional>
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  
#include "mlir/IR/Block.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/BuiltinTypeInterfaces.h"  
#include "mlir/IR/BuiltinTypes.h"  
#include "mlir/IR/Matchers.h"  
#include "mlir/IR/Operation.h"  
#include "mlir/IR/PatternMatch.h"  
#include "mlir/IR/Value.h"  
#include "mlir/Support/LLVM.h"  
#include "mlir/Support/LogicalResult.h"  
#include "mlir/Transforms/DialectConversion.h"  
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/hlo_matchers.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/util.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
namespace mlir {
namespace odml {
LogicalResult MatchReduceToArgMinMaxType2(mhlo::ReduceOp reduce_op,
                                          bool is_argmax) {
  Block& body = reduce_op.getBody().front();
  if (body.getNumArguments() != 4) return failure();
  mhlo::ReturnOp return_op = dyn_cast<mhlo::ReturnOp>(body.back());
  if (!return_op || return_op.getNumOperands() != 2) return failure();
  mhlo::SelectOp value_select = llvm::dyn_cast_or_null<mhlo::SelectOp>(
      return_op.getOperand(0).getDefiningOp());
  if (!value_select || value_select.getOnTrue() != body.getArgument(0) ||
      value_select.getOnFalse() != body.getArgument(2))
    return failure();
  auto compare_direction_included =
      is_argmax ? mhlo::ComparisonDirection::GE : mhlo::ComparisonDirection::LE;
  mhlo::CompareOp value_gt = llvm::dyn_cast_or_null<mhlo::CompareOp>(
      value_select.getOperand(0).getDefiningOp());
  if (!value_gt ||
      value_gt.getComparisonDirection() != compare_direction_included ||
      value_gt.getLhs() != body.getArgument(0) ||
      value_gt.getRhs() != body.getArgument(2))
    return failure();
  mhlo::SelectOp index_select = llvm::dyn_cast_or_null<mhlo::SelectOp>(
      return_op.getOperand(1).getDefiningOp());
  if (!index_select) return failure();
  mhlo::MinOp index_select_min = llvm::dyn_cast_or_null<mhlo::MinOp>(
      index_select.getOnTrue().getDefiningOp());
  if (!index_select_min || index_select_min.getLhs() != body.getArgument(1) ||
      index_select_min.getRhs() != body.getArgument(3))
    return failure();
  mhlo::SelectOp index_select_select = llvm::dyn_cast_or_null<mhlo::SelectOp>(
      index_select.getOnFalse().getDefiningOp());
  if (!index_select_select ||
      index_select_select.getOnTrue() != body.getArgument(1) ||
      index_select_select.getOnFalse() != body.getArgument(3) ||
      index_select_select.getOperand(0).getDefiningOp() != value_gt)
    return failure();
  mhlo::CompareOp value_eq = llvm::dyn_cast_or_null<mhlo::CompareOp>(
      index_select.getOperand(0).getDefiningOp());
  if (!value_eq ||
      value_eq.getComparisonDirection() != mhlo::ComparisonDirection::EQ ||
      value_eq.getLhs() != body.getArgument(0) ||
      value_eq.getRhs() != body.getArgument(2))
    return failure();
  return success();
}
LogicalResult MatchReduceToArgMinMaxType1(mhlo::ReduceOp reduce_op,
                                          bool is_float, bool is_argmax) {
  Block& body = reduce_op.getBody().front();
  if (body.getNumArguments() != 4) return failure();
  mhlo::ReturnOp return_op = dyn_cast<mhlo::ReturnOp>(body.back());
  if (!return_op || return_op.getNumOperands() != 2) return failure();
  mhlo::SelectOp value_select = llvm::dyn_cast_or_null<mhlo::SelectOp>(
      return_op.getOperand(0).getDefiningOp());
  if (!value_select || value_select.getOnTrue() != body.getArgument(0) ||
      value_select.getOnFalse() != body.getArgument(2))
    return failure();
  auto compare_direction =
      is_argmax ? mhlo::ComparisonDirection::GT : mhlo::ComparisonDirection::LT;
  if (is_float) {
    mhlo::OrOp value_or = llvm::dyn_cast_or_null<mhlo::OrOp>(
        value_select.getOperand(0).getDefiningOp());
    if (!value_or) return failure();
    mhlo::CompareOp value_gt = llvm::dyn_cast_or_null<mhlo::CompareOp>(
        value_or.getLhs().getDefiningOp());
    if (!value_gt || value_gt.getComparisonDirection() != compare_direction ||
        value_gt.getLhs() != body.getArgument(0) ||
        value_gt.getRhs() != body.getArgument(2))
      return failure();
    mhlo::CompareOp value_ne = llvm::dyn_cast_or_null<mhlo::CompareOp>(
        value_or.getRhs().getDefiningOp());
    if (!value_ne ||
        value_ne.getComparisonDirection() != mhlo::ComparisonDirection::NE ||
        value_ne.getLhs() != body.getArgument(0) ||
        value_ne.getRhs() != body.getArgument(0))
      return failure();
  } else {
    mhlo::CompareOp value_gt = llvm::dyn_cast_or_null<mhlo::CompareOp>(
        value_select.getOperand(0).getDefiningOp());
    if (!value_gt || value_gt.getComparisonDirection() != compare_direction ||
        value_gt.getLhs() != body.getArgument(0) ||
        value_gt.getRhs() != body.getArgument(2))
      return failure();
  }
  mhlo::SelectOp index_select = llvm::dyn_cast_or_null<mhlo::SelectOp>(
      return_op.getOperand(1).getDefiningOp());
  if (!index_select || index_select.getOnTrue() != body.getArgument(1) ||
      index_select.getOnFalse() != body.getArgument(3))
    return failure();
  mhlo::OrOp index_or = llvm::dyn_cast_or_null<mhlo::OrOp>(
      index_select.getPred().getDefiningOp());
  if (!index_or || index_or.getLhs() != value_select.getPred())
    return failure();
  mhlo::AndOp index_and =
      llvm::dyn_cast_or_null<mhlo::AndOp>(index_or.getRhs().getDefiningOp());
  if (!index_and) return failure();
  mhlo::CompareOp value_eq = llvm::dyn_cast_or_null<mhlo::CompareOp>(
      index_and.getLhs().getDefiningOp());
  if (!value_eq ||
      value_eq.getComparisonDirection() != mhlo::ComparisonDirection::EQ ||
      value_eq.getLhs() != body.getArgument(0) ||
      value_eq.getRhs() != body.getArgument(2))
    return failure();
  mhlo::CompareOp index_lt = llvm::dyn_cast_or_null<mhlo::CompareOp>(
      index_and.getRhs().getDefiningOp());
  if (!index_lt ||
      index_lt.getComparisonDirection() != mhlo::ComparisonDirection::LT ||
      index_lt.getLhs() != body.getArgument(1) ||
      index_lt.getRhs() != body.getArgument(3))
    return failure();
  return success();
}
template <typename Reduce, typename ArgReduce, typename BooleanReduce,
          bool is_argmax>
class ConvertReduceOpToArgMinMax : public OpConversionPattern<mhlo::ReduceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ReduceOp reduce_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;
  virtual bool IsValueInitValue(const DenseElementsAttr& attr) const = 0;
};
template <typename Reduce, typename ArgReduce, typename BooleanReduce,
          bool is_argmax>
LogicalResult ConvertReduceOpToArgMinMax<
    Reduce, ArgReduce, BooleanReduce,
    is_argmax>::matchAndRewrite(mhlo::ReduceOp reduce_op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const {
  if (reduce_op.getInputs().size() != 2) return failure();
  if (reduce_op.getDimensions().getNumElements() != 1) return failure();
  DenseElementsAttr operand_init;
  if (!matchPattern(reduce_op.getInitValues().front(),
                    m_Constant(&operand_init)))
    return failure();
  if (!IsValueInitValue(operand_init)) return failure();
  DenseElementsAttr iota_init;
  if (!matchPattern(reduce_op.getInitValues().back(), m_Constant(&iota_init)))
    return failure();
  if (iota_init.getValues<APInt>()[0] != 0) return failure();
  Value iota = reduce_op.getInputs().back();
  if (!MatchIota(reduce_op.getDimensions(), iota)) return failure();
  const bool is_float = mlir::isa<FloatType>(operand_init.getElementType());
  if (failed(MatchReduceToArgMinMaxType1(reduce_op, is_float, is_argmax)) &&
      failed(MatchReduceToArgMinMaxType2(reduce_op, is_argmax)))
    return rewriter.notifyMatchFailure(
        reduce_op, "Unsupported Reduce -> ArgMax/ArgMin pattern");
  Value operand = reduce_op.getInputs().front();
  int64_t axis = reduce_op.getDimensions().getValues<int64_t>()[0];
  auto dim_type = RankedTensorType::get({1}, rewriter.getI32Type());
  auto reduction_indices = rewriter.create<arith::ConstantOp>(
      reduce_op.getLoc(), dim_type,
      rewriter.getI32TensorAttr({static_cast<int32_t>(axis)}));
  if (!mlir::isa<ShapedType>(operand.getType())) return failure();
  auto operand_type = mlir::cast<ShapedType>(operand.getType());
  if (operand_type.getElementType().isInteger(1)) {
    auto tf_reduce_op = rewriter.create<BooleanReduce>(
        reduce_op.getLoc(), reduce_op->getResult(0).getType(), operand,
        reduction_indices,
        rewriter.getBoolAttr(false));
    auto tf_argreduce_op = rewriter.create<ArgReduce>(
        reduce_op.getLoc(), reduce_op->getResult(1).getType(), operand,
        reduction_indices);
    rewriter.replaceOp(reduce_op, {tf_reduce_op, tf_argreduce_op});
  } else {
    auto tf_reduce_op = rewriter.create<Reduce>(
        reduce_op.getLoc(), reduce_op->getResult(0).getType(), operand,
        reduction_indices,
        rewriter.getBoolAttr(false));
    auto tf_argreduce_op = rewriter.create<ArgReduce>(
        reduce_op.getLoc(), reduce_op->getResult(1).getType(), operand,
        reduction_indices);
    rewriter.replaceOp(reduce_op, {tf_reduce_op, tf_argreduce_op});
  }
  return success();
}
template <typename Reduce, typename ArgReduce, typename BooleanReduce>
class ConvertReduceOpToArgMax
    : public ConvertReduceOpToArgMinMax<Reduce, ArgReduce, BooleanReduce,
                                        true> {
 public:
  using ConvertReduceOpToArgMinMax<Reduce, ArgReduce, BooleanReduce,
                                   true>::ConvertReduceOpToArgMinMax;
  bool IsValueInitValue(const DenseElementsAttr& attr) const override;
};
template <typename Reduce, typename ArgReduce, typename BooleanReduce>
bool ConvertReduceOpToArgMax<Reduce, ArgReduce, BooleanReduce>::
    IsValueInitValue(const DenseElementsAttr& attr) const {
  auto element_type = attr.getType().getElementType();
  if (attr.getNumElements() != 1 || !element_type.isIntOrFloat()) return false;
  if (mlir::isa<FloatType>(element_type)) {
    auto value = *attr.value_begin<APFloat>();
    return value.isNegative() && value.isInfinity();
  } else if (element_type.isInteger(1)) {
    auto value = *attr.value_begin<APInt>();
    return value.isZero();
  } else {
    auto value = *attr.value_begin<APInt>();
    return element_type.isUnsignedInteger() ? value.isMinValue()
                                            : value.isMinSignedValue();
  }
}
template <typename Reduce, typename ArgReduce, typename BooleanReduce>
class ConvertReduceOpToArgMin
    : public ConvertReduceOpToArgMinMax<Reduce, ArgReduce, BooleanReduce,
                                        false> {
 public:
  using ConvertReduceOpToArgMinMax<Reduce, ArgReduce, BooleanReduce,
                                   false>::ConvertReduceOpToArgMinMax;
  bool IsValueInitValue(const DenseElementsAttr& attr) const override;
};
template <typename Reduce, typename ArgReduce, typename BooleanReduce>
bool ConvertReduceOpToArgMin<Reduce, ArgReduce, BooleanReduce>::
    IsValueInitValue(const DenseElementsAttr& attr) const {
  auto element_type = attr.getType().getElementType();
  if (attr.getNumElements() != 1 || !element_type.isIntOrFloat()) return false;
  if (mlir::isa<FloatType>(element_type)) {
    auto value = *attr.value_begin<APFloat>();
    return !value.isNegative() && value.isInfinity();
  } else if (element_type.isInteger(1)) {
    auto value = *attr.value_begin<APInt>();
    return value.isZero();
  } else {
    auto value = *attr.value_begin<APInt>();
    return element_type.isUnsignedInteger() ? value.isMaxValue()
                                            : value.isMaxSignedValue();
  }
}
template <typename SplatValueType>
LogicalResult GetConstantSplatValue(Value value, SplatValueType& splat_value) {
  DenseElementsAttr attr;
  if (!matchPattern(value, m_Constant(&attr)) || !attr.isSplat()) {
    return failure();
  }
  splat_value = attr.getSplatValue<SplatValueType>();
  return success();
}
template <typename ReduceOp, typename BinaryOp, bool BuilderHasFAF = false>
LogicalResult rewriteNonMatchInitValue(mhlo::ReduceOp reduce_op, Value input,
                                       arith::ConstantOp reduction_indices,
                                       ConversionPatternRewriter& rewriter) {
  Value reduce_result = rewriter.create<ReduceOp>(
      reduce_op.getLoc(), reduce_op.getType(0), input, reduction_indices,
      rewriter.getBoolAttr(false));
  if constexpr (BuilderHasFAF) {
    rewriter.replaceOpWithNewOp<BinaryOp>(reduce_op, reduce_result,
                                          reduce_op.getInitValues()[0],
                                          rewriter.getStringAttr("NONE"));
  } else {
    rewriter.replaceOpWithNewOp<BinaryOp>(reduce_op, reduce_result.getType(),
                                          reduce_result,
                                          reduce_op.getInitValues()[0]);
  }
  return success();
}
DenseIntElementsAttr GetDimsAsI32Elements(OpBuilder& b, mhlo::ReduceOp op) {
  auto dims_attr = op.getDimensions();
  const auto n_dims = dims_attr.getNumElements();
  SmallVector<int32_t> reduce_dims;
  reduce_dims.reserve(n_dims);
  for (auto dim : dims_attr.getValues<int64_t>()) {
    reduce_dims.push_back(dim);
  }
  auto dim_type = RankedTensorType::get({n_dims}, b.getI32Type());
  return DenseIntElementsAttr::get(dim_type, reduce_dims);
}
template <>
LogicalResult rewriteNonMatchInitValue<TFL::ReduceMaxOp, void>(
    mhlo::ReduceOp reduce_op, Value input, arith::ConstantOp reduction_indices,
    ConversionPatternRewriter& rewriter) {
  return failure();
}
template <>
LogicalResult rewriteNonMatchInitValue<TFL::ReduceMinOp, void>(
    mhlo::ReduceOp reduce_op, Value input, arith::ConstantOp reduction_indices,
    ConversionPatternRewriter& rewriter) {
  return failure();
}
template <>
LogicalResult rewriteNonMatchInitValue<TFL::ReduceAnyOp, void>(
    mhlo::ReduceOp reduce_op, Value input, arith::ConstantOp reduction_indices,
    ConversionPatternRewriter& rewriter) {
  return failure();
}
template <typename SrcBinaryOp, typename TargetReduceOp,
          typename TargetBinaryOp = void, bool BuilderHasFAF = false>
class ConvertReduce : public OpConversionPattern<mhlo::ReduceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ReduceOp reduce_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (failed(MatchReduceOpOperand(reduce_op))) {
      return failure();
    }
    if (failed(MatchBinaryReduceFunction<SrcBinaryOp>(reduce_op.getBody()))) {
      return failure();
    }
    auto operand = reduce_op.getInputs()[0];
    auto tfl_dims = GetDimsAsI32Elements(rewriter, reduce_op);
    auto tfl_dims_op =
        rewriter.create<arith::ConstantOp>(reduce_op.getLoc(), tfl_dims);
    if (succeeded(MatchInitValue(reduce_op.getInitValues()[0]))) {
      rewriter.replaceOpWithNewOp<TargetReduceOp>(
          reduce_op, reduce_op.getType(0), operand, tfl_dims_op,
          rewriter.getBoolAttr(false));
      return success();
    }
    return rewriteNonMatchInitValue<TargetReduceOp, TargetBinaryOp,
                                    BuilderHasFAF>(reduce_op, operand,
                                                   tfl_dims_op, rewriter);
  }
 private:
  virtual LogicalResult MatchInitValue(Value init_value) const = 0;
  LogicalResult MatchReduceOpOperand(mhlo::ReduceOp reduce_op) const {
    if (reduce_op.getInputs().size() != 1 ||
        reduce_op.getInitValues().size() != 1 ||
        reduce_op.getResults().size() != 1)
      return failure();
    if (!mlir::isa<RankedTensorType>(reduce_op.getInputs()[0].getType()))
      return failure();
    if (!mlir::isa<RankedTensorType>(reduce_op.getType(0))) return failure();
    return success();
  }
};
class ConvertReduceMul
    : public ConvertReduce<mhlo::MulOp, TFL::ReduceProdOp, TFL::MulOp, true> {
 public:
  using ConvertReduce::ConvertReduce;
  LogicalResult MatchInitValue(Value init_value) const override {
    auto type = mlir::cast<ShapedType>(init_value.getType()).getElementType();
    if (mlir::isa<FloatType>(type)) {
      float const_value;
      if (failed(GetConstantSplatValue<float>(init_value, const_value)) ||
          const_value != 1.0)
        return failure();
    } else if (mlir::isa<IntegerType>(type) && type.isSignlessInteger()) {
      int32_t const_value;
      if (failed(GetConstantSplatValue<int32_t>(init_value, const_value)) ||
          const_value != 1)
        return failure();
    } else {
      return failure();
    }
    return success();
  }
};
class ConvertReduceAdd
    : public ConvertReduce<mhlo::AddOp, TFL::SumOp, TFL::AddOp, true> {
 public:
  using ConvertReduce::ConvertReduce;
  LogicalResult MatchInitValue(Value init_value) const override {
    auto type = mlir::cast<ShapedType>(init_value.getType()).getElementType();
    if (mlir::isa<FloatType>(type)) {
      APFloat const_value(.0);
      if (failed(GetConstantSplatValue(init_value, const_value)) ||
          !const_value.isZero())
        return failure();
    } else if (mlir::isa<IntegerType>(type) && type.isSignlessInteger()) {
      APInt const_value;
      if (failed(GetConstantSplatValue(init_value, const_value)) ||
          !const_value.isZero())
        return failure();
    } else {
      return failure();
    }
    return success();
  }
};
class ConvertReduceMaxToReduceAny
    : public ConvertReduce<mhlo::MaxOp, TFL::ReduceAnyOp> {
 public:
  using ConvertReduce::ConvertReduce;
  LogicalResult MatchInitValue(Value init_value) const override {
    auto type = mlir::cast<ShapedType>(init_value.getType()).getElementType();
    if (!mlir::isa<IntegerType>(type) || !type.isSignlessInteger() ||
        !(type.getIntOrFloatBitWidth() == 1))
      return failure();
    APInt const_value;
    if (failed(GetConstantSplatValue(init_value, const_value)) ||
        (const_value == 1))
      return failure();
    return success();
  }
};
class ConvertReduceMax : public ConvertReduce<mhlo::MaxOp, TFL::ReduceMaxOp> {
 public:
  using ConvertReduce::ConvertReduce;
  LogicalResult MatchInitValue(Value init_value) const override {
    auto type = mlir::cast<ShapedType>(init_value.getType()).getElementType();
    if (mlir::isa<FloatType>(type)) {
      APFloat const_value(.0);
      if (failed(GetConstantSplatValue(init_value, const_value)) ||
          !const_value.isInfinity() || !const_value.isNegative())
        return failure();
    } else if (mlir::isa<IntegerType>(type) && type.isSignlessInteger()) {
      if (type.getIntOrFloatBitWidth() == 1) return failure();
      APInt const_value;
      if (failed(GetConstantSplatValue(init_value, const_value)) ||
          !const_value.isMinSignedValue())
        return failure();
    } else {
      return failure();
    }
    return success();
  }
};
class ConvertReduceMin : public ConvertReduce<mhlo::MinOp, TFL::ReduceMinOp> {
 public:
  using ConvertReduce::ConvertReduce;
  LogicalResult MatchInitValue(Value init_value) const override {
    auto type = mlir::cast<ShapedType>(init_value.getType()).getElementType();
    if (mlir::isa<FloatType>(type)) {
      APFloat const_value(.0);
      if (failed(GetConstantSplatValue(init_value, const_value)) ||
          !const_value.isInfinity() || const_value.isNegative())
        return failure();
    } else if (mlir::isa<IntegerType>(type) && type.isSignlessInteger()) {
      APInt const_value;
      if (failed(GetConstantSplatValue(init_value, const_value)) ||
          !const_value.isMaxSignedValue())
        return failure();
    } else {
      return failure();
    }
    return success();
  }
};
class ConvertReduceAnd
    : public ConvertReduce<mhlo::AndOp, TFL::ReduceAllOp, TFL::LogicalAndOp> {
 public:
  using ConvertReduce<mhlo::AndOp, TFL::ReduceAllOp,
                      TFL::LogicalAndOp>::ConvertReduce;
  LogicalResult MatchInitValue(Value init_value) const override {
    DenseIntElementsAttr init_attr;
    if (!matchPattern(init_value, m_Constant(&init_attr)) ||
        !init_attr.getType().getElementType().isInteger(1) ||
        !init_attr.isSplat() || !init_attr.getSplatValue<BoolAttr>().getValue())
      return failure();
    return success();
  }
};
class ConvertReduceOr
    : public ConvertReduce<mhlo::OrOp, TFL::ReduceAnyOp, TFL::LogicalOrOp> {
 public:
  using ConvertReduce<mhlo::OrOp, TFL::ReduceAnyOp,
                      TFL::LogicalOrOp>::ConvertReduce;
  LogicalResult MatchInitValue(Value init_value) const override {
    DenseIntElementsAttr init_attr;
    if (!matchPattern(init_value, m_Constant(&init_attr)) ||
        !init_attr.getType().getElementType().isInteger(1) ||
        !init_attr.isSplat() || init_attr.getSplatValue<BoolAttr>().getValue())
      return failure();
    return success();
  }
};
std::optional<bool> IsReduceOpLegal(mhlo::ReduceOp reduce_op) {
  if (succeeded(MatchReduceToArgMinMaxType1(reduce_op, true, true)) ||
      succeeded(MatchReduceToArgMinMaxType1(reduce_op, false, true)) ||
      succeeded(MatchReduceToArgMinMaxType1(reduce_op, true, false)) ||
      succeeded(MatchReduceToArgMinMaxType1(reduce_op, false, false)) ||
      succeeded(MatchReduceToArgMinMaxType2(reduce_op, false)) ||
      succeeded(MatchReduceToArgMinMaxType2(reduce_op, true))) {
    return false;
  }
  return std::nullopt;
}
template class ConvertReduceOpToArgMinMax<TFL::ReduceMaxOp, TFL::ArgMaxOp,
                                          TFL::ReduceAnyOp, true>;
template class ConvertReduceOpToArgMax<TFL::ReduceMaxOp, TFL::ArgMaxOp,
                                       TFL::ReduceAnyOp>;
template class ConvertReduceOpToArgMinMax<TFL::ReduceMinOp, TFL::ArgMinOp,
                                          TFL::ReduceAllOp, false>;
template class ConvertReduceOpToArgMin<TFL::ReduceMinOp, TFL::ArgMinOp,
                                       TFL::ReduceAllOp>;
template class ConvertReduceOpToArgMinMax<TF::MaxOp, TF::ArgMaxOp, TF::AnyOp,
                                          true>;
template class ConvertReduceOpToArgMax<TF::MaxOp, TF::ArgMaxOp, TF::AnyOp>;
template class ConvertReduceOpToArgMinMax<TF::MinOp, TF::ArgMinOp, TF::AllOp,
                                          false>;
template class ConvertReduceOpToArgMin<TF::MinOp, TF::ArgMinOp, TF::AllOp>;
void PopulateReduceArgMinMaxTFPatterns(MLIRContext* ctx,
                                       RewritePatternSet& patterns) {
  using ConvertReduceOpToTfArgmax =
      ConvertReduceOpToArgMax<TF::MaxOp, TF::ArgMaxOp, TF::AnyOp>;
  using ConvertReduceOpToTfArgmin =
      ConvertReduceOpToArgMin<TF::MinOp, TF::ArgMinOp, TF::AllOp>;
  patterns.add<ConvertReduceOpToTfArgmin, ConvertReduceOpToTfArgmax>(ctx);
}
void PopulateReducePatterns(MLIRContext* ctx, RewritePatternSet& patterns,
                            ConversionTarget& target) {
  using ConvertReduceOpToTFLiteArgmax =
      ConvertReduceOpToArgMax<TFL::ReduceMaxOp, TFL::ArgMaxOp,
                              TFL::ReduceAnyOp>;
  using ConvertReduceOpToTFLiteArgmin =
      ConvertReduceOpToArgMin<TFL::ReduceMinOp, TFL::ArgMinOp,
                              TFL::ReduceAllOp>;
  patterns.add<ConvertReduceOpToTFLiteArgmax, ConvertReduceOpToTFLiteArgmin,
               ConvertReduceMul, ConvertReduceAdd, ConvertReduceMax,
               ConvertReduceMaxToReduceAny, ConvertReduceMin, ConvertReduceAnd,
               ConvertReduceOr>(ctx);
  target.addDynamicallyLegalOp<mhlo::ReduceOp>(IsReduceOpLegal);
}
}  
}  