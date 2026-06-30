#include <memory>
#include <string>
#include <utility>
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/IR/BuiltinTypeInterfaces.h"  
#include "mlir/IR/BuiltinTypes.h"  
#include "mlir/IR/Matchers.h"  
#include "mlir/IR/Operation.h"  
#include "mlir/IR/OperationSupport.h"  
#include "mlir/IR/PatternMatch.h"  
#include "mlir/IR/TypeUtilities.h"  
#include "mlir/IR/Value.h"  
#include "mlir/Pass/Pass.h"  
#include "mlir/Support/LLVM.h"  
#include "mlir/Support/LogicalResult.h"  
#include "mlir/Transforms/DialectConversion.h"  
#include "tensorflow/compiler/mlir/quantization/stablehlo/utils/tf_type_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/lib/monitoring/counter.h"
namespace mlir::quant::stablehlo {
namespace {
using quant::tensorflow::GetDenseAttrFromTensorProtoAttr;
using quant::tensorflow::GetIntTypeFromTFQint;
using quant::tensorflow::IsTFQintType;
using quant::tensorflow::IsTFUniformQuantizedOp;
#define GEN_PASS_DEF_CONVERTTFQUANTTYPES
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h.inc"
auto *mlir_tf_quant_op_count = ::tensorflow::monitoring::Counter<1>::New(
    "/tensorflow/core/tf2xla/tf_quant_op_count" ,
    "Counts the number of ops that has qint types" ,
    "op_name" );
bool IsIllegalType(Type type) {
  return IsTFQintType(getElementTypeOrSelf(type));
}
Type ToLegalType(Type type) {
  if (IsTFQintType(type)) return GetIntTypeFromTFQint(type);
  if (auto shaped = mlir::dyn_cast<ShapedType>(type)) {
    Type elem = shaped.getElementType();
    if (IsTFQintType(elem)) return shaped.clone(ToLegalType(elem));
  }
  return type;
}
bool IsQintToIntCast(Operation *op) {
  auto cast_op = llvm::dyn_cast<TF::CastOp>(op);
  return cast_op && IsIllegalType(cast_op.getX().getType()) &&
         ToLegalType(cast_op.getX().getType()) == cast_op.getY().getType();
}
bool IsIntToQintCast(Operation *op) {
  auto cast_op = llvm::dyn_cast<TF::CastOp>(op);
  return cast_op && IsIllegalType(cast_op.getY().getType()) &&
         ToLegalType(cast_op.getY().getType()) == cast_op.getX().getType();
}
bool IsQintValueQintToIntCast(Value v) {
  if (!IsIllegalType(v.getType())) {
    return true;
  }
  if (v.getUsers().empty()) {
    return false;
  }
  return llvm::all_of(v.getUsers(), [&](OpOperand operand) {
    return IsQintToIntCast(operand.getOwner());
  });
}
bool IsQintValueDefinedByIntToQintCast(Value v) {
  if (!IsIllegalType(v.getType())) {
    return true;
  }
  if (!v.getDefiningOp() || !llvm::isa<TF::CastOp>(v.getDefiningOp())) {
    return false;
  }
  return IsIntToQintCast(v.getDefiningOp());
}
bool IsTFUniformQuantizedOpLegal(Operation *op) {
  return op && llvm::all_of(op->getResults(), IsQintValueQintToIntCast) &&
         llvm::all_of(op->getOperands(), IsQintValueDefinedByIntToQintCast);
}
bool IsCastOpLegal(TF::CastOp cast_op) {
  if (IsIllegalType(cast_op.getSrcT()) && IsIllegalType(cast_op.getDstT())) {
    return false;
  }
  if (IsIllegalType(cast_op.getSrcT()) &&
      !(cast_op.getX().getDefiningOp() &&
        IsTFUniformQuantizedOp(cast_op.getX().getDefiningOp()))) {
    return false;
  }
  if (IsIllegalType(cast_op.getDstT()) &&
      !IsTFUniformQuantizedOp(*cast_op.getY().getUsers().begin())) {
    return false;
  }
  return true;
}
class TFQuantTypeConverter : public TypeConverter {
 public:
  TFQuantTypeConverter() {
    addConversion([](Type type) -> Type {
      return IsIllegalType(type) ? ToLegalType(type) : type;
    });
  }
};
class TFQuantTypeConversionTarget : public ConversionTarget {
 public:
  explicit TFQuantTypeConversionTarget(MLIRContext &ctx,
                                       TFQuantTypeConverter &converter)
      : ConversionTarget(ctx), converter_(converter) {
    markUnknownOpDynamicallyLegal([this](Operation *op) {
      if (IsTFUniformQuantizedOp(op)) {
        return IsTFUniformQuantizedOpLegal(op);
      } else if (auto cast_op = llvm::dyn_cast<TF::CastOp>(op)) {
        return IsCastOpLegal(cast_op);
      } else if (auto const_op = llvm::dyn_cast<TF::ConstOp>(op)) {
        return !IsIllegalType(const_op.getOutput().getType());
      }
      if (auto func = dyn_cast<func::FuncOp>(op)) {
        if (!converter_.isSignatureLegal(func.getFunctionType())) return false;
      }
      return converter_.isLegal(op);
    });
  }
 private:
  TFQuantTypeConverter &converter_;
};
class TFQuantTypePattern : public ConversionPattern {
 public:
  TFQuantTypePattern(MLIRContext *ctx, TypeConverter &converter)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, ctx) {}
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (IsTFUniformQuantizedOp(op) || llvm::isa<TF::ConstOp>(op)) {
      return failure();
    }
    llvm::SmallVector<Type, 4> new_results;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                new_results)))
      return failure();
    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         new_results, op->getAttrs(), op->getSuccessors());
    for (Region &region : op->getRegions()) {
      auto new_region = std::make_unique<Region>(op);
      rewriter.inlineRegionBefore(region, *new_region, new_region->begin());
      if (failed(rewriter.convertRegionTypes(new_region.get(),
                                             *getTypeConverter()))) {
        return failure();
      }
      state.addRegion(std::move(new_region));
    }
    rewriter.replaceOp(op, rewriter.create(state)->getResults());
    mlir_tf_quant_op_count->GetCell(std::string(op->getName().getStringRef()))
        ->IncrementBy(1);
    return success();
  }
};
class TFUniformQuantizedOpsPattern : public ConversionPattern {
 public:
  explicit TFUniformQuantizedOpsPattern(MLIRContext *ctx)
      : ConversionPattern(MatchAnyOpTypeTag(), 1, ctx) {}
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (!IsTFUniformQuantizedOp(op)) {
      return failure();
    }
    llvm::SmallVector<Value, 4> new_operands;
    for (int i = 0; i < operands.size(); ++i) {
      Type orig_op_type = op->getOperandTypes()[i];
      if (IsIllegalType(orig_op_type) &&
          !IsQintValueDefinedByIntToQintCast(op->getOperand(i))) {
        new_operands.push_back(rewriter.create<TF::CastOp>(
            op->getLoc(), orig_op_type, operands[i]));
      } else {
        new_operands.push_back(operands[i]);
      }
    }
    OperationState state(op->getLoc(), op->getName().getStringRef(),
                         new_operands, op->getResultTypes(), op->getAttrs(),
                         op->getSuccessors());
    Operation *new_op = rewriter.create(state);
    llvm::SmallVector<Value, 4> new_results = new_op->getResults();
    for (int i = 0; i < new_results.size(); ++i) {
      Value &result = new_results[i];
      if (IsIllegalType(result.getType()) &&
          !IsQintValueQintToIntCast(op->getResult(i))) {
        result = rewriter.create<TF::CastOp>(
            op->getLoc(), ToLegalType(result.getType()), result);
      }
      op->getResult(i).replaceUsesWithIf(
          new_op->getResult(i), [](OpOperand &operand) {
            return IsQintToIntCast(operand.getOwner());
          });
    }
    rewriter.replaceOp(op, new_results);
    return success();
  }
};
class TFConstOpQuantToIntPattern : public OpConversionPattern<TF::ConstOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      TF::ConstOp op, TF::ConstOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!IsIllegalType(op.getOutput().getType())) return failure();
    TF::TensorProtoAttr tensor_proto_attr;
    if (!matchPattern(op.getOperation(), m_Constant(&tensor_proto_attr))) {
      return rewriter.notifyMatchFailure(op, "operand must be constant.");
    }
    auto dense_attr_or = GetDenseAttrFromTensorProtoAttr(
        tensor_proto_attr.getValue(),
        mlir::dyn_cast<TensorType>(ToLegalType(op.getOutput().getType())));
    if (failed(dense_attr_or)) {
      op->emitError("failed to get DenseElementAttr.");
      return failure();
    }
    rewriter.replaceOpWithNewOp<TF::ConstOp>(
        op, ToLegalType(op.getOutput().getType()), *dense_attr_or);
    return success();
  }
};
struct ConvertTFQuantTypes
    : public impl::ConvertTFQuantTypesBase<ConvertTFQuantTypes> {
  void runOnOperation() override;
};
void ConvertTFQuantTypes::runOnOperation() {
  TFQuantTypeConverter converter;
  RewritePatternSet patterns(&getContext());
  patterns.add<TFQuantTypePattern>(&getContext(), converter);
  patterns.add<TFConstOpQuantToIntPattern, TFUniformQuantizedOpsPattern>(
      &getContext());
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 converter);
  TFQuantTypeConversionTarget target(getContext(), converter);
  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}
}  
std::unique_ptr<OperationPass<func::FuncOp>> CreateConvertTFQuantTypesPass() {
  return std::make_unique<ConvertTFQuantTypes>();
}
}  