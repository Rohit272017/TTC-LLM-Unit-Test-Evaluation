#include <memory>
#include <string>
#include <utility>
#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  
#include "mlir/IR/Attributes.h"  
#include "mlir/IR/Builders.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/BuiltinTypes.h"  
#include "mlir/IR/MLIRContext.h"  
#include "mlir/IR/OpDefinition.h"  
#include "mlir/IR/Operation.h"  
#include "mlir/IR/OperationSupport.h"  
#include "mlir/IR/PatternMatch.h"  
#include "mlir/IR/Types.h"  
#include "mlir/Pass/Pass.h"  
#include "mlir/Pass/PassRegistry.h"  
#include "mlir/Support/LLVM.h"  
#include "mlir/Support/LogicalResult.h"  
#include "mlir/Support/TypeID.h"  
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_config.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/tf_op_quant_spec.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/framework/types.pb.h"
namespace mlir {
namespace quant {
namespace {
using ::tensorflow::quantization::OpSet;
enum QuantizationTrait { kFullQuantization, kDynamicRangeQuantization };
template <QuantizationTrait quantization_trait, typename ConcreteT,
          typename RootOpT = quantfork::DequantizeCastOp>
struct TFQuantizationBase
    : public QuantizationPattern<ConcreteT, quantfork::QuantizeCastOp,
                                 quantfork::DequantizeCastOp,
                                 void, RootOpT> {
  explicit TFQuantizationBase(MLIRContext* ctx,
                              const QuantPassSpec& quant_params)
      : QuantizationPattern<ConcreteT, quantfork::QuantizeCastOp,
                            quantfork::DequantizeCastOp,
                            void, RootOpT>(ctx, quant_params) {}
  static bool IsQuantizableCustomOp(Operation* op,
                                    const CustomMap& custom_op_map) {
    return false;
  }
  static bool AllowDynamicRangeQuantizedOperand(
      Operation* quantized_op, const CustomMap& custom_op_map) {
    auto call_op = cast<TF::PartitionedCallOp>(quantized_op);
    StringRef function_name =
        call_op.getFAttr().cast<FlatSymbolRefAttr>().getValue();
    const bool is_gather = function_name.contains("gather");
    return quantization_trait != kFullQuantization || is_gather;
  }
  static bool AllowDynamicRangeQuantizedResult(Operation* quantized_op,
                                               const CustomMap& custom_op_map) {
    auto call_op = cast<TF::PartitionedCallOp>(quantized_op);
    StringRef function_name =
        call_op.getFAttr().cast<FlatSymbolRefAttr>().getValue();
    bool is_gather = false;
    if (function_name.contains("gather")) is_gather = true;
    return quantization_trait != kFullQuantization ||
           (quantization_trait == kFullQuantization && is_gather);
  }
  static bool IsWeightOnlyOp(Operation* quantized_op,
                             absl::flat_hash_set<std::string>& ops_blocklist,
                             bool weight_only_quantization,
                             const CustomMap& custom_op_map) {
    return weight_only_quantization;
  }
};
struct TFFullQuantization
    : public TFQuantizationBase<kFullQuantization, TFFullQuantization> {
  explicit TFFullQuantization(MLIRContext* ctx,
                              const QuantPassSpec& quant_params)
      : TFQuantizationBase<kFullQuantization, TFFullQuantization>(
            ctx, quant_params) {}
};
struct TFFullQuantizationReverse
    : public TFQuantizationBase<kFullQuantization, TFFullQuantizationReverse,
                                quantfork::QuantizeCastOp> {
  explicit TFFullQuantizationReverse(MLIRContext* ctx,
                                     const QuantPassSpec& quant_params)
      : TFQuantizationBase<kFullQuantization, TFFullQuantizationReverse,
                           quantfork::QuantizeCastOp>(ctx, quant_params) {}
};
struct TFDynamicRangeQuantization
    : public TFQuantizationBase<kDynamicRangeQuantization,
                                TFDynamicRangeQuantization> {
  explicit TFDynamicRangeQuantization(MLIRContext* ctx,
                                      const quant::QuantPassSpec& quant_params)
      : TFQuantizationBase<kDynamicRangeQuantization,
                           TFDynamicRangeQuantization>(ctx, quant_params) {}
};
class RemoveUnusedQdqPattern
    : public OpRewritePattern<quantfork::DequantizeCastOp> {
 public:
  explicit RemoveUnusedQdqPattern(MLIRContext* context)
      : OpRewritePattern<quantfork::DequantizeCastOp>(context) {}
  LogicalResult matchAndRewrite(quantfork::DequantizeCastOp dq_op,
                                PatternRewriter& rewriter) const override {
    auto q_op = dq_op.getArg().getDefiningOp<quantfork::QuantizeCastOp>();
    if (!q_op) return failure();
    dq_op.replaceAllUsesWith(q_op.getArg());
    return success();
  }
};
class QuantizeSameScaleOpsPattern
    : public OpRewritePattern<quantfork::DequantizeCastOp> {
 public:
  explicit QuantizeSameScaleOpsPattern(
      MLIRContext* context, OpQuantScaleSpecGetter op_quant_scale_spec_getter,
      OpSet target_opset)
      : OpRewritePattern<quantfork::DequantizeCastOp>(context, 200),
        op_quant_scale_spec_getter_(op_quant_scale_spec_getter),
        target_opset_(target_opset) {}
  LogicalResult matchAndRewrite(quantfork::DequantizeCastOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<Operation*, 4> quantizing_ops;
    auto users = op.getResult().getUsers();
    quantizing_ops.append(users.begin(), users.end());
    bool changed = false;
    for (Operation* quantizing_op : quantizing_ops) {
      if (llvm::isa<quantfork::QuantizeCastOp, quantfork::DequantizeCastOp>(
              quantizing_op)) {
        return failure();
      }
      if (quantizing_op->hasTrait<OpTrait::IsTerminator>()) {
        return failure();
      }
      if (!op_quant_scale_spec_getter_(quantizing_op)
               ->has_same_scale_requirement) {
        continue;
      }
      if (target_opset_ == OpSet::XLA &&
          !IsConnectedWithCompsiteFunction(quantizing_op)) {
        continue;
      }
      if (target_opset_ == OpSet::UNIFORM_QUANTIZED) {
        continue;
      }
      SmallVector<Value, 4> inputs;
      inputs.reserve(quantizing_op->getNumOperands());
      for (const auto& operand : quantizing_op->getOperands()) {
        Type operand_type = operand.getType();
        if (operand_type.isa<NoneType>()) {
          inputs.push_back(operand);
          continue;
        }
        Type elem_type = operand_type.cast<TensorType>().getElementType();
        if (auto dq_op = dyn_cast_or_null<quantfork::DequantizeCastOp>(
                operand.getDefiningOp())) {
          auto dq_arg_type = dq_op.getArg().getType().cast<TensorType>();
          auto qtype = dq_arg_type.getElementType().cast<QuantizedType>();
          auto scast_op = rewriter.create<quantfork::StorageCastOp>(
              dq_op->getLoc(), dq_arg_type.clone(qtype.getStorageType()),
              dq_op.getArg());
          inputs.push_back(scast_op.getResult());
        } else if (!elem_type.isF32()) {
          inputs.push_back(operand);
        } else {
          return failure();
        }
      }
      llvm::SmallDenseMap<Value, int> outputs_replaced;
      SmallVector<Type, 4> output_types;
      output_types.reserve(quantizing_op->getNumResults());
      for (const auto& enumerated_result :
           llvm::enumerate(quantizing_op->getResults())) {
        Value result = enumerated_result.value();
        Type result_type = result.getType();
        if (result_type.isa<NoneType>()) {
          outputs_replaced.insert({result, enumerated_result.index()});
          output_types.push_back(result_type);
          continue;
        }
        auto result_tensor_type = result_type.cast<TensorType>();
        if (result.hasOneUse() &&
            llvm::isa<quantfork::QuantizeCastOp>(*result.user_begin())) {
          auto user =
              llvm::cast<quantfork::QuantizeCastOp>(*result.user_begin());
          outputs_replaced.insert(
              {user.getResult(), enumerated_result.index()});
          auto qtype = user.getType()
                           .cast<TensorType>()
                           .getElementType()
                           .cast<QuantizedType>();
          output_types.push_back(
              result_tensor_type.clone(qtype.getStorageType()));
        } else if (!result_tensor_type.getElementType().isF32()) {
          outputs_replaced.insert({result, enumerated_result.index()});
          output_types.push_back(result.getType());
        } else {
          return failure();
        }
      }
      rewriter.setInsertionPointAfter(quantizing_op);
      OperationState new_state(quantizing_op->getLoc(),
                               quantizing_op->getName().getStringRef(), inputs,
                               output_types, quantizing_op->getAttrs());
      for (int i = 0; i < quantizing_op->getNumRegions(); ++i) {
        new_state.addRegion();
      }
      Operation* quantized_op = rewriter.create(new_state);
      if (quantizing_op->getNumRegions() != 0) {
        for (const auto& indexed_regions :
             llvm::enumerate(quantizing_op->getRegions())) {
          IRMapping mapping;
          indexed_regions.value().cloneInto(
              &quantized_op->getRegion(indexed_regions.index()), mapping);
        }
      }
      for (const auto& output_index_pair : outputs_replaced) {
        Value output = output_index_pair.getFirst();
        int output_index = output_index_pair.getSecond();
        auto scast_op = rewriter.create<quantfork::StorageCastOp>(
            output.getLoc(), output.getType(),
            quantized_op->getResult(output_index));
        output.replaceAllUsesWith(scast_op);
      }
      changed = true;
    }
    return success(changed);
  }
 private:
  bool IsConnectedWithCompsiteFunction(Operation* same_scale_op) const {
    for (const auto& operand : same_scale_op->getOperands()) {
      auto dq_op = dyn_cast_or_null<quantfork::DequantizeCastOp>(
          operand.getDefiningOp());
      if (!dq_op) continue;
      Operation* preceding_op = dq_op.getArg().getDefiningOp();
      if (!preceding_op) continue;
      if (llvm::isa<TF::PartitionedCallOp>(preceding_op)) {
        auto call_op = llvm::cast<TF::PartitionedCallOp>(preceding_op);
        if (!IsCompositeFunction(call_op)) continue;
        return true;
      }
      if (llvm::isa<quantfork::StorageCastOp>(preceding_op)) {
        auto sc_op = llvm::cast<quantfork::StorageCastOp>(preceding_op);
        auto sc_arg_type = sc_op.getArg().getType().dyn_cast<TensorType>();
        if (sc_arg_type.getElementType().isInteger(8)) {
          return true;
        }
      }
    }
    for (const auto& result : same_scale_op->getResults()) {
      if (!result.hasOneUse() ||
          !llvm::isa<quantfork::QuantizeCastOp>(*result.user_begin())) {
        continue;
      }
      auto q_op = llvm::cast<quantfork::QuantizeCastOp>(*result.user_begin());
      for (auto following_op : q_op->getUsers()) {
        if (llvm::isa<TF::PartitionedCallOp>(following_op)) {
          auto call_op = llvm::cast<TF::PartitionedCallOp>(following_op);
          if (!IsCompositeFunction(call_op)) continue;
          return true;
        }
        if (llvm::isa<quantfork::StorageCastOp>(following_op)) {
          auto sc_op = llvm::cast<quantfork::StorageCastOp>(following_op);
          auto sc_arg_type = sc_op.getResult().getType().dyn_cast<TensorType>();
          if (sc_arg_type.getElementType().isInteger(8)) {
            return true;
          }
        }
      }
    }
    return false;
  }
  bool IsCompositeFunction(TF::PartitionedCallOp call_op) const {
    if (!call_op->hasAttr(kQuantTraitAttrName)) {
      return false;
    }
    const auto f_attr = call_op.getFAttr().dyn_cast<FlatSymbolRefAttr>();
    if (!f_attr || !f_attr.getValue().starts_with("composite_")) {
      return false;
    }
    bool has_quantized_types = false;
    for (Value input : call_op.getArgs()) {
      if (auto type = input.getType().dyn_cast<TensorType>()) {
        if (type.getElementType().isa<FloatType>()) {
          return false;
        }
        if (type.getElementType().isa<QuantizedType>()) {
          has_quantized_types = true;
        }
      }
    }
    for (Value output : call_op.getOutput()) {
      if (auto type = output.getType().dyn_cast<TensorType>()) {
        if (type.getElementType().isa<FloatType>()) {
          return false;
        }
        if (type.getElementType().isa<QuantizedType>()) {
          has_quantized_types = true;
        }
      }
    }
    return has_quantized_types;
  }
  OpQuantScaleSpecGetter op_quant_scale_spec_getter_;
  OpSet target_opset_;
};
struct QuantizeAvgPoolOpPattern
    : public OpRewritePattern<quantfork::StorageCastOp> {
  explicit QuantizeAvgPoolOpPattern(MLIRContext* context)
      : OpRewritePattern<quantfork::StorageCastOp>(context, 100) {}
  LogicalResult matchAndRewrite(quantfork::StorageCastOp sc_op,
                                PatternRewriter& rewriter) const override {
    auto avg_pool_op = sc_op.getArg().getDefiningOp<TF::AvgPoolOp>();
    if (!avg_pool_op) return failure();
    auto preceding_sc_op = dyn_cast_or_null<quantfork::StorageCastOp>(
        avg_pool_op.getValue().getDefiningOp());
    if (!preceding_sc_op) return failure();
    auto dq_arg_type = preceding_sc_op.getArg().getType().cast<TensorType>();
    auto qtype = dq_arg_type.getElementType().cast<QuantizedType>();
    auto q_result_type = sc_op.getType().cast<TensorType>();
    auto out_qtype = q_result_type.getElementType().cast<QuantizedType>();
    if (qtype != out_qtype) {
      avg_pool_op.emitError(
          "The preceding StorageCastOp and the following "
          "StorageCastOp must have the same quantized type");
      return failure();
    }
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(preceding_sc_op);
    auto fcast_op = rewriter.create<TF::CastOp>(
        preceding_sc_op->getLoc(), dq_arg_type.clone(rewriter.getF32Type()),
        preceding_sc_op.getResult());
    TF::AvgPoolOp float_avg_pool_op = rewriter.create<TF::AvgPoolOp>(
        avg_pool_op->getLoc(),
        avg_pool_op.getType().clone(rewriter.getF32Type()),
        fcast_op.getResult(),
        avg_pool_op->getAttrs());
    auto round_val = rewriter.create<TF::RoundOp>(
        sc_op.getLoc(), float_avg_pool_op.getOutput());
    auto icast_op = rewriter.create<TF::CastOp>(
        sc_op.getLoc(), q_result_type.clone(qtype.getStorageType()), round_val);
    avg_pool_op.getResult().replaceAllUsesWith(icast_op.getResult());
    return success();
  }
};
class QuantizePass
    : public PassWrapper<QuantizePass, OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantizePass)
  explicit QuantizePass() {
    quant_specs_.inference_type = tensorflow::DT_QINT8;
  }
  explicit QuantizePass(const QuantizationSpecs& quant_specs,
                        OpSet target_opset)
      : quant_specs_(quant_specs) {
    weight_quantization_ = quant_specs.weight_quantization;
    target_opset_ = target_opset;
  }
  QuantizePass(const QuantizePass& other) : quant_specs_(other.quant_specs_) {
    weight_quantization_ = other.weight_quantization_;
    target_opset_ = other.target_opset_;
  }
  StringRef getArgument() const final {
    return "quant-quantize";
  }
  StringRef getDescription() const final {
    return "Apply quantization on models in TensorFlow dialect";
  }
  bool shouldKeepUnusedQdqPattern();
  void runOnOperation() override;
 private:
  QuantizationSpecs quant_specs_;
  Option<bool> weight_quantization_{
      *this, "weight-quantization", llvm::cl::init(false),
      llvm::cl::desc("Whether to enable weight quantization.")};
  Option<OpSet> target_opset_{
      *this, "target-opset", llvm::cl::init(OpSet::TF),
      llvm::cl::desc("Choose target opset."),
      llvm::cl::values(
          clEnumValN(OpSet::TF, "TF",
                     "Uses TF ops that mimic quantization behavior"),
          clEnumValN(OpSet::XLA, "XLA", "Uses TF XLA ops"),
          clEnumValN(OpSet::UNIFORM_QUANTIZED, "UNIFORM_QUANTIZED",
                     "Uses TF Uniform Quantized ops"))};
};
bool QuantizePass::shouldKeepUnusedQdqPattern() {
  return target_opset_ == OpSet::XLA &&
         (quant_specs_.weight_only_quantization ||
          quant_specs_.weight_quantization);
}
void QuantizePass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();
  auto* ctx = func.getContext();
  quant_specs_.weight_quantization = weight_quantization_;
  const QuantPassSpec quant_params = {
      {quant_specs_.verify_numeric, 5.0f,
       quant_specs_.whole_model_verify, false},
      quant_specs_};
  if (quant_specs_.weight_quantization) {
    patterns.add<TFDynamicRangeQuantization>(ctx, quant_params);
  } else {
    patterns.add<TFFullQuantization, TFFullQuantizationReverse>(ctx,
                                                                quant_params);
    patterns.add<QuantizeSameScaleOpsPattern>(ctx, GetTfQuantScaleSpec,
                                              target_opset_);
    patterns.add<QuantizeAvgPoolOpPattern>(ctx);
  }
  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
    func.emitWarning("Failed to converge pattern at QuantizePass.");
  }
  if (!shouldKeepUnusedQdqPattern()) {
    RewritePatternSet patterns_2(&getContext());
    patterns_2.add<RemoveUnusedQdqPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns_2)))) {
      signalPassFailure();
    }
  }
}
}  
std::unique_ptr<OperationPass<func::FuncOp>> CreateQuantizePass() {
  QuantizationSpecs quant_specs;
  return std::make_unique<QuantizePass>(quant_specs, OpSet::TF);
}
std::unique_ptr<OperationPass<func::FuncOp>> CreateQuantizePass(
    QuantizationSpecs quant_specs, OpSet target_opset) {
  return std::make_unique<QuantizePass>(quant_specs, target_opset);
}
static PassRegistration<QuantizePass> pass;
}  
}  