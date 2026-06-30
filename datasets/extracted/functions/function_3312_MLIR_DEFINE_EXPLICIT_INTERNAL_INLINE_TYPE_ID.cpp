#include <memory>
#include <optional>
#include <utility>
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/Dialect/Quant/IR/Quant.h"  
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/BuiltinOps.h"  
#include "mlir/IR/BuiltinTypeInterfaces.h"  
#include "mlir/IR/BuiltinTypes.h"  
#include "mlir/IR/DialectRegistry.h"  
#include "mlir/IR/Matchers.h"  
#include "mlir/IR/OpDefinition.h"  
#include "mlir/IR/OperationSupport.h"  
#include "mlir/IR/PatternMatch.h"  
#include "mlir/IR/Value.h"  
#include "mlir/IR/Verifier.h"  
#include "mlir/IR/Visitors.h"  
#include "mlir/Interfaces/CallInterfaces.h"  
#include "mlir/Pass/Pass.h"  
#include "mlir/Pass/PassManager.h"  
#include "mlir/Pass/PassRegistry.h"  
#include "mlir/Rewrite/FrozenRewritePatternSet.h"  
#include "mlir/Support/LLVM.h"  
#include "mlir/Support/LogicalResult.h"  
#include "mlir/Support/TypeID.h"  
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  
#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/tf_op_quant_spec.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/tf_quantize_op.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
namespace mlir {
namespace quant {
namespace {
class QuantizeWeightsPass
    : public mlir::PassWrapper<QuantizeWeightsPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantizeWeightsPass)
  explicit QuantizeWeightsPass() : test_mode_(true) { initializeForTest(); }
  explicit QuantizeWeightsPass(
      const tensorflow::quantization::QuantizationOptions& quant_options)
      : test_mode_(false), quant_options_(quant_options) {}
  QuantizeWeightsPass(const QuantizeWeightsPass& other) {
    test_mode_ = other.test_mode_;
    quant_options_ = other.quant_options_;
    initializeForTest();
  }
  StringRef getArgument() const final {
    return "quant-quantize-weights";
  }
  StringRef getDescription() const final {
    return "Quantize weights used by quantizable ops.";
  }
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect, quant::QuantDialect>();
  }
 private:
  void runOnOperation() override;
  bool test_mode_;
  tensorflow::quantization::QuantizationOptions quant_options_;
  void initializeForTest() {
    if (!test_mode_) return;
    tensorflow::quantization::QuantizationComponentSpec quant_spec;
    quant_spec.set_quantization_component(
        tensorflow::quantization::QuantizationComponentSpec::COMPONENT_WEIGHT);
    quant_spec.set_tensor_type(
        tensorflow::quantization::QuantizationComponentSpec::TENSORTYPE_INT_8);
    auto mutable_quant_method = quant_options_.mutable_quantization_method();
    *mutable_quant_method->add_quantization_component_specs() = quant_spec;
  }
};
class QuantizeConstWeights : public OpRewritePattern<TF::ConstOp> {
 public:
  explicit QuantizeConstWeights(
      MLIRContext* context,
      const tensorflow::quantization::QuantizationOptions& quantization_options)
      : OpRewritePattern<TF::ConstOp>(context),
        quant_options_(quantization_options) {}
  LogicalResult matchAndRewrite(TF::ConstOp op,
                                PatternRewriter& rewriter) const override {
    auto weight_component_spec = GetWeightComponentSpec(quant_options_);
    if (!weight_component_spec) return failure();
    if (failed((isQuantizableWeight(op)))) {
      return failure();
    }
    if (failed(quantizeOps(rewriter, op, weight_component_spec.value()))) {
      return failure();
    }
    return success();
  }
 private:
  bool checkIfAnyUserIsConnectedToTermiantor(BlockArgument op) const {
    for (const auto& user : op.getUsers()) {
      if (user->template hasTrait<OpTrait::IsTerminator>()) return true;
      if (auto next_user = dyn_cast_or_null<TF::IdentityOp>(user)) {
        return (*(next_user->getResult(0).getUsers().begin()))
            ->template hasTrait<OpTrait::IsTerminator>();
      }
    }
    return false;
  }
  bool hasUsageFromQuantizableOp(TF::ConstOp op) const {
    llvm::SmallVector<mlir::Value> uses_at_current_level{op};
    while (!uses_at_current_level.empty()) {
      llvm::SmallVector<mlir::Value> next_values_to_visit;
      for (auto cur_op : uses_at_current_level) {
        for (auto& cur_op_use : cur_op.getUses()) {
          Operation* next_op = cur_op_use.getOwner();
          int next_op_operand_num = cur_op_use.getOperandNumber();
          if (auto call_op = llvm::dyn_cast<mlir::CallOpInterface>(next_op)) {
            mlir::func::FuncOp func =
                llvm::dyn_cast<mlir::func::FuncOp>(call_op.resolveCallable());
            if (!func) continue;
            next_values_to_visit.push_back(
                func.getArgument(next_op_operand_num));
          } else if (auto while_op =
                         llvm::dyn_cast_or_null<TF::WhileOp>(next_op)) {
            func::FuncOp func = while_op.body_function();
            auto func_argument = func.getArgument(next_op_operand_num);
            if (checkIfAnyUserIsConnectedToTermiantor(func_argument))
              next_values_to_visit.push_back(
                  func.getArgument(next_op_operand_num));
          } else if (IsOpWithQuantizableTrait(next_op)) {
            return true;
          } else if (IsOpWithDataMovementTrait(next_op)) {
            next_values_to_visit.insert(next_values_to_visit.end(),
                                        next_op->getResults().begin(),
                                        next_op->getResults().end());
          }
        }
      }
      uses_at_current_level.swap(next_values_to_visit);
    }
    return false;
  }
  LogicalResult isQuantizableWeight(TF::ConstOp op) const {
    if (!IsValueWithQuantizablePrecision(op)) return failure();
    if (!hasUsageFromQuantizableOp(op)) return failure();
    int num_elements_threshold = quant_options_.min_num_elements_for_weights();
    int num_elements = cast<ShapedType>(op.getType()).getNumElements();
    if (num_elements < num_elements_threshold) {
      op->emitRemark("Quantization is skipped because the op has ")
          << num_elements << " elements which is fewer than the threshold("
          << num_elements_threshold << " elements).";
      return failure();
    }
    return success();
  }
  LogicalResult quantizeOps(PatternRewriter& rewriter, TF::ConstOp op,
                            tensorflow::quantization::QuantizationComponentSpec&
                                weight_component_spec) const {
    if (weight_component_spec.tensor_type() ==
        tensorflow::quantization::QuantizationComponentSpec::TENSORTYPE_INT_8) {
      auto dequantized_val =
          ApplyUniformQuantization(rewriter, op, weight_component_spec);
      if (!dequantized_val.has_value()) return failure();
      op.getOutput().replaceAllUsesWith(dequantized_val.value().getResult(0));
      return success();
    }
    op->emitRemark("Not supported quantization data type.");
    return failure();
  }
 protected:
  tensorflow::quantization::QuantizationOptions quant_options_;
};
static PassRegistration<QuantizeWeightsPass> pass;
void QuantizeWeightsPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  auto module_op = getOperation();
  RewritePatternSet patterns(ctx);
  patterns.add<QuantizeConstWeights>(ctx, quant_options_);
  FrozenRewritePatternSet frozen_patterns(std::move(patterns));
  for (auto func : module_op.getOps<func::FuncOp>()) {
    if (failed(applyPatternsAndFoldGreedily(func, frozen_patterns))) {
      func.emitError() << "quant-quantize-weights failed.";
      signalPassFailure();
    }
  }
}
}  
std::unique_ptr<OperationPass<ModuleOp>> CreateQuantizeWeightsPass(
    const tensorflow::quantization::QuantizationOptions& quant_options) {
  return std::make_unique<QuantizeWeightsPass>(quant_options);
}
}  
}  