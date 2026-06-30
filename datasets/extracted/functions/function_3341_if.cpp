#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/IR/Attributes.h"  
#include "mlir/IR/Builders.h"  
#include "mlir/IR/BuiltinOps.h"  
#include "mlir/IR/PatternMatch.h"  
#include "mlir/Pass/Pass.h"  
#include "mlir/Pass/PassManager.h"  
#include "mlir/Pass/PassRegistry.h"  
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  
#include "mlir/Transforms/Passes.h"  
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#define DEBUG_TYPE "tf-gpu-op-fusion"
namespace mlir {
namespace TF {
namespace {
#define GEN_PASS_DEF_TENSORFLOWGPUFUSION
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"
class GpuOpFusionPass : public impl::TensorflowGPUFusionBase<GpuOpFusionPass> {
 public:
  void runOnOperation() final;
};
struct ReluToFusedBatchNorm : public OpRewritePattern<ReluOp> {
  using OpRewritePattern<ReluOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ReluOp relu_op,
                                PatternRewriter &rewriter) const override {
    Operation *relu_input = relu_op.getFeatures().getDefiningOp();
    if (!relu_input) return failure();
    auto batch_norm = dyn_cast_or_null<FusedBatchNormV3Op>(relu_input);
    AddV2Op add_op;
    Value side_input;
    if (!batch_norm) {
      add_op = dyn_cast_or_null<AddV2Op>(relu_input);
      if (!add_op) return failure();
      batch_norm =
          dyn_cast_or_null<FusedBatchNormV3Op>(add_op.getX().getDefiningOp());
      if (batch_norm) {
        side_input = add_op.getY();
      } else {
        batch_norm =
            dyn_cast_or_null<FusedBatchNormV3Op>(add_op.getY().getDefiningOp());
        if (!batch_norm) return failure();
        side_input = add_op.getX();
      }
    }
    assert(batch_norm);
    if (batch_norm.getIsTraining()) return failure();
    if (!batch_norm.getY().hasOneUse()) return failure();
    OperationState state(batch_norm.getLoc(),
                         _FusedBatchNormExOp::getOperationName());
    state.addOperands(batch_norm.getOperands());
    if (side_input) state.operands.push_back(side_input);
    state.addTypes(batch_norm.getResultTypes());
    state.addAttributes(batch_norm->getAttrs());
    Operation *op = rewriter.create(state);
    rewriter.replaceOp(batch_norm, op->getResults());
    if (!add_op || add_op.getZ().hasOneUse()) {
      op->setAttr("activation_mode", rewriter.getStringAttr("Relu"));
      rewriter.replaceOp(relu_op, op->getResult(0));
    }
    if (add_op) {
      rewriter.replaceOp(add_op, op->getResult(0));
    }
    return success();
  }
};
void GpuOpFusionPass::runOnOperation() {
  func::FuncOp func = getOperation();
  RewritePatternSet patterns(&getContext());
  patterns.add<ReluToFusedBatchNorm>(&getContext());
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
}  
std::unique_ptr<OperationPass<func::FuncOp>> CreateGpuOpFusionPass() {
  return std::make_unique<GpuOpFusionPass>();
}
}  
}  