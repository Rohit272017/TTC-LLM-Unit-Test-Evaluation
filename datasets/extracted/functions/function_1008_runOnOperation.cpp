#include <memory>
#include <string>
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/Operation.h"  
#include "mlir/IR/Visitors.h"  
#include "mlir/Pass/Pass.h"  
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/utils/dialect_detection_utils.h"
namespace tensorflow {
namespace tf2xla {
namespace internal {
namespace {
#define GEN_PASS_DEF_VERIFYCLUSTERINGPASS
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/clustering_passes.h.inc"
using mlir::Operation;
using mlir::WalkResult;
class VerifyClusteringPass
    : public impl::VerifyClusteringPassBase<VerifyClusteringPass> {
 public:
  void runOnOperation() override;
};
void VerifyClusteringPass::runOnOperation() {
  Operation* func_op = getOperation();
  auto walk_result = func_op->walk([&](Operation* op) {
    if (!tensorflow::tf2xla::internal::IsInBridgeAcceptableDialects(op)) {
      std::string error = "op is in dialect " +
                          op->getDialect()->getNamespace().str() +
                          " not in tf functional dialect";
      op->emitError() << error;
      return WalkResult::interrupt();
    }
    if (op->hasAttr(mlir::TF::kXlaOutsideCompilationAttr)) {
      std::string error =
          "op has outside compilation attribute _xla_outside_compilation which "
          "is not allowed after clustering";
      op->emitError() << error;
      return mlir::WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walk_result.wasInterrupted()) {
    signalPassFailure();
  }
}
}  
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateVerifyClusteringPass() {
  return std::make_unique<VerifyClusteringPass>();
}
}  
}  
}  