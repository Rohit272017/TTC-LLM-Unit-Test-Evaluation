#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/util.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Operation.h"  
#include "tensorflow/compiler/mlir/tensorflow/ir/host_runtime/tfrt_ops.h.inc"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h.inc"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h.inc"
namespace tensorflow {
namespace mlrt_compiler {
bool UseFallback(mlir::Operation *op) {
  if (!llvm::isa<mlir::TF::TensorFlowDialect>(op->getDialect())) return false;
  return !llvm::isa<
      mlir::TF::_TfrtSetResourceOp, mlir::TF::_TfrtGetResourceOp,
      mlir::TF::BatchFunctionOp, mlir::TF::CaseOp,
      mlir::TF::IfrtRestoreVariableOp, mlir::TF::StatefulPartitionedCallOp,
      mlir::TF::PartitionedCallOp, mlir::TF::LegacyCallOp, mlir::TF::IfOp,
      mlir::TF::WhileOp, mlir::TF::TPUCompileMlirAndExecuteOp>(op);
}
}  
}  