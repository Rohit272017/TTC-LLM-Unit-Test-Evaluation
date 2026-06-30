#include <memory>
#include <string>
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/IR/BuiltinOps.h"  
#include "mlir/IR/Operation.h"  
#include "mlir/Pass/Pass.h"  
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/lib/monitoring/counter.h"
namespace mlir {
namespace tf2xla {
namespace internal {
auto* has_tpu_partitioned_call_streamz =
    tensorflow::monitoring::Counter<1>::New(
        "/tensorflow/core/tf2xla/internal/inference/tpu_partitioned_call",
        "Whether the model has TPUPartitionedCallOp.",
        "has_tpu_partitioned_call");
namespace {
#define GEN_PASS_DEF_INFERENCEMETRICSPASS
#include "tensorflow/compiler/mlir/tf2xla/internal/inference/inference_passes.h.inc"
class InferenceMetricsPass
    : public impl::InferenceMetricsPassBase<InferenceMetricsPass> {
 public:
  void runOnOperation() override;
};
void InferenceMetricsPass::runOnOperation() {
  bool has_tpu_partitioned_call = false;
  ModuleOp module = getOperation();
  for (auto func_op : module.getOps<func::FuncOp>()) {
    func_op->walk(
        [&](TF::TPUPartitionedCallOp op) { has_tpu_partitioned_call = true; });
    if (has_tpu_partitioned_call) break;
  }
  std::string has_tpu_partitioned_call_str =
      has_tpu_partitioned_call ? "true" : "false";
  has_tpu_partitioned_call_streamz->GetCell(has_tpu_partitioned_call_str)
      ->IncrementBy(1);
}
}  
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateInferenceMetricsPass() {
  return std::make_unique<InferenceMetricsPass>();
}
}  
}  
}  