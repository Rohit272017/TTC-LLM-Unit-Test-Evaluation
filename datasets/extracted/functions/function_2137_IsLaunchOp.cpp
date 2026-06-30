#include <memory>
#include <string>
#include "mlir/Pass/Pass.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/Diagnostics.h"  
#include "mlir/IR/Visitors.h"  
#include "mlir/Support/LLVM.h"  
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
namespace mlir {
namespace TFDevice {
namespace {
constexpr char kXlaOutsideCompilationAttr[] = "_xla_outside_compilation";
#define GEN_PASS_DEF_VERIFYNOOUTSIDECOMPILATIONMARKERSPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_device_passes.h.inc"
class VerifyNoOutsideCompilationMarkersPass
    : public impl::VerifyNoOutsideCompilationMarkersPassBase<
          VerifyNoOutsideCompilationMarkersPass> {
 public:
  void runOnOperation() override;
};
bool IsLaunchOp(Operation& op) {
  return dyn_cast<tf_device::LaunchOp>(op) != nullptr;
}
bool IsDeviceClusterOp(Operation& op) {
  return dyn_cast<tf_device::ClusterOp>(op) != nullptr;
}
bool HasChildLaunchDeviceOp(Operation& op) {
  auto cluster_op = dyn_cast<tf_device::ClusterOp>(op);
  if (cluster_op == nullptr) return false;
  auto walk_result = cluster_op->walk([&](Operation* op) {
    if (IsLaunchOp(*op)) return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return walk_result.wasInterrupted();
}
bool HasXlaOutsideCompilationMarker(Operation& op) {
  return op.getAttrOfType<StringAttr>(kXlaOutsideCompilationAttr) != nullptr;
}
void VerifyNoOutsideCompilationMarkersPass::runOnOperation() {
  Operation* func_op = getOperation();
  auto walk_result = func_op->walk([&](Operation* op) {
    if (IsDeviceClusterOp(*op) && HasChildLaunchDeviceOp(*op)) {
      std::string launch_error =
          absl::StrCat("Node `", op->getName().getStringRef().str(), "` ",
                       "is a launch op which should have been removed by "
                       "outside compilation");
      op->emitError() << launch_error;
      LOG(ERROR) << launch_error;
      return WalkResult::interrupt();
    }
    if (HasXlaOutsideCompilationMarker(*op)) {
      std::string outside_compilation_error = absl::StrCat(
          "Node `", op->getName().getStringRef().str(), "` ",
          "has _xla_outside_compilation set which should have been removed by "
          "outside compilation");
      op->emitError() << outside_compilation_error;
      LOG(ERROR) << outside_compilation_error;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walk_result.wasInterrupted()) {
    signalPassFailure();
  }
}
}  
std::unique_ptr<mlir::OperationPass<func::FuncOp>>
CreateVerifyNoOutsideCompilationMarkersPass() {
  return std::make_unique<VerifyNoOutsideCompilationMarkersPass>();
}
}  
}  