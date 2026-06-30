#include "tensorflow/core/ir/interfaces.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"  
#include "mlir/IR/Region.h"  
#include "mlir/IR/Value.h"  
#include "mlir/Interfaces/SideEffectInterfaces.h"  
#include "mlir/Support/LLVM.h"  
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/ir/types/dialect.h"
namespace mlir {
namespace tfg {
LogicalResult ControlArgumentInterface::verifyRegion(Operation *op,
                                                     Region &region) {
  unsigned num_ctl = 0, num_data = 0;
  for (BlockArgument arg : region.getArguments()) {
    bool is_ctl = mlir::isa<tf_type::ControlType>(arg.getType());
    num_ctl += is_ctl;
    num_data += !is_ctl;
  }
  if (num_ctl != num_data) {
    return op->emitOpError("region #")
           << region.getRegionNumber()
           << " expected same number of data values and control tokens ("
           << num_data << " vs. " << num_ctl << ")";
  }
  return success();
}
void StatefulMemoryEffectInterface::getEffects(
    Operation *op,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) const {
  auto registry = dyn_cast<TensorFlowRegistryInterface>(op);
  if (!registry || registry.isStateful() || op->getParentOfType<GraphOp>()) {
    effects.emplace_back(MemoryEffects::Write::get());
  }
}
}  
}  
#include "tensorflow/core/ir/interfaces.cc.inc"