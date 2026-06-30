#include "tensorflow/core/ir/tf_op_registry.h"
#include "mlir/IR/Dialect.h"  
#include "mlir/IR/Operation.h"  
#include "mlir/Support/LLVM.h"  
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/ir/interfaces.h"
#include "tensorflow/core/ir/ops.h"
namespace mlir {
namespace tfg {
TensorFlowOpRegistryInterface::TensorFlowOpRegistryInterface(Dialect *dialect)
    : TensorFlowOpRegistryInterface(dialect, tensorflow::OpRegistry::Global()) {
}
static bool IsStatefulImpl(const tensorflow::OpRegistry *registry,
                           StringRef op_name) {
  const tensorflow::OpRegistrationData *op_reg_data =
      registry->LookUp(op_name.str());
  if (!op_reg_data) return true;
  return op_reg_data->op_def.is_stateful();
}
bool TensorFlowOpRegistryInterface::isStateful(Operation *op) const {
  if (op->hasTrait<OpTrait::IntrinsicOperation>()) return false;
  if (auto func = dyn_cast<GraphFuncOp>(op)) return func.getIsStateful();
  StringRef op_name = op->getName().stripDialect();
  if (op->getNumRegions() && op_name.ends_with("Region"))
    op_name = op_name.drop_back(6);
  return IsStatefulImpl(registry_, op_name);
}
}  
}  