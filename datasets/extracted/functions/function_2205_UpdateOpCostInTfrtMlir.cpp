#include "tensorflow/compiler/mlir/tfrt/transforms/update_op_cost_in_tfrt_mlir.h"
#include "mlir/IR/Builders.h"  
#include "mlir/IR/BuiltinOps.h"  
#include "mlir/IR/Operation.h"  
#include "tensorflow/compiler/mlir/tfrt/analysis/cost_analysis.h"
#include "tensorflow/core/tfrt/fallback/cost_recorder.h"
namespace tensorflow {
namespace tfrt_compiler {
constexpr char kCostAttrName[] = "_tfrt_cost";
constexpr char kOpKeyAttrName[] = "op_key";
void UpdateOpCostInTfrtMlir(mlir::ModuleOp op,
                            const tfrt_stub::CostRecorder& cost_recorder) {
  mlir::Builder builder(op);
  op.walk([&](mlir::Operation* op) {
    if (HasCostFunctionRegistered(op->getName().getStringRef())) return;
    const auto cost_attr = op->getAttrOfType<mlir::IntegerAttr>(kCostAttrName);
    if (!cost_attr) return;
    const auto op_key_attr =
        op->getAttrOfType<mlir::IntegerAttr>(kOpKeyAttrName);
    if (!op_key_attr) return;
    const int64_t op_key = op_key_attr.getInt();
    op->setAttr(kCostAttrName, builder.getI64IntegerAttr(
                                   cost_recorder.GetCost(op_key)));
  });
}
}  
}  