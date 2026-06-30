#include "tensorflow/compiler/mlir/tensorflow/utils/shape_inference_utils.h"
#include <optional>
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"
#define DEBUG_TYPE "tf-shape-inference-utils"
namespace mlir {
namespace TF {
LogicalResult InferReturnTypeComponentsForTFOp(
    std::optional<Location> location, Operation* op, int64_t graph_version,
    tfg::OperandAsConstantFn operand_as_constant_fn,
    tfg::OpResultAsShapeFn op_result_as_shape_fn,
    tfg::ResultElementTypeFn result_element_type_fn,
    SmallVectorImpl<ShapedTypeComponents>& inferred_return_shapes) {
  assert(op->getName().getDialectNamespace() ==
         TensorFlowDialect::getDialectNamespace());
  return tfg::InferReturnTypeComponentsForTFOp(
      location, op, op->getOperands(), graph_version, operand_as_constant_fn,
      op_result_as_shape_fn, result_element_type_fn,
      tensorflow::GetAttrValuesFromOperation, inferred_return_shapes);
}
}  
}  