#include "tensorflow/compiler/mlir/tensorflow/transforms/constant_fold.h"
#include <algorithm>
#include <cstdint>
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"  
#include "mlir/IR/BuiltinAttributeInterfaces.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/BuiltinTypeInterfaces.h"  
#include "mlir/IR/BuiltinTypes.h"  
#include "mlir/IR/OpDefinition.h"  
#include "mlir/IR/TypeRange.h"  
#include "mlir/IR/Types.h"  
#include "mlir/Support/LLVM.h"  
#include "mlir/Support/LogicalResult.h"  
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/constant_fold_utils.h"
namespace mlir {
namespace TF {
static bool IsFoldedByDefaultPolicy(Operation* inst) {
  auto get_size = [&](TypeRange types) {
    int64_t size = 0;
    for (auto t : types) {
      auto tensor_type = mlir::cast<TensorType>(t);
      if (!tensor_type.getElementType().isIntOrFloat()) continue;
      if (!tensor_type.hasStaticShape()) continue;
      size += tensor_type.getNumElements() *
              tensor_type.getElementType().getIntOrFloatBitWidth();
    }
    return size;
  };
  int64_t results_size = get_size(inst->getResultTypes());
  int64_t operands_size = get_size(inst->getOperandTypes());
  constexpr int kSizeFactor = 2;
  constexpr int64_t kResultsSizeThreshold = (1 << 16);   
  constexpr int64_t kOperandsSizeThreshold = (1 << 30);  
  return (operands_size <= kOperandsSizeThreshold) &&
         ((results_size <= kResultsSizeThreshold) ||
          (results_size <= kSizeFactor * operands_size));
}
LogicalResult ConstantFoldFallbackHook(
    Operation* inst, ArrayRef<Attribute> operands,
    SmallVectorImpl<OpFoldResult>& results) {  
  if (!CanBeFolded(inst)) return failure();
  if (!IsFoldedByDefaultPolicy(inst)) return failure();
  bool has_empty_numerical_results =
      llvm::all_of(inst->getResultTypes(), [](Type ty) {
        ShapedType shaped_ty = mlir::cast<ShapedType>(ty);
        Type element_ty = shaped_ty.getElementType();
        return shaped_ty.hasStaticShape() && shaped_ty.getNumElements() == 0 &&
               element_ty.isIntOrFloat();
      });
  if (has_empty_numerical_results &&
      inst->isRegistered()) {
    for (Type ty : inst->getResultTypes()) {
      auto shaped_ty = mlir::cast<ShapedType>(ty);
      results.push_back(
          DenseElementsAttr::get(shaped_ty, llvm::ArrayRef<Attribute>()));
    }
    return success();
  }
  if (std::any_of(operands.begin(), operands.end(), [](Attribute attr) {
        return !attr || !mlir::isa<ElementsAttr>(attr);
      }))
    return failure();
  SmallVector<ElementsAttr, 4> inputs;
  inputs.reserve(operands.size());
  for (auto input : operands) {
    inputs.push_back(mlir::cast<ElementsAttr>(input));
  }
  SmallVector<Attribute> constants;
  LogicalResult status = EvaluateOperation(inst, inputs, constants);
  results.assign(constants.begin(), constants.end());
  return status;
}
static bool init_hooks = ([] () {
  TensorFlowDialect::RegisterConstantFoldHook(ConstantFoldFallbackHook);
}(), true);
}  
}  