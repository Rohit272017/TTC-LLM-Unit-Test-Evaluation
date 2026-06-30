#include "tensorflow/compiler/mlir/tfrt/transforms/utils.h"
#include <optional>
#include <string>
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/BuiltinOps.h"  
#include "mlir/IR/BuiltinTypes.h"  
#include "mlir/IR/OpDefinition.h"  
#include "mlir/IR/OperationSupport.h"  
#include "mlir/IR/SymbolTable.h"  
#include "mlir/IR/Value.h"  
#include "mlir/Support/LLVM.h"  
#include "tensorflow/compiler/mlir/tensorflow/ir/host_runtime/tfrt_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
namespace tensorflow {
bool IsResourceArgument(mlir::Value value) {
  auto arg = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!arg) return false;
  auto func = llvm::cast<mlir::func::FuncOp>(arg.getOwner()->getParentOp());
  return func.getArgAttr(arg.getArgNumber(), "tf.resource_name") != nullptr;
}
bool IsResultVariable(const mlir::Value &original_operand,
                      const mlir::Value &operand) {
  if (mlir::isa<mlir::OpResult>(original_operand)) {
    auto defining_op = original_operand.getDefiningOp();
    if (llvm::isa<mlir::TF::ReadVariableOp>(defining_op) &&
        defining_op->getNumOperands() == 1) {
      return true;
    } else if (llvm::isa<mlir::TF::_TfrtGetResourceOp>(defining_op)) {
      return true;
    }
    return false;
  }
  return IsResourceArgument(operand);
}
std::optional<std::string> CanonicalizeTensorflowFunctionName(
    const mlir::SymbolTable &symbol_table, absl::string_view mlir_func_name,
    bool use_mlir_func_name) {
  if (use_mlir_func_name) {
    return std::string(mlir_func_name);
  }
  auto callee =
      symbol_table.lookup<mlir::func::FuncOp>(std::string(mlir_func_name));
  if (!callee) return std::nullopt;
  mlir::StringAttr original_func_name =
      callee->getAttrOfType<mlir::StringAttr>("tf._original_func_name");
  if (!original_func_name) {
    mlir_func_name.remove_suffix(1);
    return std::string(mlir_func_name);
  }
  return original_func_name.str();
}
bool IsSessionInitializer(mlir::func::FuncOp op) {
  auto session_initializer_op = mlir::tf_saved_model::GetSessionInitializerOp(
      op->getParentOfType<mlir::ModuleOp>());
  if (!session_initializer_op) return false;
  for (auto sym_ref : session_initializer_op.getInitializers()) {
    if (op.getSymName() ==
        mlir::cast<mlir::FlatSymbolRefAttr>(sym_ref).getValue())
      return true;
  }
  return false;
}
}  