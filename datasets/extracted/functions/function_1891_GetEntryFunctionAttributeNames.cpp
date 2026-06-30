#include <vector>
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/IR/BuiltinOps.h"  
#include "mlir/Support/LLVM.h"  
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
namespace mlir {
std::vector<llvm::StringRef> GetEntryFunctionAttributeNames() {
  return {"tf.entry_function",
          tf_saved_model::kTfSavedModelInitializerTypeAttr};
}
bool IsEntryFunction(func::FuncOp func) {
  for (const auto &attr : GetEntryFunctionAttributeNames()) {
    if (func->hasAttr(attr)) {
      return true;
    }
  }
  return false;
}
llvm::SmallVector<func::FuncOp> GetEntryFunctions(ModuleOp module) {
  llvm::SmallVector<func::FuncOp> entry_funcs;
  module.walk([&](func::FuncOp func) {
    if (IsEntryFunction(func)) {
      entry_funcs.push_back(func);
    }
  });
  return entry_funcs;
}
LogicalResult GetCallees(SymbolUserOpInterface op, SymbolTable &symtab,
                         llvm::SmallVector<func::FuncOp> &callees) {
  for (auto attr : op->getAttrs()) {
    auto sym = mlir::dyn_cast<SymbolRefAttr>(attr.getValue());
    if (!sym) continue;
    auto callee = symtab.lookup<func::FuncOp>(sym.getRootReference());
    if (!callee) {
      return op->emitError()
             << "Cannot find function " << sym.getRootReference();
    }
    callees.push_back(callee);
  }
  return success();
}
bool HasSingleBlock(func::FuncOp func) {
  return func->getNumRegions() == 1 && func.getBody().hasOneBlock();
}
}  