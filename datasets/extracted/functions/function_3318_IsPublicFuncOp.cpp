#include "tensorflow/compiler/mlir/quantization/common/func.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/IR/BuiltinOps.h"  
#include "mlir/IR/SymbolTable.h"  
#include "mlir/Support/LLVM.h"  
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
namespace mlir::quant {
namespace {
using ::tensorflow::kDefaultServingSignatureDefKey;
using ::tensorflow::kImportModelDefaultGraphFuncName;
bool IsPublicFuncOp(func::FuncOp func_op) {
  return SymbolTable::getSymbolVisibility(&*func_op) ==
         SymbolTable::Visibility::Public;
}
}  
func::FuncOp FindMainFuncOp(ModuleOp module_op) {
  if (const auto main_func_op = module_op.lookupSymbol<func::FuncOp>(
          kImportModelDefaultGraphFuncName);
      main_func_op != nullptr && IsPublicFuncOp(main_func_op)) {
    return main_func_op;
  }
  if (const auto serving_default_func_op =
          module_op.lookupSymbol<func::FuncOp>(kDefaultServingSignatureDefKey);
      serving_default_func_op != nullptr &&
      IsPublicFuncOp(serving_default_func_op)) {
    return serving_default_func_op;
  }
  return nullptr;
}
}  