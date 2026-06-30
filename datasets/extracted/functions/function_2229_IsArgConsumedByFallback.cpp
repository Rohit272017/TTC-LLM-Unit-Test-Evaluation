#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_util.h"
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.h"
namespace tfrt {
namespace fallback_async {
bool IsArgConsumedByFallback(mlir::func::FuncOp func, int arg_index) {
  auto arg = func.getArgument(arg_index);
  for (mlir::Operation *user : arg.getUsers()) {
    if (llvm::isa<FallbackAsyncDialect>(user->getDialect())) return true;
  }
  return false;
}
void ForEachArgConsumedByFallback(
    mlir::func::FuncOp func, llvm::function_ref<void(int arg_index)> action) {
  for (int arg_index = 0; arg_index < func.getNumArguments(); ++arg_index) {
    if (IsArgConsumedByFallback(func, arg_index)) action(arg_index);
  }
}
void ForEachArgConsumedByFallback(
    mlir::ModuleOp module,
    llvm::function_ref<void(llvm::StringRef func_name, int arg_index)> action) {
  for (auto func : module.getOps<mlir::func::FuncOp>()) {
    ForEachArgConsumedByFallback(
        func, [func_name = func.getName(), action](int arg_index) {
          action(func_name, arg_index);
        });
  }
}
}  
}  