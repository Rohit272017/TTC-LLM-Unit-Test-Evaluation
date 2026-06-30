#include "xla/mlir/tools/mlir_interpreter/framework/registration.h"
#include <cassert>
#include <functional>
#include <utility>
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value.h"
namespace mlir {
namespace interpreter {
namespace detail {
namespace {
DenseMap<llvm::StringRef, llvm::StringRef>& GetOpAliases() {
  static DenseMap<llvm::StringRef, llvm::StringRef>* aliases = nullptr;
  if (!aliases) {
    aliases = new DenseMap<llvm::StringRef, llvm::StringRef>();
  }
  return *aliases;
}
DenseMap<llvm::StringRef, InterpreterFunction>& GetFunctions() {
  static DenseMap<llvm::StringRef, InterpreterFunction>* functions = nullptr;
  if (!functions) {
    functions = new DenseMap<llvm::StringRef, InterpreterFunction>();
  }
  return *functions;
}
}  
InterpreterFunction GetFunction(llvm::StringRef name) {
  const auto& fns = GetFunctions();
  auto fn = fns.find(name);
  if (fn != fns.end()) {
    return fn->second;
  }
  const auto& aliases = GetOpAliases();
  auto alias = aliases.find(name);
  if (alias != aliases.end()) {
    return fns.find(alias->second)->second;
  }
  return nullptr;
}
void RegisterInterpreterOp(llvm::StringRef name,
                           InterpreterValue (*fn)(const InterpreterValue&)) {
  RegisterInterpreterOp(
      name,
      [fn](MutableArrayRef<InterpreterValue> operands, mlir::Operation*,
           InterpreterState&) -> SmallVector<InterpreterValue> {
        assert(operands.size() == 1 && "unexpected number of operands");
        return {fn(operands[0])};
      });
}
void RegisterInterpreterOp(llvm::StringRef name,
                           InterpreterValue (*fn)(const InterpreterValue&,
                                                  const InterpreterValue&)) {
  RegisterInterpreterOp(
      name,
      [fn](MutableArrayRef<InterpreterValue> operands, mlir::Operation*,
           InterpreterState&) -> SmallVector<InterpreterValue> {
        assert(operands.size() == 2 && "unexpected number of operands");
        return {fn(operands[0], operands[1])};
      });
}
void RegisterInterpreterOp(
    llvm::StringRef name,
    InterpreterValue (*fn)(MutableArrayRef<InterpreterValue>)) {
  RegisterInterpreterOp(
      name,
      [fn](MutableArrayRef<InterpreterValue> operands, mlir::Operation*,
           InterpreterState&) -> SmallVector<InterpreterValue> {
        return {fn(operands)};
      });
}
void RegisterInterpreterOp(
    llvm::StringRef name,
    std::function<llvm::SmallVector<InterpreterValue>(
        MutableArrayRef<InterpreterValue>, mlir::Operation*, InterpreterState&)>
        fn) {
  GetFunctions()[name] = std::move(fn);
}
void RegisterInterpreterOp(llvm::StringRef name, llvm::StringRef original) {
  GetOpAliases()[name] = original;
}
}  
}  
}  