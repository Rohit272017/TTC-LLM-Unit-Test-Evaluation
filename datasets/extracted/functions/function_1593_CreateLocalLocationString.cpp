#include "tensorflow/dtensor/mlir/dtensor_location.h"
#include <algorithm>
#include <queue>
#include <string>
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LLVM.h"  
#include "tensorflow/compiler/mlir/utils/name_utils.h"
namespace tensorflow {
namespace dtensor {
namespace {
std::string CreateLocalLocationString(mlir::FileLineColLoc loc) {
  return llvm::formatv(">> {0}:{1}:{2}", loc.getFilename(), loc.getLine(),
                       loc.getColumn())
      .str();
}
}  
mlir::Location DTensorLocation(mlir::Location loc, llvm::StringRef file,
                               unsigned int line, llvm::StringRef name) {
  auto split = file.rsplit("/");
  if (!split.second.empty()) file = split.second;
  mlir::Location callee_loc =
      mlir::FileLineColLoc::get(loc.getContext(), file, line, 0);
  std::string new_name = GetNameFromLoc(loc);
  if (!new_name.empty()) {
    if (!name.empty()) {
      new_name = llvm::formatv("{0}/{1}", new_name, name).str();
    }
    callee_loc = mlir::NameLoc::get(
        mlir::StringAttr::get(loc.getContext(), new_name), callee_loc);
  }
  return mlir::CallSiteLoc::get(callee_loc, loc);
}
mlir::Location DTensorLocation(mlir::Operation* op, llvm::StringRef file,
                               unsigned int line, llvm::StringRef name) {
  return DTensorLocation(op->getLoc(), file, line, name);
}
std::string DTensorLocationToString(mlir::Location loc) {
  llvm::SmallVector<std::string, 4> stack;
  std::queue<mlir::Location> queue;
  queue.push(loc);
  while (!queue.empty()) {
    mlir::Location& front = queue.front();
    if (auto name_loc = mlir::dyn_cast<mlir::NameLoc>(front)) {
      queue.push(name_loc.getChildLoc());
    } else if (auto callsite_loc = mlir::dyn_cast<mlir::CallSiteLoc>(front)) {
      queue.push(callsite_loc.getCallee());
      queue.push(callsite_loc.getCaller());
    } else if (auto line_loc = mlir::dyn_cast<mlir::FileLineColLoc>(front)) {
      stack.push_back(CreateLocalLocationString(line_loc));
    }
    queue.pop();
  }
  std::reverse(stack.begin(), stack.end());
  std::string s;
  llvm::raw_string_ostream ss(s);
  llvm::interleave(stack, ss, "\n");
  return ss.str();
}
}  
}  