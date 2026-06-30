#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include <string_view>
#include "absl/status/status.h"
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/Diagnostics.h"  
#include "mlir/Support/LLVM.h"  
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/util/managed_stack_trace.h"
namespace mlir {
StatusScopedDiagnosticHandler::StatusScopedDiagnosticHandler(
    MLIRContext* context, bool propagate, bool filter_stack)
    : BaseScopedDiagnosticHandler(context, propagate, filter_stack) {
  if (filter_stack) {
    this->shouldShowLocFn = [](Location loc) -> bool {
      if (FileLineColLoc fileLoc = mlir::dyn_cast<FileLineColLoc>(loc)) {
        return !tensorflow::IsInternalFrameForFilename(
            fileLoc.getFilename().str());
      } else {
        return true;
      }
    };
  }
  setHandler([this](Diagnostic& diag) { return this->handler(&diag); });
}
Status StatusScopedDiagnosticHandler::ConsumeStatus() {
  return BaseScopedDiagnosticHandler::ConsumeStatus();
}
Status StatusScopedDiagnosticHandler::Combine(Status status) {
  absl::Status absl_s = BaseScopedDiagnosticHandler::Combine(status);
  return absl_s;
}
}  