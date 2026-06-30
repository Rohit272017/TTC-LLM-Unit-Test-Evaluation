#include "tensorflow/compiler/mlir/quantization/stablehlo/instrumentations/save_report.h"
#include <optional>
#include <string>
#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  
#include "mlir/IR/Operation.h"  
#include "mlir/Pass/Pass.h"  
#include "mlir/Support/LLVM.h"  
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/report.h"
namespace mlir::quant::stablehlo {
namespace {
std::optional<std::string> OptionalStringViewToOptionalString(
    std::optional<absl::string_view> view) {
  if (view == std::nullopt) return std::nullopt;
  return std::make_optional<std::string>(*view);
}
bool IsQuantizeCompositeFunctionPass(absl::Nullable<Pass*> pass,
                                     absl::Nullable<Operation*> op) {
  return pass != nullptr &&
         pass->getArgument() == "stablehlo-quantize-composite-functions" &&
         isa_and_nonnull<ModuleOp>(op);
}
bool ShouldSaveReport(absl::Nullable<Pass*> pass, absl::Nullable<Operation*> op,
                      const std::optional<std::string>& file_path) {
  return file_path != std::nullopt && IsQuantizeCompositeFunctionPass(pass, op);
}
void SaveReport(const QuantizationReport& report,
                const absl::string_view file_path) {
  if (const absl::Status save_status = report.Save(file_path);
      save_status.ok()) {
    LOG(INFO) << "Successfully saved quantization report to: " << file_path;
  } else {
    LOG(ERROR) << "Failed to save quantization report to: " << file_path
               << " with status: " << save_status;
  }
}
}  
SaveQuantizationReportInstrumentation::SaveQuantizationReportInstrumentation(
    std::optional<absl::string_view> file_path)
    : file_path_(OptionalStringViewToOptionalString(file_path)) {}
void SaveQuantizationReportInstrumentation::runAfterPass(Pass* pass,
                                                         Operation* op) {
  if (!IsQuantizeCompositeFunctionPass(pass, op)) return;
  auto module_op = cast<ModuleOp>(op);
  const QuantizationReport report(module_op);
  report.Print();
  if (!ShouldSaveReport(pass, op, file_path_)) return;
  SaveReport(report, *file_path_);
}
}  