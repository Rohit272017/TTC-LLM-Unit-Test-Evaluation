#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/report.h"
#include <optional>
#include <string>
#include <utility>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  
#include "mlir/IR/BuiltinAttributes.h"  
#include "mlir/IR/BuiltinOps.h"  
#include "mlir/IR/Visitors.h"  
#include "mlir/Support/LLVM.h"  
#include "tensorflow/compiler/mlir/quantization/common/lift_as_function_call.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/io.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tsl/platform/protobuf.h"  
namespace mlir::quant::stablehlo {
namespace {
using ::stablehlo::quantization::Method;
using ::stablehlo::quantization::QuantizationResult;
using ::stablehlo::quantization::QuantizationResults;
using ::stablehlo::quantization::io::WriteStringToFile;
using ::tsl::protobuf::TextFormat;
std::string GetCompositeFunctionName(const StringRef quantized_func_name) {
  return Twine(kCompositeFuncPrefix)
      .concat(quantized_func_name.rsplit(kQuantizedFuncPrefix).second)
      .str();
}
std::optional<QuantizationResult> GetQuantizationResult(func::CallOp call_op) {
  const StringRef callee_name = call_op.getCalleeAttr().getValue();
  if (!callee_name.starts_with(kQuantizedFuncPrefix)) {
    return std::nullopt;  
  }
  absl::StatusOr<Method> method = GetQuantizationMethod(call_op);
  if (!method.ok()) {
    call_op->emitError() << "Failed to get quantization method: "
                         << method.status().ToString();
    return std::nullopt;
  }
  QuantizationResult result{};
  result.mutable_quantizable_unit()->set_name(
      GetCompositeFunctionName(callee_name));
  *result.mutable_method() = std::move(*method);
  return result;
}
std::optional<QuantizationResult> GetQuantizationResult(
    TF::XlaCallModuleOp xla_call_module_op) {
  const StringAttr callee_name_attr =
      mlir::dyn_cast_or_null<StringAttr>(xla_call_module_op->getDiscardableAttr(
          kOriginalStablehloEntryFunctionAttrName));
  if (callee_name_attr == nullptr) return std::nullopt;
  if (callee_name_attr.getValue().starts_with(kCompositeFuncPrefix)) {
    QuantizationResult result{};
    result.mutable_quantizable_unit()->set_name(
        callee_name_attr.getValue().str());
    result.mutable_method()->mutable_no_quantization();
    return result;
  } else {
    return std::nullopt;
  }
}
void PopulateQuantizedResults(ModuleOp module_op,
                              QuantizationResults& results) {
  module_op.walk([&results](func::CallOp call_op) {
    std::optional<QuantizationResult> result = GetQuantizationResult(call_op);
    if (result == std::nullopt) return WalkResult::skip();
    *results.add_results() = std::move(*result);
    return WalkResult::advance();
  });
}
void PopulateNonQuantizedResults(ModuleOp module_op,
                                 QuantizationResults& results) {
  module_op.walk([&results](TF::XlaCallModuleOp xla_call_module_op) {
    std::optional<QuantizationResult> result =
        GetQuantizationResult(xla_call_module_op);
    if (result == std::nullopt) return WalkResult::skip();
    *results.add_results() = std::move(*result);
    return WalkResult::advance();
  });
}
}  
QuantizationReport::QuantizationReport(ModuleOp module_op)
    : quantization_results_(CollectResultsFromModuleOp(module_op)) {}
QuantizationResults QuantizationReport::CollectResultsFromModuleOp(
    ModuleOp module_op) const {
  QuantizationResults results{};
  PopulateQuantizedResults(module_op, results);
  PopulateNonQuantizedResults(module_op, results);
  return results;
}
void QuantizationReport::AddQuantizationResult(QuantizationResult&& result) {
  *quantization_results_.add_results() = std::move(result);
}
std::string QuantizationReport::ToString() const {
  std::string results_str{};
  TextFormat::PrintToString(quantization_results_, &results_str);
  return absl::StrCat("===== Quantization Report =====\n\n", results_str,
                      "\n===== Quantization Report End =====\n\n");
}
void QuantizationReport::Print() const {
  llvm::outs() << ToString();
  llvm::outs().flush();  
}
absl::Status QuantizationReport::Save(const StringRef file_path) const {
  std::string results_str{};
  TextFormat::PrintToString(GetQuantizationResults(), &results_str);
  return WriteStringToFile(file_path, results_str);
}
}  