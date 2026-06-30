#include "tensorflow/compiler/mlir/tensorflow/utils/bridge_logger.h"
#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include "absl/strings/str_split.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Operation.h"  
#include "mlir/IR/OperationSupport.h"  
#include "mlir/Pass/Pass.h"  
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
namespace tensorflow {
static std::atomic<int> log_counter(0);
BridgeLoggerConfig::BridgeLoggerConfig(bool print_module_scope,
                                       bool print_after_only_on_change,
                                       mlir::OpPrintingFlags op_printing_flags)
    : mlir::PassManager::IRPrinterConfig(
          print_module_scope, print_after_only_on_change,
          false, op_printing_flags),
      pass_filter_(GetFilter("MLIR_BRIDGE_LOG_PASS_FILTER")),
      string_filter_(GetFilter("MLIR_BRIDGE_LOG_STRING_FILTER")) {}
inline static void Log(BridgeLoggerConfig::PrintCallbackFn print_callback,
                       mlir::Pass* pass, mlir::Operation* op,
                       llvm::StringRef file_suffix) {
  std::string pass_name = pass->getName().str();
  std::string name = llvm::formatv("{0,0+4}_mlir_bridge_{1}_{2}", log_counter++,
                                   pass_name, file_suffix);
  std::unique_ptr<llvm::raw_ostream> os;
  std::string filepath;
  if (CreateFileForDumping(name, &os, &filepath).ok()) {
    print_callback(*os);
    LOG(INFO) << "Dumped MLIR module to " << filepath;
  }
}
void BridgeLoggerConfig::printBeforeIfEnabled(mlir::Pass* pass,
                                              mlir::Operation* op,
                                              PrintCallbackFn print_callback) {
  if (ShouldPrint(pass, op)) Log(print_callback, pass, op, "before");
}
void BridgeLoggerConfig::printAfterIfEnabled(mlir::Pass* pass,
                                             mlir::Operation* op,
                                             PrintCallbackFn print_callback) {
  if (ShouldPrint(pass, op)) Log(print_callback, pass, op, "after");
}
std::vector<std::string> BridgeLoggerConfig::GetFilter(
    const std::string& env_var) {
  std::vector<std::string> filter;
  const char* filter_str = getenv(env_var.c_str());
  if (filter_str) {
    filter = absl::StrSplit(filter_str, ';', absl::SkipWhitespace());
  }
  return filter;
}
bool BridgeLoggerConfig::MatchesFilter(const std::string& str,
                                       const std::vector<std::string>& filter,
                                       bool exact_match) {
  if (filter.empty()) return true;
  for (const std::string& filter_str : filter) {
    if (str == filter_str) return true;
    if (!exact_match && str.find(filter_str) != std::string::npos) return true;
  }
  return false;
}
bool BridgeLoggerConfig::ShouldPrint(mlir::Pass* pass, mlir::Operation* op) {
  std::string pass_name = pass->getName().str();
  if (!MatchesFilter(pass_name, pass_filter_, true)) {
    VLOG(1) << "Not logging invocation of pass `" << pass_name
            << "` because the pass name does not match any string in "
               "`MLIR_BRIDGE_LOG_PASS_FILTER`";
    return false;
  }
  if (!string_filter_.empty()) {
    std::string serialized_op;
    llvm::raw_string_ostream os(serialized_op);
    op->print(os);
    if (!MatchesFilter(serialized_op, string_filter_, false)) {
      VLOG(1) << "Not logging invocation of pass `" << pass_name
              << "` because the serialized operation on which the pass is "
                 "invoked does not contain any of the strings specified by "
                 "MLIR_BRIDGE_LOG_STRING_FILTER";
      return false;
    }
  }
  return true;
}
}  