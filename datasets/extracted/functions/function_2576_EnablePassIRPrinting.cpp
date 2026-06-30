#include "tensorflow/compiler/mlir/tf2xla/internal/logging_hooks.h"
#include <memory>
#include <string>
#include "llvm/ADT/StringRef.h"
#include "mlir/Pass/PassManager.h"  
#include "tensorflow/compiler/mlir/tensorflow/utils/data_dumper_logger_config.h"
#include "tensorflow/core/util/debug_data_dumper.h"
namespace tensorflow {
namespace tf2xla {
namespace internal {
using mlir::PassManager;
void EnablePassIRPrinting(PassManager& pm, const std::string& dump_group_name,
                          llvm::StringRef module_name) {
  pm.getContext()->disableMultithreading();
  pm.enableIRPrinting(std::make_unique<::tensorflow::DataDumperLoggerConfig>(
      [module_name, dump_group_name](const std::string& pass_tag_name,
                                     mlir::Operation* op) {
        return DEBUG_DATA_DUMPER()->GetDumpFilename(
            module_name.str(), dump_group_name, pass_tag_name);
      },
      "",
      true));
  pm.enableTiming();
}
};  
};  
};  