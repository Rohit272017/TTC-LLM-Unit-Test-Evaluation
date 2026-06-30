#include "tensorflow/compiler/mlir/tensorflow/utils/dump_graph.h"
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"  
#include "mlir/IR/Verifier.h"  
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/ir/importexport/graphdef_import.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/util/dump_graph.h"
namespace tensorflow {
namespace {
struct WritableFileRawStream : public llvm::raw_ostream {
  explicit WritableFileRawStream(WritableFile* file) : file(file) {
    SetUnbuffered();
  }
  ~WritableFileRawStream() override = default;
  uint64_t current_pos() const override { return 0; }
  void write_impl(const char* ptr, size_t size) override {
    if (file) {
      Status s = file->Append(StringPiece(ptr, size));
      if (!s.ok()) {
        LOG(WARNING) << "Write failed: " << s;
        file = nullptr;
      }
    }
  }
  WritableFile* file;
};
}  
Status DumpTextualIRToFile(const MlirDumpConfig& config, const Graph& graph,
                           const FunctionLibraryDefinition* flib_def,
                           WritableFile* file) {
  WritableFileRawStream os(std::move(file));
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (flib_def) {
    flib_def = &graph.flib_def();
  }
  auto convert = [&]() -> Status {
    mlir::StatusScopedDiagnosticHandler status_handler(&context);
    GraphDebugInfo debug_info;
    switch (config.dialect) {
      case MlirDumpConfig::Dialect::kTFG: {
        TF_ASSIGN_OR_RETURN(module,
                            mlir::tfg::ImportGraphAndFunctionsToMlir(
                                &context, debug_info, graph,
                                flib_def ? *flib_def : graph.flib_def()));
        break;
      }
    }
    if (failed(mlir::verify(*module))) {
      return status_handler.ConsumeStatus();
    }
    return status_handler.ConsumeStatus();
  };
  TF_RETURN_IF_ERROR(convert());
  module->print(os, config.op_printing_flags);
  return absl::OkStatus();
}
void UseMlirForGraphDump(const MlirDumpConfig& config) {
  SetGraphDumper(
      [config](const Graph& graph, const FunctionLibraryDefinition* flib_def,
               WritableFile* file) -> Status {
        return DumpTextualIRToFile(config, graph, flib_def, file);
      },
      ".mlir");
}
}  