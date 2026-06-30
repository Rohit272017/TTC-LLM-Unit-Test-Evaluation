#include "tensorflow/core/transforms/graph_transform_wrapper.h"
#include <initializer_list>
#include "absl/memory/memory.h"
#include "mlir/IR/MLIRContext.h"  
#include "mlir/Pass/PassManager.h"  
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/ir/importexport/graphdef_export.h"
#include "tensorflow/core/ir/importexport/graphdef_import.h"
#include "tensorflow/core/platform/statusor.h"
namespace mlir {
namespace tfg {
tensorflow::Status RunTransformOnGraph(
    tensorflow::Graph* graph,
    const std::initializer_list<
        llvm::function_ref<std::unique_ptr<mlir::Pass>()>>& passes,
    const tensorflow::GraphDebugInfo& debug_info) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  TF_ASSIGN_OR_RETURN(OwningOpRef<ModuleOp> module,
                      ImportGraphAndFunctionsToMlir(&context, debug_info,
                                                    *graph, graph->flib_def()));
  PassManager pm((*module)->getName(), mlir::PassManager::Nesting::Explicit);
  for (auto& pass : passes) pm.addPass(pass());
  mlir::StatusScopedDiagnosticHandler error_handler(&context);
  if (failed(pm.run(*module)))
    return error_handler.Combine(
        tensorflow::errors::InvalidArgument("MLIR Graph Optimizer failed: "));
  tensorflow::GraphDef graphdef;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(ConvertToGraphDef(*module, &graphdef),
                                  "when exporting MLIR module to GraphDef");
  graph->Clear();
  graph->mutable_flib_def()->Clear();
  tensorflow::GraphConstructorOptions opts;
  return ConvertGraphDefToGraph(opts, graphdef, graph);
}
}  
}  