#include "tensorflow/core/common_runtime/isolate_placer_inspection_required_ops_pass.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/placer_inspection_required_ops_utils.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/dump_graph.h"
namespace tensorflow {
Status IsolatePlacerInspectionRequiredOpsPass::Run(
    const GraphOptimizationPassOptions& options) {
  if (options.graph == nullptr) {
    VLOG(1) << "Not running IsolatePlacerInspectionRequiredOpsPass because no "
               "graph is provided";
    return absl::OkStatus();
  }
  VLOG(1) << "IsolatePlacerInspectionRequiredOpsPass::Run";
  Graph* graph = options.graph->get();
  if (VLOG_IS_ON(3)) {
    DumpGraphToFile("isolate_deep_ops_before", *graph, nullptr, "/tmp");
  }
  Status status = IsolatePlacerInspectionRequiredOps(*options.flib_def, graph);
  if (VLOG_IS_ON(3) && status.ok()) {
    DumpGraphToFile("isolate_deep_ops_after", *graph, nullptr, "/tmp");
  }
  return status;
}
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 35,
                      IsolatePlacerInspectionRequiredOpsPass);
}  