#include "tensorflow/core/grappler/verifiers/structure_verifier.h"
#include <string>
#include <vector>
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/validate.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/grappler/verifiers/graph_verifier.h"
#include "tensorflow/core/lib/core/status.h"
namespace tensorflow {
namespace grappler {
Status StructureVerifier::Verify(const GraphDef& graph) {
  StatusGroup status_group;
  FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                             graph.library());
  status_group.Update(tensorflow::graph::ValidateGraphDefAgainstOpRegistry(
      graph, function_library));
  status_group.Update(tensorflow::graph::VerifyNoDuplicateNodeNames(graph));
  std::vector<const NodeDef*> topo_order;
  status_group.Update(ComputeTopologicalOrder(graph, &topo_order));
  return status_group.as_concatenated_status();
}
}  
}  