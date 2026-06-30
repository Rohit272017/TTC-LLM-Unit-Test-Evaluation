#include "tensorflow/core/grappler/optimizers/data/batch_parallelization.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
namespace tensorflow {
namespace grappler {
namespace {
constexpr char kBatchDataset[] = "BatchDatasetV2";
constexpr char kParallelBatchDataset[] = "ParallelBatchDataset";
NodeDef MakeParallelBatch(const string& name, MutableGraphView* graph) {
  int index = graph_utils::FindGraphNodeWithName(name, *graph->graph());
  DCHECK_NE(index, -1) << "Failed to find node " << name
                       << " in the optimized graph.";
  NodeDef parallel_batch = graph->graph()->node(index);
  graph_utils::SetUniqueGraphNodeName(kParallelBatchDataset, graph->graph(),
                                      &parallel_batch);
  parallel_batch.set_op(kParallelBatchDataset);
  auto* num_parallel_calls =
      graph_utils::AddScalarConstNode(data::model::kAutotune, graph);
  string drop_remainder_name = parallel_batch.input(2);
  parallel_batch.set_input(2, num_parallel_calls->name());
  parallel_batch.add_input(drop_remainder_name);
  return parallel_batch;
}
}  
Status BatchParallelization::OptimizeAndCollectStats(Cluster* cluster,
                                                     const GrapplerItem& item,
                                                     GraphDef* output,
                                                     OptimizationStats* stats) {
  *output = item.graph;
  if (!autotune_) {
    VLOG(1) << "The optimization batch_parallelization is not applied if "
               "autotune is off.";
    return absl::OkStatus();
  }
  MutableGraphView graph(output);
  if (graph_utils::IsItemDerivedFromFunctionDef(item, graph))
    return absl::OkStatus();
  absl::flat_hash_set<string> nodes_to_delete;
  FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                             item.graph.library());
  auto get_batch_node = [](const NodeDef& node) -> const NodeDef* {
    if (node.op() == kBatchDataset) return &node;
    return nullptr;
  };
  for (const NodeDef& node : item.graph.node()) {
    const NodeDef* batch_node = get_batch_node(node);
    if (!batch_node) continue;
    auto* parallel_batch =
        graph.AddNode(MakeParallelBatch(batch_node->name(), &graph));
    TF_RETURN_IF_ERROR(
        graph.UpdateFanouts(batch_node->name(), parallel_batch->name()));
    nodes_to_delete.insert(batch_node->name());
    stats->num_changes++;
  }
  TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));
  return absl::OkStatus();
}
REGISTER_GRAPH_OPTIMIZER_AS(BatchParallelization, "batch_parallelization");
}  
}  