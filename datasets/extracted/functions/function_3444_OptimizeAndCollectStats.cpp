#include "tensorflow/core/grappler/optimizers/data/parallel_batch.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
namespace tensorflow {
namespace grappler {
Status ParallelBatch::OptimizeAndCollectStats(Cluster* cluster,
                                              const GrapplerItem& item,
                                              GraphDef* output,
                                              OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);
  for (NodeDef& node : *output->mutable_node()) {
    if (node.op() == "BatchDatasetV2" || node.op() == "PaddedBatchDatasetV2") {
      (*node.mutable_attr())["parallel_copy"].set_b(true);
      stats->num_changes++;
    }
  }
  return absl::OkStatus();
}
REGISTER_GRAPH_OPTIMIZER_AS(ParallelBatch, "parallel_batch");
}  
}  