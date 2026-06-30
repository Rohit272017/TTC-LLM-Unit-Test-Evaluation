#include "tensorflow/core/grappler/optimizers/data/replicate_on_split.h"
#include "absl/log/log.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
namespace tensorflow {
namespace grappler {
Status ReplicateOnSplit::OptimizeAndCollectStats(Cluster* cluster,
                                                 const GrapplerItem& item,
                                                 GraphDef* output,
                                                 OptimizationStats* stats) {
  VLOG(1) << "Running replicate on split optimization";
  *output = item.graph;
  MutableGraphView graph(output);
  for (NodeDef& node : *output->mutable_node()) {
    if (graph_utils::HasReplicateOnSplitAttr(node.op())) {
      (*node.mutable_attr())["replicate_on_split"].set_b(true);
      stats->num_changes++;
    }
  }
  return absl::OkStatus();
}
REGISTER_GRAPH_OPTIMIZER_AS(ReplicateOnSplit, "replicate_on_split");
}  
}  