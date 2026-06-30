#include "tensorflow/core/grappler/optimizers/data/make_sloppy.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
namespace tensorflow {
namespace grappler {
Status MakeSloppy::OptimizeAndCollectStats(Cluster* cluster,
                                           const GrapplerItem& item,
                                           GraphDef* output,
                                           OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);
  for (NodeDef& node : *output->mutable_node()) {
    if (graph_utils::HasSloppyAttr(node.op())) {
      (*node.mutable_attr())["sloppy"].set_b(true);
      stats->num_changes++;
    }
    if (graph_utils::HasDeterministicAttr(node.op()) &&
        node.attr().at("deterministic").s() == "default") {
      (*node.mutable_attr())["deterministic"].set_s("false");
      stats->num_changes++;
    }
  }
  return absl::OkStatus();
}
REGISTER_GRAPH_OPTIMIZER_AS(MakeSloppy, "make_sloppy");
}  
}  