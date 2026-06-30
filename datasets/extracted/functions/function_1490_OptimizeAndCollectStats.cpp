#include "tensorflow/core/grappler/optimizers/data/disable_prefetch_legacy_autotune.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"
namespace tensorflow {
namespace grappler {
namespace {
constexpr char kLegacyAutotune[] = "legacy_autotune";
constexpr char kPrefetchDataset[] = "PrefetchDataset";
}  
Status DisablePrefetchLegacyAutotune::OptimizeAndCollectStats(
    Cluster* cluster, const GrapplerItem& item, GraphDef* output,
    OptimizationStats* stats) {
  *output = item.graph;
  if (!autotune_) {
    VLOG(1) << "The optimization disable_prefetch_legacy_autotune is not "
               "applied if autotune is off.";
    return absl::OkStatus();
  }
  MutableGraphView graph(output);
  for (NodeDef& node : *output->mutable_node()) {
    if (node.op() == kPrefetchDataset) {
      if (node.attr().find(kLegacyAutotune) == node.attr().end() ||
          node.attr().at(kLegacyAutotune).b()) {
        (*node.mutable_attr())[kLegacyAutotune].set_b(false);
        stats->num_changes++;
      }
    }
  }
  return absl::OkStatus();
}
REGISTER_GRAPH_OPTIMIZER_AS(DisablePrefetchLegacyAutotune,
                            "disable_prefetch_legacy_autotune");
}  
}  