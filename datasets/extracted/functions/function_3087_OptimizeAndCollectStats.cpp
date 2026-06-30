#include "tensorflow/core/grappler/optimizers/data/enable_gradient_descent.h"
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
constexpr char kAlgorithm[] = "algorithm";
constexpr char kModelDataset[] = "ModelDataset";
constexpr int64_t HILL_CLIMB = 0;
constexpr int64_t GRADIENT_DESCENT = 1;
}  
Status EnableGradientDescent::OptimizeAndCollectStats(
    Cluster* cluster, const GrapplerItem& item, GraphDef* output,
    OptimizationStats* stats) {
  *output = item.graph;
  if (!autotune_) {
    VLOG(1) << "The optimization enable_gradient_descent is not applied if "
               "autotune is off.";
    return absl::OkStatus();
  }
  MutableGraphView graph(output);
  if (graph_utils::IsItemDerivedFromFunctionDef(item, graph))
    return absl::OkStatus();
  int index = graph_utils::FindGraphNodeWithOp(kModelDataset, *output);
  NodeDef& model_node = *(output->mutable_node(index));
  if (model_node.attr().at(kAlgorithm).i() == HILL_CLIMB) {
    (*model_node.mutable_attr())[kAlgorithm].set_i(GRADIENT_DESCENT);
    stats->num_changes++;
  }
  return absl::OkStatus();
}
REGISTER_GRAPH_OPTIMIZER_AS(EnableGradientDescent, "enable_gradient_descent");
}  
}  