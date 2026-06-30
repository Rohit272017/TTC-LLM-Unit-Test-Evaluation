#include "tensorflow/core/grappler/optimizers/data/disable_intra_op_parallelism.h"
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
constexpr char kMaxIntraOpParallelismDataset[] = "MaxIntraOpParallelismDataset";
constexpr char kModelDataset[] = "ModelDataset";
constexpr std::array<const char*, 2> kMaxIntraOpParallelismDatasetOps = {
    "MaxIntraOpParallelismDataset",
    "ExperimentalMaxIntraOpParallelismDataset",
};
}  
Status DisableIntraOpParallelism::OptimizeAndCollectStats(
    Cluster* cluster, const GrapplerItem& item, GraphDef* output,
    OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);
  if (graph_utils::IsItemDerivedFromFunctionDef(item, graph))
    return absl::OkStatus();
  if (item.fetch.size() != 1) {
    return errors::InvalidArgument(
        "Expected only one fetch node but there were ", item.fetch.size(), ": ",
        absl::StrJoin(item.fetch, ", "));
  }
  for (const NodeDef& node : item.graph.node()) {
    for (const auto& target_dataset_op : kMaxIntraOpParallelismDatasetOps) {
      if (node.op() == target_dataset_op) {
        return absl::OkStatus();
      }
    }
  }
  NodeDef* sink_node = graph.GetNode(item.fetch.at(0));
  NodeDef* last_node = graph_utils::GetInputNode(*sink_node, graph);
  if (last_node->op() == kModelDataset) {
    last_node = graph_utils::GetInputNode(*last_node, graph);
  }
  NodeDef* max_parallelism_value =
      graph_utils::AddScalarConstNode(int64_t{1}, &graph);
  NodeDef insert_node;
  graph_utils::SetUniqueGraphNodeName("intra_op_parallelism", graph.graph(),
                                      &insert_node);
  insert_node.set_op(kMaxIntraOpParallelismDataset);
  *insert_node.mutable_input()->Add() = last_node->name();
  *insert_node.mutable_input()->Add() = max_parallelism_value->name();
  if (!graph_utils::CopyShapesAndTypesAttrs(*last_node, &insert_node))
    return absl::OkStatus();
  auto* added_node = graph.AddNode(std::move(insert_node));
  TF_RETURN_IF_ERROR(
      graph.UpdateFanouts(last_node->name(), added_node->name()));
  stats->num_changes++;
  return absl::OkStatus();
}
REGISTER_GRAPH_OPTIMIZER_AS(DisableIntraOpParallelism,
                            "disable_intra_op_parallelism");
}  
}  