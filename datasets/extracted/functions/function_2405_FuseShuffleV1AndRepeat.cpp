#include "tensorflow/core/grappler/optimizers/data/shuffle_and_repeat_fusion.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/strcat.h"
namespace tensorflow {
namespace grappler {
namespace {
constexpr char kShuffleDataset[] = "ShuffleDataset";
constexpr char kShuffleDatasetV2[] = "ShuffleDatasetV2";
constexpr char kShuffleDatasetV3[] = "ShuffleDatasetV3";
constexpr char kRepeatDataset[] = "RepeatDataset";
constexpr char kShuffleAndRepeatDataset[] = "ShuffleAndRepeatDataset";
constexpr char kShuffleAndRepeatDatasetV2[] = "ShuffleAndRepeatDatasetV2";
constexpr char kReshuffleEachIteration[] = "reshuffle_each_iteration";
Status FuseShuffleV1AndRepeat(const NodeDef& shuffle_node,
                              const NodeDef& repeat_node,
                              MutableGraphView* graph, GraphDef* output,
                              NodeDef* fused_node) {
  fused_node->set_op(kShuffleAndRepeatDataset);
  graph_utils::SetUniqueGraphNodeName(kShuffleAndRepeatDataset, output,
                                      fused_node);
  fused_node->add_input(shuffle_node.input(0));
  fused_node->add_input(shuffle_node.input(1));
  fused_node->add_input(shuffle_node.input(2));
  fused_node->add_input(shuffle_node.input(3));
  fused_node->add_input(repeat_node.input(1));
  graph_utils::CopyShapesAndTypesAttrs(shuffle_node, fused_node);
  graph_utils::CopyAttribute(kReshuffleEachIteration, shuffle_node, fused_node);
  graph_utils::MaybeSetFusedMetadata(shuffle_node, repeat_node, fused_node);
  return absl::OkStatus();
}
Status FuseShuffleV2AndRepeat(const NodeDef& shuffle_node,
                              const NodeDef& repeat_node,
                              MutableGraphView* graph, GraphDef* output,
                              NodeDef* fused_node) {
  fused_node->set_op(kShuffleAndRepeatDatasetV2);
  graph_utils::SetUniqueGraphNodeName(kShuffleAndRepeatDatasetV2, output,
                                      fused_node);
  NodeDef zero_node = *graph_utils::AddScalarConstNode<int64_t>(0, graph);
  fused_node->add_input(shuffle_node.input(0));
  fused_node->add_input(shuffle_node.input(1));
  fused_node->add_input(zero_node.name());
  fused_node->add_input(zero_node.name());
  fused_node->add_input(repeat_node.input(1));
  fused_node->add_input(shuffle_node.input(2));
  graph_utils::CopyShapesAndTypesAttrs(shuffle_node, fused_node);
  (*fused_node->mutable_attr())[kReshuffleEachIteration].set_b(true);
  graph_utils::MaybeSetFusedMetadata(shuffle_node, repeat_node, fused_node);
  return absl::OkStatus();
}
Status FuseShuffleV3AndRepeat(const NodeDef& shuffle_node,
                              const NodeDef& repeat_node,
                              MutableGraphView* graph, GraphDef* output,
                              NodeDef* fused_node) {
  fused_node->set_op(kShuffleAndRepeatDatasetV2);
  graph_utils::SetUniqueGraphNodeName(kShuffleAndRepeatDataset, output,
                                      fused_node);
  fused_node->add_input(shuffle_node.input(0));
  fused_node->add_input(shuffle_node.input(1));
  fused_node->add_input(shuffle_node.input(2));
  fused_node->add_input(shuffle_node.input(3));
  fused_node->add_input(repeat_node.input(1));
  fused_node->add_input(shuffle_node.input(4));
  graph_utils::CopyShapesAndTypesAttrs(shuffle_node, fused_node);
  graph_utils::CopyAttribute(kReshuffleEachIteration, shuffle_node, fused_node);
  graph_utils::MaybeSetFusedMetadata(shuffle_node, repeat_node, fused_node);
  return absl::OkStatus();
}
}  
Status ShuffleAndRepeatFusion::OptimizeAndCollectStats(
    Cluster* cluster, const GrapplerItem& item, GraphDef* output,
    OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);
  absl::flat_hash_set<string> nodes_to_delete;
  for (const NodeDef& repeat_node : item.graph.node()) {
    if (repeat_node.op() != kRepeatDataset) {
      continue;
    }
    const NodeDef& shuffle_node =
        *graph_utils::GetInputNode(repeat_node, graph);
    NodeDef fused_node;
    if (shuffle_node.op() == kShuffleDataset) {
      TF_RETURN_IF_ERROR(FuseShuffleV1AndRepeat(shuffle_node, repeat_node,
                                                &graph, output, &fused_node));
    } else if (shuffle_node.op() == kShuffleDatasetV2) {
      TF_RETURN_IF_ERROR(FuseShuffleV2AndRepeat(shuffle_node, repeat_node,
                                                &graph, output, &fused_node));
    } else if (shuffle_node.op() == kShuffleDatasetV3) {
      TF_RETURN_IF_ERROR(FuseShuffleV3AndRepeat(shuffle_node, repeat_node,
                                                &graph, output, &fused_node));
    } else {
      continue;
    }
    NodeDef& shuffle_and_repeat_node = *graph.AddNode(std::move(fused_node));
    TF_RETURN_IF_ERROR(graph.UpdateFanouts(repeat_node.name(),
                                           shuffle_and_repeat_node.name()));
    TF_RETURN_IF_ERROR(graph.UpdateFanouts(shuffle_node.name(),
                                           shuffle_and_repeat_node.name()));
    const auto nodes_to_preserve = item.NodesToPreserve();
    if (nodes_to_preserve.find(shuffle_node.name()) ==
            nodes_to_preserve.end() &&
        nodes_to_preserve.find(repeat_node.name()) == nodes_to_preserve.end()) {
      nodes_to_delete.insert(shuffle_node.name());
      nodes_to_delete.insert(repeat_node.name());
    }
    stats->num_changes++;
  }
  TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));
  return absl::OkStatus();
}
REGISTER_GRAPH_OPTIMIZER_AS(ShuffleAndRepeatFusion,
                            "shuffle_and_repeat_fusion");
}  
}  