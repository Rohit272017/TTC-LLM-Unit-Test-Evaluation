#include "tensorflow/core/grappler/optimizers/memory_optimizer.h"
#include <algorithm>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.pb.h"  
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/costs/graph_memory.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/costs/utils.h"
#include "tensorflow/core/grappler/graph_topology_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/static_schedule.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/grappler/utils/traversal.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/util/device_name_utils.h"
namespace tensorflow {
namespace grappler {
namespace {
const char* kRecomputedNodePrefix = "Recomputed";
const char* kRecomputeTriggerNodePrefix = "RecomputeTrigger";
const char* kRecomputeHint = "_recompute_hint";
std::unordered_set<string> GetCheapToRecomputeOps() {
  std::unordered_set<string> cheap_ops = {"Add",
                                          "AddN",
                                          "BiasAdd",
                                          "Cast",
                                          "Fill",
                                          "FloorDiv",
                                          "FloorMod",
                                          "FusedBatchNorm",
                                          "LeakyRelu",
                                          "Mul",
                                          "Neg",
                                          "RealDiv",
                                          "Reciprocal",
                                          "Relu",
                                          "Relu6",
                                          "Reshape",
                                          "Rsqrt",
                                          "Sigmoid",
                                          "Sqrt",
                                          "Square",
                                          "SquaredDifference",
                                          "Sub",
                                          "Tile",
                                          "Transpose"};
  return cheap_ops;
}
std::unordered_set<const NodeDef*> FindCandidateRecomputeNodes(
    const NodeMap& node_map, const GraphDef* graph,
    const std::function<bool(const NodeDef&)>& is_candidate,
    const std::function<bool(const NodeDef&)>& is_target) {
  std::unordered_set<const NodeDef*> candidate_recompute_nodes;
  for (const auto& node : graph->node()) {
    if (!is_candidate(node)) {
      continue;
    }
    bool has_target_output = false;
    for (const NodeDef* output : node_map.GetOutputs(node.name())) {
      if (is_target(*output)) {
        has_target_output = true;
        break;
      }
    }
    if (!has_target_output) {
      continue;
    }
    bool has_target_input = false;
    for (const string& input_name : node.input()) {
      const NodeDef* input_node = node_map.GetNode(input_name);
      if (is_target(*input_node)) {
        has_target_input = true;
        break;
      }
    }
    if (has_target_input) {
      continue;
    }
    candidate_recompute_nodes.insert(&node);
  }
  return candidate_recompute_nodes;
}
void connected_subgraph(const NodeMap& node_map, bool collect_inputs,
                        bool collect_outputs,
                        const std::function<bool(const NodeDef&)>& is_candidate,
                        std::unordered_set<const NodeDef*>* expanded_nodes) {
  std::queue<const NodeDef*> to_visit;
  for (const NodeDef* starting_node : *expanded_nodes) {
    to_visit.push(starting_node);
  }
  expanded_nodes->clear();
  while (!to_visit.empty()) {
    const NodeDef* current_node = to_visit.front();
    to_visit.pop();
    if (!expanded_nodes->insert(current_node).second) {
      continue;
    }
    if (collect_inputs) {
      for (const string& input_name_raw : current_node->input()) {
        const NodeDef* input_node = node_map.GetNode(input_name_raw);
        if (expanded_nodes->count(input_node) == 0 &&
            is_candidate(*input_node)) {
          to_visit.push(input_node);
        }
      }
    }
    if (collect_outputs) {
      for (const NodeDef* output : node_map.GetOutputs(current_node->name())) {
        if (expanded_nodes->count(output) == 0 && is_candidate(*output)) {
          to_visit.push(output);
        }
      }
    }
  }
}
struct RecomputedSubGraph {
  std::unordered_set<const NodeDef*> recomputed_source_nodes;
  std::unordered_set<NodeDef*> target_nodes;
};
std::vector<RecomputedSubGraph> GetOpGroupsToRecompute(
    const GraphDef* graph, const NodeMap& node_map,
    const std::function<bool(const NodeDef&)>& should_recompute,
    const std::function<bool(const NodeDef&)>& is_target) {
  std::unordered_set<const NodeDef*> visited_nodes;
  std::vector<RecomputedSubGraph> subgraphs_to_recompute;
  std::unordered_set<const NodeDef*> candidate_recompute_nodes =
      FindCandidateRecomputeNodes(node_map, graph, should_recompute, is_target);
  for (const NodeDef* recompute_node : candidate_recompute_nodes) {
    if (visited_nodes.count(recompute_node) > 0) {
      continue;
    }
    RecomputedSubGraph current_recomputation;
    std::unordered_set<const NodeDef*> unpruned_recompute_nodes;
    unpruned_recompute_nodes.insert(recompute_node);
    connected_subgraph(node_map,
                       true,  
                       true,  
                       should_recompute, &unpruned_recompute_nodes);
    visited_nodes.insert(unpruned_recompute_nodes.begin(),
                         unpruned_recompute_nodes.end());
    for (const NodeDef* unpruned_recompute_node : unpruned_recompute_nodes) {
      bool inserted_feed = false;
      for (NodeDef* output :
           node_map.GetOutputs(unpruned_recompute_node->name())) {
        if (is_target(*output)) {
          current_recomputation.target_nodes.insert(output);
          if (!inserted_feed) {
            current_recomputation.recomputed_source_nodes.insert(
                unpruned_recompute_node);
            inserted_feed = true;
          }
        }
      }
    }
    connected_subgraph(
        node_map,
        true,   
        false,  
        [&unpruned_recompute_nodes](const NodeDef& node) {
          return unpruned_recompute_nodes.count(&node) != 0;
        },
        &current_recomputation.recomputed_source_nodes);
    if (current_recomputation.target_nodes.empty()) {
      continue;
    }
    subgraphs_to_recompute.push_back(current_recomputation);
  }
  return subgraphs_to_recompute;
}
std::unordered_map<const NodeDef*, int> GetMaxDownstreamComponents(
    const std::unordered_set<const NodeDef*>& recomputed_source_nodes,
    const std::unordered_set<NodeDef*>& target_nodes, const NodeMap& node_map,
    const std::unordered_map<const NodeDef*, int>& components) {
  std::unordered_map<const NodeDef*, int> recomputed_node_components;
  for (const NodeDef* original_recompute_node : recomputed_source_nodes) {
    int max_target_component = -1;
    for (NodeDef* output :
         node_map.GetOutputs(original_recompute_node->name())) {
      if (target_nodes.count(output) != 0) {
        int current_target_component = components.find(output)->second;
        if (current_target_component > max_target_component) {
          max_target_component = current_target_component;
        }
      }
    }
    if (max_target_component > -1) {
      recomputed_node_components[original_recompute_node] =
          max_target_component;
    }
  }
  std::vector<const NodeDef*> recomputed_source_nodes_topological(
      recomputed_source_nodes.begin(), recomputed_source_nodes.end());
  std::sort(recomputed_source_nodes_topological.begin(),
            recomputed_source_nodes_topological.end(),
            [&components](const NodeDef* first, const NodeDef* second) {
              return components.find(first)->second <
                     components.find(second)->second;
            });
  for (const NodeDef* original_recompute_node :
       recomputed_source_nodes_topological) {
    int max_component;
    auto recomputed_component_iterator =
        recomputed_node_components.find(original_recompute_node);
    if (recomputed_component_iterator != recomputed_node_components.end()) {
      max_component = recomputed_component_iterator->second;
    } else {
      max_component = -1;
    }
    for (NodeDef* output :
         node_map.GetOutputs(original_recompute_node->name())) {
      if (recomputed_source_nodes.count(output) == 0) {
        continue;
      }
      auto child_component_iterator = recomputed_node_components.find(output);
      CHECK(child_component_iterator != recomputed_node_components.end());
      int child_component = child_component_iterator->second;
      if (child_component > max_component) {
        max_component = child_component;
      }
    }
    CHECK_GE(max_component, 0);
    recomputed_node_components[original_recompute_node] = max_component;
  }
  return recomputed_node_components;
}
std::unordered_map<const NodeDef*, const NodeDef*>
AddRecomputeControlDependencyNodes(
    const std::unordered_set<const NodeDef*>& recomputed_source_nodes,
    const std::unordered_set<NodeDef*>& target_nodes, const NodeMap& node_map,
    const std::unordered_map<const NodeDef*, int>& components,
    const std::unordered_map<const NodeDef*, int>&
        recomputed_node_max_feed_components,
    GraphDef* graph) {
  std::vector<const NodeDef*> recomputed_source_nodes_topological(
      recomputed_source_nodes.begin(), recomputed_source_nodes.end());
  std::sort(recomputed_source_nodes_topological.begin(),
            recomputed_source_nodes_topological.end(),
            [&recomputed_node_max_feed_components](const NodeDef* first,
                                                   const NodeDef* second) {
              int first_component =
                  recomputed_node_max_feed_components.find(first)->second;
              int second_component =
                  recomputed_node_max_feed_components.find(second)->second;
              return first_component > second_component
                     || (first_component == second_component &&
                         first->name() > second->name());
            });
  std::vector<const NodeDef*> target_inputs_topological;
  for (const NodeDef* target_node : target_nodes) {
    for (const string& target_input_name_raw : target_node->input()) {
      const NodeDef* target_input = node_map.GetNode(target_input_name_raw);
      if (target_input == nullptr ||
          recomputed_source_nodes.count(target_input) != 0 ||
          components.find(target_node)->second ==
              components.find(target_input)->second) {
        continue;
      }
      target_inputs_topological.push_back(target_input);
    }
  }
  std::sort(target_inputs_topological.begin(), target_inputs_topological.end(),
            [&components](const NodeDef* first, const NodeDef* second) {
              return components.find(first)->second >
                     components.find(second)->second;
            });
  auto target_input_iterator = target_inputs_topological.begin();
  NodeDef* current_trigger_node = nullptr;
  std::unordered_map<const NodeDef*, const NodeDef*> triggers;
  for (const NodeDef* original_recomputed_node :
       recomputed_source_nodes_topological) {
    NodeDef* new_trigger_node = graph->add_node();
    new_trigger_node->set_name(AddPrefixToNodeName(
        original_recomputed_node->name(), kRecomputeTriggerNodePrefix));
    new_trigger_node->set_op("NoOp");
    new_trigger_node->set_device(original_recomputed_node->device());
    if (current_trigger_node != nullptr) {
      *new_trigger_node->add_input() =
          strings::StrCat("^", current_trigger_node->name());
    }
    current_trigger_node = new_trigger_node;
    triggers[original_recomputed_node] = current_trigger_node;
    for (;
         target_input_iterator != target_inputs_topological.end() &&
         components.find(*target_input_iterator)->second >
             recomputed_node_max_feed_components.find(original_recomputed_node)
                 ->second;
         ++target_input_iterator) {
      *current_trigger_node->add_input() =
          strings::StrCat("^", (*target_input_iterator)->name());
      VLOG(2) << "  Recomputation trigger " << current_trigger_node->name()
              << " depends on " << (*target_input_iterator)->name();
    }
  }
  return triggers;
}
string RecomputedOrOriginalNodeName(
    const std::unordered_set<string>& recomputed_node_names,
    const string& original_node_name) {
  if (recomputed_node_names.find(original_node_name) ==
      recomputed_node_names.end()) {
    return original_node_name;
  } else {
    return AddPrefixToNodeName(original_node_name, kRecomputedNodePrefix);
  }
}
void RecomputeSubgraph(
    const std::unordered_set<const NodeDef*>& recomputed_source_nodes,
    const std::unordered_set<NodeDef*>& target_nodes, const NodeMap& node_map,
    const std::unordered_map<const NodeDef*, int>& components,
    GraphDef* graph) {
  std::unordered_set<string> recomputed_node_names;
  VLOG(1) << "Recomputing a " << recomputed_source_nodes.size()
          << " node subgraph";
  std::unordered_map<const NodeDef*, int> recomputed_node_components =
      GetMaxDownstreamComponents(recomputed_source_nodes, target_nodes,
                                 node_map, components);
  for (const NodeDef* original_node : recomputed_source_nodes) {
    VLOG(2) << "  " << original_node->name();
    recomputed_node_names.insert(original_node->name());
  }
  std::unordered_map<const NodeDef*, const NodeDef*> triggers =
      AddRecomputeControlDependencyNodes(recomputed_source_nodes, target_nodes,
                                         node_map, components,
                                         recomputed_node_components, graph);
  for (const NodeDef* original_node : recomputed_source_nodes) {
    NodeDef* copied_node = graph->add_node();
    copied_node->set_name(
        AddPrefixToNodeName(original_node->name(), kRecomputedNodePrefix));
    copied_node->set_op(original_node->op());
    *copied_node->mutable_attr() = original_node->attr();
    copied_node->set_device(original_node->device());
    for (const string& original_input_name : original_node->input()) {
      *copied_node->add_input() = RecomputedOrOriginalNodeName(
          recomputed_node_names, original_input_name);
    }
    *copied_node->add_input() =
        strings::StrCat("^", triggers[original_node]->name());
  }
  for (NodeDef* target_node : target_nodes) {
    for (string& target_input_name : *target_node->mutable_input()) {
      target_input_name = RecomputedOrOriginalNodeName(recomputed_node_names,
                                                       target_input_name);
    }
  }
}
void RecomputationRewritingPass(RewriterConfig::MemOptType optimization_level,
                                const string& recomputation_targets_name_scope,
                                GraphDef* graph, const GrapplerItem& item) {
  TF_CHECK_OK(TopologicalSort(graph));
  NodeMap node_map(graph);
  std::vector<RecomputedSubGraph> recomputed_subgraphs;
  std::unordered_set<string> feeds;
  for (const auto& feed : item.feed) {
    feeds.insert(NodeName(feed.first));
  }
  std::function<bool(const NodeDef&)> is_target =
      [&recomputation_targets_name_scope](const NodeDef& node) {
        return absl::StartsWith(node.name(),
                                recomputation_targets_name_scope) ||
               static_cast<int>(node.name().find(
                   "/" + recomputation_targets_name_scope)) != -1;
      };
  if (optimization_level == RewriterConfig::RECOMPUTATION_HEURISTICS ||
      optimization_level == RewriterConfig::HEURISTICS) {
    std::unordered_set<string> cheap_to_recompute_ops =
        GetCheapToRecomputeOps();
    recomputed_subgraphs = GetOpGroupsToRecompute(
        graph, node_map,
        [&cheap_to_recompute_ops, &feeds, &is_target](const NodeDef& node) {
          return !is_target(node) && feeds.count(node.name()) == 0 &&
                 (cheap_to_recompute_ops.count(node.op()) > 0 ||
                  node.attr().count(kRecomputeHint) > 0);
        },
        is_target);
  } else if (optimization_level == RewriterConfig::MANUAL) {
    recomputed_subgraphs = GetOpGroupsToRecompute(
        graph, node_map,
        [&feeds, &is_target](const NodeDef& node) {
          return !is_target(node) && feeds.count(node.name()) == 0 &&
                 node.attr().count(kRecomputeHint) > 0;
        },
        is_target);
  }
  if (!recomputed_subgraphs.empty()) {
    std::unordered_map<const NodeDef*, int> topological_numbering;
    for (int node_number = 0; node_number < graph->node().size();
         ++node_number) {
      topological_numbering[graph->mutable_node(node_number)] =
          graph->node().size() - node_number - 1;
    }
    for (const RecomputedSubGraph& subgraph : recomputed_subgraphs) {
      RecomputeSubgraph(subgraph.recomputed_source_nodes, subgraph.target_nodes,
                        node_map, topological_numbering, graph);
    }
  }
}
bool SchedulingPass(Cluster* cluster, std::unique_ptr<GraphMemory>* memory_ptr,
                    GrapplerItem* item) {
  MutableGraphView view(&item->graph);
  std::unordered_map<string, std::unordered_set<NodeDef*>> addn_list;
  for (NodeDef& node : *item->graph.mutable_node()) {
    if (!IsAddN(node) && node.op() != "AccumulateNV2") {
      continue;
    }
    if (view.NumFanins(node, false) <= 2) {
      continue;
    }
    for (const auto& input : view.GetFanins(node, false)) {
      if (input.node->device() == node.device()) {
        string tensor_name =
            strings::StrCat(input.node->name(), ":", input.port_id);
        addn_list[tensor_name].insert(&node);
      }
    }
  }
  if (addn_list.empty()) {
    return false;
  }
  if ((*memory_ptr) == nullptr) {
    memory_ptr->reset(new GraphMemory(*item));
    Status s = (*memory_ptr)->InferStatically(cluster->GetDevices());
    if (!s.ok()) {
      memory_ptr->reset();
      VLOG(1) << "Failed to infer memory usage: " << s.message();
      return false;
    }
  }
  const GraphMemory& memory = **memory_ptr;
  std::unordered_set<NodeDef*> addn_to_rewrite;
  for (const auto& device : cluster->GetDevices()) {
    const string& name = device.first;
    const DeviceProperties& prop = device.second;
    if (prop.memory_size() <= 0) {
      VLOG(1) << "Available memory unknown for device " << name;
      continue;
    }
    const GraphMemory::MemoryUsage& mem_usage = memory.GetPeakMemoryUsage(name);
    if (mem_usage.used_memory <= prop.memory_size() * 0.8) {
      continue;
    }
    for (const auto& live : mem_usage.live_tensors) {
      string tensor_name = strings::StrCat(live.node, ":", live.output_id);
      auto it = addn_list.find(tensor_name);
      if (it != addn_list.end()) {
        addn_to_rewrite.insert(it->second.begin(), it->second.end());
      }
    }
  }
  if (addn_to_rewrite.empty()) {
    return false;
  }
  GraphProperties properties(*item);
  Status s = properties.InferStatically(false,
                                        false,
                                        false);
  if (!s.ok()) {
    VLOG(1) << "Failed to infer shapes: " << s.message();
    return false;
  }
  GraphTopologyView graph_topology;
  Status initialized_topology = graph_topology.InitializeFromGraph(item->graph);
  if (!initialized_topology.ok()) {
    VLOG(1) << "Failed to initialize graph topology view: "
            << initialized_topology.message();
    return false;
  }
  bool updated_graph = false;
  for (NodeDef* node : addn_to_rewrite) {
    if (!properties.HasOutputProperties(node->name())) {
      VLOG(1) << "Missing properties for " << node->name();
      continue;
    }
    const TensorShapeProto& shape =
        properties.GetOutputProperties(node->name())[0].shape();
    PartialTensorShape shp(shape);
    if (!shp.IsFullyDefined()) {
      VLOG(1) << "Shape not fully known for " << node->name();
      continue;
    }
    DataType dtype = node->attr().at("T").type();
    if (dtype != DT_HALF && dtype != DT_FLOAT && dtype != DT_DOUBLE &&
        dtype != DT_INT64) {  
      VLOG(1) << "Unsupported dtype for " << node->name();
      continue;
    }
    std::unordered_map<const NodeDef*, int> topo_order;
    DfsTraversal(graph_topology, {node}, TraversalDirection::kFollowInputs,
                 DfsCallbacks::PostOrder([&topo_order](const NodeDef* n) {
                   int topo_index = static_cast<int>(topo_order.size());
                   topo_order[n] = topo_index;
                 }));
    std::vector<int> input_topo_index;
    for (int i = 0; i < node->input_size(); ++i) {
      const string& input = node->input(i);
      const string node_name = NodeName(input);
      const NodeDef* node = view.GetNode(node_name);
      input_topo_index.push_back(topo_order.at(node));
    }
    int min_input_topo_index = INT_MAX;
    int min_input_id = -1;
    for (int i = 0; i < node->input_size(); ++i) {
      if (IsControlInput(node->input(i))) {
        break;
      }
      const int current = input_topo_index[i];
      if (current < min_input_topo_index) {
        min_input_topo_index = current;
        min_input_id = i;
      }
    }
    CHECK_LE(0, min_input_id);
    std::vector<string> pre_ctrl_deps;
    std::vector<string> post_ctrl_deps;
    for (int i = node->input_size() - 1; i >= 0; --i) {
      if (!IsControlInput(node->input(i))) {
        break;
      }
      if (input_topo_index[i] < min_input_topo_index) {
        pre_ctrl_deps.push_back(node->input(i));
      } else {
        post_ctrl_deps.push_back(node->input(i));
      }
    }
    const string& device = node->device();
    const string tmp_var_name = strings::StrCat(node->name(), "/tmp_var");
    if (view.GetNode(tmp_var_name) != nullptr) {
      VLOG(1) << "Temporary variable already exists " << tmp_var_name;
      return false;
    }
    NodeDef* tmp_var = item->graph.add_node();
    tmp_var->set_name(tmp_var_name);
    tmp_var->set_op("TemporaryVariable");
    tmp_var->set_device(device);
    (*tmp_var->mutable_attr())["dtype"].set_type(dtype);
    *(*tmp_var->mutable_attr())["shape"].mutable_shape() = shape;
    (*tmp_var->mutable_attr())["var_name"].set_s(tmp_var->name());
    for (const string& ctrl_dep : pre_ctrl_deps) {
      *tmp_var->add_input() = ctrl_dep;
    }
    *tmp_var->add_input() =
        AsControlDependency(NodeName(node->input(min_input_id)));
    NodeDef* zeros = item->graph.add_node();
    zeros->set_name(strings::StrCat(node->name(), "/tmp_var_zeros"));
    zeros->set_op("ZerosLike");
    zeros->set_device(device);
    (*zeros->mutable_attr())["T"].set_type(dtype);
    *zeros->add_input() = node->input(min_input_id);
    NodeDef* initialize = item->graph.add_node();
    initialize->set_name(strings::StrCat(node->name(), "/tmp_var_initializer"));
    initialize->set_op("Assign");
    initialize->set_device(device);
    (*initialize->mutable_attr())["T"].set_type(dtype);
    (*initialize->mutable_attr())["use_locking"].set_b(false);
    (*initialize->mutable_attr())["validate_shape"].set_b(false);
    *initialize->add_input() = tmp_var->name();
    *initialize->add_input() = zeros->name();
    std::vector<NodeDef*> accumulates;
    for (int i = 0; i < node->input_size(); ++i) {
      const string& input = node->input(i);
      if (!IsControlInput(input)) {
        NodeDef* accumulate = item->graph.add_node();
        accumulate->set_name(
            strings::StrCat(node->name(), "/tmp_var_accum_", i));
        accumulate->set_op("AssignAdd");
        accumulate->set_device(device);
        (*accumulate->mutable_attr())["T"].set_type(dtype);
        (*accumulate->mutable_attr())["use_locking"].set_b(true);
        *accumulate->add_input() = initialize->name();
        *accumulate->add_input() = input;
        accumulates.push_back(accumulate);
      }
    }
    node->set_op("DestroyTemporaryVariable");
    node->clear_input();
    EraseRegularNodeAttributes(node);
    (*node->mutable_attr())["T"].set_type(dtype);
    (*node->mutable_attr())["var_name"].set_s(tmp_var->name());
    *node->add_input() = initialize->name();
    for (const NodeDef* accum : accumulates) {
      *node->add_input() = AsControlDependency(accum->name());
    }
    for (const string& ctrl_dep : post_ctrl_deps) {
      *node->add_input() = ctrl_dep;
    }
    updated_graph = true;
  }
  return updated_graph;
}
Status BuildSwapPair(NodeDef* node, int input_to_swap,
                     const std::unordered_map<string, const NodeDef*>& name_map,
                     GraphDef* graph,
                     std::pair<NodeDef*, NodeDef*>* swap_pair) {
  string task, device;
  if (!DeviceNameUtils::SplitDeviceName(node->device(), &task, &device) ||
      !absl::StrContains(device, DEVICE_GPU)) {
    return errors::InvalidArgument("Can't swap input ", input_to_swap,
                                   " of node ", node->name(),
                                   " since it is not on GPU");
  }
  const OpDef* op_def;
  TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUpOpDef(node->op(), &op_def));
  DataType input_type;
  TF_RETURN_IF_ERROR(
      InputTypeForNode(*node, *op_def, input_to_swap, &input_type));
  if (IsRefType(input_type)) {
    return errors::InvalidArgument("Can't swap input ", input_to_swap,
                                   " of node ", node->name(),
                                   " since it expects a reference");
  }
  string tensor_to_swap = strings::StrCat(node->name(), "_", input_to_swap);
  string swap_out_name = strings::StrCat("swap_out_", tensor_to_swap);
  string swap_in_name = strings::StrCat("swap_in_", tensor_to_swap);
  if (name_map.find(swap_out_name) != name_map.end() ||
      name_map.find(swap_in_name) != name_map.end()) {
    return errors::InvalidArgument("Input ", input_to_swap, " of node ",
                                   node->name(), " is already swapped");
  }
  NodeDef* swap_out_node = graph->add_node();
  swap_out_node->set_name(swap_out_name);
  swap_out_node->set_op("_CopyFromGpuToHost");
  NodeDef* swap_in_node = graph->add_node();
  swap_in_node->set_name(swap_in_name);
  swap_in_node->set_op("_CopyFromHostToGpu");
  *swap_in_node->add_input() = swap_out_node->name();
  swap_out_node->set_device(node->device());
  swap_in_node->set_device(node->device());
  string coloc_group = strings::StrCat("loc@", tensor_to_swap);
  (*swap_out_node->mutable_attr())["_class"].mutable_list()->add_s(coloc_group);
  (*swap_in_node->mutable_attr())["_class"].mutable_list()->add_s(coloc_group);
  (*node->mutable_attr())["_class"].mutable_list()->add_s(coloc_group);
  (*swap_in_node->mutable_attr())["T"].set_type(input_type);
  (*swap_out_node->mutable_attr())["T"].set_type(input_type);
  *swap_pair = std::make_pair(swap_out_node, swap_in_node);
  return absl::OkStatus();
}
struct SwapInfo {
  std::vector<int> inputs_to_swap;
  Costs::NanoSeconds time_to_swap = 0;
};
static const NodeDef* FindSwapInTrigger(
    const NodeDef* node, const SwapInfo& swap_info,
    const std::unordered_map<string, const NodeDef*>& name_map,
    const std::unordered_map<const NodeDef*, Costs::NanoSeconds>&
        execution_times) {
  Costs::NanoSeconds max_trigger_time(0);
  std::set<string> possible_inputs;
  for (int i = 0; i < node->input_size(); ++i) {
    const string input_node_name = NodeName(node->input(i));
    auto it1 = name_map.find(input_node_name);
    if (it1 == name_map.end()) {
      return nullptr;
    }
    const NodeDef* input_node = it1->second;
    auto it2 = execution_times.find(input_node);
    if (it2 == execution_times.end()) {
      return nullptr;
    }
    max_trigger_time = std::max(max_trigger_time, it2->second);
    possible_inputs.insert(input_node_name);
  }
  for (const int i : swap_info.inputs_to_swap) {
    const string input_node_name = NodeName(node->input(i));
    possible_inputs.erase(input_node_name);
  }
  if (possible_inputs.empty()) {
    return nullptr;
  }
  max_trigger_time -= swap_info.time_to_swap;
  std::map<Costs::NanoSeconds, const NodeDef*> candidates;
  std::set<string> already_processed;
  while (!possible_inputs.empty()) {
    const string input_node_name = *possible_inputs.begin();
    possible_inputs.erase(possible_inputs.begin());
    already_processed.insert(input_node_name);
    auto it1 = name_map.find(input_node_name);
    if (it1 == name_map.end()) {
      return nullptr;
    }
    const NodeDef* input_node = it1->second;
    if (ModifiesFrameInfo(*input_node) || IsSwitch(*input_node) ||
        IsMerge(*input_node)) {
      continue;
    }
    auto it2 = execution_times.find(input_node);
    if (it2 == execution_times.end()) {
      return nullptr;
    }
    if (it2->second < max_trigger_time) {
      candidates[it2->second] = input_node;
    } else {
      for (const string& fanin : input_node->input()) {
        string name = NodeName(fanin);
        if (already_processed.find(name) == already_processed.end()) {
          possible_inputs.insert(name);
        }
      }
    }
  }
  if (!candidates.empty()) {
    return candidates.rbegin()->second;
  }
  return nullptr;
}
static bool IsSwappable(const MutableGraphView& graph,
                        MutableGraphView::OutputPort output) {
  const NodeDef& node = *output.node;
  if (IsPersistent(node)) {
    return false;
  }
  const OpDef* op_def;
  if (!OpRegistry::Global()->LookUpOpDef(node.op(), &op_def).ok()) {
    return false;
  }
  DataType dtype;
  if (!OutputTypeForNode(node, *op_def, output.port_id, &dtype).ok()) {
    return false;
  }
  if (IsRefType(dtype)) {
    return false;
  }
  if (output.node->op() == "Identity" || output.node->op() == "Reshape") {
    MutableGraphView::InputPort input;
    input.node = output.node;
    input.port_id = 0;
    MutableGraphView::OutputPort fanin = graph.GetRegularFanin(input);
    if (fanin.node->device() == node.device()) {
      return IsSwappable(graph, fanin);
    }
  }
  return true;
}
static NodeDef* FindSwapOutTrigger(
    const NodeDef* node, int input_id, const MutableGraphView& view,
    const std::unordered_map<const NodeDef*, Costs::NanoSeconds>&
        execution_times) {
  MutableGraphView::InputPort swap;
  swap.node = const_cast<NodeDef*>(node);
  swap.port_id = input_id;
  MutableGraphView::OutputPort generator = view.GetRegularFanin(swap);
  if (!generator.node) {
    return nullptr;
  }
  const absl::flat_hash_set<MutableGraphView::InputPort>& fanout =
      view.GetFanout(generator);
  NodeDef* trigger = nullptr;
  Costs::NanoSeconds earliest_fanout(Costs::NanoSeconds::infinity());
  for (const auto& port : fanout) {
    if (port.node == node) {
      continue;
    }
    auto it = execution_times.find(port.node);
    if (it != execution_times.end() && it->second < earliest_fanout) {
      earliest_fanout = it->second;
      trigger = port.node;
    }
  }
  return trigger;
}
static bool IsSwappable(MutableGraphView::InputPort input) {
  const NodeDef& node = *input.node;
  const OpDef* op_def;
  if (!OpRegistry::Global()->LookUpOpDef(node.op(), &op_def).ok()) {
    return false;
  }
  DataType dtype;
  if (!InputTypeForNode(node, *op_def, input.port_id, &dtype).ok()) {
    return false;
  }
  return !IsRefType(dtype);
}
struct MemInfo {
  MutableGraphView::OutputPort port;
  int64_t memory_used;
  std::vector<MutableGraphView::InputPort> uses_left;
  double fitness;
  bool operator<(const MemInfo& other) const { return fitness < other.fitness; }
};
static bool IdentifySwappingCandidates(
    Cluster* cluster, GrapplerItem* item,
    std::unique_ptr<GraphMemory>* memory_ptr,
    std::unordered_set<string>* skip_list,
    std::unordered_map<NodeDef*, SwapInfo>* nodes_to_swap) {
  if ((*memory_ptr) == nullptr) {
    memory_ptr->reset(new GraphMemory(*item));
    Status s = (*memory_ptr)->InferStatically(cluster->GetDevices());
    if (!s.ok()) {
      memory_ptr->reset();
      VLOG(1) << "Failed to infer memory usage: " << s.message();
      return false;
    }
  }
  const GraphMemory& memory = **memory_ptr;
  bool updated_graph = false;
  for (const auto& device : cluster->GetDevices()) {
    const string& name = device.first;
    const DeviceProperties& prop = device.second;
    if (prop.type() != "GPU") {
      continue;
    }
    if (prop.memory_size() <= 0) {
      VLOG(1) << "Peak memory usage unknown for device " << name;
      continue;
    }
    const GraphMemory::MemoryUsage& mem_usage = memory.GetPeakMemoryUsage(name);
    if (mem_usage.used_memory <= prop.memory_size()) {
      continue;
    }
    int64_t required_savings = mem_usage.used_memory - prop.memory_size();
    std::unordered_map<string, Costs::NanoSeconds> op_completion_times;
    {
      VirtualCluster vcluster(cluster->GetDevices());
      if (!vcluster.Provision().ok()) {
        return false;
      }
      if (!vcluster.Initialize(*item).ok()) {
        return false;
      }
      RunMetadata metadata;
      Status s = vcluster.Run(item->graph, item->feed, item->fetch, &metadata);
      if (!s.ok() && s.code() != error::RESOURCE_EXHAUSTED) {
        return false;
      }
      for (const auto& dev_stats : metadata.step_stats().dev_stats()) {
        for (const auto& node_stats : dev_stats.node_stats()) {
          Costs::NanoSeconds exec_time =
              Costs::NanoSeconds(1) +
              Costs::MicroSeconds(node_stats.all_start_micros() +
                                  node_stats.op_end_rel_micros());
          op_completion_times.emplace(node_stats.node_name(), exec_time);
        }
      }
    }
    Costs::Duration peak_time = -1;
    for (const auto& live_tensor : mem_usage.live_tensors) {
      if (live_tensor.allocation_time > peak_time) {
        peak_time = live_tensor.allocation_time;
      }
    }
    std::vector<MemInfo> mem_state;
    MutableGraphView graph(&item->graph);
    for (const auto& live_tensor : mem_usage.live_tensors) {
      if (live_tensor.memory_used <= 1024) {
        continue;
      }
      if (live_tensor.deallocation_time - live_tensor.allocation_time <=
          Costs::Duration(1e6)) {
        VLOG(1) << "Not enough time to swap: skipping " << live_tensor.node;
        continue;
      }
      if (skip_list->find(live_tensor.node) != skip_list->end()) {
        continue;
      }
      MutableGraphView::OutputPort port =
          graph.GetOutputPort(live_tensor.node, live_tensor.output_id);
      if (!IsSwappable(graph, port)) {
        continue;
      }
      MemInfo mem_info;
      mem_info.port = port;
      mem_info.memory_used = live_tensor.memory_used;
      Costs::Duration allocation_time = live_tensor.allocation_time;
      Costs::Duration earliest_use(Costs::Duration::infinity());
      bool valid = true;
      for (MutableGraphView::InputPort input : graph.GetFanout(port)) {
        auto it = op_completion_times.find(input.node->name());
        if (it == op_completion_times.end()) {
          valid = false;
          break;
        }
        if (it->second <= peak_time) {
          continue;
        }
        if (skip_list->find(input.node->name()) != skip_list->end()) {
          valid = false;
          break;
        }
        string input_name =
            strings::StrCat(input.node->name(), ":", input.port_id);
        if (skip_list->find(input_name) != skip_list->end()) {
          valid = false;
          break;
        }
        if (!IsSwappable(input)) {
          valid = false;
          break;
        }
        mem_info.uses_left.emplace_back(input);
        earliest_use = std::min(earliest_use, it->second);
      }
      if (valid && !mem_info.uses_left.empty()) {
        mem_info.fitness =
            MathUtil::IPow<double>((earliest_use - peak_time).count(), 2) /
                MathUtil::IPow<double>(mem_info.uses_left.size(), 2) +
            MathUtil::IPow<double>((allocation_time - peak_time).count(), 2);
        mem_info.fitness = -mem_info.fitness;
        mem_state.push_back(mem_info);
      }
    }
    std::sort(mem_state.begin(), mem_state.end());
    for (const MemInfo& mem_info : mem_state) {
      for (const MutableGraphView::InputPort fanout_to_swap :
           mem_info.uses_left) {
        VLOG(1) << "Will swap fanout " << fanout_to_swap.node->name() << ":"
                << fanout_to_swap.port_id << " of tensor "
                << mem_info.port.node->name() << ":" << mem_info.port.port_id
                << " of size " << mem_info.memory_used;
        (*nodes_to_swap)[fanout_to_swap.node].inputs_to_swap.push_back(
            fanout_to_swap.port_id);
      }
      required_savings -= mem_info.memory_used;
      updated_graph = true;
      if (required_savings < 0) {
        break;
      }
    }
  }
  return updated_graph;
}
bool SwappingPass(RewriterConfig::MemOptType optimization_level,
                  Cluster* cluster, std::unique_ptr<GraphMemory>* memory,
                  GrapplerItem* item, std::unordered_set<string>* skip_list) {
  std::unordered_map<NodeDef*, SwapInfo> nodes_to_swap;
  if (optimization_level == RewriterConfig::DEFAULT_MEM_OPT ||
      optimization_level == RewriterConfig::SWAPPING_HEURISTICS ||
      optimization_level == RewriterConfig::HEURISTICS) {
    IdentifySwappingCandidates(cluster, item, memory, skip_list,
                               &nodes_to_swap);
  }
  for (auto& node : *item->graph.mutable_node()) {
    if (node.attr().count("_swap_to_host") != 0) {
      SwapInfo& swap_info = nodes_to_swap[&node];
      const AttrValue& val = node.attr().at("_swap_to_host");
      if (val.has_list()) {
        for (int64_t input_id : val.list().i()) {
          swap_info.inputs_to_swap.push_back(input_id);
        }
      } else {
        int64_t input_id = val.i();
        swap_info.inputs_to_swap.push_back(input_id);
      }
    }
  }
  if (nodes_to_swap.empty()) {
    return false;
  }
  GraphProperties properties(*item);
  if (!properties
           .InferStatically(true,
                            false,
                            false)
           .ok()) {
    return false;
  }
  for (auto& swap : nodes_to_swap) {
    const NodeDef* node = swap.first;
    const std::vector<OpInfo::TensorProperties>& props =
        properties.GetInputProperties(node->name());
    SwapInfo& swap_info = swap.second;
    int64_t bytes_to_swap = 0;
    for (int64_t input_id : swap_info.inputs_to_swap) {
      const OpInfo::TensorProperties& t = props[input_id];
      bytes_to_swap += CalculateTensorSize(t);
    }
    swap_info.time_to_swap = bytes_to_swap / 16;
  }
  std::unordered_map<const NodeDef*, Costs::NanoSeconds> execution_times;
  if (!EstimateEarliestExecutionTimes(*item, cluster, &execution_times).ok()) {
    return false;
  }
  std::unordered_map<string, const NodeDef*> name_map;
  for (const auto& node : item->graph.node()) {
    name_map[node.name()] = &node;
  }
  MutableGraphView view(&item->graph);
  bool updated_graph = false;
  for (auto& swap : nodes_to_swap) {
    NodeDef* node = swap.first;
    const SwapInfo& swap_info = swap.second;
    if (skip_list->find(node->name()) != skip_list->end()) {
      continue;
    }
    const NodeDef* in_trigger =
        FindSwapInTrigger(node, swap_info, name_map, execution_times);
    if (!in_trigger) {
      skip_list->insert(node->name());
      continue;
    }
    for (int input_id : swap_info.inputs_to_swap) {
      string input_name = strings::StrCat(node->name(), ":", input_id);
      if (skip_list->find(input_name) != skip_list->end()) {
        continue;
      } else {
        skip_list->insert(input_name);
      }
      NodeDef* out_trigger =
          FindSwapOutTrigger(node, input_id, view, execution_times);
      if (!out_trigger) {
        continue;
      }
      std::pair<NodeDef*, NodeDef*> swap_nodes;
      if (!BuildSwapPair(node, input_id, name_map, &item->graph, &swap_nodes)
               .ok()) {
        continue;
      }
      *swap_nodes.first->add_input() = node->input(input_id);
      *node->mutable_input(input_id) = swap_nodes.second->name();
      out_trigger->add_input(strings::StrCat("^", swap_nodes.first->name()));
      swap_nodes.second->add_input(strings::StrCat("^", in_trigger->name()));
      skip_list->insert(swap_nodes.first->name());
      skip_list->insert(swap_nodes.second->name());
    }
  }
  return updated_graph;
}
bool CrossesTaskOrCpuGpuBoundary(const NodeDef& node1, const NodeDef& node2) {
  string task1;
  string device1;
  DeviceNameUtils::SplitDeviceName(node1.device(), &task1, &device1);
  string task2;
  string device2;
  DeviceNameUtils::SplitDeviceName(node2.device(), &task2, &device2);
  return task1 != task2 ||
         (absl::StrContains(device1, DEVICE_CPU) &&
          absl::StrContains(device2, DEVICE_GPU)) ||
         (absl::StrContains(device1, DEVICE_GPU) &&
          absl::StrContains(device2, DEVICE_CPU));
}
void RelaxAssignNodes(const std::set<int>& nodes_to_relax,
                      GraphDef* optimized_graph) {
  for (int idx : nodes_to_relax) {
    NodeDef* assign_node = optimized_graph->mutable_node(idx);
    (*assign_node->mutable_attr())["_grappler_relax_allocator_constraints"]
        .set_b(true);
  }
}
Status FindAssignNodesToRelax(const GraphDef& graph,
                              std::set<int>* nodes_to_relax) {
  std::unordered_set<string> devices;
  std::vector<int> assign_nodes;
  bool found_send = false;
  for (int i = 0; i < graph.node_size(); ++i) {
    const NodeDef& node = graph.node(i);
    devices.insert(node.device());
    if (IsAssign(node)) {
      assign_nodes.push_back(i);
    }
    if (IsSend(node)) {
      found_send = true;
      break;
    }
  }
  if (!found_send && devices.size() == 1) {
    nodes_to_relax->insert(assign_nodes.begin(), assign_nodes.end());
    return absl::OkStatus();
  }
  GraphTopologyView graph_view;
  TF_RETURN_IF_ERROR(
      graph_view.InitializeFromGraph(graph, true));
  std::unordered_set<const NodeDef*> optimized_nodes;
  for (int i : assign_nodes) {
    const NodeDef& assign_node = graph.node(i);
    if (optimized_nodes.find(&assign_node) == optimized_nodes.end()) {
      std::vector<const NodeDef*> assign_nodes_in_fanout;
      optimized_nodes.insert(&assign_node);
      assign_nodes_in_fanout.push_back(&assign_node);
      std::vector<const NodeDef*> transitive_fanout;
      DfsTraversal(graph_view, {graph_view.GetNode(i)},
                   TraversalDirection::kFollowOutputs,
                   DfsPredicates::Advance([&](const NodeDef* node) {
                     return !NeverForwardsInputs(*node);
                   }),
                   DfsCallbacks::PreOrder([&](const NodeDef* node) {
                     transitive_fanout.push_back(node);
                   }));
      bool relax_constraint = true;
      for (const NodeDef* fanout_node : transitive_fanout) {
        if (relax_constraint &&
            (IsSend(*fanout_node) ||
             CrossesTaskOrCpuGpuBoundary(*fanout_node, assign_node))) {
          relax_constraint = false;
          break;
        }
        if (optimized_nodes.find(fanout_node) == optimized_nodes.end() &&
            IsAssign(*fanout_node)) {
          assign_nodes_in_fanout.push_back(fanout_node);
        }
      }
      if (relax_constraint) {
        for (const NodeDef* assign_node_in_fanout : assign_nodes_in_fanout) {
          optimized_nodes.insert(assign_node_in_fanout);
          const absl::optional<int> assign_node_idx =
              graph_view.GetNodeIndex(*assign_node_in_fanout);
          nodes_to_relax->insert(assign_node_idx.value());
        }
      }
    }
  }
  return absl::OkStatus();
}
}  
Status MemoryOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* optimized_graph) {
  std::set<int> nodes_to_relax;
  TF_RETURN_IF_ERROR(FindAssignNodesToRelax(item.graph, &nodes_to_relax));
  bool run_recomputation_pass =
      (optimization_level_ == RewriterConfig::RECOMPUTATION_HEURISTICS ||
       optimization_level_ == RewriterConfig::HEURISTICS ||
       optimization_level_ == RewriterConfig::MANUAL);
  if (!run_recomputation_pass && nodes_to_relax.empty() && item.fetch.empty()) {
    return errors::Aborted("Nothing to do.");
  }
  GrapplerItem optimized_item(item);
  RelaxAssignNodes(nodes_to_relax, &optimized_item.graph);
  if (run_recomputation_pass) {
    RecomputationRewritingPass(optimization_level_,
                               recomputation_targets_name_scope_,
                               &optimized_item.graph, item);
  }
  std::unordered_set<string> skip_list;
  std::unique_ptr<GraphMemory> memory;
  if (!item.fetch.empty() && cluster != nullptr) {
    bool updated_graph = true;
    for (int i = 0; i < 25 && updated_graph; ++i) {
      GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();
      updated_graph = false;
      if ((optimization_level_ == RewriterConfig::DEFAULT_MEM_OPT ||
           optimization_level_ == RewriterConfig::SCHEDULING_HEURISTICS ||
           optimization_level_ == RewriterConfig::HEURISTICS) &&
          cluster != nullptr) {
        if (SchedulingPass(cluster, &memory, &optimized_item)) {
          memory.reset();
          updated_graph = true;
        }
      }
      GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();
      if ((optimization_level_ == RewriterConfig::DEFAULT_MEM_OPT ||
           optimization_level_ == RewriterConfig::SWAPPING_HEURISTICS ||
           optimization_level_ == RewriterConfig::HEURISTICS ||
           optimization_level_ == RewriterConfig::MANUAL) &&
          cluster != nullptr) {
        if (SwappingPass(optimization_level_, cluster, &memory, &optimized_item,
                         &skip_list)) {
          memory.reset();
          updated_graph = true;
        }
      }
    }
  }
  optimized_graph->Swap(&optimized_item.graph);
  return absl::OkStatus();
}
}  
}  