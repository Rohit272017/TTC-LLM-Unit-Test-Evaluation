#include "tensorflow/core/grappler/optimizers/model_pruner.h"
#include <unordered_set>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/transitive_fanin.h"
namespace tensorflow {
namespace grappler {
namespace {
bool IsTrivialIdentity(const NodeDef& node, const GraphView& graph_view) {
  for (const auto input :
       graph_view.GetFanins(node, true)) {
    if (input.port_id == Graph::kControlSlot) {
      return false;
    } else if (IsSwitch(*input.node)) {  
      return false;
    }
  }
  for (const auto output :
       graph_view.GetFanouts(node, true)) {
    if (output.port_id == Graph::kControlSlot) {
      return false;
    } else if (IsMerge(*output.node)) {  
      return false;
    }
  }
  return true;
}
bool IsTrivialOp(const NodeDef& node, const GraphView& graph_view) {
  if (IsStopGradient(node)) {
    return true;
  }
  if (IsIdentity(node) || IsIdentityNSingleInput(node)) {
    return IsTrivialIdentity(node, graph_view);
  }
  if (IsNoOp(node) && node.input().empty()) {
    return true;
  }
  if (IsConstant(node) && node.input().empty() &&
      graph_view.NumFanouts(node, false) == 0) {
    return true;
  }
  return IsAddN(node) && NumNonControlInputs(node) <= 1;
}
bool RemovalIncreasesEdgeCount(const NodeDef& node,
                               const GraphView& graph_view) {
  int in_degree =
      graph_view.NumFanins(node, true);
  int out_degree =
      graph_view.NumFanouts(node, true);
  return in_degree * out_degree > in_degree + out_degree;
}
bool IsOutputPortRefValue(const NodeDef& node, int port_id,
                          const OpRegistryInterface& op_registry) {
  const OpRegistrationData* op_reg_data = nullptr;
  Status s = op_registry.LookUp(node.op(), &op_reg_data);
  if (s.ok()) {
    DataType output_type;
    s = OutputTypeForNode(node, op_reg_data->op_def, port_id, &output_type);
    if (s.ok() && IsRefType(output_type)) {
      return true;
    }
  }
  return false;
}
bool CanRemoveNode(const NodeDef& node, const GraphView& graph_view,
                   const absl::flat_hash_set<string>& function_names,
                   const OpRegistryInterface& op_registry) {
  if (IsNoOp(node) &&
      (node.input().empty() ||
       graph_view.NumFanouts(node, true) == 0)) {
    return true;
  }
  if (IsConstant(node) && node.input().empty() &&
      graph_view.NumFanouts(node, false) == 0) {
    return true;
  }
  if (RemovalIncreasesEdgeCount(node, graph_view)) {
    return false;
  }
  for (const auto input :
       graph_view.GetFanins(node, true)) {
    if (node.device() != input.node->device()) {
      return false;
    } else if (input.port_id == Graph::kControlSlot) {
      continue;
    } else if (function_names.find(input.node->op()) != function_names.end()) {
      return false;
    } else if (IsOutputPortRefValue(*input.node, input.port_id, op_registry)) {
      return false;
    }
  }
  for (const auto output :
       graph_view.GetFanouts(node, false)) {
    if (function_names.find(output.node->op()) != function_names.end()) {
      return false;
    }
  }
  return true;
}
void ForwardInputsInternal(
    const NodeDef& node,
    const absl::flat_hash_set<const NodeDef*>& nodes_to_delete,
    bool add_as_control, NodeDef* new_node,
    const absl::flat_hash_map<string, const NodeDef*>& optimized_nodes,
    const GraphView& graph_view) {
  auto itr = optimized_nodes.find(node.name());
  if (itr != optimized_nodes.end()) {
    for (const string& input : itr->second->input()) {
      *new_node->add_input() =
          add_as_control ? AsControlDependency(NodeName(input)) : input;
    }
    return;
  }
  for (const auto& input : node.input()) {
    const NodeDef* input_node = graph_view.GetNode(NodeName(input));
    if (input_node == nullptr) {
      *new_node->add_input() =
          add_as_control ? AsControlDependency(NodeName(input)) : input;
      continue;
    }
    if (nodes_to_delete.find(input_node) != nodes_to_delete.end()) {
      ForwardInputsInternal(*input_node, nodes_to_delete,
                            add_as_control || IsControlInput(input), new_node,
                            optimized_nodes, graph_view);
    } else {
      *new_node->add_input() =
          add_as_control ? AsControlDependency(NodeName(input)) : input;
    }
  }
}
void ForwardInputs(const NodeDef& original_node,
                   const absl::flat_hash_set<const NodeDef*>& nodes_to_delete,
                   NodeDef* new_node,
                   absl::flat_hash_map<string, const NodeDef*>* optimized_nodes,
                   const GraphView& graph_view) {
  ForwardInputsInternal(original_node, nodes_to_delete,
                        false, new_node, *optimized_nodes,
                        graph_view);
  if (!new_node->name().empty()) {
    (*optimized_nodes)[new_node->name()] = new_node;
  }
  int pos = 0;
  for (int i = 0; i < new_node->input_size(); ++i) {
    if (!IsControlInput(new_node->input(i))) {
      new_node->mutable_input()->SwapElements(pos, i);
      ++pos;
    }
  }
  DedupControlInputs(new_node);
}
absl::flat_hash_map<string, absl::flat_hash_set<int>> IdentityNTerminalPorts(
    const NodeMap& node_map, const std::vector<string>& terminal_nodes,
    int graph_size) {
  std::vector<string> to_visit;
  to_visit.reserve(graph_size);
  absl::flat_hash_set<string> visited(terminal_nodes.begin(),
                                      terminal_nodes.end());
  for (const string& terminal_node : terminal_nodes) {
    NodeDef* node = node_map.GetNode(terminal_node);
    if (node == nullptr) {
      continue;
    }
    for (const string& input : node->input()) {
      to_visit.push_back(input);
    }
  }
  absl::flat_hash_set<string> identity_n_fanouts;
  while (!to_visit.empty()) {
    string curr = to_visit.back();
    to_visit.pop_back();
    NodeDef* curr_node = node_map.GetNode(curr);
    if (curr_node == nullptr ||
        visited.find(curr_node->name()) != visited.end()) {
      continue;
    }
    if (IsIdentityN(*curr_node)) {
      if (identity_n_fanouts.find(curr) == identity_n_fanouts.end()) {
        identity_n_fanouts.emplace(curr);
        int pos = NodePositionIfSameNode(curr, curr_node->name());
        if (pos >= 0) {
          to_visit.push_back(curr_node->input(pos));
        }
        for (const string& input : curr_node->input()) {
          if (IsControlInput(input) &&
              identity_n_fanouts.find(input) == identity_n_fanouts.end()) {
            to_visit.push_back(input);
          }
        }
      }
    } else {
      for (const string& input : curr_node->input()) {
        to_visit.push_back(input);
      }
      visited.emplace(curr_node->name());
    }
  }
  absl::flat_hash_map<string, absl::flat_hash_set<int>> identity_n_ports;
  for (const auto& fanout : identity_n_fanouts) {
    int pos;
    string node_name = ParseNodeName(fanout, &pos);
    if (node_name.empty() || pos < 0) {  
      continue;
    }
    if (identity_n_ports.find(node_name) == identity_n_ports.end()) {
      identity_n_ports[node_name] = {pos};
    } else {
      identity_n_ports[node_name].emplace(pos);
    }
  }
  return identity_n_ports;
}
string NewIdentityFromIdentityN(int pos, const NodeDef& identity_n,
                                GraphDef* graph, NodeMap* node_map) {
  string new_node_name =
      strings::StrCat(identity_n.name(), "-", pos, "-grappler-ModelPruner");
  if (node_map->NodeExists(new_node_name)) {
    return "";
  }
  NodeDef* new_node = graph->add_node();
  Status status = NodeDefBuilder(new_node_name, "Identity")
                      .Input(identity_n.input(pos), 0,
                             identity_n.attr().at("T").list().type(pos))
                      .Device(identity_n.device())
                      .Finalize(new_node);
  if (!status.ok()) {
    return "";
  }
  node_map->AddNode(new_node->name(), new_node);
  node_map->AddOutput(NodeName(new_node->input(0)), new_node->name());
  return new_node->name();
}
Status RewriteIdentityNAndInputsOutputs(
    NodeDef* node, int num_non_control_inputs,
    const absl::flat_hash_set<int>& terminal_ports, GraphDef* graph,
    NodeMap* node_map) {
  struct NodeOutputUpdate {
    string input;
    string output;
  };
  absl::flat_hash_map<int, int> terminal_input_pos;
  absl::flat_hash_map<int, string> new_identities;
  int new_idx = 0;
  for (int i = 0; i < num_non_control_inputs; i++) {
    if (terminal_ports.find(i) != terminal_ports.end()) {
      terminal_input_pos[i] = new_idx++;
    } else {
      string identity = NewIdentityFromIdentityN(i, *node, graph, node_map);
      if (identity.empty()) {
        return errors::Internal(
            "Could not create Identity node from IdentityN node ", node->name(),
            " at port ", i);
      }
      new_identities[i] = identity;
    }
  }
  std::vector<NodeOutputUpdate> updates;
  for (NodeDef* output : node_map->GetOutputs(node->name())) {
    for (int i = 0; i < output->input_size(); i++) {
      string input = output->input(i);
      if (IsControlInput(input)) {
        continue;
      }
      TensorId input_tensor = ParseTensorName(input);
      if (input_tensor.node() == node->name()) {
        if (terminal_ports.find(input_tensor.index()) == terminal_ports.end()) {
          string new_identity = new_identities[input_tensor.index()];
          output->set_input(i, new_identity);
          updates.push_back({new_identity, output->name()});
        } else {
          int new_pos = terminal_input_pos[input_tensor.index()];
          string updated_input_name =
              new_pos > 0 ? strings::StrCat(node->name(), ":", new_pos)
                          : node->name();
          output->set_input(i, updated_input_name);
        }
      }
    }
  }
  for (const NodeOutputUpdate& update : updates) {
    node_map->AddOutput(update.input, update.output);
  }
  const int num_inputs = node->input_size();
  int curr_pos = 0;
  auto mutable_inputs = node->mutable_input();
  auto mutable_types =
      node->mutable_attr()->at("T").mutable_list()->mutable_type();
  for (int i = 0; i < num_non_control_inputs; i++) {
    if (terminal_input_pos.find(i) != terminal_input_pos.end()) {
      mutable_inputs->SwapElements(i, curr_pos);
      mutable_types->SwapElements(i, curr_pos);
      curr_pos++;
    }
  }
  mutable_types->Truncate(curr_pos);
  for (int i = num_non_control_inputs; i < num_inputs; i++) {
    mutable_inputs->SwapElements(i, curr_pos++);
  }
  mutable_inputs->DeleteSubrange(curr_pos, num_inputs - curr_pos);
  return absl::OkStatus();
}
Status SplitIdentityNInputs(GraphDef* graph,
                            const std::vector<string>& terminal_nodes,
                            bool* updated_graph) {
  NodeMap node_map(graph);
  for (auto const& terminal :
       IdentityNTerminalPorts(node_map, terminal_nodes, graph->node_size())) {
    NodeDef* node = node_map.GetNode(terminal.first);
    if (node == nullptr) {
      continue;
    }
    const int num_non_control_inputs = NumNonControlInputs(*node);
    const int terminal_second_size = terminal.second.size();
    if (node->attr().count("T") == 0 ||
        node->attr().at("T").list().type_size() != num_non_control_inputs ||
        terminal_second_size >= num_non_control_inputs) {
      continue;
    }
    TF_RETURN_IF_ERROR(RewriteIdentityNAndInputsOutputs(
        node, num_non_control_inputs, terminal.second, graph, &node_map));
    *updated_graph = true;
  }
  return absl::OkStatus();
}
}  
Status ModelPruner::Optimize(Cluster* cluster, const GrapplerItem& item,
                             GraphDef* optimized_graph) {
  const std::unordered_set<string> nodes_to_preserve = item.NodesToPreserve();
  std::unique_ptr<GraphDef> pruned_graph_release;
  GraphDef* pruned_graph;
  if (!nodes_to_preserve.empty()) {
    pruned_graph_release.reset(new GraphDef());
    pruned_graph = pruned_graph_release.get();
    pruned_graph->mutable_node()->Reserve(item.graph.node_size());
    std::vector<string> terminal_nodes(nodes_to_preserve.begin(),
                                       nodes_to_preserve.end());
    std::sort(terminal_nodes.begin(), terminal_nodes.end());
    TF_RETURN_IF_ERROR(
        SetTransitiveFaninGraph(item.graph, pruned_graph, terminal_nodes));
    bool did_split_identity_n = false;
    TF_RETURN_IF_ERROR(SplitIdentityNInputs(pruned_graph, terminal_nodes,
                                            &did_split_identity_n));
    if (did_split_identity_n) {
      GraphDef fanin_split_identity_n_graph;
      TF_RETURN_IF_ERROR(SetTransitiveFaninGraph(
          *pruned_graph, &fanin_split_identity_n_graph, terminal_nodes));
      pruned_graph->Swap(&fanin_split_identity_n_graph);
    }
    GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();
  } else {
    pruned_graph = const_cast<GraphDef*>(&item.graph);
  }
  GraphView graph_view(pruned_graph);
  absl::flat_hash_set<string> function_names;
  for (const auto& function : item.graph.library().function()) {
    function_names.insert(function.signature().name());
  }
  OpRegistryInterface* op_registry = OpRegistry::Global();
  absl::flat_hash_set<const NodeDef*> nodes_to_delete;
  for (int i = 0; i < pruned_graph->node_size(); ++i) {
    NodeDef* node = pruned_graph->mutable_node(i);
    DedupControlInputs(node);
    if (!IsTrivialOp(*node, graph_view)) {
      VLOG(3) << node->name() << " is not trivial.";
      continue;
    }
    if (nodes_to_preserve.find(node->name()) != nodes_to_preserve.end()) {
      continue;
    }
    if (CanRemoveNode(*node, graph_view, function_names, *op_registry)) {
      nodes_to_delete.insert(node);
    } else {
      VLOG(3) << node->name() << " cannot be removed";
    }
  }
  if (nodes_to_delete.empty() && nodes_to_preserve.empty()) {
    return errors::Aborted("Nothing to do.");
  }
  optimized_graph->Clear();
  *optimized_graph->mutable_library() = item.graph.library();
  *optimized_graph->mutable_versions() = item.graph.versions();
  if (nodes_to_delete.empty()) {
    optimized_graph->mutable_node()->Swap(pruned_graph->mutable_node());
    return absl::OkStatus();
  }
  const bool fetches_are_known = !item.fetch.empty();
  absl::flat_hash_map<string, const NodeDef*> optimized_nodes;
  optimized_graph->mutable_node()->Reserve(pruned_graph->node_size());
  for (const auto& node : pruned_graph->node()) {
    if (!fetches_are_known ||
        nodes_to_delete.find(&node) == nodes_to_delete.end()) {
      NodeDef* new_node = optimized_graph->add_node();
      *new_node = node;
      new_node->clear_input();
      ForwardInputs(node, nodes_to_delete, new_node, &optimized_nodes,
                    graph_view);
    }
  }
  VLOG(1) << "Pruned " << nodes_to_delete.size()
          << " nodes from the graph. The graph now contains "
          << optimized_graph->node_size() << " nodes.";
  if (optimized_graph->node_size() > item.graph.node_size()) {
    return errors::Internal("Pruning increased graph size.");
  }
  return absl::OkStatus();
}
}  
}  