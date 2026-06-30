#include "tensorflow/core/common_runtime/replicate_per_replica_nodes.h"
#include <algorithm>
#include <queue>
#include "absl/strings/str_cat.h"
#include "tensorflow/core/common_runtime/optimize_cross_host_control_deps.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/platform/errors.h"
namespace tensorflow {
namespace {
constexpr int kOptimizeCrossHostEdgesTheshold = 8;
constexpr int kOptimizeCrossHostDataEdgesTheshold = 2;
class ReplicateHelper {
 public:
  Status InitializeNode(const Node* node, int num_allowed_devices) {
    if (replicated_nodes_map_.find(node) != replicated_nodes_map_.end()) {
      return errors::InvalidArgument("Node ", node->name(),
                                     " has been replicated.");
    }
    std::vector<Node*> replicated_nodes(num_allowed_devices, nullptr);
    replicated_nodes_map_.emplace(node, std::move(replicated_nodes));
    return absl::OkStatus();
  }
  Status ReplicateNode(const Node* node,
                       const std::vector<string>& allowed_devices,
                       int allowed_device_index, Graph* graph) {
    auto& replicated_nodes = replicated_nodes_map_.at(node);
    if (replicated_nodes[allowed_device_index] != nullptr) {
      return absl::OkStatus();
    }
    const auto& device = allowed_devices.at(allowed_device_index);
    NodeDef node_def = node->def();
    const string suffix = strings::StrCat("/R", allowed_device_index);
    node_def.set_name(graph->NewName(strings::StrCat(node_def.name(), suffix)));
    TF_ASSIGN_OR_RETURN(Node * replicated_node, graph->AddNode(node_def));
    replicated_node->set_assigned_device_name(device);
    if (replicated_node->IsArg()) {
      replicated_node->AddAttr("sub_index", allowed_device_index);
    }
    replicated_nodes[allowed_device_index] = replicated_node;
    return absl::OkStatus();
  }
  void ReplicateFromRegularDeviceToCompositeDevice(const Edge* edge,
                                                   Graph* graph) const {
    Node* src = edge->src();
    const std::vector<Node*>& dst_replicated_nodes =
        replicated_nodes_map_.at(edge->dst());
    for (Node* dst : dst_replicated_nodes) {
      if (dst == nullptr) {
        continue;
      }
      graph->AddEdge(src, edge->src_output(), dst, edge->dst_input());
    }
  }
  Status ReplicateFromCompositeDeviceToCompositeDevice(
      const Edge* edge, const std::vector<string>& allowed_devices,
      Graph* graph) {
    const std::vector<Node*>& src_replicated_nodes =
        replicated_nodes_map_.at(edge->src());
    const std::vector<Node*>& dst_replicated_nodes =
        replicated_nodes_map_.at(edge->dst());
    if (src_replicated_nodes.size() != dst_replicated_nodes.size()) {
      return errors::InvalidArgument(
          "Nodes assigned to the same composite device should have the "
          "same number of replicated nodes. Found an edge from node ",
          edge->src()->name(), " (", src_replicated_nodes.size(),
          " replicated nodes) to node ", edge->dst()->name(), " (",
          dst_replicated_nodes.size(), " replicated nodes).");
    }
    for (int i = 0; i < src_replicated_nodes.size(); ++i) {
      Node* dst = dst_replicated_nodes.at(i);
      if (dst == nullptr) {
        continue;
      }
      TF_RETURN_IF_ERROR(ReplicateNode(edge->src(), allowed_devices, i, graph));
      graph->AddEdge(src_replicated_nodes.at(i), edge->src_output(), dst,
                     edge->dst_input());
    }
    return absl::OkStatus();
  }
  Status ReplicateFromCompositeDeviceToRegularDevice(
      const Edge* edge, const std::vector<string>& allowed_devices,
      Graph* graph) {
    const std::vector<Node*>& src_replicated_nodes =
        replicated_nodes_map_.at(edge->src());
    Node* dst = edge->dst();
    const string& dst_device = dst->assigned_device_name();
    bool found_src_node = false;
    for (int i = 0; i < allowed_devices.size(); ++i) {
      if (allowed_devices.at(i) == dst_device) {
        TF_RETURN_IF_ERROR(
            ReplicateNode(edge->src(), allowed_devices, i, graph));
        graph->AddEdge(src_replicated_nodes.at(i), edge->src_output(), dst,
                       edge->dst_input());
        found_src_node = true;
        break;
      }
    }
    if (!found_src_node) {
      for (int i = 0; i < allowed_devices.size(); ++i) {
        TF_RETURN_IF_ERROR(
            ReplicateNode(edge->src(), allowed_devices, i, graph));
      }
      if (edge->IsControlEdge()) {
        for (Node* replicated_node : src_replicated_nodes) {
          graph->AddControlEdge(replicated_node, dst,
                                true);
        }
        return absl::OkStatus();
      }
      if (edge->src()->type_string() == "_Arg") {
        NodeDefBuilder pack_builder(
            graph->NewName(absl::StrCat(edge->src()->name(), "/Packed")),
            "Pack");
        const int num_replicas = src_replicated_nodes.size();
        pack_builder.Attr("N", num_replicas);
        const DataType dtype = edge->src()->output_type(edge->src_output());
        pack_builder.Attr("T", dtype);
        std::vector<NodeDefBuilder::NodeOut> inputs;
        inputs.reserve(src_replicated_nodes.size());
        for (Node* replicated_node : src_replicated_nodes) {
          inputs.emplace_back(NodeDefBuilder::NodeOut{
              replicated_node->name(), edge->src_output(), dtype});
        }
        pack_builder.Input(inputs);
        NodeDef pack_def;
        TF_RETURN_IF_ERROR(pack_builder.Finalize(&pack_def));
        TF_ASSIGN_OR_RETURN(Node * pack_node, graph->AddNode(pack_def));
        pack_node->set_assigned_device_name(dst->assigned_device_name());
        for (int i = 0; i < src_replicated_nodes.size(); ++i) {
          graph->AddEdge(src_replicated_nodes[i], edge->src_output(), pack_node,
                         i);
        }
        graph->AddEdge(pack_node, 0, dst, edge->dst_input());
      } else {
        return errors::InvalidArgument(
            "Dst node should be assigned to an allowed device. Found an "
            "edge from node ",
            edge->src()->name(), " assigned to ",
            edge->src()->assigned_device_name(), " to node ", dst->name(),
            " assigned to ", dst_device);
      }
    }
    return absl::OkStatus();
  }
 private:
  absl::flat_hash_map<const Node*, std::vector<Node*>> replicated_nodes_map_;
};
Status ReplicateNodesAndEdges(const std::vector<string>& allowed_devices,
                              absl::flat_hash_map<Node*, int>* cluster_nodes,
                              ReplicateHelper* helper, Graph* graph) {
  std::queue<Node*> nodes_ready_to_delete;
  for (auto& pair : *cluster_nodes) {
    Node* node = pair.first;
    for (const Edge* edge : node->out_edges()) {
      Node* dst = edge->dst();
      if (dst->assigned_device_name() != node->assigned_device_name()) {
        TF_RETURN_IF_ERROR(helper->ReplicateFromCompositeDeviceToRegularDevice(
            edge, allowed_devices, graph));
        --pair.second;
      }
    }
    if (cluster_nodes->at(node) == 0) {
      nodes_ready_to_delete.push(node);
    }
  }
  while (!nodes_ready_to_delete.empty()) {
    Node* node = nodes_ready_to_delete.front();
    nodes_ready_to_delete.pop();
    for (const Edge* edge : node->in_edges()) {
      Node* src = edge->src();
      if (src->assigned_device_name() != node->assigned_device_name()) {
        helper->ReplicateFromRegularDeviceToCompositeDevice(edge, graph);
      } else {
        TF_RETURN_IF_ERROR(
            helper->ReplicateFromCompositeDeviceToCompositeDevice(
                edge, allowed_devices, graph));
        if (--(*cluster_nodes)[src] == 0) {
          nodes_ready_to_delete.push(src);
        }
      }
    }
    cluster_nodes->erase(node);
    graph->RemoveNode(node);
  }
  return absl::OkStatus();
}
}  
Status ReplicatePerReplicaNodesInFunctionGraph(
    const absl::flat_hash_map<string, const std::vector<string>*>&
        composite_devices,
    Graph* graph) {
  VLOG(1) << "Starting ReplicatePerReplicaNodesInFunctionGraph";
  VLOG(1) << "Graph #nodes " << graph->num_nodes() << " #edges "
          << graph->num_edges();
  std::set<string> composite_device_names;
  for (const auto& it : composite_devices) {
    composite_device_names.insert(it.first);
  }
  absl::flat_hash_map<string, absl::flat_hash_map<Node*, int>>
      composite_device_to_cluster_nodes;
  for (Node* n : graph->op_nodes()) {
    if (composite_device_names.find(n->assigned_device_name()) !=
        composite_device_names.end()) {
      composite_device_to_cluster_nodes[n->assigned_device_name()].emplace(
          n, n->out_edges().size());
    }
  }
  if (composite_device_to_cluster_nodes.empty()) {
    VLOG(1) << "No nodes with composiste device found.";
    return absl::OkStatus();
  }
  for (auto& it : composite_device_to_cluster_nodes) {
    const std::vector<string>& allowed_devices =
        *composite_devices.at(it.first);
    if (allowed_devices.empty()) {
      return errors::InvalidArgument("No allowed device of composite device: ",
                                     it.first);
    }
    absl::flat_hash_map<Node*, int>& cluster_nodes = it.second;
    if (allowed_devices.size() == 1) {
      for (const auto& pair : it.second) {
        Node* n = pair.first;
        n->set_assigned_device_name(allowed_devices.at(0));
        if (n->IsArg()) {
          n->AddAttr("sub_index", 0);
        }
      }
      continue;
    }
    ReplicateHelper helper;
    for (const auto& pair : cluster_nodes) {
      TF_RETURN_IF_ERROR(
          helper.InitializeNode(pair.first, allowed_devices.size()));
    }
    TF_RETURN_IF_ERROR(ReplicateNodesAndEdges(allowed_devices, &cluster_nodes,
                                              &helper, graph));
    if (!cluster_nodes.empty()) {
      return errors::InvalidArgument(
          "There are still ", cluster_nodes.size(),
          " nodes on CompositiveDevice ",
          cluster_nodes.begin()->first->assigned_device_name());
    }
  }
  TF_RETURN_IF_ERROR(OptimizeCrossHostControlOutputEdges(
      graph, kOptimizeCrossHostEdgesTheshold));
  TF_RETURN_IF_ERROR(OptimizeCrossHostControlInputEdges(
      graph, kOptimizeCrossHostEdgesTheshold));
  TF_RETURN_IF_ERROR(OptimizeCrossHostDataOutputEdges(
      graph, kOptimizeCrossHostDataEdgesTheshold));
  VLOG(1) << "Finished ReplicatePerReplicaNodesInFunctionGraph";
  VLOG(1) << "Graph #nodes " << graph->num_nodes() << " #edges "
          << graph->num_edges();
  return absl::OkStatus();
}
}  