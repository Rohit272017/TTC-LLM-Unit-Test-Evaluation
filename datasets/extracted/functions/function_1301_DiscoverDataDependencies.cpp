#include "tensorflow/core/graph/collective_order.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/graph/algorithm.h"
namespace tensorflow {
namespace {
Status DiscoverDataDependencies(
    const Graph* graph, std::vector<Node*>* collective_nodes,
    std::vector<int32>* instance_keys,
    absl::flat_hash_map<Node*, absl::flat_hash_set<int32>>* data_dependencies) {
  Status s;
  auto node_leave = [collective_nodes, instance_keys, data_dependencies,
                     &s](Node* node) {
    int32_t instance_key;
    bool enter_node =
        node->IsCollective() && node->type_string() == "CollectiveReduce";
    if (enter_node) {
      Status get_attr_status =
          GetNodeAttr(node->attrs(), "instance_key", &instance_key);
      s.Update(get_attr_status);
      collective_nodes->push_back(node);
      instance_keys->push_back(instance_key);
      VLOG(2) << "collective node " << node->DebugString();
    }
    data_dependencies->reserve(data_dependencies->size() + 1 +
                               node->out_edges().size());
    const auto& node_deps = (*data_dependencies)[node];
    for (const Edge* out_edge : node->out_edges()) {
      auto& child_deps = (*data_dependencies)[out_edge->dst()];
      child_deps.insert(node_deps.begin(), node_deps.end());
      if (enter_node && s.ok()) {
        child_deps.insert(instance_key);
      }
    }
  };
  ReverseDFS(*graph, nullptr, node_leave);
  return s;
}
Status CreateControlDependencies(
    const std::vector<Node*>& collective_nodes,
    const std::vector<int32>& instance_keys,
    absl::flat_hash_map<Node*, absl::flat_hash_set<int32>>* data_dependencies,
    absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>>* dependency_edges) {
  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>> all_paths;
  for (int i = 0; i < collective_nodes.size() - 1; i++) {
    if (!collective_nodes[i]->IsCollective() ||
        collective_nodes[i]->type_string() != "CollectiveReduce") {
      return errors::Internal("Unexpected node ",
                              collective_nodes[i]->DebugString());
    }
    const auto& deps_i = (*data_dependencies)[collective_nodes[i]];
    for (int j = i + 1; j < collective_nodes.size(); j++) {
      if (collective_nodes[i]->requested_device() !=
          collective_nodes[j]->requested_device()) {
        continue;
      }
      if (instance_keys[i] == instance_keys[j]) {
        return errors::Internal("Unexpected same instance_key ",
                                instance_keys[i],
                                " on 2 nodes with the same device ",
                                collective_nodes[i]->requested_device());
      }
      const auto& deps_j = (*data_dependencies)[collective_nodes[j]];
      if (deps_i.find(instance_keys[j]) == deps_i.end() &&
          deps_j.find(instance_keys[i]) == deps_j.end()) {
        int src_idx = instance_keys[i] > instance_keys[j] ? i : j;
        int dst_idx = instance_keys[i] > instance_keys[j] ? j : i;
        Node* src_node = collective_nodes[src_idx];
        Node* dst_node = collective_nodes[dst_idx];
        VLOG(1) << "Adding control dependency from node " << src_node->name()
                << " instance " << instance_keys[src_idx] << " to node "
                << dst_node->name() << " instance " << instance_keys[dst_idx];
        (*dependency_edges)[src_node].insert(dst_node);
        auto& src_paths = all_paths[src_node];
        src_paths.insert(dst_node);
        for (Node* downstream_node : all_paths[dst_node]) {
          src_paths.insert(downstream_node);
        }
      }
    }
  }
  for (int i = 0; i < collective_nodes.size(); ++i) {
    Node* node = collective_nodes[i];
    auto& neighbor_set = (*dependency_edges)[node];
    std::vector<Node*> neighbor_list(neighbor_set.begin(), neighbor_set.end());
    for (int j = 0; j < neighbor_list.size(); ++j) {
      Node* n1 = neighbor_list[j];
      if (n1 == nullptr) continue;
      auto& n1_paths = all_paths[n1];
      for (int k = 0; k < neighbor_list.size(); ++k) {
        Node* n2 = neighbor_list[k];
        if (j == k || n2 == nullptr) continue;
        if (n1_paths.find(n2) != n1_paths.end()) {
          neighbor_set.erase(n2);
          neighbor_list[k] = nullptr;
        }
      }
    }
  }
  return absl::OkStatus();
}
Status InsertControlDependencies(
    Graph* graph, GraphCollectiveOrder order_type,
    const absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>>&
        dependency_edges) {
  if (order_type == GraphCollectiveOrder::kEdges) {
    for (const auto& pair : dependency_edges) {
      Node* src_node = pair.first;
      for (Node* dst_node : pair.second) {
        graph->AddControlEdge(src_node, dst_node);
      }
    }
  } else if (order_type == GraphCollectiveOrder::kAttrs) {
    absl::flat_hash_map<Node*, absl::flat_hash_set<int32>> wait_for;
    for (const auto& pair : dependency_edges) {
      int32_t src_instance;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(pair.first->attrs(), "instance_key", &src_instance));
      for (Node* dst_node : pair.second) {
        wait_for[dst_node].insert(src_instance);
      }
    }
    for (const auto& pair : wait_for) {
      std::vector<int32> wait_for_list(pair.second.begin(), pair.second.end());
      pair.first->ClearAttr("wait_for");
      pair.first->AddAttr("wait_for", wait_for_list);
    }
  } else {
    return errors::Internal("Unexpected GraphCollectiveOrder type ",
                            static_cast<int>(order_type));
  }
  return absl::OkStatus();
}
}  
Status OrderCollectives(Graph* graph, GraphCollectiveOrder order_type) {
  std::vector<Node*> collective_nodes;
  std::vector<int32> instance_keys;
  absl::flat_hash_map<Node*, absl::flat_hash_set<int32>> data_dependencies;
  TF_RETURN_IF_ERROR(DiscoverDataDependencies(
      graph, &collective_nodes, &instance_keys, &data_dependencies));
  if (collective_nodes.empty()) return absl::OkStatus();
  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>> dependency_edges;
  TF_RETURN_IF_ERROR(CreateControlDependencies(
      collective_nodes, instance_keys, &data_dependencies, &dependency_edges));
  return InsertControlDependencies(graph, order_type, dependency_edges);
}
}  