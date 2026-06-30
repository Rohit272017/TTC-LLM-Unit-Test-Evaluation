#include "tensorflow/compiler/mlir/tensorflow/translate/node_order.h"
#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
namespace tensorflow {
void TopologicalOrdering(
    const Graph& g, const std::function<void(Node*)>& emit,
    const std::function<std::string(Node*)>& get_grouping_key) {
  std::unordered_map<std::string, int> group_key_string_to_integer;
  absl::flat_hash_map<Node*, int> node_to_group;
  absl::flat_hash_map<Node*, int> remaining_incoming_nodes;
  absl::flat_hash_map<Node*, int> node_to_position;
  using Ready = std::vector<Node*>;
  std::vector<Ready> group_members_that_are_ready;
  std::set<int> groups_that_are_ready;
  int i = 0;
  DFS(
      g, [](Node*) {},
      [&](Node* n) {
        std::string group_key_string = get_grouping_key(n);
        auto entry = group_key_string_to_integer.try_emplace(
            group_key_string, group_key_string_to_integer.size());
        int group_key = entry.first->second;
        node_to_position[n] = i++;
        node_to_group[n] = group_key;
        if (entry.second) {
          group_members_that_are_ready.push_back({});
        }
        auto in_nodes = n->in_nodes();
        int num_incoming = std::distance(in_nodes.begin(), in_nodes.end());
        remaining_incoming_nodes[n] = num_incoming;
        if (num_incoming == 0) {
          group_members_that_are_ready[group_key].push_back(n);
          groups_that_are_ready.emplace(group_key);
        }
      },
      [](const Node* n1, const Node* n2) { return n1->name() < n2->name(); });
  assert(group_key_string_to_integer.size() ==
         group_members_that_are_ready.size());
  int num_nodes = remaining_incoming_nodes.size();
  int current_group = 0;
  for (int i = 0; i < num_nodes; i++) {
    if (groups_that_are_ready.find(current_group) ==
        groups_that_are_ready.end()) {
      current_group = *groups_that_are_ready.begin();
    }
    int size = group_members_that_are_ready[current_group].size();
    assert(size);
    Node* node = group_members_that_are_ready[current_group][--size];
    group_members_that_are_ready[current_group].pop_back();
    if (size == 0) {
      groups_that_are_ready.erase(current_group);
    }
    emit(node);
    auto out_nodes = node->out_nodes();
    std::vector<Node*> nodes_sorted(out_nodes.begin(), out_nodes.end());
    std::sort(nodes_sorted.begin(), nodes_sorted.end(), [&](Node* a, Node* b) {
      return node_to_position[a] < node_to_position[b];
    });
    for (Node* out : nodes_sorted) {
      remaining_incoming_nodes[out]--;
      if (remaining_incoming_nodes[out] == 0) {
        int group_key = node_to_group[out];
        if (group_members_that_are_ready[group_key].empty()) {
          groups_that_are_ready.emplace(group_key);
        }
        group_members_that_are_ready[group_key].push_back(out);
      }
    }
  }
}
}  