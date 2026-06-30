#include "tensorflow/core/grappler/utils/colocation.h"
#include <cstring>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/utils.h"
namespace tensorflow {
namespace grappler {
namespace {
string GetColocationGroupRoot(std::unordered_map<string, string>* map,
                              const string& node_name) {
  if (map->find(node_name) == map->end()) {
    map->insert({node_name, node_name});
    return node_name;
  }
  std::list<string> nodes_to_root;
  string cur = node_name;
  while ((*map)[cur] != cur) {
    nodes_to_root.push_back(cur);
    cur = (*map)[cur];
  }
  if (!nodes_to_root.empty()) {
    nodes_to_root.pop_back();
    for (const string& node : nodes_to_root) {
      (*map)[node] = cur;
    }
  }
  return cur;
}
void MergeColocationGroup(std::unordered_map<string, string>* map,
                          const string& left, const string& right) {
  if (map->find(left) == map->end() || map->find(right) == map->end()) {
    return;
  }
  if (left != right) {
    map->at(right) = left;
  }
}
}  
void ReassignColocation(GraphDef* graph) {
  constexpr char kClassAttr[] = "_class";
  constexpr char kColocPrefix[] = "loc:@";
  std::unordered_map<string, string> coloc_groups;
  NodeMap node_map(graph);
  for (const auto& node : graph->node()) {
    auto iter = node.attr().find(kClassAttr);
    if (iter != node.attr().end() && iter->second.has_list()) {
      for (const auto& str : iter->second.list().s()) {
        size_t pos = str.find(kColocPrefix);
        if (pos == 0) {
          string colocate_node = str.substr(pos + strlen(kColocPrefix));
          MergeColocationGroup(
              &coloc_groups, GetColocationGroupRoot(&coloc_groups, node.name()),
              GetColocationGroupRoot(&coloc_groups, colocate_node));
        }
      }
    }
  }
  for (const auto& pair : coloc_groups) {
    if (pair.first != pair.second) {
      NodeDef* node = node_map.GetNode(pair.first);
      if (node) {
        AttrValue new_value;
        new_value.mutable_list()->add_s(
            kColocPrefix + GetColocationGroupRoot(&coloc_groups, pair.first));
        node->mutable_attr()->erase(kClassAttr);
        node->mutable_attr()->insert({kClassAttr, new_value});
      }
    } else {
      NodeDef* node = node_map.GetNode(pair.first);
      if (node) {  
        node->mutable_attr()->erase(kClassAttr);
      }
    }
  }
}
}  
}  