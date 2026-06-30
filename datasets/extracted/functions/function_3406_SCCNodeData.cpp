#include "tensorflow/core/grappler/utils/scc.h"
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
namespace tensorflow {
namespace grappler {
struct SCCNodeData {
  SCCNodeData()
      : node(nullptr),
        index(-1),
        lowlink(-1),
        onstack(false),
        caller(nullptr),
        caller_loop_location(-1) {}
  void ResetStack(int new_index, SCCNodeData* new_caller) {
    index = new_index;
    lowlink = new_index;
    onstack = true;
    caller = new_caller;
    caller_loop_location = 0;
  }
  const NodeDef* node;
  int index;
  int lowlink;
  bool onstack;
  std::vector<SCCNodeData*> children;
  SCCNodeData* caller;       
  int caller_loop_location;  
};
void StrongConnect(SCCNodeData* v, std::stack<SCCNodeData*>* stack, int* index,
                   std::unordered_map<const NodeDef*, int>* components,
                   int* scc_index) {
  v->ResetStack(*index , nullptr );
  ++*index;
  stack->push(v);
  v->caller = nullptr;
  v->caller_loop_location = 0;
  SCCNodeData* last = v;
  while (true) {
    if (last->caller_loop_location < last->children.size()) {
      SCCNodeData* w = last->children[last->caller_loop_location];
      ++(last->caller_loop_location);  
      if (w->index == -1) {
        w->ResetStack(*index , last );
        ++*index;
        stack->push(w);
        last = w;
      } else if (w->onstack == true) {
        last->lowlink = std::min(last->lowlink, w->index);
      }
    } else {
      if (last->lowlink == last->index) {
        SCCNodeData* top;
        while (true) {
          top = stack->top();
          stack->pop();
          top->onstack = false;
          (*components)[top->node] = *scc_index;
          if (top == last) {
            break;
          }
        }
        ++*scc_index;
      }
      SCCNodeData* next_last = last->caller;
      if (next_last == nullptr) {
        break;
      } else {
        next_last->lowlink = std::min(next_last->lowlink, last->lowlink);
        last = next_last;
      }
    }
  }
}
void StronglyConnectedComponents(
    const GraphDef& graph, std::unordered_map<const NodeDef*, int>* components,
    int* num_components) {
  std::stack<SCCNodeData*> stack;
  std::unordered_map<string, SCCNodeData*> name_to_data;
  std::vector<SCCNodeData> node_data_container;
  node_data_container.reserve(graph.node_size());
  std::unordered_map<const NodeDef*, SCCNodeData*> node_to_data;
  for (const NodeDef& node : graph.node()) {
    SCCNodeData node_data;
    node_data.node = &node;
    node_data_container.push_back(node_data);
    name_to_data[node.name()] = &(*node_data_container.rbegin());
    node_to_data[&node] = &(*node_data_container.rbegin());
  }
  for (const NodeDef& node : graph.node()) {
    for (const string& input : node.input()) {
      auto it = name_to_data.find(NodeName(input));
      if (it != name_to_data.end()) {
        it->second->children.push_back(node_to_data[&node]);
      }
    }
  }
  components->clear();
  *num_components = 0;
  int index = 0;
  for (auto& v : node_data_container) {
    if (v.index == -1) {
      StrongConnect(&v, &stack, &index, components, num_components);
    }
  }
  std::vector<int> counts_per_component(*num_components, 0);
  for (auto& component : *components) {
    DCHECK(component.second >= 0);
    DCHECK(component.second < *num_components);
    counts_per_component[component.second]++;
  }
  bool has_single_element_component = false;
  for (auto& component : *components) {
    if (counts_per_component[component.second] == 1) {
      component.second = -1;
      (*num_components)--;
      has_single_element_component = true;
    }
  }
  if (has_single_element_component) {
    (*num_components) += 1;
  }
}
int IdentifyLoops(const GraphDef& graph,
                  std::unordered_map<const NodeDef*, std::vector<int>>* loops) {
  int num_components = 0;
  std::unordered_map<const NodeDef*, int> components;
  StronglyConnectedComponents(graph, &components, &num_components);
  if (num_components <= 1) {
    if (!components.empty() && components.begin()->second == -1) {
      return 0;
    }
  }
  std::unordered_map<int, std::vector<const NodeDef*>> component_ids;
  for (const auto it : components) {
    int id = it.second;
    if (id < 0) {
      continue;
    }
    component_ids[id].push_back(it.first);
  }
  int loop_id = 0;
  for (const auto& component : component_ids) {
    const std::vector<const NodeDef*>& component_nodes = component.second;
    std::vector<std::pair<NodeDef*, string>> next_iter_nodes;
    GraphDef subgraph;
    std::unordered_map<const NodeDef*, const NodeDef*> subgraph_mapping;
    for (const auto& component_node : component_nodes) {
      NodeDef* node = subgraph.add_node();
      *node = *component_node;
      subgraph_mapping[node] = component_node;
      if (IsNextIteration(*node)) {
        CHECK_EQ(1, node->input_size());
        next_iter_nodes.emplace_back(node, node->input(0));
      }
    }
    if (next_iter_nodes.size() == 1) {
      for (const auto& component_node : component_nodes) {
        (*loops)[component_node].push_back(loop_id);
      }
      ++loop_id;
    } else {
      for (int i = 0; i < next_iter_nodes.size(); ++i) {
        for (int j = 0; j < next_iter_nodes.size(); ++j) {
          next_iter_nodes[j].first->clear_input();
          if (i == j) {
            *next_iter_nodes[j].first->add_input() = next_iter_nodes[j].second;
          }
        }
        int num_components = 0;
        std::unordered_map<const NodeDef*, int> components;
        StronglyConnectedComponents(subgraph, &components, &num_components);
        CHECK_GE(num_components, 1);
        for (const auto it : components) {
          int id = it.second;
          if (id < 0) {
            continue;
          }
          (*loops)[subgraph_mapping[it.first]].push_back(loop_id);
        }
        ++loop_id;
      }
    }
  }
  return loop_id;
}
}  
}  