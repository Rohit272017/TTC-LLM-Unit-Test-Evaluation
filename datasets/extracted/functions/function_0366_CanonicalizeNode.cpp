#include "tensorflow/core/grappler/utils/canonicalizer.h"
#include <algorithm>
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
namespace tensorflow {
namespace grappler {
void CanonicalizeNode(NodeDef* node) {
  if (node->input_size() < 2) return;
  int index = 0;
  for (; index < node->input_size(); ++index) {
    if (IsControlInput(node->input(index))) {
      break;
    }
  }
  auto* input = node->mutable_input();
  if (IsCommutative(*node) && index > 0) {
    std::sort(input->begin(), input->begin() + index);
  }
  if (index < node->input_size()) {
    std::sort(input->begin() + index, input->end());
    input->erase(std::unique(input->begin() + index, input->end()),
                 input->end());
  }
}
void CanonicalizeGraph(GraphDef* graph) {
  for (int i = 0; i < graph->node_size(); ++i) {
    CanonicalizeNode(graph->mutable_node(i));
  }
}
void CompressConstants(GraphDef* graph) {
  for (int i = 0; i < graph->node_size(); ++i) {
    NodeDef* node = graph->mutable_node(i);
    if ((IsConstant(*node) || IsHostConstant(*node)) &&
        HasNodeAttr(*node, "value")) {
      AttrValue& attr_val = (*node->mutable_attr())["value"];
      if (attr_val.has_tensor()) {
        tensor::CompressTensorProtoInPlace(attr_val.mutable_tensor());
      }
    }
  }
}
}  
}  