#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"
namespace tensorflow {
namespace graph_transforms {
Status SetDevice(const GraphDef& input_graph_def,
                 const TransformFuncContext& context,
                 GraphDef* output_graph_def) {
  string new_device;
  TF_RETURN_IF_ERROR(context.GetOneStringParameter("device", "", &new_device));
  bool if_default;
  TF_RETURN_IF_ERROR(
      context.GetOneBoolParameter("if_default", false, &if_default));
  output_graph_def->Clear();
  for (const NodeDef& node : input_graph_def.node()) {
    NodeDef* new_node = output_graph_def->mutable_node()->Add();
    *new_node = node;
    if (!if_default || (node.device().empty())) {
      new_node->set_device(new_device);
    }
  }
  return OkStatus();
}
REGISTER_GRAPH_TRANSFORM("set_device", SetDevice);
}  
}  