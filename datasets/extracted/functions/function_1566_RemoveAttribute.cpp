#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/fold_constants_lib.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"
namespace tensorflow {
namespace graph_transforms {
Status RemoveAttribute(const GraphDef& input_graph_def,
                       const TransformFuncContext& context,
                       GraphDef* output_graph_def) {
  if (!context.params.count("attribute_name") ||
      (context.params.at("attribute_name").size() != 1)) {
    return errors::InvalidArgument(
        "remove_attribute expects exactly one 'attribute_name' "
        "argument, e.g. remove_attribute(op_name=Mul, attribute_name=foo)");
  }
  string op_name;
  if (context.params.count("op_name")) {
    if (context.params.at("op_name").size() != 1) {
      return errors::InvalidArgument(
          "remove_attribute expects a single op_name argument, but found ",
          context.params.at("op_name").size());
    }
    op_name = context.params.at("op_name")[0];
  } else {
    op_name = "*";
  }
  const string attribute_name = context.params.at("attribute_name")[0];
  output_graph_def->Clear();
  for (const NodeDef& node : input_graph_def.node()) {
    NodeDef* new_node = output_graph_def->mutable_node()->Add();
    *new_node = node;
    if (((op_name == "*") || (op_name == node.op())) &&
        (node.attr().count(attribute_name))) {
      new_node->mutable_attr()->erase(attribute_name);
    }
  }
  return OkStatus();
}
REGISTER_GRAPH_TRANSFORM("remove_attribute", RemoveAttribute);
}  
}  