#include "tensorflow/core/grappler/optimizers/debug_stripper.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"
namespace tensorflow {
namespace grappler {
Status DebugStripper::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* output) {
  bool can_optimize = false;
  for (const NodeDef& node : item.graph.node()) {
    if (IsAssert(node) || IsCheckNumerics(node) || IsPrint(node)) {
      can_optimize = true;
      break;
    }
  }
  if (!can_optimize) {
    return errors::Aborted("Nothing to do.");
  }
  *output = item.graph;
  for (NodeDef& node : *output->mutable_node()) {
    if (IsAssert(node) || node.op() == "PrintV2") {
      node.set_op("NoOp");
      EraseRegularNodeAttributes(&node);
      for (string& inp : *node.mutable_input()) {
        if (!IsControlInput(inp)) {
          inp = AsControlDependency(NodeName(inp));
        }
      }
    } else if (IsCheckNumerics(node) || node.op() == "Print") {
      node.set_op("Identity");
      protobuf::Map<string, AttrValue> new_attr;
      if (node.attr().find("T") != node.attr().end()) {
        new_attr.insert({"T", node.attr().at("T")});
      }
      node.mutable_attr()->swap(new_attr);
      for (int i = 1, end = node.input_size(); i < end; ++i) {
        if (!IsControlInput(node.input(i))) {
          *node.mutable_input(i) = AsControlDependency(NodeName(node.input(i)));
        }
      }
    }
  }
  return absl::OkStatus();
}
}  
}  