#include "tensorflow/core/grappler/utils/tpu.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
namespace tensorflow {
namespace grappler {
bool IsLegacyTPUBridgeGraphDef(const GraphDef& def) {
  for (const auto& node : def.node()) {
    if (node.op() == "TPUCompile" || node.op() == "TPUPartitionedCall") {
      return true;
    }
  }
  if (!def.has_library()) return false;
  for (const auto& function_def : def.library().function()) {
    for (const auto& node : function_def.node_def()) {
      if (node.op() == "TPUCompile" || node.op() == "TPUPartitionedCall") {
        return true;
      }
    }
  }
  return false;
}
}  
}  