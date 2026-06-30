#include "tensorflow/core/graph/regularization/simple_delete.h"
#include <cstdint>
#include <string>
#include "absl/status/statusor.h"
#include "absl/strings/strip.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/regularization/util.h"
#include "tensorflow/core/grappler/op_types.h"
namespace tensorflow::graph_regularization {
namespace {
void RegularizeNodes(GraphDef* graph_def) {
  for (NodeDef& node : *graph_def->mutable_node()) {
    if (grappler::IsPartitionedCall(node) ||
        grappler::IsStatefulPartitionedCall(node)) {
      std::string function_name = node.attr().find("f")->second.func().name();
      absl::StatusOr<int64_t> uid = GetSuffixUID(function_name);
      if (uid.ok()) {
        node.mutable_attr()->find("f")->second.mutable_func()->set_name(
            std::string(
                absl::StripSuffix(function_name, std::to_string(*uid))));
      }
      auto node_config_proto = node.mutable_attr()->find("config_proto");
      if (node_config_proto != node.attr().end()) {
        node_config_proto->second.mutable_s()->erase();
      }
    }
    if (grappler::IsConstant(node)) {
      if (node.attr().at("dtype").type() == DT_STRING) {
        node.mutable_attr()->find("value")->second.clear_value();
      }
    }
  }
}
}  
void SimpleDelete(GraphDef& graph_def) {
  RegularizeNodes(&graph_def);
  graph_def.mutable_library()->Clear();
  graph_def.mutable_versions()->Clear();
}
}  