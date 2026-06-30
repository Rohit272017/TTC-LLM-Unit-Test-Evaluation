#include "tensorflow/core/common_runtime/optimized_function_graph_info.h"
#include <memory>
#include <utility>
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/op.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
namespace tensorflow {
OptimizedFunctionGraph OptimizedFunctionGraphInfo::ToProto(
    const OptimizedFunctionGraphInfo& info) {
  OptimizedFunctionGraph proto;
  proto.set_name(info.name);
  GraphDef* function_graph_def = proto.mutable_function_graph();
  info.function_graph->ToGraphDef(function_graph_def);
  *function_graph_def->mutable_library() = info.lib_def.ToProto();
  *proto.mutable_ret_types() = {info.ret_types.begin(), info.ret_types.end()};
  proto.set_num_return_nodes(info.num_return_nodes);
  *proto.mutable_node_name_to_control_ret() = {
      info.node_name_to_control_ret.begin(),
      info.node_name_to_control_ret.end()};
  proto.set_optimization_time_usecs(info.optimization_duration_usecs);
  proto.set_source(info.optimization_source);
  return proto;
}
absl::StatusOr<OptimizedFunctionGraphInfo>
OptimizedFunctionGraphInfo::FromProto(OptimizedFunctionGraph&& proto) {
  FunctionLibraryDefinition lib_def(OpRegistry::Global());
  FunctionDefLibrary proto_library;
  std::swap(proto_library, *proto.mutable_function_graph()->mutable_library());
  TF_RETURN_IF_ERROR(lib_def.AddLibrary(std::move(proto_library)));
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  graph->mutable_flib_def()->set_default_registry(&lib_def);
  GraphConstructorOptions options;
  options.allow_internal_ops = true;
  options.expect_device_spec = true;
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(
      options, std::move(*proto.mutable_function_graph()), graph.get()));
  graph->mutable_flib_def()->set_default_registry(nullptr);
  graph->mutable_flib_def()->Clear();
  const int num_ret_types = proto.ret_types_size();
  DataTypeVector data_type_vector(num_ret_types);
  for (int i = 0; i < num_ret_types; ++i) {
    data_type_vector[i] = static_cast<DataType>(proto.ret_types().at(i));
  }
  return OptimizedFunctionGraphInfo(
      proto.name(), std::move(graph), std::move(lib_def),
      {proto.node_name_to_control_ret().begin(),
       proto.node_name_to_control_ret().end()},
      std::move(data_type_vector), proto.num_return_nodes(),
      proto.optimization_time_usecs(), proto.source());
}
}  