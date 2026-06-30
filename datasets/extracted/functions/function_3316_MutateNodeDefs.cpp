#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_GRAPH_DEF_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_GRAPH_DEF_H_
#include <type_traits>
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
namespace stablehlo::quantization {
template <typename FuncT, typename = std::enable_if_t<std::is_invocable_r_v<
                              void, FuncT, tensorflow::NodeDef&>>>
void MutateNodeDefs(tensorflow::GraphDef& graph_def, FuncT&& func) {
  for (tensorflow::NodeDef& node_def : *graph_def.mutable_node()) {
    func(node_def);
  }
  for (tensorflow::FunctionDef& function_def :
       *graph_def.mutable_library()->mutable_function()) {
    for (tensorflow::NodeDef& node_def : *function_def.mutable_node_def()) {
      func(node_def);
    }
  }
}
}  
#endif  