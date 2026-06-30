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
Status SortByExecutionOrderWithUnusedContext(
    const GraphDef& input_graph_def, const TransformFuncContext& unused_context,
    GraphDef* output_graph_def) {
  return SortByExecutionOrder(input_graph_def, output_graph_def);
}
REGISTER_GRAPH_TRANSFORM("sort_by_execution_order",
                         SortByExecutionOrderWithUnusedContext);
}  
}  