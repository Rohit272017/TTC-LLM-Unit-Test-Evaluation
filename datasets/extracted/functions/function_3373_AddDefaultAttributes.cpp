#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"
namespace tensorflow {
namespace graph_transforms {
Status AddDefaultAttributes(const GraphDef& input_graph_def,
                            const TransformFuncContext& context,
                            GraphDef* output_graph_def) {
  std::unique_ptr<FunctionLibraryDefinition> flib_def(
      new FunctionLibraryDefinition(OpRegistry::Global(),
                                    input_graph_def.library()));
  *output_graph_def = input_graph_def;
  TF_RETURN_IF_ERROR(AddDefaultAttrsToGraphDef(output_graph_def, *flib_def, 0));
  return OkStatus();
}
REGISTER_GRAPH_TRANSFORM("add_default_attributes", AddDefaultAttributes);
}  
}  