#include "tensorflow/core/framework/node_properties.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
namespace tensorflow {
Status NodeProperties::CreateFromNodeDef(
    NodeDef node_def, const OpRegistryInterface* op_registry,
    std::shared_ptr<const NodeProperties>* props) {
  const OpDef* op_def;
  TF_RETURN_IF_ERROR(op_registry->LookUpOpDef(node_def.op(), &op_def));
  DataTypeVector input_types;
  DataTypeVector output_types;
  TF_RETURN_IF_ERROR(
      InOutTypesForNode(node_def, *op_def, &input_types, &output_types));
  props->reset(new NodeProperties(op_def, std::move(node_def),
                                  std::move(input_types),
                                  std::move(output_types)));
  return absl::OkStatus();
}
}  