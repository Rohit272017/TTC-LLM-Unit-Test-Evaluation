#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/graph/graph_def_builder.h"
namespace tensorflow {
Status Var::AsGraphDef(GraphDefBuilder* builder, Node** out) const {
  Node* var = ops::SourceOp(
      "VarHandleOp",
      builder->opts()
          .WithAttr("dtype", tensor_.dtype())
          .WithAttr("shape", tensor_.shape())
          .WithAttr("shared_name", ResourceHandle::ANONYMOUS_NAME));
  Node* value = ops::SourceOp("Const", builder->opts()
                                           .WithAttr("dtype", tensor_.dtype())
                                           .WithAttr("value", tensor_));
  Node* assign =
      ops::BinaryOp("AssignVariableOp", var, value,
                    builder->opts().WithAttr("dtype", tensor_.dtype()));
  *out =
      ops::UnaryOp("Identity", var, builder->opts().WithControlInput(assign));
  return absl::OkStatus();
}
std::string Var::MakeRefCountingHandleName(int64_t resource_id) const {
  std::string handle_name = absl::StrFormat("%s%d", debug_name_, resource_id);
  return handle_name;
}
}  