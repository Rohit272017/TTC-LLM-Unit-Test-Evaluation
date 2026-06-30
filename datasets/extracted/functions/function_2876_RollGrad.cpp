#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/manip_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
namespace tensorflow {
namespace ops {
namespace {
Status RollGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
  auto shift = op.input(1);
  auto axis = op.input(2);
  auto grad_op = Roll(scope, grad_inputs[0], Neg(scope, shift), axis);
  grad_outputs->push_back(grad_op);
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("Roll", RollGrad);
}  
}  
}  