#include "tensorflow/cc/framework/while_gradients.h"
#include <string>
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/while_loop.h"
namespace tensorflow {
namespace {
using ops::BodyGraphBuilderFn;
using ops::BuildWhileLoop;
using ops::CondGraphBuilderFn;
Output ToOutput(OutputTensor output_tensor) {
  return Output(const_cast<Node*>(output_tensor.node), output_tensor.index);
}
std::vector<Output> ToOutputVector(
    const std::vector<OutputTensor>& output_tensors) {
  const int n = output_tensors.size();
  std::vector<Output> result;
  result.reserve(n);
  for (int i = 0; i < n; ++i) result.push_back(ToOutput(output_tensors[i]));
  return result;
}
string BackPropFrameName(const string& forward_frame_name) {
  return strings::StrCat(forward_frame_name, "_backprop");
}
Status AddForwardLoopCounter(WhileContext* while_ctx, const Scope& scope,
                             Output* count) {
  Output zero = ops::Const(scope, 0, {});
  CondGraphBuilderFn cond_fn = [while_ctx](const Scope& scope,
                                           const std::vector<Output>& inputs,
                                           Output* output) {
    *output = ToOutput(while_ctx->cond_output());
    return absl::OkStatus();
  };
  BodyGraphBuilderFn body_fn = [](const Scope& scope,
                                  const std::vector<Output>& inputs,
                                  std::vector<Output>* outputs) {
    DCHECK_EQ(inputs.size(), 1);
    outputs->emplace_back(ops::Add(scope, inputs[0], 1));
    return scope.status();
  };
  std::vector<Output> outputs;
  TF_RETURN_IF_ERROR(BuildWhileLoop(scope, {zero}, cond_fn, body_fn,
                                    while_ctx->frame_name(), &outputs,
                                     false));
  *count = outputs[0];
  return absl::OkStatus();
}
Status AddBackPropLoopCounter(WhileContext* while_ctx, const Output& loop_count,
                              const Scope& scope,
                              Output* backprop_execution_pred) {
  CondGraphBuilderFn cond_fn = [](const Scope& scope,
                                  const std::vector<Output>& inputs,
                                  Output* output) {
    DCHECK_EQ(inputs.size(), 1);
    *output = ops::Greater(scope, inputs[0], 0);
    return scope.status();
  };
  BodyGraphBuilderFn body_fn = [](const Scope& scope,
                                  const std::vector<Output>& inputs,
                                  std::vector<Output>* outputs) {
    DCHECK_EQ(inputs.size(), 1);
    outputs->emplace_back(ops::Subtract(scope, inputs[0], 1));
    return scope.status();
  };
  string frame_name = BackPropFrameName(while_ctx->frame_name());
  std::vector<Output> outputs;
  TF_RETURN_IF_ERROR(BuildWhileLoop(
      scope, {loop_count}, cond_fn, body_fn, frame_name, &outputs,
       false, backprop_execution_pred));
  return absl::OkStatus();
}
Status AddWhileGradientLoop(WhileContext* while_ctx,
                            const std::vector<Output>& grad_inputs,
                            const Output& backprop_execution_pred,
                            const Scope& parent_scope,
                            std::vector<Output>* grad_outputs) {
  DCHECK_EQ(grad_inputs.size(), while_ctx->body_outputs().size());
  DCHECK_EQ(while_ctx->body_inputs().size(), while_ctx->body_outputs().size());
  Scope scope = parent_scope.NewSubScope("while");
  CondGraphBuilderFn cond_fn = [backprop_execution_pred](
                                   const Scope& scope,
                                   const std::vector<Output>& inputs,
                                   Output* output) {
    *output = backprop_execution_pred;
    return absl::OkStatus();
  };
  BodyGraphBuilderFn body_fn = [while_ctx](const Scope& scope,
                                           const std::vector<Output>& inputs,
                                           std::vector<Output>* outputs) {
    std::vector<Output> body_outputs =
        ToOutputVector(while_ctx->body_outputs());
    std::vector<Output> body_inputs = ToOutputVector(while_ctx->body_inputs());
    return AddSymbolicGradients(scope, body_outputs, body_inputs, inputs,
                                outputs);
  };
  string frame_name = BackPropFrameName(while_ctx->frame_name());
  TF_RETURN_IF_ERROR(BuildWhileLoop(scope, grad_inputs, cond_fn, body_fn,
                                    frame_name, grad_outputs,
                                     false));
  return absl::OkStatus();
}
}  
Status AddWhileLoopGradient(WhileContext* while_ctx, const Scope& scope,
                            const std::vector<Output>& grad_inputs,
                            std::vector<Output>* grad_outputs) {
  Output forward_loop_count;
  TF_RETURN_IF_ERROR(AddForwardLoopCounter(
      while_ctx, scope.NewSubScope("ForwardLoopCounter"), &forward_loop_count));
  Output backprop_counter_cond;
  TF_RETURN_IF_ERROR(AddBackPropLoopCounter(
      while_ctx, forward_loop_count, scope.NewSubScope("BackPropLoopCounter"),
      &backprop_counter_cond));
  return AddWhileGradientLoop(while_ctx, grad_inputs, backprop_counter_cond,
                              scope, grad_outputs);
}
}  