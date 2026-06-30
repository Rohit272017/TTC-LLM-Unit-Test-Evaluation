#include "tensorflow/cc/ops/data_flow_ops.h"
#include "tensorflow/cc/ops/data_flow_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"
namespace tensorflow {
namespace ops {
namespace {
REGISTER_NO_GRADIENT_OP("Queue");
REGISTER_NO_GRADIENT_OP("QueueEnqueue");
REGISTER_NO_GRADIENT_OP("QueueEnqueueMany");
REGISTER_NO_GRADIENT_OP("QueueDequeue");
REGISTER_NO_GRADIENT_OP("QueueDequeueMany");
REGISTER_NO_GRADIENT_OP("QueueDequeueUpTo");
REGISTER_NO_GRADIENT_OP("QueueClose");
REGISTER_NO_GRADIENT_OP("QueueSize");
REGISTER_NO_GRADIENT_OP("Stack");
REGISTER_NO_GRADIENT_OP("StackPush");
REGISTER_NO_GRADIENT_OP("StackPop");
REGISTER_NO_GRADIENT_OP("StackClose");
REGISTER_NO_GRADIENT_OP("GetSessionHandle");
REGISTER_NO_GRADIENT_OP("GetSessionHandleV2");
REGISTER_NO_GRADIENT_OP("GetSessionTensor");
REGISTER_NO_GRADIENT_OP("DeleteSessionTensor");
Status DynamicPartitionGrad(const Scope& scope, const Operation& op,
                            const std::vector<Output>& grad_inputs,
                            std::vector<Output>* grad_outputs) {
  auto data = op.input(0);
  auto partitions = op.input(1);
  int32_t num_partitions;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "num_partitions", &num_partitions));
  auto partitions_shape = Shape(scope, partitions);
  auto zero = Const(scope, 0);
  auto one = Const(scope, 1);
  auto original_indices = Reshape(
      scope, Range(scope, zero, Prod(scope, partitions_shape, zero), one),
      partitions_shape);
  auto partitioned_indices =
      DynamicPartition(scope, original_indices, partitions, num_partitions);
  auto reconstructed =
      DynamicStitch(scope, partitioned_indices.outputs, grad_inputs);
  grad_outputs->push_back(Reshape(scope, reconstructed, Shape(scope, data)));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("DynamicPartition", DynamicPartitionGrad);
Status DynamicStitchGrad(const Scope& scope, const Operation& op,
                         const std::vector<Output>& grad_inputs,
                         std::vector<Output>* grad_outputs) {
  int32_t num_values = op.num_inputs() / 2;
  for (int32_t i = 0; i < num_values; i++) {
    grad_outputs->push_back(NoGradient());
  }
  for (int32_t i = 0; i < num_values; i++) {
    auto index = op.input(i);
    if (index.type() != DT_INT32) {
      index = Cast(scope, index, DT_INT32);
    }
    grad_outputs->push_back(Gather(scope, grad_inputs[0], index));
  }
  return scope.status();
}
REGISTER_GRADIENT_OP("DynamicStitch", DynamicStitchGrad);
REGISTER_GRADIENT_OP("ParallelDynamicStitch", DynamicStitchGrad);
}  
}  
}  